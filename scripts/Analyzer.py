import cv2
import numpy as np
from io import BytesIO
import streamlit as st
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
from scripts.config import KEYPOINT
from scripts.utils import calculate_circle_center_cords

plt.rcParams.update({
    "axes.facecolor":    (0.054, 0.066, 0.090, 1.0),  # same as streamlit dark style color
    "savefig.facecolor": (0.054, 0.066, 0.090, 1.0),  # same as streamlit dark style color
})


class Analyzer:
    """
    This class used to make analysis based on predictions and segments information.

    Attributes:
        segments_df: (pd.Dataframe): dataframe of segments information
        first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
        num_frames (int): number of frames in the video
        frames_per_second (float): number of frames per second
    """

    def __init__(self, video, segments_df, first_image):
        """
        initialize analyzer class with streamlit widgets and markdowns

        args:
            video (cv2.VideoCapture): Video file to process.
            segments_df (pd.Dataframe): dataframe of segments information
            first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
        """

        st.markdown("<h3 style='text-align: center; color: #FFB266;'>Behavior report</h3>", unsafe_allow_html=True)
        self.img_placeholder = st.empty()

        self.segments_df = segments_df
        self.first_image = first_image
        self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_per_second = video.get(cv2.CAP_PROP_FPS)

    def draw_tracked_road(self, predictions):
        """Draw the entire route covered by the mouse"""
        self.first_image = self.first_image.resize((704, 396))
        draw = ImageDraw.Draw(self.first_image, "RGBA")

        for keypoint_x, keypoint_y in predictions:
            draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                          (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                         outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        self.img_placeholder.image(self.first_image)

    def _count_elapsed_n_frames(self, segment, predictions):
        """count quantity of frames when mouse is in segment"""
        x, y = predictions[:, 0], predictions[:, 1]

        if segment["type"] == "rect":
            x1, y1 = segment["left"], segment["top"]
            x2, y2 = segment["left"]+segment["width"], segment["top"]+segment["height"]
            is_in_segment = (x > x1) & (x < x2) & (y > y1) & (y < y2)
        else:
            circle_x, circle_y = calculate_circle_center_cords(segment)
            rad = segment["radius"]
            # Compare radius of circle with distance of its center from given point
            is_in_segment = (x - circle_x) ** 2 + (y - circle_y) ** 2 <= rad ** 2

        in_segment = sum(is_in_segment)
        return in_segment

    def count_elapsed_time_in_segments(self, predictions):
        if self.segments_df.empty:
            return

        predictions = np.array(predictions)
        self.segments_df['elapsed_n_frames'] = self.segments_df.apply(
            lambda segment: self._count_elapsed_n_frames(segment, predictions), axis=1)
        self.segments_df['elapsed_sec'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.frames_per_second)
        self.segments_df['elapsed_sec%'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.num_frames*100)

    def show_elapsed_time_in_segments(self, predictions):
        """count elapsed time in each segment and plot bars"""
        self.count_elapsed_time_in_segments(predictions)
        st.dataframe(self.segments_df[["type", "elapsed_n_frames", "elapsed_sec", "elapsed_sec%"]])

        fig, ax = plt.subplots(figsize=(8, 6), facecolor='blue')
        ax.set_title("elapsed time percentage in segments", fontsize=16, color="#FFB266")
        bars = plt.bar(self.segments_df["type"], self.segments_df["elapsed_sec%"], color=["orange"])

        # # get rid of the frame
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # add value on top off the bar
        for b in bars:
            height = b.get_height()
            plt.gca().text(b.get_x() + b.get_width() / 2, b.get_height() - 3, str(int(height)),
                           ha='center', color='white', fontsize=20)
        # remove ticks
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True, labelsize=12)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
