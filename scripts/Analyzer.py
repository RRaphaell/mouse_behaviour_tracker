import cv2
import numpy as np
import streamlit as st
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
from scripts.config import KEYPOINT, CANVAS
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

    def __init__(self, video, segments_df, first_image, segment_colors, report):
        """
        initialize analyzer class with streamlit widgets and markdowns

        args:
            video (cv2.VideoCapture): Video file to process.
            segments_df (pd.Dataframe): dataframe of segments information
            first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
            segment_colors (dict[str, list[float]]): color for each unique segment
        """

        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Behavior report</h3>", unsafe_allow_html=True)
        self.col1, self.col2, self.col3 = st.columns(3)

        self.img_placeholder = self.col1.empty()

        self.segments_df = segments_df
        self.first_image = first_image
        self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_per_second = video.get(cv2.CAP_PROP_FPS)
        self.segment_colors = segment_colors
        self.report = report

    def draw_tracked_road(self, predictions):
        """Draw the entire route covered by the mouse"""
        self.first_image = self.first_image.resize((CANVAS.width, CANVAS.height))
        draw = ImageDraw.Draw(self.first_image, "RGBA")

        for keypoint_x, keypoint_y in predictions:
            draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                          (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                         outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        self.report.road_passed(self.first_image)

        # self.img_placeholder.image(self.first_image)

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

        # in_segment = sum(is_in_segment)
        return is_in_segment

    def _count_elapsed_time_in_segments(self, predictions):
        """Count the number of frames and the amount of time spent when the mouse is in a segment"""
        if self.segments_df.empty:
            return

        self.segments_df['elapsed_n_frames'] = self.segments_df.apply(
            lambda segment: sum(self._count_elapsed_n_frames(segment, predictions)), axis=1)
        self.segments_df['elapsed_sec'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.frames_per_second)
        self.segments_df['elapsed_sec%'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.num_frames*100)

    # def _plot_bars(self, x, y, title):
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     bars = plt.bar(x, y, color=rgba_0_255_to_0_1(self.segment_colors.values()))
    #
    #     ax.set_title(title, fontsize=18, color="white", pad=15)
    #     # # get rid of the frame
    #     for spine in plt.gca().spines.values():
    #         spine.set_visible(False)
    #
    #     # add value on top off the bar
    #     for b in bars:
    #         height = b.get_height()
    #         plt.gca().text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(height)),
    #                        ha='center', color='white', fontsize=20)
    #     # remove ticks
    #     ax.tick_params(axis='x', colors='white', rotation=5 if len(x) > 5 else 0)
    #     ax.tick_params(top=False, bottom=True, left=False, right=False, labelleft=False, labelbottom=True, labelsize=10)
    #
    #     buf = BytesIO()
    #     fig.savefig(buf, format="png")
    #     st.image(buf)

    def show_elapsed_time_in_segments(self, predictions):
        """count elapsed time in each segment and plot bars"""

        self._count_elapsed_time_in_segments(predictions)

        # sum up values for same segments
        self.segments_df["elapsed_sec%"] = self.segments_df.groupby('segment key')["elapsed_sec%"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'elapsed_sec%'])

        self.report.time_spent(df)

        # with self.col2:
        #     self._plot_bars(x=[f"{i}" for i in df["segment key"]],
        #                     y=df["elapsed_sec%"],
        #                     title="elapsed time(%)")

            # self.col2.dataframe(self.segments_df[["type", "elapsed_n_frames", "elapsed_sec", "elapsed_sec%"]])

    def show_n_crossing_in_segments(self, predictions):
        """count number of crossing in each segment and plot bars"""

        self.segments_df['n_crossing'] = self.segments_df.apply(
            lambda segment: sum(np.diff(self._count_elapsed_n_frames(segment, predictions))), axis=1)

        # sum up values for same segments
        self.segments_df["n_crossing"] = self.segments_df.groupby('segment key')["n_crossing"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'n_crossing'])

        self.report.n_crossing(df)

        #
        # with self.col3:
        #     self._plot_bars(x=[f"{i}" for i in df["segment key"]],
        #                     y=df["n_crossing"],
        #                     title="number of crossings")
