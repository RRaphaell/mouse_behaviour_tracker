import cv2
import numpy as np
import matplotlib.pyplot as plt
from scripts.config import CANVAS
from scripts.utils import calculate_circle_center_cords

plt.rcParams.update({
    "axes.facecolor":    (0.054, 0.066, 0.090, 1.0),  # same as streamlit dark style color
    "savefig.facecolor": (0.054, 0.066, 0.090, 1.0),  # same as streamlit dark style color
})


class Analyzer:
    """
    This class used to make analysis based on predictions and segments information.

    Attributes:
        segments_df (pd.Dataframe): dataframe of segments information
        first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
        num_frames (int): number of frames in the video
        frame_width (int): video frames width
        frame_heigt (int): video frames height
        frames_per_second (float): number of frames per second
        segment_colors (dict[str, list[float]]): color for each unique segment
        report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
    """

    def __init__(self, video, segments_df, first_image, segment_colors, report):
        """
        initialize analyzer class with streamlit widgets and markdowns

        args:
            video (cv2.VideoCapture): Video file to process.
            segments_df (pd.Dataframe): dataframe of segments information
            first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
            segment_colors (dict[str, list[float]]): color for each unique segment
            report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
        """

        self.segments_df = segments_df
        self.first_image = first_image
        self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_heigt = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames_per_second = video.get(cv2.CAP_PROP_FPS)
        self.segment_colors = segment_colors
        self.report = report

    def draw_tracked_road(self, predictions):
        """Draw the entire route covered by the mouse"""
        for x, y in predictions:
            if (y, x) != [0, 0]:  # if model doesn't predict any part it returns [0,0]
                self.first_image = cv2.circle(np.array(self.first_image), (x, y), 7, (255, 0, 0), -1)

        self.first_image = cv2.resize(self.first_image, (CANVAS.height, CANVAS.width), interpolation=cv2.INTER_NEAREST)

        self.report.road_passed(self.first_image)

    def _count_elapsed_n_frames(self, segment, predictions):
        """count quantity of frames when mouse is in segment"""
        predictions = np.stack(predictions)
        x, y = predictions[:, 0], predictions[:, 1]
        x = x * CANVAS.width / self.frame_width
        y = y * CANVAS.height / self.frame_heigt
        
        if segment["type"] == "rect":
            x1, y1 = segment["left"], segment["top"]
            x2, y2 = segment["left"]+segment["width"], segment["top"]+segment["height"]
            is_in_segment = (x > x1) & (x < x2) & (y > y1) & (y < y2)
        else:
            circle_x, circle_y = calculate_circle_center_cords(segment)
            rad = segment["radius"]
            # Compare radius of circle with distance of its center from given point
            is_in_segment = (x - circle_x) ** 2 + (y - circle_y) ** 2 <= rad ** 2

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

    def show_elapsed_time_in_segments(self, predictions):
        """count elapsed time in each segment and plot bars"""

        self._count_elapsed_time_in_segments(predictions)

        # sum up values for same segments
        self.segments_df["elapsed_sec%"] = self.segments_df.groupby('segment key')["elapsed_sec%"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'elapsed_sec%'])
        df = df.append({'segment key': "Other", 'elapsed_sec%': 100-df["elapsed_sec%"].sum()}, ignore_index=True)

        self.report.time_spent(df)

    def show_n_crossing_in_segments(self, predictions):
        """count number of crossing in each segment and plot bars"""

        self.segments_df['n_crossing'] = self.segments_df.apply(
            lambda segment: sum(np.diff(self._count_elapsed_n_frames(segment, predictions))), axis=1)

        # sum up values for same segments
        self.segments_df["n_crossing"] = self.segments_df.groupby('segment key')["n_crossing"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'n_crossing'])

        self.report.n_crossing(df)
