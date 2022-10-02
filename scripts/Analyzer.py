import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.config import CANVAS
from scripts.utils import calculate_circle_center_cords
from types import SimpleNamespace
from typing import Dict, List, Iterable
from streamlit import session_state


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
        frame_height (int): video frames height
        frames_per_second (float): number of frames per second
        segment_colors (dict[str, list[float]]): color for each unique segment
        report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
    """

    def __init__(self,
                 video_params: dict,
                 segments_df: pd.DataFrame,
                 first_image: np.ndarray,
                 segment_colors: Dict[str, List[float]],
                 report: SimpleNamespace,
                 show_report: bool
                 ):
        """
        initialize analyzer class with streamlit widgets and markdowns

        args:
            video_params (dict): dictionary of video parameters like frame number width height
            segments_df (pd.Dataframe): dataframe of segments information
            first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
            segment_colors (dict[str, list[float]]): color for each unique segment
            report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
        """

        self.segments_df = segments_df
        self.first_image = first_image
        self.num_frames = video_params["num_frames"]
        self.frame_width = video_params["frame_width"]
        self.frame_height = video_params["frame_height"]
        self.frames_per_second = video_params["frames_per_second"]
        self.segment_colors = segment_colors
        self.report = report
        self.show_report = show_report

    def draw_tracked_road(self, predictions: np.ndarray) -> None:
        """Draw the entire route covered by the mouse"""
        for x, y in predictions:
            if (y, x) != [0, 0]:  # if model doesn't predict any part it returns [0,0]
                self.first_image = cv2.circle(np.array(self.first_image), (x, y), 7, (255, 0, 0), -1)

        self.first_image = cv2.resize(self.first_image, (CANVAS.width, CANVAS.height), interpolation=cv2.INTER_NEAREST)

        session_state["tracked_road"] = self.first_image
        session_state["predictions"] = predictions

        if self.show_report:
            self.report.road_passed(pd.DataFrame(predictions, columns=["x", "y"]), self.first_image)

    def _count_elapsed_n_frames(self, segment: pd.Series, predictions: np.ndarray) -> Iterable:
        """count quantity of frames when mouse is in segment"""
        predictions = np.stack(predictions)
        x, y = predictions[:, 0], predictions[:, 1]
        x = x * CANVAS.width / self.frame_width
        y = y * CANVAS.height / self.frame_height

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

    def _count_elapsed_time_in_segments(self, predictions: np.ndarray) -> None:
        """Count the number of frames and the amount of time spent when the mouse is in a segment"""
        if self.segments_df.empty:
            return

        self.segments_df['elapsed_n_frames'] = self.segments_df.apply(
            lambda segment: sum(self._count_elapsed_n_frames(segment, predictions)), axis=1)
        self.segments_df['elapsed_sec'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.frames_per_second)
        self.segments_df['elapsed_sec%'] = self.segments_df['elapsed_n_frames'].apply(
            lambda n_frames: n_frames/self.num_frames*100)

    def show_elapsed_time_in_segments(self, predictions: np.ndarray) -> None:
        """count elapsed time in each segment and plot bars"""

        self._count_elapsed_time_in_segments(predictions)

        # sum up values for same segments
        self.segments_df["elapsed_sec%"] = self.segments_df.groupby('segment key')["elapsed_sec%"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'elapsed_sec%'])
        df = df.append({'segment key': "Other", 'elapsed_sec%': 100-df["elapsed_sec%"].sum()}, ignore_index=True)

        session_state["time_df"] = df

        if self.show_report:
            self.report.time_spent(df)

    def show_n_crossing_in_segments(self, predictions: np.ndarray) -> None:
        """count number of crossing in each segment and plot bars"""

        self.segments_df['n_crossing'] = self.segments_df.apply(
            lambda segment: int(np.ceil(sum(np.diff(self._count_elapsed_n_frames(segment, predictions))) / 2)), axis=1)

        # sum up values for same segments
        self.segments_df["n_crossing"] = self.segments_df.groupby('segment key')["n_crossing"].transform('sum')
        df = self.segments_df.drop_duplicates(subset=['segment key', 'n_crossing'])

        session_state["crossing_df"] = df

        if self.show_report:
            self.report.n_crossing(df)
