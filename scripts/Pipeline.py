import cv2
import numpy as np
import pandas as pd
from scripts.Tracker import Tracker
from scripts.Analyzer import Analyzer
from scripts.utils import generate_segments_colors, create_video_output_file, convert_mp4_standard_format
from scripts.Report import Bar, Card, Dashboard, Pie
from scripts.config import CANVAS
import streamlit as st
from types import SimpleNamespace
from streamlit_elements import elements
from typing import Callable


class Pipeline:
    """
    The class contains the entire structure.
    It runs the video, makes predictions for each frame, and analyzes the results.

    Attributes:
        file (st.file_uploader): uploaded file with streamlit widget
        video (cv2.VideoCapture): Video file to process.
        frame_size (int): video frame size
        report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
        tracker (Tracker): adds segment and predictions to video stream
        analyzer (Analyzer): analyze predictions and show some results
        first_image (np.array): first image from video. Used as background for canvas and results placed on that also
    """

    def __init__(self,
                 video: cv2.VideoCapture,
                 segments_df: pd.DataFrame,
                 first_image: np.ndarray,
                 file: Callable
                 ):
        """
        Initialize pipeline from video and segments information

        Args:
            video (cv2.VideoCapture): Video file to process.
            segments_df (pd.DataFrame): dataframe of segments information
            first_image (np.ndarray): first image of the video
            file (st.file_uploader): uploaded file with streamlit widget
        """

        segment_colors = generate_segments_colors(segments_df)
        self.file = file
        self.video = video
        self.frame_size = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.report = SimpleNamespace(
            dashboard=Dashboard(),
            n_crossing=Bar(0, 0, 6, 8, minW=3, minH=4),
            time_spent=Pie(6, 0, 6, 8, minW=3, minH=4),
            road_passed=Card(3, 8, 6, 14, minW=2, minH=4)
        )

        self.tracker = Tracker(segments_df, segment_colors)
        self.analyzer = Analyzer(video, segments_df, first_image, segment_colors, self.report)
        self.first_image = first_image
        self.file_out, self.out = create_video_output_file(25.0, CANVAS.width, CANVAS.height)

    def show_report(self) -> None:
        """this functions calls all functions to show reports"""
        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Behavior report</h3>", unsafe_allow_html=True)

        predictions = np.array(self.tracker.get_predictions())

        with elements("demo"):
            with self.report.dashboard(rowHeight=57):
                self.analyzer.draw_tracked_road(predictions)
                self.analyzer.show_elapsed_time_in_segments(predictions)
                self.analyzer.show_n_crossing_in_segments(predictions)

    def run(self) -> None:
        """this function runs video stream, use tracker to show segments, predictions and also analyzes them"""

        curr_frame_idx, progress_bar = 0, st.progress(0)
        while True:
            success, img = self.video.read()
            curr_frame_idx += 1
            progress_bar.progress(curr_frame_idx / self.frame_size)
            if not success:
                progress_bar.empty()
                break

            img = self.tracker.draw_predictions(img)
            self.out.write(img)

        self.release_videos()

        # show annotated video
        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Video streaming</h3>", unsafe_allow_html=True)
        video_file = convert_mp4_standard_format(self.file_out)
        st.video(video_file)

        # show tracked road, elapsed time in segments and etc
        self.show_report()

    def release_videos(self) -> None:
        """call video destructors"""
        self.video.release()
        self.out.release()
        cv2.destroyAllWindows()
