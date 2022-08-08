import cv2
import numpy as np
import pandas as pd
from scripts.Models.Controller.Controller import Controller
from scripts.Tracker import Tracker
from scripts.Analyzer import Analyzer
from scripts.utils import generate_segments_colors, create_video_output_file, convert_mp4_standard_format
from scripts.Report import Bar, Card, Dashboard, Pie
from scripts.config import CANVAS
import streamlit as st
from types import SimpleNamespace
from streamlit_elements import elements
from streamlit import session_state


class Pipeline:
    """
    The class contains the entire structure.
    It runs the video, makes predictions for each frame, and analyzes the results.

    Attributes:
        num_frames (int): video frame size
        report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
        tracker (Tracker): adds segment and predictions to video stream
        analyzer (Analyzer): analyze predictions and show some results
    """

    def __init__(self,
                 video_params: dict,
                 segments_df: pd.DataFrame,
                 first_image: np.ndarray,
                 show_tracked_video: bool,
                 show_report: bool,
                 analysis_df: pd.DataFrame
                 ):
        """
        Initialize pipeline from video and segments information

        Args:
            video_params (dict): dictionary of video parameters like frame number width height.
            segments_df (pd.DataFrame): dataframe of segments information
            first_image (np.ndarray): first image of the video
        """

        self.controller = Controller()
        self.analysis_df = analysis_df
        self.show_tracked_video = show_tracked_video
        self.show_report = show_report
        segment_colors = generate_segments_colors(segments_df)
        self.num_frames = video_params["num_frames"]
        self.report = SimpleNamespace(
            dashboard=Dashboard(),
            n_crossing=Bar(0, 0, 6, 8, minW=3, minH=4),
            time_spent=Pie(6, 0, 6, 8, minW=3, minH=4),
            road_passed=Card(3, 8, 6, 14, minW=2, minH=4)
        )

        session_state["report"] = self.report
        session_state["video_name"] = video_params["video_name"]

        self.tracker = Tracker(segments_df, segment_colors)
        self.file_out, self.out = create_video_output_file(25.0, CANVAS.width, CANVAS.height)
        self.analyzer = Analyzer(video_params, segments_df, first_image, segment_colors, self.report, self.show_report)

    def run(self, video: cv2.VideoCapture) -> None:
        """this function runs video stream, use tracker to show segments, predictions and also analyzes them"""

        curr_frame_idx, progress_bar = 0, st.progress(0)
        while True:
            success, img = video.read()
            curr_frame_idx += 1
            progress_bar.progress(curr_frame_idx / self.num_frames)
            if not success:
                progress_bar.empty()
                break

            coords = self.controller.predict_img(img)

            if self.show_tracked_video:
                img = self.tracker.draw_predictions(img, coords)
                self.out.write(img)

        if self.show_tracked_video:
            self.generate_tracked_video()

        # show tracked road, elapsed time in segments and etc
        self.analyse()

    def analyse(self) -> None:
        """this functions calls all functions to show reports"""
        # st.markdown("<h3 style='text-align: center; color: #FF8000;'>Behavior report</h3>", unsafe_allow_html=True)

        predictions = np.array(self.controller.get_predictions())

        with elements("demo"):
            with self.report.dashboard(rowHeight=57):
                self.analyzer.draw_tracked_road(predictions)
                self.analyzer.show_elapsed_time_in_segments(predictions)
                self.analyzer.show_n_crossing_in_segments(predictions)

    def generate_tracked_video(self):
        self.release_videos()

        # show annotated video
        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Video streaming</h3>", unsafe_allow_html=True)
        video_file = convert_mp4_standard_format(self.file_out)
        st.video(video_file)
        session_state["generated_video"] = video_file

    def release_videos(self) -> None:
        """call video destructors"""
        self.out.release()
        cv2.destroyAllWindows()
