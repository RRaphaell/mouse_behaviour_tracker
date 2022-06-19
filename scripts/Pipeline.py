import os
import cv2
import tempfile
import numpy as np
from scripts.Tracker import Tracker
from scripts.Analyzer import Analyzer
from scripts.utils import generate_segments_colors
from scripts.Report import Bar, Card, Dashboard, Pie
from scripts.config import CANVAS
import streamlit as st
from types import SimpleNamespace
from streamlit_elements import elements


class Pipeline:
    """
    The class contains the entire structure.
    It runs the video, makes predictions for each frame, and analyzes the results.

    Attributes:
        file (st.file_uploader): uploaded file with streamlit widget
        video (cv2.VideoCapture): Video file to process.
        report: (SimpleNamespace): namespace which contains several streamlit_elements to show some behaviour analysis
        tracker (Tracker): adds segment and predictions to video stream
        analyzer (Analyzer): analyze predictions and show some results
        first_image (np.array): first image from video. Used as background for canvas and results placed on that also
    """

    def __init__(self, video, segments_df, first_image, file):
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
        self.report = SimpleNamespace(
            dashboard=Dashboard(),
            n_crossing=Bar(0, 0, 6, 8, minW=3, minH=4),
            time_spent=Pie(6, 0, 6, 8, minW=3, minH=4),
            road_passed=Card(3, 8, 6, 12, minW=2, minH=4)
        )

        self.tracker = Tracker(segments_df, segment_colors)
        self.analyzer = Analyzer(video, segments_df, first_image, segment_colors, self.report)
        self.first_image = first_image

    def run(self):
        """this function runs video stream, use tracker to show segments, predictions and also analyzes them"""

        frame_size = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_idx = 0
        progress_bar = st.progress(0)

        file_out = tempfile.NamedTemporaryFile(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_out.name, fourcc, 25.0, (CANVAS.height, CANVAS.width))

        while True:
            success, img = self.video.read()
            curr_frame_idx += 1
            progress_bar.progress(curr_frame_idx/frame_size)
            if not success:
                progress_bar.empty()
                break

            img = self.tracker.draw_predictions(img)
            out.write(np.array(img))

        self.video.release()
        out.release()
        cv2.destroyAllWindows()

        os.system(f"ffmpeg -i {file_out.name} -c:v libx264 -c:a copy -f mp4 {file_out.name}_new")
        video_file = open(f"{file_out.name}_new", "rb")

        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Video streaming</h3>", unsafe_allow_html=True)
        st.video(video_file)

        predictions = np.array(self.tracker.get_predictions())

        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Behavior report</h3>", unsafe_allow_html=True)
        with elements("demo"):
            with self.report.dashboard(rowHeight=57):
                self.analyzer.draw_tracked_road(predictions)
                self.analyzer.show_elapsed_time_in_segments(predictions)
                self.analyzer.show_n_crossing_in_segments(predictions)

        self.video.release()
        cv2.destroyAllWindows()
