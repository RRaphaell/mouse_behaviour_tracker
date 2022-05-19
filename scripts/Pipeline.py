import cv2
import numpy as np
from scripts.Video import VideoStream
from scripts.Tracker import Tracker
from scripts.Analyzer import Analyzer
from scripts.utils import generate_segments_colors
from scripts.Report import Bar, Card, Dashboard, Pie
from types import SimpleNamespace
from streamlit_elements import elements


class Pipeline:
    """
    The class contains the entire structure.
    It runs the video, makes predictions for each frame, and analyzes the results.

    Attributes:
        video_stream (VideoStream): runs a video stream from a video object in thread
        tracker (Tracker): adds segment and predictions to video stream
        analyzer (Analyzer): analyze predictions and show some results
        first_image (np.array): first image from video. Used as background for canvas and results placed on that also
    """

    def __init__(self, video, segments_df, first_image):
        """
        Initialize pipeline from video and segments information

        Args:
            video (cv2.VideoCapture): Video file to process.
            segments_df (pd.DataFrame): dataframe of segments information
            first_image (np.ndarray): first image of the video
        """

        segment_colors = generate_segments_colors(segments_df)

        self.report = SimpleNamespace(
            dashboard=Dashboard(),
            n_crossing=Bar(0, 0, 6, 7, minW=3, minH=4),
            time_spent=Pie(6, 0, 6, 7, minW=3, minH=4),
            road_passed=Card(3, 7, 6, 7, minW=2, minH=4)
        )

        self.video_stream = VideoStream(video)
        self.tracker = Tracker(segments_df, segment_colors)
        self.analyzer = Analyzer(video, segments_df, first_image, segment_colors, self.report)
        self.first_image = first_image

    def run(self):
        """this function runs video stream, use tracker to show segments, predictions and also analyzes them"""
        self.video_stream.start()
        while not self.video_stream.stopped():
            frame = self.video_stream.read()
            if frame is None:
                break

            self.tracker.draw_predictions(frame)
            predictions = self.tracker.get_predictions()

        predictions = np.array(predictions)

        with elements("demo"):
            with self.report.dashboard(rowHeight=57):
                self.analyzer.draw_tracked_road(predictions)
                self.analyzer.show_elapsed_time_in_segments(predictions)
                self.analyzer.show_n_crossing_in_segments(predictions)

        cv2.destroyAllWindows()
        self.video_stream.stop()
