import cv2
from scripts.Video import VideoStream
from scripts.Tracker import Tracker
from scripts.Analyzer import Analyzer


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

        self.video_stream = VideoStream(video)
        self.tracker = Tracker(segments_df)
        self.analyzer = Analyzer(first_image)
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

        self.analyzer.draw_tracked_road(predictions)

        cv2.destroyAllWindows()
        self.video_stream.stop()
