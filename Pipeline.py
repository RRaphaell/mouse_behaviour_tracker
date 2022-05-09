import cv2
from Video import VideoStream
from Tracker import Tracker
from Analyzer import Analyzer


class Pipeline:
    def __init__(self, video, objects, first_image):
        self.video_stream = VideoStream(video)
        self.tracker = Tracker(objects)
        self.analyzer = Analyzer()
        self.first_image = first_image

    def run(self):
        self.video_stream.start()

        while not self.video_stream.stopped():
            frame = self.video_stream.read()
            if frame is None:
                break

            self.tracker.draw_predictions(frame)
            predictions = self.tracker.get_predictions()

        self.analyzer.draw_tracked_road(predictions, self.first_image)

        cv2.destroyAllWindows()
        self.video_stream.stop()
