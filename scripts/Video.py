import cv2
import time
import threading


class VideoStream:
    """
    this class creates video stream from video object, reads frame by frame in thread

    Attributes:
        name (str, default='VideoStream'): Name for the thread.
        stream (cv2.VideoCapture): Video file stream.
        real_time (bool, default='True'): Defines if the video is going to
            be read at full speed or adjusted to the original frame rate.
        frame_rate (float): Frame rate of the video.
        grabbed (bool): Tells if the current frame's been correctly read.
        frame (nparray): OpenCV image containing the current frame.
        lock (_thread.lock): Lock to avoid race condition.
        _stop_event (threading.Event): Event used to gently stop the thread.
    """

    # Opens a video with OpenCV from file in a thread
    def __init__(self, video, name="VideoStream", real_time=True):
        """
        Initialize the video stream from a video object

        Args:
            video (cv2.VideoCapture): Video file to process.
            name (str, default='VideoStream'): Name for the thread.
            real_time (bool, default='True'): Defines if the video is going to
                be read at full speed or adjusted to the original frame rate.
        """

        self.name = name
        self.stream = video
        self.real_time = real_time
        self.frame_rate = self.stream.get(cv2.CAP_PROP_FPS)
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self):
        # Start the thread to read frames from the video stream with target function update
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuously iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                if self.real_time:
                    self.grabbed, self.frame = self.stream.read()
                    # Wait to match the original video frame rate
                    time.sleep(1.0 / self.frame_rate)
                else:
                    self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stop()

    def read(self):
        if self.stopped():
            print("Video ended")
        return self.frame

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()
