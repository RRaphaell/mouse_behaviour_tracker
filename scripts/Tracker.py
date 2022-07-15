import cv2
import numpy as np
import pandas as pd
from scripts.Models.Controller.Controller import Controller
from scripts.config import KEYPOINT, SEGMENTS, CANVAS
from scripts.utils import calculate_circle_center_cords
from typing import Dict, List


class Tracker:
    """
    This class used to draw segments on video stream, including everything drawn on canvas and all predictions

    Attributes:
        controller (Type[Controller]): model controller class to use both model in proper way
        segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
        segment_colors (dict[str, list[float]]): color for each unique segment
    """

    def __init__(self, segments_df: pd.DataFrame, segment_colors: Dict[str, List[float]]):
        """
        initialize tracker class with streamlit widgets and markdowns

        Args:
            segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
            segment_colors (dict[str, list[float]]): color for each unique segment
        """

        self.controller = Controller()
        self.segments_df = segments_df
        self.segment_colors = segment_colors

    def _draw_keypoints(self, frame: np.ndarray, coords: list) -> None:
        """draw model prediction keypoint"""
        for c in coords:
            if c != [0, 0]:  # if model doesn't predict any part it returns [0,0]
                frame = cv2.circle(frame, c, KEYPOINT.radius, KEYPOINT.fill, -1)

    def _draw_segments(self, img: np.ndarray) -> np.ndarray:
        """Draw all segments on the video stream that were drawn on the canvas"""

        overlay = img.copy()
        for index, segment in self.segments_df.iterrows():
            color = tuple(self.segment_colors[segment["segment key"]])  # color for each segment
            color = list(map(int, color))
            if segment["type"] == "rect":
                cv2.rectangle(overlay,
                              (int(segment["left"]), int(segment["top"])),
                              (int(segment["left"] + segment["width"]), int(segment["top"] + segment["height"])),
                              color=color[:-1], thickness=-1)

                img = cv2.addWeighted(overlay, SEGMENTS.alpha, img, 1-SEGMENTS.alpha, 1.0)
            else:
                center_x, center_y = calculate_circle_center_cords(segment)
                cv2.circle(overlay,
                           (int(center_x), int(center_y)),
                           int(segment["radius"]), color=color[:-1], thickness=-1)
                img = cv2.addWeighted(overlay, SEGMENTS.alpha, img, 1-SEGMENTS.alpha, 1.0)

        return img

    def draw_predictions(self, frame: np.ndarray) -> np.ndarray:
        """draw all segments and predictions on video stream"""
        coords = self.controller.predict_img(frame)
        self._draw_keypoints(frame, coords)
        predicted_image = cv2.resize(frame, (CANVAS.height, CANVAS.width), interpolation=cv2.INTER_NEAREST)
        predicted_image = self._draw_segments(predicted_image)
        return predicted_image

    def get_predictions(self):
        return self.controller.get_predictions()
