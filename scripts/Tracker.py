import cv2
import numpy as np
import pandas as pd
from scripts.config import KEYPOINT, SEGMENTS, CANVAS
from scripts.utils import calculate_circle_center_cords
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw


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

        self.segments_df = segments_df
        self.segment_colors = segment_colors

    def _draw_keypoints(self, frame: np.ndarray, coords: list, target_coord: tuple) -> None:
        """draw model prediction keypoint"""
        for c in coords:
            frame = cv2.circle(frame, c, KEYPOINT.radius, KEYPOINT.fill, -1)

        if target_coord:
            frame = cv2.circle(frame, target_coord, KEYPOINT.radius, KEYPOINT.target_fill, -1)

    def _draw_segments(self, img: np.ndarray) -> np.ndarray:
        """Draw all segments on the video stream that were drawn on the canvas"""

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img, "RGBA")

        for index, segment in self.segments_df.iterrows():
            color = tuple(self.segment_colors[segment["segment key"]])  # color for each segment
            color = list(map(int, color))
            if segment["type"] == "rect":
                draw.polygon([(int(segment["x1"]), int(segment["y1"])),
                              (int(segment["x2"]), int(segment["y2"])),
                              (int(segment["x3"]), int(segment["y3"])),
                              (int(segment["x4"]), int(segment["y4"]))],
                             fill=tuple(color[:-1])+(125,))
            else:
                center_x, center_y = calculate_circle_center_cords(segment)

                draw.ellipse((int(center_x - segment["radius"] * segment["scaleX"]),
                              int(center_y - segment["radius"] * segment["scaleY"]),
                              int(center_x + segment["radius"] * segment["scaleX"]),
                              int(center_y + segment["radius"] * segment["scaleY"])),
                             fill=tuple(color[:-1])+(125,))

        return np.array(img)

    def draw_predictions(self, frame: np.ndarray, coords: List[Tuple[int, int]], target_coord: Tuple[int, int]) -> np.ndarray:
        """draw all segments and predictions on video stream"""
        self._draw_keypoints(frame, coords, target_coord)
        predicted_image = cv2.resize(frame, (CANVAS.width, CANVAS.height), interpolation=cv2.INTER_NEAREST)
        predicted_image = self._draw_segments(predicted_image)
        return predicted_image
