import cv2
import numpy as np
from scripts.Models.Controller.Controller import Controller
from scripts.config import KEYPOINT, SEGMENTS, CANVAS
from scripts.utils import calculate_circle_center_cords

import albumentations as A
from scripts.Models.CenterDetector.config import CFG as CCFG
from scripts.Models.PartsDetector.config import CFG as PCFG


class Tracker:
    """
    This class used to draw segments on video stream, including everything drawn on canvas and all predictions

    Attributes:
        model (Model): model class for predict mouse keypoint from frame
        segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
    """

    def __init__(self, segments_df, segment_colors):
        """
        initialize tracker class with streamlit widgets and markdowns

        Args:
            segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
            segment_colors (dict[str, list[float]]): color for each unique segment
        """

        center_transforms = A.Compose([
            A.Resize(*CCFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.Normalize()
        ], p=1.0)

        parts_transforms = A.Compose([
            A.Normalize(),
            A.PadIfNeeded(PCFG.img_size[0], PCFG.img_size[1],
                          position=A.transforms.PadIfNeeded.PositionType.BOTTOM_RIGHT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
        ], p=1.0)

        self.controller = Controller("scripts/Models/CenterDetector/weights.bin",
                                     "scripts/Models/PartsDetector/weights.bin",
                                     center_transforms,
                                     parts_transforms)
        self.segments_df = segments_df
        self.segment_colors = segment_colors

    def _predict_keypoints(self):
        """Save and return predictions from the model"""
        pass

    def _draw_keypoints(self, draw):
        """draw model prediction keypoint"""
        pass

    def _draw_segments(self, img):
        """Draw all segments on the video stream that were drawn on the canvas"""
        if self.segments_df.empty:
            return

        overlay = img.copy()
        for index, segment in self.segments_df.iterrows():
            color = tuple(self.segment_colors[segment["segment key"]])  # color for each segment
            color = list(map(int, color))
            if segment["type"] == "rect":
                cv2.rectangle(overlay,
                              (segment["left"], segment["top"]),
                              (segment["left"] + segment["width"], segment["top"] + segment["height"]),
                              color=color[:-1], thickness=-1)

                img_ = cv2.addWeighted(overlay, 0.6, img, 0.4, 1.0)
            else:
                center_x, center_y = calculate_circle_center_cords(segment)
                print((center_x, center_y), segment["radius"])
                cv2.circle(overlay,
                           (int(center_x), int(center_y)),
                           int(segment["radius"]), color=color[:-1], thickness=-1)
                img_ = cv2.addWeighted(overlay, 0.6, img, 0.4, 1.0)

        return img_

    def draw_predictions(self, frame):
        """draw all segments and predictions on video stream"""
        predicted_image = self.controller.get_predicted_image(np.array(frame))
        predicted_image = cv2.resize(predicted_image, (CANVAS.height, CANVAS.width), interpolation=cv2.INTER_NEAREST)
        predicted_image = self._draw_segments(predicted_image)
        return predicted_image

    def get_predictions(self):
        return self.controller.get_predictions()
