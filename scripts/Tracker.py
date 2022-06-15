import cv2
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
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
        predictions (list): list of all prediction coordinates
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
        self.predictions = []

    def _predict_keypoints(self):
        """Save and return predictions from the model"""
        pred_x, pred_y = self.model.predict()
        self.predictions.append((pred_x, pred_y))
        return pred_x, pred_y

    def _draw_keypoints(self, draw):
        """draw model prediction keypoint"""
        keypoint_x, keypoint_y = self._predict_keypoints()

        draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                      (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                     outline=KEYPOINT.outline, fill=KEYPOINT.fill)

    def _draw_segments(self, draw):
        """Draw all segments on the video stream that were drawn on the canvas"""
        if self.segments_df.empty:
            return

        for index, segment in self.segments_df.iterrows():
            color = tuple(self.segment_colors[segment["segment key"]])  # color for each segment

            if segment["type"] == "rect":
                draw.rectangle([(segment["left"], segment["top"]),
                                (segment["left"] + segment["width"], segment["top"] + segment["height"])],
                               outline=SEGMENTS.stroke_color, fill=color, width=SEGMENTS.stroke_width)
            else:
                center_x, center_y = calculate_circle_center_cords(segment)
                draw.ellipse([(center_x - segment["radius"], center_y - segment["radius"]),
                              (center_x + segment["radius"], center_y + segment["radius"])],
                             outline=SEGMENTS.stroke_color, fill=color, width=SEGMENTS.stroke_width)

    def draw_predictions(self, frame):
        """draw all segments and predictions on video stream"""
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image_pil = Image.fromarray(frame)
        # image_pil = image_pil.resize((CANVAS.width, CANVAS.height))
        # draw = ImageDraw.Draw(image_pil, "RGBA")

        # import streamlit as st
        # st.image(np.array(frame), width=832)
        predicted_image = self.controller.get_predicted_image(np.array(frame))
        predicted_image = A.Resize(CANVAS.height, CANVAS.width, interpolation=cv2.INTER_NEAREST)(image=predicted_image)["image"]
        # predicted_image = self.model.predict(image_pil)
        # ind = np.unravel_index(np.argmax(predicted_image, axis=None), predicted_image.shape)
        # ind = np.array(ind)
        # ind *= 2
        #
        # draw.ellipse([(ind[1] - KEYPOINT.radius, ind[0] - KEYPOINT.radius),
        #               (ind[1] + KEYPOINT.radius, ind[0] + KEYPOINT.radius)],
        #              outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        # self._draw_segments(draw)
        # self.predictions.append((ind[0], ind[1]))
        return predicted_image

    def get_predictions(self):
        return self.predictions
