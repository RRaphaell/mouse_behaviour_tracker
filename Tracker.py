import cv2
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import streamlit as st
from Model import Model
from config import KEYPOINT, SEGMENTS


class Tracker:
    """
    This class used to draw segments on video stream, including everything drawn on canvas and all predictions

    Attributes:
        model (Model): model class for predict mouse keypoint from frame
        segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
        img_placeholder (float): streamlit empty space where the frames will be placed
        predictions (list): list of all prediction coordinates
    """

    def __init__(self, segments_df):
        """
        initialize tracker class with streamlit widgets and markdowns

        Args:
            segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
        """

        st.markdown("<h3 style='text-align: center; color: #FFB266;'>Video streaming</h3>", unsafe_allow_html=True)
        predictions_img_placeholder = st.empty()

        self.model = Model()
        self.segments_df = segments_df
        self.img_placeholder = predictions_img_placeholder
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
            if segment["type"] == "rect":
                draw.rectangle([(segment["left"], segment["top"]),
                                (segment["left"] + segment["width"], segment["top"] + segment["height"])],
                               outline=SEGMENTS.stroke_color, fill=SEGMENTS.fill, width=SEGMENTS.stroke_width)
            else:
                center_x = segment["left"] + segment["radius"] * np.cos(np.deg2rad(segment["angle"]))
                center_y = segment["top"] + segment["radius"] * np.sin(np.deg2rad(segment["angle"]))
                draw.ellipse([(center_x - segment["radius"], center_y - segment["radius"]),
                              (center_x + segment["radius"], center_y + segment["radius"])],
                             outline=SEGMENTS.stroke_color, fill=SEGMENTS.fill, width=SEGMENTS.stroke_width)

    def draw_predictions(self, frame):
        """draw all segments and predictions on video stream"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
        image_pil = image_pil.resize((704, 396))
        draw = ImageDraw.Draw(image_pil, "RGBA")

        self._draw_keypoints(draw)
        self._draw_segments(draw)

        self.img_placeholder.image(image_pil)

    def get_predictions(self):
        return self.predictions
