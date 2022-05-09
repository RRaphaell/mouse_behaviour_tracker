import cv2
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import streamlit as st
from Model import Model
from config import KEYPOINT, SEGMENTS


class Tracker:
    def __init__(self, segments_df):
        st.markdown("<h3 style='text-align: center; color: #FFB266;'>Video streaming</h3>", unsafe_allow_html=True)
        predictions_img_placeholder = st.empty()

        self.model = Model()
        self.segments_df = segments_df
        self.img_placeholder = predictions_img_placeholder
        self.predictions = []

    # dummy function until model is ready
    def _predict_keypoints(self):
        pred_x, pred_y = self.model.predict()
        self.predictions.append((pred_x, pred_y))
        return pred_x, pred_y

    def _draw_keypoints(self, draw):
        keypoint_x, keypoint_y = self._predict_keypoints()

        draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                      (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                     outline=KEYPOINT.outline, fill=KEYPOINT.fill)

    def _draw_segments(self, draw):
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
        image_pil = image_pil.resize((704, 396))
        draw = ImageDraw.Draw(image_pil, "RGBA")

        self._draw_keypoints(draw)
        self._draw_segments(draw)

        self.img_placeholder.image(image_pil)

    def get_predictions(self):
        return self.predictions
