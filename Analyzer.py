import streamlit as st
import PIL.ImageDraw as ImageDraw
from config import KEYPOINT


class Analyzer:
    def __init__(self):
        st.markdown("<h3 style='text-align: center; color: #FFB266;'>Behavior report</h3>", unsafe_allow_html=True)
        self.img_placeholder = st.empty()

    def draw_tracked_road(self, predictions, image_pil):
        image_pil = image_pil.resize((704, 396))
        draw = ImageDraw.Draw(image_pil, "RGBA")

        for keypoint_x, keypoint_y in predictions:
            draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                          (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                         outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        self.img_placeholder.image(image_pil)
