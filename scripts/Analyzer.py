import streamlit as st
import PIL.ImageDraw as ImageDraw
from scripts.config import KEYPOINT


class Analyzer:
    """
    This class used to make analysis based on predictions and segments information.

    Attributes:
        first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
    """

    def __init__(self, first_image):
        """
        initialize analyzer class with streamlit widgets and markdowns

        args:
            first_image (np.ndarray): first image of the video. Used as background for canvas and results placed on that also
        """

        st.markdown("<h3 style='text-align: center; color: #FFB266;'>Behavior report</h3>", unsafe_allow_html=True)
        self.img_placeholder = st.empty()

        self.first_image = first_image

    def draw_tracked_road(self, predictions):
        """Draw the entire route covered by the mouse"""
        self.first_image = self.first_image.resize((704, 396))
        draw = ImageDraw.Draw(self.first_image, "RGBA")

        for keypoint_x, keypoint_y in predictions:
            draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                          (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                         outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        self.img_placeholder.image(self.first_image)
