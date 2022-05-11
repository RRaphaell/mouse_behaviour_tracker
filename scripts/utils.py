import cv2
import numpy as np
import pandas as pd
import tempfile
import streamlit as st
from PIL import Image


def show_canvas_info(canvas_result):
    """
    creates dataframe object from segments drawn on the canvas.
    each row is a segment information such as coordinates, radius etc.

    Args:
        canvas_result (streamlit_drawable_canvas): Streamlit drawable canvas to track all drawing segments

    Returns:
        pd.DataFrame: dataframe containing segment information
    """

    objects = pd.DataFrame()
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")

        with st.expander("Segments information"):
            if objects.empty:
                st.dataframe(objects)
            else:
                st.dataframe(objects[["type", "left", "top", "width", "height", "scaleX", "scaleY", "angle"]])

    return objects


def read_video(file):
    """
    create cv2 video object from uploaded file

    Args:
        file (streamlit.UploadedFile): uploaded mp4 file

    Returns:
        cv2.VideoCapture: cv2 video object
        np.ndarray: first image of the video
    """

    video, first_image = None, None

    if file:
        t_file = tempfile.NamedTemporaryFile(delete=False)
        t_file.write(file.read())
        video = cv2.VideoCapture(t_file.name)
        _, first_image = video.read()
        # canvas needs pil image
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        first_image = Image.fromarray(first_image)

    return video, first_image


def calculate_circle_center_cords(segment):
    center_x = segment["left"] + segment["radius"] * np.cos(np.deg2rad(segment["angle"]))
    center_y = segment["top"] + segment["radius"] * np.sin(np.deg2rad(segment["angle"]))
    return center_x, center_y

