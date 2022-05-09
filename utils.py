import cv2
import pandas as pd
import tempfile
import streamlit as st
from PIL import Image


def show_canvas_info(canvas_result):
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

