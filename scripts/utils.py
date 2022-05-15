import cv2
import numpy as np
import pandas as pd
import tempfile
import streamlit as st
from PIL import Image
from st_aggrid import AgGrid    # for editable dataframe
import matplotlib as plt


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
                # add segment key column which would be weights of each segment
                objects.insert(loc=0, column='segment key', value=range(len(objects)))

                # change dataframe configuration as editable only "segment key" column
                visible_columns = ["type", "left", "top", "width", "height", "scaleX", "scaleY", "angle"]
                grid_options = {"columnDefs": [{"field": "segment key", "editable": True}]}
                grid_options["columnDefs"] += [{"field": c, "editable": False} for c in visible_columns]

                # streamlit doesn't have interactive dataframe so using agrid dataframe.
                grid_return = AgGrid(objects, grid_options)
                objects = grid_return["data"]
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
    """calculate circle center based on radius, angle and corner coordinates using pythagorean theorem"""
    center_x = segment["left"] + segment["radius"] * np.cos(np.deg2rad(segment["angle"]))
    center_y = segment["top"] + segment["radius"] * np.sin(np.deg2rad(segment["angle"]))
    return center_x, center_y


def rgba_0_1_to_0_255(color_palette):
    color_palette = np.maximum(0, np.minimum(255, (color_palette * 256.0).astype(int)))
    return color_palette


def rgba_0_255_to_0_1(color_palette):
    color_palette = np.array(list(color_palette))/255
    return color_palette


def generate_segments_colors(segments_df):
    """generate different colors for each unique segments"""
    color_palette = plt.cm.get_cmap("tab20", segments_df["segment key"].nunique()).colors  # color for each segment
    color_palette = rgba_0_1_to_0_255(color_palette)
    color_palette[:, -1] = 100  # add transparency
    segment_colors = dict(zip(segments_df["segment key"].unique(), color_palette))

    return segment_colors

