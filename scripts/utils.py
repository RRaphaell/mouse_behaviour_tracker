import os
import gc
import cv2
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import streamlit as st
import PIL
from PIL import Image
from st_aggrid import AgGrid    # for editable dataframe
from scripts.config import COLOR_PALETTE
from typing import Tuple


def read_markdown(markdown_file):
    return Path(markdown_file).read_text()


def show_canvas_info(canvas_result) -> pd.DataFrame:
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
                grid_return = AgGrid(objects, grid_options, theme='streamlit')
                objects = grid_return["data"]
    return objects


def read_video(file) -> Tuple[cv2.VideoCapture, dict, PIL.Image.Image]:
    """
    create cv2 video object from uploaded file

    Args:
        file (streamlit.UploadedFile): uploaded mp4 file

    Returns:
        cv2.VideoCapture: cv2 video object
        np.ndarray: first image of the video
    """

    video, video_params, first_image = None, None, None

    if file:
        t_file = tempfile.NamedTemporaryFile()
        t_file.write(file.read())
        video = cv2.VideoCapture(t_file.name)
        _, first_image = video.read()
        # canvas needs pil image
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        first_image = Image.fromarray(first_image)

        video_params = {"num_frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "frame_width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "frame_height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "frames_per_second": video.get(cv2.CAP_PROP_FPS)}

    return video, video_params, first_image


def calculate_circle_center_cords(segment: pd.Series) -> Tuple[int, int]:
    """calculate circle center based on radius, angle and corner coordinates using pythagorean theorem"""
    center_x = segment["left"] + segment["radius"] * np.cos(np.deg2rad(segment["angle"]))
    center_y = segment["top"] + segment["radius"] * np.sin(np.deg2rad(segment["angle"]))
    return center_x, center_y


def color_to_rgb_str():
    color_palette = [f"rgb{str(c)}" for c in COLOR_PALETTE]
    return color_palette


def color_to_hex():
    color_palette = ['#%02x%02x%02x' % c for c in COLOR_PALETTE]
    return color_palette


def generate_segments_colors(segments_df: pd.DataFrame) -> dict:
    """generate different colors for each unique segments"""
    color_palette = np.array(COLOR_PALETTE)
    color_palette[:, [2, 0]] = color_palette[:, [0, 2]]             # video writer converts rgb2gbr
    transparency = np.full((color_palette.shape[0], 1), 100)        # transparency array
    color_palette = np.append(color_palette, transparency, axis=1)  # add transparency to color_palette
    segment_colors = dict(zip(segments_df["segment key"].unique(), color_palette))
    return segment_colors


def create_video_output_file(frame_rate: float, height: int, width: int) -> Tuple[tempfile.NamedTemporaryFile, cv2.VideoWriter]:
    file_out = tempfile.NamedTemporaryFile(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_out.name, fourcc, frame_rate, (height, width))
    return file_out, out


def convert_mp4_standard_format(file_out: tempfile.NamedTemporaryFile):
    os.system(f"ffmpeg -i -y {file_out.name} -c:v libx264 -c:a copy -f mp4 temp")
    video_file = open(f"temp", "rb")
    os.remove(file_out.name)
    gc.collect()
    return video_file
