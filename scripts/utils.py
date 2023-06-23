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
from scripts.config import COLOR_PALETTE
from typing import Tuple
from streamlit_elements import elements
from streamlit import session_state


def redraw_after_refresh(show_tracked_video_btn, show_report_btn):
    if show_tracked_video_btn:
        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Video streaming</h3>", unsafe_allow_html=True)
        st.video(session_state.generated_video)

    if show_report_btn:
        st.markdown("<h3 style='text-align: center; color: #FF8000;'>Behavior report</h3>", unsafe_allow_html=True)
        report = session_state.report
        crossing_df = session_state.crossing_df
        time_df = session_state.time_df
        tracked_road = session_state.tracked_road
        predictions = session_state.predictions

        with elements("demo"):
            with report.dashboard(rowHeight=57):
                report.road_passed(pd.DataFrame(predictions, columns=["x", "y"]), tracked_road)
                report.time_spent(time_df)
                report.n_crossing(crossing_df)


# @st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def get_analysis_df(group_type, series):
    crossing_df = session_state.crossing_df[["segment key", "n_crossing"]]
    time_df = session_state.time_df[["segment key", "elapsed_sec%", "elapsed_sec"]]
    merge_df = pd.merge(crossing_df, time_df, on='segment key')
    merge_df["group_type"] = group_type
    merge_df["series"] = series

    return convert_df(merge_df)


def redraw_export_btn(placeholder, group_type_text, series_text):
    df = get_analysis_df(group_type_text, series_text)

    # clear and redraw download button with new generated data
    placeholder.empty()
    export_analysis_btn = placeholder.download_button(label="export",
                                                      data=df,
                                                      file_name=f'{session_state["video_name"]}_analysis.csv',
                                                      mime='text/csv')


def read_markdown(markdown_file):
    return Path(markdown_file).read_text()


def get_rect_all_coords(x1, y1, width, height, scale_x, scale_y):
    width, height = width * scale_x, height * scale_y
    p1 = np.array([x1, y1])
    p2 = np.array([x1 + width, y1])
    p3 = np.array([x1 + width, y1 + height])
    p4 = np.array([x1, y1 + height])
    return p1, p2, p3, p4


# https://www.geeksforgeeks.org/check-whether-given-point-lies-inside-rectangle-not/
def get_rect_coords_after_rotation(x1, y1, width, height, scale_x, scale_y, angle):
    p1, p2, p3, p4 = get_rect_all_coords(x1, y1, width, height, scale_x, scale_y)
    # center = np.array([x1 + width / 2, y1 + height / 2])
    center = p1
    angle = angle  # formula works for counter-clockwise
    R = np.array([[np.cos(np.deg2rad(angle)), -1*np.sin(np.deg2rad(angle))],
                  [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

    p1 = center + np.matmul(R, (p1 - center))
    p2 = center + np.matmul(R, (p2 - center))
    p3 = center + np.matmul(R, (p3 - center))
    p4 = center + np.matmul(R, (p4 - center))
    return p1, p2, p3, p4


# TODO: object dataframeshi bevri obieqti sheidzleba iyos amito eg gavasworo
def add_df_rect_coords(objects):
    objects[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]] = 0

    for index, row in objects.iterrows():
        if row["type"] == "rect":
            p1, p2, p3, p4 = get_rect_coords_after_rotation(float(row["left"]), float(row["top"]),
                                                            float(row["width"]), float(row["height"]),
                                                            float(row["scaleX"]), float(row["scaleY"]),
                                                            float(row["angle"]))

            objects.at[index, "x1"], objects.at[index, "y1"] = p1
            objects.at[index, "x2"], objects.at[index, "y2"] = p2
            objects.at[index, "x3"], objects.at[index, "y3"] = p3
            objects.at[index, "x4"], objects.at[index, "y4"] = p4


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
                objects.insert(loc=0, column='segment key', value=list(map(str, range(len(objects)))))
                add_df_rect_coords(objects)

                objects = st.data_editor(objects, disabled=objects.columns.difference(["segment key"]))

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

        video_params = {"video_name": file.name,
                        "num_frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "frame_width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "frame_height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "frames_per_second": video.get(cv2.CAP_PROP_FPS)}

    return video, video_params, first_image


def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)


# https://math.stackexchange.com/questions/4240275/calculating-xy-coordinates-of-a-rectangle-that-is-rotated-with-a-given-rotation
# A function to check whether point P(x, y) lies inside the rectangle
# formed by A(x1, y1), B(x2, y2), C(x3, y3) and D(x4, y4)
def is_in_rect(pred_x, pred_y, segment):
    x1, y1 = segment["x1"], segment["y1"]
    x2, y2 = segment["x2"], segment["y2"]
    x3, y3 = segment["x3"], segment["y3"]
    x4, y4 = segment["x4"], segment["y4"]

    # Calculate area of rectangle ABCD
    A = (triangle_area(x1, y1, x2, y2, x3, y3) +
         triangle_area(x1, y1, x4, y4, x3, y3))

    A1 = triangle_area(pred_x, pred_y, x1, y1, x2, y2)  # area of PAB
    A2 = triangle_area(pred_x, pred_y, x2, y2, x3, y3)  # area of PBC
    A3 = triangle_area(pred_x, pred_y, x3, y3, x4, y4)  # area of PCD
    A4 = triangle_area(pred_x, pred_y, x1, y1, x4, y4)  # area of PAD

    # Check if sum of A1, A2, A3 and A4 is same as A
    # print(A, A1 + A2 + A3 + A4, A-(A1 + A2 + A3 + A4))
    return abs(A - (A1 + A2 + A3 + A4)) < 0.000001


def calculate_circle_center_cords(segment: pd.Series) -> Tuple[int, int]:
    """calculate circle center based on radius, angle and corner coordinates using pythagorean theorem"""
    center_x = segment["left"] + segment["radius"] * segment["scaleX"] * np.cos(np.deg2rad(segment["angle"]))
    center_y = segment["top"] + segment["radius"] * segment["scaleY"] * np.sin(np.deg2rad(segment["angle"]))
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


def create_video_output_file(frame_rate: float,
                             height: int,
                             width: int) -> Tuple[tempfile.NamedTemporaryFile, cv2.VideoWriter]:
    file_out = tempfile.NamedTemporaryFile(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_out.name, fourcc, frame_rate, (height, width))
    return file_out, out


def convert_mp4_standard_format(file_out: tempfile.NamedTemporaryFile):
    if not os.path.exists('videos'):
        os.makedirs("videos")

    os.system(f"ffmpeg -i {file_out.name} -c:v libx264 -c:a copy -f mp4 -y videos/generated_video")
    video_file = open("videos/generated_video", "rb")
    gc.collect()
    return video_file
