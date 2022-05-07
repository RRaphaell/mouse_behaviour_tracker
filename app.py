import cv2
import tempfile
import pandas as pd
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from Video import VideoStream
from Tracker import Tracker
from Analyzer import get_dummy_plots
from config import SEGMENTS
st.set_page_config(page_title="Mouse behaviour analysis")


# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "circle", "transform"))

f = st.sidebar.file_uploader("Upload video:", type=["mp4"])
first_image = None
if f:
    t_file = tempfile.NamedTemporaryFile(delete=False)
    t_file.write(f.read())
    video = cv2.VideoCapture(t_file.name)
    ret, first_image = video.read()
    # canvas needs pil image
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    first_image = Image.fromarray(first_image)

    # if we need that
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


# Create a canvas component
canvas_result = st_canvas(
    fill_color=SEGMENTS.fill_color,  # Fixed fill color with some opacity
    stroke_width=SEGMENTS.stroke_width,
    stroke_color=SEGMENTS.stroke_color,
    background_image=first_image,
    update_streamlit=True,
    height=396,
    width=704,
    drawing_mode=drawing_mode,
    key="canvas")


start_btn = st.button("Start")

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)


if start_btn:
    video_stream = VideoStream(t_file.name)
    video_stream.start()

    img_placeholder = st.empty()
    tracker = Tracker(objects, img_placeholder)

    while not video_stream.stopped():
        # Camera detection loop
        frame = video_stream.read()
        if frame is None:
            break

        tracker.draw_predictions(frame)

    cv2.destroyAllWindows()
    video_stream.stop()

    left_plot, right_plot = st.columns([5, 5])
    buf = get_dummy_plots()
    left_plot.image(buf)
    right_plot.image(buf)
