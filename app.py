import streamlit as st
from streamlit_drawable_canvas import st_canvas
from Pipeline import Pipeline
from config import SEGMENTS
from utils import show_canvas_info, read_video


st.set_page_config(page_title="Mouse behavior analysis")
st.markdown("<h1 style='text-align: center; color: #FF8000;'>Mouse behavior analysis </h1>", unsafe_allow_html=True)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "circle", "transform"))
file = st.sidebar.file_uploader("Upload video:", type=["mp4"])

video, first_image = read_video(file)

# Create a canvas component
canvas_result = st_canvas(
    fill_color=SEGMENTS.fill_color,
    stroke_width=SEGMENTS.stroke_width,
    stroke_color=SEGMENTS.stroke_color,
    background_image=first_image,
    update_streamlit=True,
    height=396,
    width=704,
    drawing_mode=drawing_mode,
    key="canvas")

start_btn = st.button("Start")
objects = show_canvas_info(canvas_result)

if start_btn:
    if not video:
        st.warning("Please upload video first!")
    else:
        pipeline = Pipeline(video, objects, first_image)
        pipeline.run()
