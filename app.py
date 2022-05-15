import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scripts.Pipeline import Pipeline
from scripts.config import SEGMENTS, CANVAS
from scripts.utils import show_canvas_info, read_video


st.set_page_config(page_title="Mouse behavior analysis", layout="wide")
st.markdown("<h1 style='text-align: center; color: #FF8000;'>Mouse behavior analysis </h1>", unsafe_allow_html=True)

# # add styling
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "circle", "transform"))
file = st.sidebar.file_uploader("Upload video:", type=["mp4"])

video, first_image = read_video(file)

_, center_layout, _ = st.columns([1, 3, 1])
with center_layout:
    # Create center_layout canvas component
    canvas_result = st_canvas(
        fill_color=SEGMENTS.fill_color,
        stroke_width=SEGMENTS.stroke_width,
        stroke_color=SEGMENTS.stroke_color,
        background_image=first_image,
        update_streamlit=True,
        height=CANVAS.height,
        width=CANVAS.width,
        drawing_mode=drawing_mode,
        key="canvas")


start_btn = center_layout.button("Start")
with center_layout:
    objects = show_canvas_info(canvas_result)

if start_btn:
    if not video:
        st.warning("Please upload video first!")    # if user did not upload a video
    else:
        pipeline = Pipeline(video, objects, first_image)
        pipeline.run()
