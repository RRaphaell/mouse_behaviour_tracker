import pandas as pd
import cv2 as cv
import tempfile
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
st.set_page_config(page_title="Mouse behaviour analysis", layout="wide")


# add styling
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("rect", "circle", "transform"))

f = st.sidebar.file_uploader("Upload video:", type=["mp4"])
first_image = None
img_height, image_width = 500, 1000
if f:
    t_file = tempfile.NamedTemporaryFile(delete=False)
    t_file.write(f.read())
    video = cv.VideoCapture(t_file.name)
    ret, first_image = video.read()
    img_height, image_width = first_image.shape[0]//2, first_image.shape[1]//2
    # canvas needs pil image
    first_image = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)
    first_image = Image.fromarray(first_image)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=first_image,
    update_streamlit=True,
    height=img_height,
    width=image_width,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
