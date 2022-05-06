import cv2
import tempfile
import numpy as np
import pandas as pd

from PIL import Image
import PIL.ImageDraw as ImageDraw

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from Video import VideoStream
st.set_page_config(page_title="Mouse behaviour analysis", layout="wide")

# add styling
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=first_image,
    update_streamlit=True,
    height=1080,
    width=1920,
    drawing_mode=drawing_mode,
    key="canvas")

start_btn = st.button("Start")

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)

if f:
    # st_video = open(t_file.name, 'rb')
    # video_bytes = st_video.read()
    # st.video(video_bytes)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if start_btn:
    video_stream = VideoStream(t_file.name)
    video_stream.start()

    img_placeholder = st.empty()

    keypoint_x, keypoint_y = 100, 100
    while not video_stream.stopped():
        # Camera detection loop
        frame = video_stream.read()
        if frame is None:
            print("Frame stream interrupted")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
        draw = ImageDraw.Draw(image_pil, "RGBA")
        draw.ellipse([(keypoint_x - 10, keypoint_y - 10),
                      (keypoint_x + 10, keypoint_y + 10)],
                     outline="red", fill="red")
        keypoint_x += 20

        # Display the image with the detections in the Streamlit app
        # draw = ImageDraw.Draw(image_pil)

        draw.ellipse([(objects["left"], objects["top"]),
                      (objects["left"] + 2*objects["radius"],  objects["top"] + 2*objects["radius"])],
                     outline="red", fill=(255, 178, 102, 100), width=4)

        np.copyto(frame, np.array(image_pil))
        img_placeholder.image(image_pil)

    cv2.destroyAllWindows()
    video_stream.stop()
