import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scripts.Pipeline import Pipeline
from scripts.config import CANVAS
from scripts.utils import show_canvas_info, read_video, read_markdown


def set_page_config():
    # add page general config
    st.set_page_config(page_title="Mouse behavior analysis", page_icon="🐀", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FF8000;'>Mouse behavior analysis 🐀 </h1>", unsafe_allow_html=True)

    # add styling
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox("Drawing tool: 🖼", ("rect", "circle", "transform"))

    # create UI to uploading video
    file = st.sidebar.file_uploader("Upload video: 💾", type=["mp4"])

    with st.sidebar:
        if not file:
            st.warning("upload video \t ⚠️")
        # About
        st.markdown(read_markdown("docs/about.md"), unsafe_allow_html=True)

    video, video_params, first_image = read_video(file)

    # Create center_layout canvas component
    canvas_result = st_canvas(
        fill_color=CANVAS.fill_color,
        stroke_width=CANVAS.stroke_width,
        stroke_color=CANVAS.stroke_color,
        background_image=first_image,
        update_streamlit=True,
        height=CANVAS.height,
        width=CANVAS.width,
        drawing_mode=drawing_mode,
        key="canvas")

    start_btn = st.button("Start")

    # show table of segments information
    objects = show_canvas_info(canvas_result)

    if start_btn:
        if not file:
            st.warning("Please upload video first!")    # if the user did not upload a video
        elif objects.empty:
            st.warning("add at least one segment on canvas")  # if the user did not add the segment at all
        else:
            pipeline = Pipeline(video_params, objects, first_image)
            pipeline.run(video)


if __name__ == "__main__":
    set_page_config()
    main()

