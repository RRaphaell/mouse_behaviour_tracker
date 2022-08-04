import gc
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scripts.Pipeline import Pipeline
from scripts.config import CANVAS
from scripts.utils import show_canvas_info, read_video, read_markdown
from streamlit import session_state
from streamlit_elements import elements


def set_page_config():
    # add page general config
    st.set_page_config(page_title="Mouse behavior analysis", page_icon="🐀", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FF8000;'>Mouse behavior analysis 🐀 </h1>", unsafe_allow_html=True)

    # add styling
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.info("""The project is in progress, we trained the model with a few images of rats,
        so it would be inaccurate frequently, but we update it periodically.
        If you have data that could be helpful, please contact us at raffo.kalandadze@gmail.com""")

    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox("Drawing tool: 🖼", ("rect", "circle", "transform"))

    # create UI to uploading video
    file = st.sidebar.file_uploader("Upload video: 💾", type=["mp4"])

    with st.sidebar:
        # example video option
        example_btn = st.checkbox("use example", disabled=bool(file), value=not bool(file),
                                  help="If you don't have a video, use our example")

        if not file and not example_btn:
            st.warning(" \t ⚠️ upload video file or use example")

        # About
        st.markdown(read_markdown("docs/about.rst"), unsafe_allow_html=True)

    file = open("examples/example.mp4", "rb") if example_btn else file
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
        if not file and not example_btn:
            st.warning(" \t ⚠️ Please upload video first or use example!")    # if the user did not upload a video
        elif objects.empty:
            st.warning("add at least one segment on canvas")  # if the user did not add the segment at all
        else:
            pipeline = Pipeline(video_params, objects, first_image)
            pipeline.run(video)
            gc.collect()
    elif "report" in session_state:
            st.markdown("<h3 style='text-align: center; color: #FF8000;'>Video streaming</h3>", unsafe_allow_html=True)
            st.video(session_state.generated_video)

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


if __name__ == "__main__":
    set_page_config()
    main()

