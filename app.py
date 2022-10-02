import gc
import cv2
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from scripts.Pipeline import Pipeline
from scripts.config import CANVAS
from scripts.utils import show_canvas_info, read_video, read_markdown, \
    redraw_after_refresh, convert_df, redraw_export_btn
from streamlit import session_state
gc.enable()


def set_page_config():
    # add page general config
    st.set_page_config(page_title="Mouse behavior analysis", page_icon="🐀", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #FF8000;'>Mouse behavior analysis 🐀 </h1>", unsafe_allow_html=True)

    # add styling
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def info():
    with st.sidebar:
        st.info("""The project is in progress, we trained the model with a few images of rats,
        so it would be inaccurate frequently, but we update it periodically.
        If you have data that could be helpful, please contact us at raffo.kalandadze@gmail.com""")


def user_input_form(file):
    with st.sidebar.form(key="User input form"):
        # example video option
        example_btn = st.checkbox("use example",
                                  disabled=bool(file),
                                  value=not bool(file),
                                  help="If you don't have a video, use our example")

        if not file and not example_btn:
            st.warning(" \t ⚠️ upload video file or use example")

        show_tracked_video_btn = st.checkbox("Show tracked video",
                                             value=False,
                                             help="mark if you want to see the output video")

        show_report_btn = st.checkbox("Show report",
                                      value=False,
                                      help="mark if you want to see the analysis report")

        start_btn = st.form_submit_button(label="▶ Start")

    return example_btn, show_tracked_video_btn, show_report_btn, start_btn


def export_form():
    analysis_df = convert_df(pd.DataFrame())

    with st.sidebar:
        group_type_text = st.text_input("Group type", help="The name of the group that participated in the test")
        series_text = st.text_input("Series", help="What session is the experiment in?")

        # As the download button needs the df at the beginning, we use this trick to set the analysis after generating
        export_btn_placeholder = st.empty()
        export_analysis_btn = export_btn_placeholder.download_button(label="📩 export",
                                                                     data=analysis_df,
                                                                     file_name='analysis.csv',
                                                                     mime='text/csv')

    with st.sidebar:
        # About
        st.markdown(read_markdown("docs/about.rst"), unsafe_allow_html=True)

    return group_type_text, series_text, export_analysis_btn, analysis_df, export_btn_placeholder


def main():
    # ********************************** sidebar **********************************

    info()
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox("Drawing tool: 🖼", ("rect", "circle", "transform"))
    # create UI to uploading video
    file = st.sidebar.file_uploader("Upload video: 💾", type=["mp4"])
    example_btn, show_tracked_video_btn, show_report_btn, start_btn = user_input_form(file)
    group_type_text, series_text, export_analysis_btn, analysis_df, export_btn_placeholder = export_form()

    # ********************************** mainbar **********************************

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

    # show table of segments information
    objects = show_canvas_info(canvas_result)
    pipeline = None

    if start_btn:
        if not file and not example_btn:
            st.warning(" \t ⚠️ Please upload video first or use example!")  # if the user did not upload a video
        elif objects.empty:
            st.warning("add at least one segment on canvas")  # if the user did not add the segment at all
        else:
            pipeline = Pipeline(video_params, objects, first_image, show_tracked_video_btn, show_report_btn, analysis_df)
            pipeline.run(video)
            redraw_export_btn(export_btn_placeholder, group_type_text, series_text)

    # redraw widgets if app refreshed
    elif "report" in session_state:
        redraw_after_refresh(show_tracked_video_btn, show_report_btn)
        redraw_export_btn(export_btn_placeholder, group_type_text, series_text)

    if video:
        video.release()
        cv2.destroyAllWindows()
        file.close()
        del video_params
        del first_image

    if pipeline:
        del pipeline
    gc.collect()


if __name__ == "__main__":
    set_page_config()
    main()
