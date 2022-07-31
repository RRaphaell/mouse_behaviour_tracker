import numpy as np
from PIL import Image
from streamlit_elements import mui
from scripts.Report.Dashboard import Dashboard
from .utils import download_file
from functools import partial


class Card(Dashboard.Item):

    def __call__(self, data, image):

        im = Image.fromarray(np.array(image))
        im.save("/home/appuser/venv/lib/python3.9/site-packages/streamlit_elements/frontend/build/temp.png")

        with mui.Card(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar(partial(download_file, data[["x", "y"]])):
                mui.icon.Route()
                mui.Typography("Road that passe", sx={"flex": 1})

            mui.CardMedia(
                component="img",
                src="temp.png",
            )
