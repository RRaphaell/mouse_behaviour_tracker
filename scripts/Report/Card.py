import numpy as np
from PIL import Image
from streamlit_elements import mui
from scripts.Report.Dashboard import Dashboard


class Card(Dashboard.Item):

    def __call__(self, image):

        im = Image.fromarray(np.array(image))
        im.save("/home/appuser/venv/bin/python/site-packages/streamlit_elements/frontend/build/temp.png")

        with mui.Card(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            mui.CardHeader(
                title="Road that passed",
                avatar=mui.icon.Route(),
                className=self._draggable_class,
            )

            mui.CardMedia(
                component="img",
                src="temp.png",
            )
