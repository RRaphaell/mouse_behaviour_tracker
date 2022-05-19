from streamlit_elements import nivo, mui
from scripts.Report.Dashboard import Dashboard
from .utils import df_to_dict

COLOR_PALETTE = ["#7042fa", "#b600f9", "#8201b0", "#9064a3", "#1c0525",
                 "#fff4da", "#f9cf7f", "#aa3c3c", "#398a93", "#012b52",
                 "#3dc531", "#ea3838", "#ff933b", "#cbca34"]


class Bar(Dashboard.Item):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dark_theme = {
            "background": "#252526",
            "textColor": "#FAFAFA",
            "tooltip": {
                "container": {
                    "background": "#3F3F3F",
                    "color": "FAFAFA",
                }
            }
        }

    def __call__(self, data):
        data_dict = df_to_dict(data, col="n_crossing")

        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar():
                mui.icon.BarChart()
                mui.Typography("Number of segment crossing", sx={"flex": 1})

            with mui.Box(sx={"flex": 1, "minHeight": 0}):
                nivo.Bar(
                    data=data_dict,
                    theme=self.dark_theme,
                    keys=list(data["segment key"]),
                    indexBy="segment key",
                    margin={"top": 70, "right": 80, "bottom": 40, "left": 80},
                    colors=COLOR_PALETTE,
                    borderWidth=1,
                    enableGridY=False,
                    fontSize=100,
                    borderColor={
                        "from": "color",
                        "modifiers": [
                            [
                                "darker",
                                0.2,
                            ]
                        ]
                    },
                    axisLeft={
                        "tickSize": 5,
                        "tickPadding": 5,
                        "tickRotation": 0,
                        "legendPosition": 'middle',
                        "legendOffset": -40
                    },
                    legends=[
                        {
                            "dataFrom": 'keys',
                            "anchor": 'bottom-right',
                            "direction": 'column',
                            "justify": False,
                            "translateX": 100,
                            "translateY": 0,
                            "itemsSpacing": 2,
                            "itemWidth": 100,
                            "itemHeight": 20,
                            "itemDirection": 'left-to-right',
                            "itemOpacity": 0.85,
                            "symbolSize": 20,
                            "effects": [
                                {
                                    "on": 'hover',
                                    "style": {
                                        "itemOpacity": 1
                                    }
                                }
                            ]
                        }
                    ]
                )
