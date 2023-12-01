# imports
import os

import pandas as pd
import param
import pydeck as pdk
import panel as pn
from param import depends


class PyDeckMap(pn.viewable.Viewer):

    map_data = param.DataFrame(default=pd.DataFrame())

    heatmap_weight = param.String(default="temperature")

    def __init__(self, **params):

        super().__init__(**params)

    @depends("map_data", "heatmap_weight", watch=True)
    def deck(self):

        if self.map_data.empty:
            lat = 53.946813
            lon = -1.030806
        else:
            lat = self.map_data["lat"].mean()
            lon = self.map_data["lon"].mean()

        return pdk.Deck(
            api_keys={
                "mapbox": os.getenv("MAPBOX_API_KEY")
            },
            layers=[self.heatmap_layer()],
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=11,
                max_zoom=16,
                pitch=45,
                bearing=0
            ),
            map_provider="mapbox",
            map_style=pdk.map_styles.MAPBOX_DARK,
        )

    @depends("map_data", "heatmap_weight", watch=True)
    def heatmap_layer(self):
        return pdk.Layer(
            "HeatmapLayer",
            data=self.map_data,
            opacity=0.9,
            get_position=["lon", "lat"],
            threshold=0.1,
            get_weight=self.heatmap_weight,
            pickable=True
        )

    def __panel__(self):
        return pn.pane.DeckGL(self.deck, sizing_mode="stretch_both")
