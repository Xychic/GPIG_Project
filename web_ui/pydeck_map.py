# imports
import os

import pydeck as pdk
import panel as pn


class PyDeckMap:

    def __init__(self):

        INITIAL_VIEW_STATE = pdk.ViewState(
            latitude=53.946813,
            longitude=-1.030806,
            zoom=11,
            max_zoom=16,
            pitch=45,
            bearing=0
        )

        self.deck = pdk.Deck(
            api_keys={
                "mapbox": os.getenv("MAPBOX_API_KEY")
            },
            layers=[],
            initial_view_state=INITIAL_VIEW_STATE,
            map_provider="mapbox",
            map_style=pdk.map_styles.MAPBOX_SATELLITE,
        )

    def __panel__(self):

        return pn.pane.DeckGL(self.deck, sizing_mode="stretch_both")
