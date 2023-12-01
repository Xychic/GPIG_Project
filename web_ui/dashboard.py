# imports
import os

import numpy as np
import panel as pn
import param
from dotenv import load_dotenv

from web_ui.pydeck_map import PyDeckMap
from database.database_handler import DatabaseHandler


class Dashboard:

    def __init__(self):

        load_dotenv("web_ui/ui.env")

        self.database_handler = DatabaseHandler("database/db.ini")

        self.ui = None
        self.site_select = None
        self.data_type_select = None
        self.map_object = None
        self.sidebar = None

    def start_ui(self):

        # define UI objects -------------------------------------------------------------------------------------------

        self.site_select = pn.widgets.Select(name="Site", options=self.database_handler.get_all_sites())
        self.data_type_select = pn.widgets.Select(name="Data Type",
                                                  value="temperature",
                                                  options=["co2_level", "ozone_level", "temperature",
                                                           "humidity", "co_level", "so2_level",
                                                           "no2_level", "soil_moisture_level",
                                                           "soil_temperature", "soil_humidity",
                                                           "soil_ph"],)

        self.map_object = PyDeckMap()

        # initialise UI template --------------------------------------------------------------------------------------

        self.ui = pn.template.FastGridTemplate(
            title="LORACHS Dashboard",
            corner_radius=5,
            header_background="#9db8c9",
            logo="web_ui/images/icon.png",
            favicon="web_ui/images/icon.png",
            sidebar=pn.Column(self.site_select, self.data_type_select)
        )

        # define UI structure -----------------------------------------------------------------------------------------

        self.ui.main[:2, :6] = self.map_object

        # define UI events --------------------------------------------------------------------------------------------

        self.site_select.param.watch(self.site_changed_event, "value")
        self.data_type_select.param.watch(self.data_type_changed_event, "value")

        return self.ui

    def update_ui_event(self):

        df = self.database_handler.pandas_sensordata_test(site_id=self.site_select.value.site_id)

        self.map_object.map_data = df

    def site_changed_event(self, event: param.Event):

        print(event.obj.value)

        self.update_ui_event()

    def data_type_changed_event(self, event: param.Event):

        self.map_object.heatmap_weight = event.obj.value


