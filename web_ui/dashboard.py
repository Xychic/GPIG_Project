# imports
import os

import numpy as np
import panel as pn
from dotenv import load_dotenv

from pydeck_map import PyDeckMap

class Dashboard:

    def __init__(self):

        load_dotenv("ui.env")

        self.ui = pn.template.FastGridTemplate(
            title="LORACHS Dashboard",
            corner_radius=5,
            header_background="#9db8c9",
            logo="images/icon.png",
            favicon="images/icon.png"

        )

        self.map_object = None

    def start_ui(self):

        self.map_object = PyDeckMap()

        self.ui.main[:2, :6] = self.map_object

        return self.ui



