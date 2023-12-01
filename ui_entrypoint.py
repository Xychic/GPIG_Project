# imports
from web_ui.dashboard import Dashboard


dashboard = Dashboard()

ui = dashboard.start_ui()

ui.servable()
