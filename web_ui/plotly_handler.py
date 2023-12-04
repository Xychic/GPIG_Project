# imports


import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo


class PlotlyHandler:

    def __init__(self):

        pass

    def gen_sensordata_line_chart(self, sensordata_df: pd.DataFrame, data_limits_df: pd.DataFrame, column_name: str) -> pgo.Figure:

        output_graph = pgo.Figure()

        output_graph.add_trace(
            pgo.Scatter(
                x=sensordata_df["date_recorded"],
                y=sensordata_df[column_name],
                mode="lines+markers",
                name="sensordata"
            )
        )

        output_graph.add_shape(
            type="line",
            x0=sensordata_df["date_recorded"].min(), y0=data_limits_df[f"{column_name}_min"][0],
            x1=sensordata_df["date_recorded"].max(), y1=data_limits_df[f"{column_name}_min"][0],
            line=dict(color="red", width=3),
            name="temp"
        )

        output_graph.add_shape(
            type="line",
            x0=sensordata_df["date_recorded"].min(), y0=data_limits_df[f"{column_name}_max"][0],
            x1=sensordata_df["date_recorded"].max(), y1=data_limits_df[f"{column_name}_max"][0],
            line=dict(color="red", width=3),
            name="temp"
        )

        output_graph.update_layout(
            title="Sensordata Over Time",
            xaxis_title="Date Recorded",
            yaxis_title="Data Values",
            legend_title="Legend",
            template="plotly_dark",
        )

        output_graph.layout.autosize = True

        return output_graph
