# imports
import time
import random
import datetime

from database_handler import DatabaseHandler
from objects.sensordata import SensorData


class RandomDataHandler:
    def __init__(self):
        self.database_handler = DatabaseHandler("database/db.ini")

    def generate_random_forest_location(self):
        min_lat, max_lat = 50.5, 51.5
        min_lon, max_lon = -0.5, 0.5

        lat1 = round(random.uniform(min_lat, max_lat), 4)
        lon1 = round(random.uniform(min_lon, max_lon), 4)

        square_size = 0.02

        lat2 = lat1 + square_size
        lon2 = lon1 + square_size

        return lat1, lon1, lat2, lon2

    def generate_random_sensordata(self):
        return SensorData(
            co2_level=round(random.uniform(300, 2200), 3),
            ozone_level=round(random.uniform(80, 230), 3),
            temperature=round(random.uniform(-20, 50), 3),
            humidity=round(random.uniform(0, 100), 3),
            co_level=round(random.uniform(0, 105), 3),
            so2_level=round(random.uniform(25, 100), 3),
            no2_level=round(random.uniform(50, 200), 3),
            soil_moisture_level=round(random.uniform(25, 300), 3),
            soil_temperature_level=round(random.uniform(-20, 50), 3),
            soil_humidity_level=round(random.uniform(0, 100), 3),
            soil_ph=round(random.uniform(0, 14), 3),
            date=str(datetime.datetime.now())[:-7],
            anomalous=False,
        )

    def generate_site_and_nodes(self, site_name, node_count):
        site = self.database_handler.insert_site(site_name=site_name)

        print(f"New Site Generated: {site}")

        lat_min, lon_min, lat_max, lon_max = self.generate_random_forest_location()

        nodes = []
        for i in range(node_count):
            new_node = self.database_handler.insert_node(
                lat=round(random.uniform(lat_min, lat_max), 4),
                lon=round(random.uniform(lon_min, lon_max), 4),
                site_id=site.site_id,
            )

            nodes.append(new_node)

            print(f"New Node Generated: {new_node}")

        return site, nodes

    def simulate_live_data_insertion(self, site_name: str, node_count: int = 20):
        site, nodes = self.generate_site_and_nodes(site_name, node_count)

        while True:
            if random.uniform(0, 1) > 0.8:
                new_sensordata = self.generate_random_sensordata()

                node_to_use = nodes[random.randint(0, node_count - 1)]

                new_sensordata = self.database_handler.insert_sensordata(new_sensordata)

                self.database_handler.insert_node_to_data_link(
                    int(node_to_use.id), new_sensordata.id
                )

                print(f"Added new sensordata {new_sensordata} for node {node_to_use}")

            else:
                print("No new sensordata added")

            time.sleep(2)

    def generate_populated_site(
        self, site_name: str, node_count: int = 20, sensordata_per_node: int = 5
    ):
        site, nodes = self.generate_site_and_nodes(site_name, node_count)

        for node in nodes:
            for i in range(sensordata_per_node):
                new_sensordata = self.generate_random_sensordata()

                new_sensordata = self.database_handler.insert_sensordata(new_sensordata)

                self.database_handler.insert_node_to_data_link(
                    int(node.id), new_sensordata.id
                )

                print(f"Added new sensordata {new_sensordata} for node {node}")


if __name__ == "__main__":
    random_data_handler = RandomDataHandler()

    random_data_handler.generate_populated_site("Populated test")
