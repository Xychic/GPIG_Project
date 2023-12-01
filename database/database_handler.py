# default imports
import configparser

# pip imports
import psycopg2
import pandas as pd

# local imports
from objects.site import Site
from objects.node import Node
from objects.species import Species
from objects.plant import Plant
from objects.sensordata import SensorData


class DatabaseHandler:

    def __init__(self, db_ini_path: str):

        self.conf = configparser.ConfigParser()
        self.conf.read(db_ini_path)

    def execute_expression(self, sql_expression: str, sql_parameters: tuple):

        with psycopg2.connect(dbname=self.conf["DATABASE_CONNECTION_INFO"]["DB_NAME"],
                              user=self.conf["DATABASE_CONNECTION_INFO"]["DB_USER"],
                              password=self.conf["DATABASE_CONNECTION_INFO"]["DB_PASS"],
                              host=self.conf["DATABASE_CONNECTION_INFO"]["DB_HOST"],
                              port=self.conf["DATABASE_CONNECTION_INFO"]["DB_PORT"]) as conn:
            with conn.cursor() as cursor:
                try:

                    cursor.execute(sql_expression, sql_parameters)

                except (Exception, psycopg2.DatabaseError) as error:

                    print(error)

                try:

                    results = cursor.fetchall()
                    return results

                except psycopg2.ProgrammingError as error:

                    # query returns no results, not really an error I just don't know if there's a better way to check
                    pass

    def execute_expression_to_pandas(self, sql_expression: str, sql_parameters: tuple):

        with psycopg2.connect(dbname=self.conf["DATABASE_CONNECTION_INFO"]["DB_NAME"],
                              user=self.conf["DATABASE_CONNECTION_INFO"]["DB_USER"],
                              password=self.conf["DATABASE_CONNECTION_INFO"]["DB_PASS"],
                              host=self.conf["DATABASE_CONNECTION_INFO"]["DB_HOST"],
                              port=self.conf["DATABASE_CONNECTION_INFO"]["DB_PORT"]) as conn:

            # TODO exception catching

            output = pd.read_sql_query(sql=sql_expression, params=list(sql_parameters), con=conn)

            return output

    # INSERTION FUNCTIONS

    def insert_site(self, site_name: str) -> Site:

        sql_expression = '''
            insert into sites (site_name) values (%s);
            select id from sites where site_name = %s;
        '''

        sql_parameters = (site_name, site_name)

        site_id = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)[0][0]

        return Site(site_id=site_id, site_name=site_name)

    def insert_node(self, lat: float, lon: float, site_id: int) -> Node:

        sql_expression = '''
            insert into nodes (lat, lon, site_id) values (%s, %s, %s);
            select id from nodes where lat = %s and lon = %s and site_id = %s;
        '''

        sql_parameters = (lat, lon, site_id, lat, lon, site_id)

        node_id = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)[0][0]

        return Node(id=node_id, lat=lat, lon=lon)

    def insert_species(self, species_name) -> Species:

        sql_expression = '''
            insert into species (species_name) values (%s);
            select id from species where species_name = %s;
        '''

        sql_parameters = (species_name, species_name)

        species_id = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)[0][0]

        return Species(species_id=species_id, species_name=species_name)

    def insert_plant(self, plant: Plant) -> Plant:

        sql_expression = '''
            insert into plants (species_id, diseased, date_recorded) values (%s, %s, %s);
            select id from plants where date_recorded = %s;
        '''

        sql_parameters = plant.to_tuple() + (plant.date_recorded,)

        plant_id = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)[0][0]

        plant.plant_id = plant_id

        return plant

    def insert_sensordata(self, sensordata_obj: SensorData) -> SensorData:

        sql_expression = '''
            insert into 
                sensor_data (
                    co2_level, ozone_level, temperature, humidity, co_level, so2_level, no2_level, soil_moisture_level,
                    soil_temperature, soil_humidity, soil_ph, date_recorded, anomalous
                )
            values (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) returning id;
        '''

        sql_parameters = sensordata_obj.to_tuple()

        print(sql_parameters)

        sensor_data_id = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)[0][0]

        sensordata_obj.id = sensor_data_id

        return sensordata_obj

    def insert_node_to_node_link(self, node_1_id: int, node_2_id: int, weight: float) -> None:

        sql_expression = '''
            insert into node_to_node_links (node_1_id, node_2_id, weight) values (%s, %s, %s);
        '''

        sql_parameters = (node_1_id, node_2_id, weight)

        self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

    def insert_node_to_data_link(self, node_id: int, data_id: int) -> None:

        sql_expression = '''
            insert into node_to_data_links (node_id, data_id) values (%s, %s);
        '''

        sql_parameters = (node_id, data_id)

        self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

    def insert_node_to_plant_link(self, node_id: int, plant_id: int, heading: float):

        sql_expression = '''
            insert into node_to_plant_links (node_id, plant_id, heading) values (%s, %s, %s);
        '''

        sql_parameters = (node_id, plant_id, heading)

        self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

    # QUERY FUNCTIONS

    def get_all_sites(self):

        sql_expression = '''
            select * from sites;
        '''

        sites = self.execute_expression(sql_expression=sql_expression, sql_parameters=())

        sites = [Site(site_id=x[0], site_name=x[1]) for x in sites]

        return sites

    def get_nodes(self, site_id: int):

        sql_expression = '''
            select * from nodes where site_id = %s;
        '''

        sql_parameters = (site_id, )

        nodes = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

        nodes = [Node(id=x[0], lat=x[1], lon=x[2]) for x in nodes]

        return nodes

    def get_sensordata(self, node_id: int) -> [SensorData]:

        sql_expression = '''
            select 
                s.* 
            from 
                sensor_data sd 
            inner join 
                node_to_data_links ntdl 
            on 
                ntdl.data_id = sd.id 
            where 
                ntdl.node_id = %s;
        '''

        sql_parameters = (node_id, )

        sensordata = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

        sensordata = [SensorData(id=x[0],
                                 co2_level=x[1],
                                 ozone_level=x[2],
                                 temperature=x[3],
                                 humidity=x[4],
                                 co_level=x[5],
                                 so2_level=x[6],
                                 no2_level=x[7],
                                 soil_moisture_level=x[8],
                                 soil_temperature_level=x[9],
                                 soil_humidity_level=x[10],
                                 soil_ph=x[11],
                                 date=x[12],
                                 anomalous=x[13]) for x in sensordata]

        return sensordata

    def get_plants(self, node_id: int) -> [Plant]:

        sql_expression = '''
            select 
                p.* 
            from 
                plants p  
            inner join 
                node_to_plant_links ntpl 
            on 
                ntpl.plant_id  = p.id 
            where 
                ntpl.node_id = %s;
        '''

        sql_parameters = (node_id, )

        plants = self.execute_expression(sql_expression=sql_expression, sql_parameters=sql_parameters)

        plants = [Plant(plant_id=x[0], species_id=x[1], is_diseased=x[2], date_recorded=x[3]) for x in plants]

        return plants

    def get_node_to_node_links(self):

        # TODO unsure of desired format

        pass

    def pandas_sensordata_test(self, site_id: int) -> pd.DataFrame:

        sql_expression = '''
            select ntdl.node_id, n.lat, n.lon, sd.*
            from sensor_data sd
            left join node_to_data_links ntdl ON ntdl.data_id = sd.id
            left join nodes n ON n.id = ntdl.node_id
            where n.site_id = %s;
        '''

        sql_parameters = (site_id, )

        output_df = self.execute_expression_to_pandas(sql_expression=sql_expression, sql_parameters=sql_parameters)

        return output_df


if __name__ == "__main__":
    test_db_handler = DatabaseHandler("database/db.ini")

    print(test_db_handler.get_plants(1))
