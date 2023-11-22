import psycopg2
from datetime import datetime
import sensordata
DB_USER ="postgres"
DB_PASS = "example"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME="postgres"


try:
	conn = psycopg2.connect(dbname=DB_NAME,
							user=DB_USER,
							password=DB_PASS,
							host=DB_HOST,
							port=DB_PORT)
	print("Database connected successfully")
except:
    print("Database not connected successfully")
    exit()
def getSchema():
	query:str = """
			SELECT table_name, column_name, data_type
			FROM information_schema.columns
			WHERE table_schema = 'public'  -- Change 'public' to your schema name if needed
			ORDER BY table_name, ordinal_position
			"""
	with conn.cursor() as curs:
		try:
			curs.execute(query)
			schema_info = curs.fetchall()
			for table_name, column_name, data_type in schema_info:
				print(f"Table: {table_name}, Column: {column_name}, Type: {data_type}")
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
			


def writeSensorDataExample():
	with conn.cursor() as curs:
		insert_query = """
		INSERT INTO sensor_data (co2_level, ozone_level,temperature,humidity,co_level,so2_level,no2_level,soil_moisture_level,soil_temperature,soil_humidity,soil_ph,date_recorded,anomalous)
		VALUES (%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s);
		"""
		data=(0,1,2,3,4,5,6,7,8,9,10,datetime.now(),True)
		try:
			curs.execute(insert_query,data)
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
def writeSensorData(sensorData:sensordata.SensorData):
	with conn.cursor() as curs:
		insert_query = """
		INSERT INTO sensor_data (co2_level, ozone_level,temperature,humidity,co_level,so2_level,no2_level,soil_moisture_level,soil_temperature,soil_humidity,soil_ph,date_recorded,anomalous)
		VALUES (%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s);
		"""
		#data=(sensorData.co2_level,sensorData.ozone_level,sensorData.temperature,sensorData.co_level,sensorData.so2_level,sensorData,sensorData.no2_level,sensorData.soil_moisture_level,sensorData.soil_temperature_level,sensorData.soil_humidity_level,sensorData.soil_ph,sensorData.date,sensorData.anomalous)
		data = sensorData.toTupleForDataBase();
		try:
			curs.execute(insert_query,data)
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
def writeNodeData():
	with conn.cursor() as curs:
		insert_query = """
		INSERT INTO nodes ( lat, lon)
		VALUES (%s, %s);
		"""
		data=(1,1)
		try:
			curs.execute(insert_query,data)
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)
def selectAllFromTable(table:str):
	select_query = f"SELECT * FROM {table};"
	with conn.cursor() as curs:
		try:
			curs.execute(select_query)
			records = curs.fetchall()
			for record in records:
				print(f"{record} \n") 
		except (Exception, psycopg2.DatabaseError) as error:
			print(error)

writeSensorDataExample()
selectAllFromTable("sensor_data")
conn.commit()
conn.close()
