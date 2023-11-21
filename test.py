import psycopg2


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
		

with conn.cursor() as curs:
	try:
		curs.execute
	except (Exception, psycopg2.DatabaseError) as error:
		print(error)
		



conn.close()
