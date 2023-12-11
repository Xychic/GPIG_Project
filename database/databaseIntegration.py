from sqlalchemy import create_engine, MetaData, Table, insert

# Replace 'username', 'password', 'host', 'port', and 'database_name' with your PostgreSQL credentials
DATABASE_URL = "postgresql://postgres:example@localhost:5432/postgres"

# Create an engine to connect to the PostgreSQL database
engine = create_engine(DATABASE_URL)

# Create a MetaData instance
metadata = MetaData()
# Reflect all tables in the database
metadata.reflect(bind=engine)
# Reflect the necessary tables from the database
# Replace 'table_name' with the actual table name you want to work with

your_table = Table("sites", metadata, autoload=True, autoload_with=engine)


# Perform queries using the engine
def example_read():
    with engine.connect() as connection:
        # Example query
        query = your_table.select().limit(10)  # Select the first 10 rows from the table
        result = connection.execute(query)

        # Fetch and print the results
        for row in result:
            print(row)
            dict = row._asdict()
            x = sites.site(dict["id"], dict["site_name"])
            print(x)


def readTable(tablename: str, limit: int = 100):
    your_table: Table = Table(tablename, metadata, autoload=True, autoload_with=engine)

    with engine.connect() as connection:
        # Example query
        query = your_table.select().limit(10)  # Select the first 10 rows from the table
        result = connection.execute(query)
        for row in result:
            print(row)


def example_write():
    with engine.connect() as connection:
        query = insert(your_table).values(site_name="test")

        result = connection.execute(query)
        connection.commit()
        example_read()


example_read()
