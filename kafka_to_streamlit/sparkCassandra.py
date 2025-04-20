from pyspark.sql import SparkSession
import uuid
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Create SparkSession
spark = SparkSession.builder \
    .appName("TestSparkCassandraConnection") \
    .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.13:3.5.1") \
    .config("spark.cassandra.connection.host", "172.17.0.2") \
    .config("spark.cassandra.connection.port", "9042") \
    .getOrCreate()

# Define test data
test_data = [
    (str(uuid.uuid4()), "Alice", 42.5),
    (str(uuid.uuid4()), "Bob", 55.0),
    (str(uuid.uuid4()), "Charlie", 29.8)
]
schema = StructType([
    StructField("id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("value", DoubleType(), True)
])

# Create DataFrame
df = spark.createDataFrame(test_data, schema)

# Write test data to Cassandra
df.write \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="test_table", keyspace="test_keyspace") \
    .mode("append") \
    .save()

print("Data written to Cassandra successfully.")

# Read data from Cassandra
read_df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="test_table", keyspace="test_keyspace") \
    .load()

print("Data read from Cassandra:")
read_df.show()

# Stop the Spark session
spark.stop()