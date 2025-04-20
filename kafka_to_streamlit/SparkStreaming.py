from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import MinMaxScalerModel,VectorAssembler
from pyspark.sql.functions import col,from_json,to_json,struct,current_timestamp,udf,when
from pyspark.sql.types import StructField,StructType, DoubleType,ArrayType,StringType
from datetime import datetime
import time

POSTGRES_URL = 'jdbc:postgresql://host.docker.internal:5432/bigdataproject'
POSTGRES_PROPERTIES = {
    'user': 'postgres',
    'password': 'root',
    'driver': 'org.postgresql.Driver'
}

# Initialize Spark with trigger settings for micro-batches
spark = SparkSession.builder \
        .appName("RealTimePrediction") \
        .master("local[*]") \
        .config("spark.sql.streaming.checkpointLocation", "file:///tmp/spark-checkpoints") \
        .config('spark.jars.packages', 'org.postgresql:postgresql:42.6.0') \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()

# Load my saved model
model = LogisticRegressionModel.load("file:///root/myproject/Model/LR")
scaler_model = MinMaxScalerModel.load("file:///root/myproject/Model/Scaler") 

# kafka topic and broker settings
INPUT_KAFKA_TOPIC = "prediction-topic"
OUTPUT_KAFKA_TOPIC = "prediction-results-topic"
KAFKA_BROKER = "hadoop-master:9092"

# Cassandra Settings
CASSANDRA_KEYSPACE = 'bigdataproject'
CASSANDRA_TABLE = "predictions_table"

# Define the schema for incoming data
schema = StructType([
    StructField("radius_mean", DoubleType(), True),
    StructField("texture_mean", DoubleType(), True),
    StructField("perimeter_mean", DoubleType(), True),
    StructField("area_mean",DoubleType(), True),
    StructField("smoothness_mean", DoubleType(), True),
    StructField("compactness_mean", DoubleType(), True),
    StructField("concavity_mean", DoubleType(), True),
    StructField("concave_points_mean", DoubleType(), True),
    StructField("symmetry_mean", DoubleType(), True),
    StructField("fractal_dimension_mean",DoubleType(),True),
    StructField("radius_se", DoubleType(), True),
    StructField("texture_se", DoubleType(),True),
    StructField("perimeter_se", DoubleType(), True),
    StructField("area_se", DoubleType(), True),
    StructField("smoothness_se", DoubleType(), True),
    StructField("compactness_se", DoubleType(), True),
    StructField("concavity_se", DoubleType(), True),
    StructField("concave_points_se", DoubleType(), True),
    StructField("symmetry_se", DoubleType(), True),
    StructField("fractal_dimension_se", DoubleType(), True),
    StructField("radius_worst", DoubleType(), True),
    StructField("texture_worst", DoubleType(), True),
    StructField("perimeter_worst", DoubleType(), True),
    StructField("area_worst", DoubleType(), True),
    StructField("smoothness_worst", DoubleType(), True),
    StructField("compactness_worst", DoubleType(), True),
    StructField("concavity_worst", DoubleType(), True),
    StructField("concave_points_worst", DoubleType(), True),
    StructField("symmetry_worst", DoubleType(), True),
    StructField("fractal_dimension_worst", DoubleType(), True)
])

# UDF to convert Vector to Array
vector_to_array = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

def write_to_postgres(batch_df, batch_id):
    try:
        # Force eager execution of the batch
        if batch_df.count() > 0:  # Only process if there's data
            # Convert probability vector to array
            batch_df = batch_df.withColumn("probability_array", vector_to_array("probability"))
            
            # Prepare DataFrame for PostgreSQL
            postgres_df = batch_df.select(
                current_timestamp().alias("timestamp"),
                col("prediction").cast("integer").alias("predicted_label"),
                col("probability_array").getItem(0).alias("benign_probability"),
                col("probability_array").getItem(1).alias("malignant_probability"),
                *[col(c) for c in numerical_cols]
            )
            
            # Write to PostgreSQL in append mode with smaller batch size
            postgres_df.coalesce(1).write \
                .option("batchsize", "100") \
                .option("isolationLevel", "READ_COMMITTED") \
                .jdbc(url=POSTGRES_URL,
                      table="predictions",
                      mode="append",
                      properties=POSTGRES_PROPERTIES)
            
            print(f"✅ Batch {batch_id}: Successfully wrote {postgres_df.count()} rows to PostgreSQL")
            
            # Force cleanup of the batch
            batch_df.unpersist()
            postgres_df.unpersist()
    except Exception as e:
        print(f"❌ Batch {batch_id}: Error writing to PostgreSQL - {str(e)}")
        import traceback
        traceback.print_exc()

# Read Streaming Data from Kafka producer
raw_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers",KAFKA_BROKER) \
            .option("subscribe", INPUT_KAFKA_TOPIC) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false")  \
            .load()

print(f"Connecting To KAFKA BROKER : {KAFKA_BROKER}")

# Parse the JSON data from kafka messages
parsed_stream = raw_stream.selectExpr("CAST(value AS STRING)") \
                .select(from_json(col("value"), schema).alias("data")) \
                .select("data.*")

# Define numerical columns
numerical_cols = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Handle null values by replacing them with 0
for col_name in numerical_cols:
    parsed_stream = parsed_stream.withColumn(
        col_name,
        when(col(col_name).isNull(), 0.0).otherwise(col(col_name))
    )

# Apply same feature preprocessing pipeline
assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol="features",
    handleInvalid="keep"  
)
assembled_stream = assembler.transform(parsed_stream)

# Apply MinMaxScaler to scale features
scaled_stream = scaler_model.transform(assembled_stream)

# Make real-time predictions using the loaded model
predictions_stream = model.transform(scaled_stream)

# Format The output to include original data, prediction, and probabilities
result_stream = predictions_stream.select(
    to_json(
        struct(
            *[col(c) for c in numerical_cols],  # original features
            col("prediction").alias("predicted_label"),
            col("probability").alias("prediction_probability")
        )
    ).alias("value")
)

# Write the result to kafka topic
kafka_query = result_stream.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers",KAFKA_BROKER) \
            .option("topic", OUTPUT_KAFKA_TOPIC) \
            .option("checkpointLocation", "file:///tmp/checkpoint") \
            .trigger(processingTime='500 milliseconds') \
            .outputMode("append") \
            .start()

postgres_query = predictions_stream \
                .writeStream \
                .foreachBatch(write_to_postgres) \
                .trigger(processingTime='500 milliseconds') \
                .option("checkpointLocation", "file:///tmp/checkpoint-postgres") \
                .option("maxOffsetPerTrigger", "100") \
                .start()

# Monitor both queries
try:
    while True:
        # Print progress information
        print("\nStreaming Status:")
        print(f"Kafka Query: {kafka_query.status}")
        print(f"PostgreSQL Query: {postgres_query.status}")
        
        # Sleep for a short interval
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down gracefully...")
    # Stop both queries
    kafka_query.stop()
    postgres_query.stop()
    spark.stop()