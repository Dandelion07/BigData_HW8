from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import os

os.environ["JAVA_HOME"] = r'path_to_JAVA'

# Create a Spark session
spark = SparkSession.builder.appName('UberKMeans').getOrCreate()

# Load the Uber travel data into a PySpark dataframe
df = spark.read.format("csv").option("header", "true").load(r"path/to/uber/data.csv")

# Convert latitude and longitude columns to numeric types
df = df.withColumn("Lat", col("Lat").cast("double"))
df = df.withColumn("Lon", col("Lon").cast("double"))

# Create a VectorAssembler to combine the latitude and longitude columns into a feature vector
assembler = VectorAssembler(inputCols=["Lat", "Lon"], outputCol="features")
df = assembler.transform(df)

# Set the number of clusters to 8
k = 8

# Split the data into training and test sets
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

# Train the KMeans model on the training data
kmeans = KMeans().setK(k).setSeed(42)
model = kmeans.fit(trainingData)

# Load new data to be clustered
new_data = spark.read.format('csv').options(header='true', inferSchema='true').load("<path_to_data_csv>")

# Assemble the feature vector for the new data
assembler = VectorAssembler(inputCols=['Lat', 'Lon'], outputCol='features')
new_data = assembler.transform(new_data).select('features')

# Use the pre-trained model to cluster the new data
clustered_data = model.transform(new_data)

# Group the data based on the cluster number and count the number of data in each cluster
cluster_counts = df.groupBy("prediction").count()

# Display the cluster count
cluster_counts.show()
