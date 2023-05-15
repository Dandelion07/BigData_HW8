# Load the pre-trained KMeans model
from pyspark.sql.functions import concat_ws
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, to_date, current_date, datediff
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import count
import os

os.environ["JAVA_HOME"] = r'path_to_JAVA'

# Create a Spark session
spark = SparkSession.builder.appName('UberKMeans').getOrCreate()

# Load the Uber travel data into a PySpark dataframe
df = spark.read.format("csv").option("header", "true").load(r"<path_to_data_csv>")

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

# Load the data from the Excel file and filter for the last 10 days
data = spark.read.format('csv').options(header='true', inferSchema='true').load("<path_to_data_csv>")
data = data.filter(datediff(current_date(), to_date(col('Date/Time'), 'yyyy-MM-dd HH:mm:ss')) <= 10)

# Apply the KMeans model to the filtered data
assembler = VectorAssembler(inputCols=['Lat', 'Lon'], outputCol='features')
data = assembler.transform(data).select('features')
clustered_data = model.transform(data)

# Count the number of data points in each cluster and find the cluster with the highest count
cluster_counts = clustered_data.groupBy('prediction').agg(count('*').alias('count'))
peak_cluster = cluster_counts.orderBy(col('count').desc()).first()['prediction']

# Filter the data to keep only the data points in the cluster with the highest count
filtered_data = clustered_data.filter(col('prediction') == peak_cluster)

# Group the data by the (Lat, Lon) coordinates and count the number of data points in each coordinate
coordinates = filtered_data.select(concat_ws(',', col('Lat'), col('Lon')).alias('coordinates'))
coordinate_counts = coordinates.groupBy('coordinates').agg(count('*').alias('count'))

# Sort the data points by the count in descending order and select the top 10 points
top_coordinates = coordinate_counts.orderBy(col('count').desc()).limit(10).selectExpr('split(coordinates, ",")[0] as Lat', 'split(coordinates, ",")[1] as Lon', 'count')
