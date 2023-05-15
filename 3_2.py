from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from sklearn.cluster import KMeans
from pyspark.sql.functions import count
from pyspark.sql.window import Window
import pyspark.sql.functions as F
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

# Load new data to be clustered
new_data = spark.read.format('csv').options(header='true', inferSchema='true').load(r"<path_to_data_csv>")

# Assemble the feature vector for the new data
assembler = VectorAssembler(inputCols=['Lat', 'Lon'], outputCol='features')
new_data = assembler.transform(new_data).select('features')

# Use the pre-trained model to cluster the new data
clustered_data = model.transform(new_data)

# Group the data by cluster and date/time, and count the number of trips in each group
counted_data = clustered_data.groupBy('cluster', 'Date/Time').agg(count('*').alias('count'))

# Define a window specification to partition the data by cluster and order by count in descending order
w = Window.partitionBy('cluster').orderBy(F.desc('count'))

# Get the peak cluster by selecting the first row for each cluster based on the window specification
peak_cluster = counted_data.select('*', F.rank().over(w).alias('rank')).filter(F.col('rank') == 1).select('cluster').collect()[0][0]
df['cluster'] = kmeans.predict(df[['Lat', 'Lon']])
filtered_data = df[df['cluster'] == peak_cluster]

# Extract the week information from the 'Date/Time' column and group the filtered data by the week and the 'Base' column
filtered_data['week'] = filtered_data['Date/Time'].dt.week
grouped_data = filtered_data.groupby(['week', 'Base'])

# Count the number of data points in each group to get the number of available services during each week for each service area
service_counts = grouped_data.size().reset_index(name='counts')
