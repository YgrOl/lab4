from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import os

# Створення SparkSession
spark = SparkSession.builder \
    .appName("TripAnalysis") \
    .getOrCreate()

# Шлях до CSV-файлу
data_path = "/opt/bitnami/spark/jobs/data.csv"

# Читання даних
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Попереднє очищення: заміна "," на "" в tripduration і приведення типу
df = df.withColumn("tripduration", regexp_replace("tripduration", ",", "").cast("float"))

# Додавання колонки дати і місяця
df = df.withColumn("start_date", to_date("start_time")) \
    .withColumn("month", date_format("start_date", "yyyy-MM"))

# Функція a: Середня тривалість поїздки на день
def average_trip_duration_per_day(df):
    return df.groupBy("start_date").agg(avg("tripduration").alias("average_duration"))

# Функція b: Кількість поїздок на день
def trips_per_day(df):
    return df.groupBy("start_date").count().withColumnRenamed("count", "trips_count")

# Функція c: Найпопулярніша стартова станція щомісяця
def most_popular_station_per_month(df):
    return df.groupBy("month", "from_station_name") \
        .count() \
        .withColumn("rank", row_number().over(Window.partitionBy("month").orderBy(desc("count")))) \
        .filter(col("rank") == 1) \
        .drop("rank")

# Функція d: Топ-3 станції за останні 14 днів
def top3_stations_last_14_days(df):
    latest_date = df.agg(max("start_date")).first()[0]
    df_recent = df.filter(col("start_date") >= date_sub(lit(latest_date), 13))
    return df_recent.groupBy("start_date", "from_station_name") \
        .count() \
        .withColumn("rank", row_number().over(Window.partitionBy("start_date").orderBy(desc("count")))) \
        .filter(col("rank") <= 3) \
        .drop("rank")

# Функція e: Хто їздить довше — чоловіки чи жінки
def avg_trip_by_gender(df):
    return df.groupBy("gender").agg(avg("tripduration").alias("avg_duration"))

# Створення вихідного каталогу
output_dir = "/opt/bitnami/spark/jobs/out"
os.makedirs(output_dir, exist_ok=True)

# Запис результатів
average_trip_duration_per_day(df).coalesce(1).write.csv(f"{output_dir}/avg_duration_per_day", header=True, mode="overwrite")
trips_per_day(df).coalesce(1).write.csv(f"{output_dir}/trips_per_day", header=True, mode="overwrite")
most_popular_station_per_month(df).coalesce(1).write.csv(f"{output_dir}/popular_station_per_month", header=True, mode="overwrite")
top3_stations_last_14_days(df).coalesce(1).write.csv(f"{output_dir}/top3_stations_14_days", header=True, mode="overwrite")
avg_trip_by_gender(df).coalesce(1).write.csv(f"{output_dir}/avg_by_gender", header=True, mode="overwrite")

spark.stop()
