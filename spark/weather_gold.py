from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("UrbanMobility-Weather-Gold")
    .config("spark.hadoop.fs.permissions.umask-mode", "000")
    .getOrCreate()
)

INPUT_PATH = "data/silver/weather/weather.parquet"
OUTPUT_PATH = "data/gold/weather"


weather = spark.read.parquet(INPUT_PATH)

print("Silver weather schema:")
weather.printSchema()

print("Weather rows:", weather.count())


gold_weather = (
    weather
    .select(
        F.col("time").alias("timestamp"),
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "latitude",
        "longitude"
    )

    .withColumn(
        "date",
        F.to_date("timestamp")
    )

    .withColumn(
        "hour",
        F.hour("timestamp")
    )

    .withColumn(
        "year",
        F.year("timestamp")
    )

    .withColumn(
        "month",
        F.month("timestamp")
    )

    .withColumn(
        "day_of_week",
        F.dayofweek("timestamp")
    )

    .withColumn(
        "is_weekend",
        F.when(
            F.dayofweek("timestamp").isin(1, 7),
            1
        ).otherwise(0)
    )

    .withColumn(
        "is_raining",
        F.when(
            F.col("precipitation") > 0,
            1
        ).otherwise(0)
    )
)


print("\nGold weather schema:")
gold_weather.printSchema()

gold_weather.show(10, truncate=False)


(
    gold_weather.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(f"\nGold weather written to: {OUTPUT_PATH}")

spark.stop()