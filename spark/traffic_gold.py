from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("UrbanMobility-Traffic-Gold")
    .config("spark.hadoop.fs.permissions.umask-mode", "000")
    .getOrCreate()
)

INPUT_PATH = "data/silver/traffic/traffic.parquet"
OUTPUT_PATH = "data/gold/traffic"


traffic = spark.read.parquet(INPUT_PATH)

print("Silver traffic schema:")
traffic.printSchema()

print("Traffic rows:", traffic.count())


gold_traffic = (
    traffic
    .select(
        F.col("varco").alias("traffic_location"),
        F.col("data").alias("date"),
        F.col("passaggi_24h").alias("traffic_count_24h")
    )

    .withColumn(
        "year",
        F.year("date")
    )

    .withColumn(
        "month",
        F.month("date")
    )

    .withColumn(
        "day",
        F.dayofmonth("date")
    )

    .withColumn(
        "day_of_week",
        F.dayofweek("date")
    )

    .withColumn(
        "week_of_year",
        F.weekofyear("date")
    )

    .withColumn(
        "is_weekend",
        F.when(
            F.dayofweek("date").isin(1, 7),
            1
        ).otherwise(0)
    )
)


print("\nGold traffic schema:")
gold_traffic.printSchema()

gold_traffic.show(10, truncate=False)


(
    gold_traffic.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(f"\nGold traffic written to: {OUTPUT_PATH}")

spark.stop()