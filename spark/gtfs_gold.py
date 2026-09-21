from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("UrbanMobility-GTFS-Gold")
    .getOrCreate()
)

BASE_PATH = "data/silver/gtfs"

routes = spark.read.parquet(f"{BASE_PATH}/routes.parquet")
stops = spark.read.parquet(f"{BASE_PATH}/stops.parquet")
trips = spark.read.parquet(f"{BASE_PATH}/trips.parquet")
stop_times = spark.read.parquet(f"{BASE_PATH}/stop_times.parquet")
shapes = spark.read.parquet(f"{BASE_PATH}/shapes.parquet")


print("Routes:", routes.count())
print("Stops:", stops.count())
print("Trips:", trips.count())
print("Stop times:", stop_times.count())
print("Shape points:", shapes.count())


# --------------------------------------------------
# Connect stop_times → trips → routes
# --------------------------------------------------

trip_routes = (
    trips
    .select(
        "trip_id",
        "route_id",
        "shape_id",
        "direction_id"
    )
)


stop_services = (
    stop_times
    .select(
        "trip_id",
        "stop_id",
        "arrival_time",
        "departure_time",
        "stop_sequence"
    )
    .join(
        trip_routes,
        on="trip_id",
        how="left"
    )
)


# --------------------------------------------------
# Aggregate service intensity by stop
# --------------------------------------------------

stop_metrics = (
    stop_services
    .groupBy("stop_id")
    .agg(
        F.countDistinct("trip_id").alias("trip_count"),
        F.countDistinct("route_id").alias("route_count"),
        F.countDistinct("shape_id").alias("shape_count"),
        F.min("arrival_time").alias("first_service_time"),
        F.max("departure_time").alias("last_service_time")
    )
)


# --------------------------------------------------
# Attach stop coordinates
# --------------------------------------------------

gold_transit = (
    stops
    .select(
        "stop_id",
        "stop_name",
        F.col("stop_lat").alias("latitude"),
        F.col("stop_lon").alias("longitude"),
        "wheelchair_boarding"
    )
    .join(
        stop_metrics,
        on="stop_id",
        how="left"
    )
    .fillna(
        {
            "trip_count": 0,
            "route_count": 0,
            "shape_count": 0
        }
    )
)


print("\nGold transit schema:")
gold_transit.printSchema()

gold_transit.show(10, truncate=False)


OUTPUT_PATH = "data/gold/transit"

(
    gold_transit.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(f"\nGold transit written to: {OUTPUT_PATH}")

spark.stop()