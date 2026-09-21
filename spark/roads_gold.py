from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("UrbanMobility-Roads-Gold")
    .getOrCreate()
)

INPUT_PATH = "data/silver/osm/roads.parquet"
OUTPUT_PATH = "data/gold/roads"


roads = spark.read.parquet(INPUT_PATH)

print("Silver OSM schema:")
roads.printSchema()

print("Silver road count:", roads.count())


gold_roads = (
    roads
    .select(
        F.col("osm_id").alias("road_segment_id"),
        "geometry",
        "highway",
        "name",
        "ref",
        "maxspeed",
        "lanes",
        "oneway",
        "surface",
        "bridge",
        "tunnel",
        "length_m"
    )

    # Broad analytical road class
    .withColumn(
        "road_class",
        F.when(
            F.col("highway").isin(
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link"
            ),
            "major_highway"
        )
        .when(
            F.col("highway").isin(
                "primary",
                "primary_link",
                "secondary",
                "secondary_link"
            ),
            "arterial"
        )
        .when(
            F.col("highway").isin(
                "tertiary",
                "tertiary_link"
            ),
            "collector"
        )
        .when(
            F.col("highway").isin(
                "residential",
                "living_street",
                "unclassified"
            ),
            "local"
        )
        .otherwise("other")
    )

    # Missing-value indicators useful later for ML
    .withColumn(
        "maxspeed_missing",
        F.col("maxspeed").isNull().cast("integer")
    )
    .withColumn(
        "lanes_missing",
        F.col("lanes").isNull().cast("integer")
    )

    # Basic derived values
    .withColumn(
        "is_major_road",
        F.when(
            F.col("road_class").isin(
                "major_highway",
                "arterial"
            ),
            1
        ).otherwise(0)
    )
)


print("\nGold road schema:")
gold_roads.printSchema()

print("Gold road count:", gold_roads.count())

gold_roads.show(10, truncate=False)


(
    gold_roads.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(f"\nGold roads written to: {OUTPUT_PATH}")

spark.stop()