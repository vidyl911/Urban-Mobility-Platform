from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("UrbanMobility-POIs-Gold")
    .config("spark.hadoop.fs.permissions.umask-mode", "000")
    .getOrCreate()
)

INPUT_PATH = "data/silver/pois/pois.parquet"
OUTPUT_PATH = "data/gold/pois"


pois = spark.read.parquet(INPUT_PATH)

print("Silver POI schema:")
pois.printSchema()

print("POI rows:", pois.count())


gold_pois = (
    pois
    .select(
        "osm_id",
        "element_type",
        "latitude",
        "longitude",
        "poi_category",
        "amenity",
        "shop",
        "tourism",
        "leisure",
        "office",
        "name"
    )

    .withColumn(
        "poi_group",

        F.when(
            F.col("amenity").isNotNull(),
            "amenity"
        )

        .when(
            F.col("shop").isNotNull(),
            "retail"
        )

        .when(
            F.col("tourism").isNotNull(),
            "tourism"
        )

        .when(
            F.col("leisure").isNotNull(),
            "leisure"
        )

        .when(
            F.col("office").isNotNull(),
            "office"
        )

        .otherwise("other")
    )
)


print("\nGold POI schema:")
gold_pois.printSchema()

print("\nPOI group distribution:")

(
    gold_pois
    .groupBy("poi_group")
    .count()
    .orderBy(F.desc("count"))
    .show()
)


(
    gold_pois.write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(f"\nGold POIs written to: {OUTPUT_PATH}")

spark.stop()