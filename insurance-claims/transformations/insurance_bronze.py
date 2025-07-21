import dlt
from pyspark.sql.functions import col, current_timestamp, input_file_name, expr

# Bronze Layer: Raw Data Ingestion
@dlt.table(
    comment="Raw healthcare insurance data with ingestion metadata",
    table_properties={
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }
)
def insurance_bronze():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("header", "true")
        .load("/Volumes/juan_dev/ml/data/inputs/insurance/")
        .withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_file", col("_metadata.file_path"))
        # .withColumn("customer_id", expr("uuid()"))
    )