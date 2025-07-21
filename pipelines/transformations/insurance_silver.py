import dlt
from pyspark.sql.functions import col, upper, expr
from pyspark.sql.window import Window
from utilities import utils

# age,sex,bmi,children,smoker,region,charges
# Silver Layer: Data Quality and Healthcare Compliance
@dlt.table(
    comment="Validated insurance data with healthcare compliance checks"
)
# @dlt.expect_or_fail("valid_customer_id", "customer_id IS NOT NULL")
@dlt.expect_or_fail("valid_age", "age BETWEEN 18 AND 100")
@dlt.expect_or_fail("valid_bmi", "bmi BETWEEN 10 AND 60")
@dlt.expect_or_drop("valid_charges", "charges > 0")
@dlt.expect("reasonable_charges", "charges < 100000")  # Warning only
def insurance_silver():
    
    return (
        dlt.read_stream("insurance_bronze")
        .filter(col("age").between(18, 100))  # Filter out invalid ages
        .filter(col("bmi").between(10, 60))  # Filter out invalid bmi
        .select(
            col("customer_id").cast("bigint"),
            # col("customer_id"),
            col("age").cast("int"),
            upper(col("sex")).alias("sex"),
            col("bmi").cast("double"),
            col("children").cast("int"),
            expr("CASE WHEN lower(smoker) = 'yes' THEN true ELSE false END").alias("smoker"),
            upper(col("region")).alias("region"),
            col("charges").cast("double"),
            # col("medical_history"),
            # col("family_history"),
            # col("lifestyle_factors"),
            # col("policy_type"),
            # col("coverage_amount").cast("double"),
            # col("deductible").cast("double"),
            # col("timestamp").cast("timestamp"),
            col("ingestion_timestamp")
        )
        .withColumn("age_group", 
                   expr("CASE WHEN age < 30 THEN 'young' " +
                        "WHEN age < 50 THEN 'middle' " +
                        "ELSE 'senior' END"))
    )