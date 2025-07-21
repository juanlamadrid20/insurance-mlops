# Databricks notebook source
from databricks.feature_engineering import FeatureEngineeringClient
import mlflow.pyfunc
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, FloatType, DoubleType

class HealthcareBatchInference:

    def __init__(self, model_name="juan_dev.ml.healthcare_insurance_model", model_alias="champion"):
        self.model_name = model_name
        self.model_alias = model_alias
        self.fe = FeatureEngineeringClient()
        
        # Spark optimization for batch processing
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
    
    def run_batch_inference(self, input_table, output_table):
        """Execute batch inference with automatic feature lookup"""
      
        # Load model from Unity Catalog
        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        model_version = mlflow.MlflowClient().get_model_version_by_alias(
            self.model_name, self.model_alias
        ).version

        print(f"Loading model version {model_version} from {model_uri}")
        
        # Ensure input data contains all required columns
        required_columns = ['smoker', 'sex', 'customer_id', 'age_group', 'region']

        input_df = spark.table(input_table)
        
        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")
        
        # Ensure input data only contains numeric values and required columns
        numeric_columns = [field.name for field in input_df.schema.fields if field.dataType in [IntegerType(), FloatType(), DoubleType()]]
        # Exclude non-numeric required columns
        numeric_columns = [col for col in numeric_columns if col not in required_columns]
        input_df = input_df.select(*required_columns, *numeric_columns)
        # Convert categorical columns to string type and customer_id to float if necessary
        input_df = input_df.withColumn("smoker", col("smoker").cast("string")) \
                           .withColumn("sex", col("sex").cast("string")) \
                           .withColumn("age_group", col("age_group").cast("string")) \
                           .withColumn("region", col("region").cast("string"))

        # Batch scoring with feature engineering integration
        predictions_df = self.fe.score_batch(
            df=input_df,
            model_uri=model_uri
        )

        display(predictions_df)

        # display(predictions_df)
        
        # # Add metadata and business logic
        # final_predictions = (
        #     predictions_df
        #     .withColumn("prediction_timestamp", current_timestamp())
        #     .withColumn("model_version", lit(model_version))
        #     .withColumn("model_name", lit(self.model_name))
            
        #     # Business rule: minimum charge threshold
        #     .withColumn("adjusted_prediction", 
        #                expr("GREATEST(prediction, 500)"))
            
        #     # Risk categorization
        #     .withColumn("cost_risk_category",
        #                expr("CASE WHEN adjusted_prediction < 2000 THEN 'low' " +
        #                     "WHEN adjusted_prediction < 8000 THEN 'medium' " +
        #                     "WHEN adjusted_prediction < 20000 THEN 'high' " +
        #                     "ELSE 'very_high' END"))
            
        #     # Confidence intervals (approximate)
        #     .withColumn("prediction_lower_bound", 
        #                expr("adjusted_prediction * 0.85"))
        #     .withColumn("prediction_upper_bound", 
        #                expr("adjusted_prediction * 1.15"))
        # )
        
        # Write results with Delta Lake optimization
        # (final_predictions
        #  .write
        #  .mode("overwrite")
        #  .option("overwriteSchema", "true")
        #  .saveAsTable(output_table))
        
        # Log batch inference metrics
        # with mlflow.start_run(run_name="batch_inference"):
        #     inference_count = final_predictions.count()
        #     avg_prediction = final_predictions.agg(
        #         avg("adjusted_prediction")
        #     ).collect()[0][0]
            
        #     mlflow.log_metrics({
        #         "batch_inference_count": inference_count,
        #         "average_predicted_cost": avg_prediction,
        #         "model_version": float(model_version)
        #     })
        
        # display(final_predictions)
        # return final_predictions

# Example batch inference execution
batch_inference = HealthcareBatchInference()

results = batch_inference.run_batch_inference(
    input_table="juan_dev.ml.insurance_silver",
    output_table="juan_dev.ml.cost_predictions"
)

# COMMAND ----------

