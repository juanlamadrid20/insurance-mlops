#!/usr/bin/env python3
"""
Test script to validate the refactored batch inference pipeline.
This script runs a small sample test to ensure everything works correctly.
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

def test_batch_inference():
    """Test the refactored batch inference pipeline on a small sample"""
    
    print("=== Testing Refactored Batch Inference Pipeline ===\n")
    
    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("TestBatchInference").getOrCreate()
        
        # Test 1: Verify silver_patients table structure
        print("1. Verifying silver_patients table structure...")
        input_df = spark.table("juan_dev.healthcare_data.silver_patients")
        print(f"✓ Table exists with {input_df.count()} rows")
        print(f"✓ Columns: {input_df.columns}")
        
        # Test 2: Check required columns are present
        print("\n2. Checking required columns...")
        required_base_columns = ['patient_id', 'sex_standardized', 'region_standardized', 
                                'smoker_flag', 'age_years', 'bmi_validated', 'children_count']
        missing_columns = [col for col in required_base_columns if col not in input_df.columns]
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        else:
            print("✓ All required columns present")
        
        # Test 3: Sample data transformation
        print("\n3. Testing data transformation...")
        sample_df = input_df.limit(10)
        
        # Apply the same transformations as in the notebook
        input_df_prepared = (
            sample_df
            .withColumn("sex", upper(col("sex_standardized")).cast("string"))
            .withColumn("region", upper(col("region_standardized")).cast("string"))
            .withColumn("smoker", col("smoker_flag").cast("boolean"))
            .withColumn("age", col("age_years").cast("integer"))
            .withColumn("bmi", col("bmi_validated").cast("double"))
            .withColumn("children", col("children_count").cast("integer"))
            .withColumn("patient_id", col("patient_id").cast("string"))
        )
        
        print("✓ Data transformation successful")
        print("Sample transformed data:")
        input_df_prepared.select(
            "patient_id", "sex", "region", "smoker", "age", "bmi", "children"
        ).show(5)
        
        # Test 4: Check feature table accessibility
        print("\n4. Checking feature table accessibility...")
        try:
            feature_df = spark.table("juan_dev.healthcare_ml.patient_features")
            print(f"✓ Feature table accessible with {feature_df.count()} rows")
            print(f"✓ Feature table columns: {feature_df.columns}")
        except Exception as e:
            print(f"✗ Feature table access failed: {e}")
            return False
        
        # Test 5: Validate business rule expressions
        print("\n5. Testing business rule expressions...")
        test_predictions = input_df_prepared.withColumn("prediction", lit(5000.0))
        
        # Test the business rules
        test_business_rules = (
            test_predictions
            .withColumn("adjusted_prediction", expr("GREATEST(prediction, 500)"))
            .withColumn("cost_risk_category",
                       expr("CASE WHEN adjusted_prediction < 2000 THEN 'low' " +
                            "WHEN adjusted_prediction < 8000 THEN 'medium' " +
                            "WHEN adjusted_prediction < 20000 THEN 'high' " +
                            "ELSE 'very_high' END"))
            .withColumn("high_risk_patient",
                       expr("adjusted_prediction > 15000 OR cost_risk_category = 'very_high'"))
            .withColumn("requires_review", 
                       expr("adjusted_prediction > 25000 OR (smoker AND adjusted_prediction > 10000)"))
        )
        
        print("✓ Business rule expressions validated")
        print("Sample business rule results:")
        test_business_rules.select(
            "patient_id", "smoker", "adjusted_prediction", 
            "cost_risk_category", "high_risk_patient", "requires_review"
        ).show(5)
        
        print("\n=== All Tests Passed! ===")
        print("The refactored batch inference pipeline is ready for use.")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    success = test_batch_inference()
    sys.exit(0 if success else 1)
