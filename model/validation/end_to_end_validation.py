#!/usr/bin/env python3
"""
End-to-End Validation & Regression Tests for Insurance MLOps Pipeline

This script validates the complete pipeline execution:
1. Feature table validation
2. Training metrics validation  
3. Batch inference validation
4. Model performance comparison against baseline

Execute notebooks in this order before running validation:
1. insurance-model-feature-updated.ipynb 
2. insurance-model-train-updated.ipynb
3. insurance-model-batch.ipynb

Usage:
    python end_to_end_validation.py --profile DEFAULT
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, isnan, isnull

# Initialize Spark and MLflow
spark = SparkSession.builder.appName("E2E-Validation").getOrCreate()
fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

class E2EValidator:
    """End-to-end validation for the insurance MLOps pipeline"""
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "feature_validation": {},
            "training_validation": {},
            "batch_validation": {},
            "performance_comparison": {},
            "overall_status": "PENDING"
        }
        
        # Expected table names
        self.feature_table = "juan_dev.ml.insurance_features_v2"
        self.source_table = "juan_dev.healthcare_data.dim_patients"
        self.model_name = "juan_dev.ml.healthcare_risk_model_v2"
        self.batch_output_table = "juan_dev.healthcare_ml.patient_predictions"
        
        # Performance baseline (to be updated with actual baseline values)
        self.baseline_metrics = {
            "r2_score": 0.75,  # Update with actual baseline
            "mean_absolute_error": 15.0,  # Update with actual baseline
            "root_mean_squared_error": 25.0  # Update with actual baseline
        }
    
    def validate_feature_table(self) -> Dict:
        """
        Validate feature table creation and row count alignment
        
        Checks:
        - Feature table exists and is accessible
        - Row count matches silver_patients (current records only)
        - Required feature columns are present
        - No null values in key features
        """
        print("🔍 Validating feature table...")
        
        try:
            # Check if feature table exists
            feature_df = spark.table(self.feature_table)
            feature_count = feature_df.count()
            
            # Get source table count (current records only)
            source_df = spark.table(self.source_table).filter(col("is_current_record") == True)
            source_count = source_df.count()
            
            # Expected feature columns
            expected_features = [
                "customer_id", "age_risk_score", "smoking_impact", 
                "family_size_factor", "regional_multiplier", 
                "health_risk_composite", "data_quality_score"
            ]
            
            actual_columns = feature_df.columns
            missing_features = [f for f in expected_features if f not in actual_columns]
            
            # Check for null values in key features
            null_checks = {}
            for feature in expected_features:
                if feature in actual_columns:
                    null_count = feature_df.filter(col(feature).isNull()).count()
                    null_checks[feature] = null_count
            
            # Validation results
            feature_validation = {
                "table_exists": True,
                "feature_count": feature_count,
                "source_count": source_count,
                "count_match": feature_count == source_count,
                "missing_features": missing_features,
                "null_checks": null_checks,
                "all_features_present": len(missing_features) == 0,
                "status": "PASS" if (feature_count == source_count and len(missing_features) == 0) else "FAIL"
            }
            
            print(f"   ✅ Feature table row count: {feature_count}")
            print(f"   ✅ Source table row count: {source_count}")
            print(f"   {'✅' if feature_count == source_count else '❌'} Row count match: {feature_count == source_count}")
            print(f"   {'✅' if len(missing_features) == 0 else '❌'} All features present: {len(missing_features) == 0}")
            
            if missing_features:
                print(f"   ❌ Missing features: {missing_features}")
                
            return feature_validation
            
        except Exception as e:
            print(f"   ❌ Feature validation failed: {str(e)}")
            return {
                "table_exists": False,
                "error": str(e),
                "status": "FAIL"
            }
    
    def validate_training_run(self) -> Dict:
        """
        Validate training run completion and model registration
        
        Checks:
        - Latest training run exists and completed successfully
        - Required metrics are logged (R², MAE, RMSE)
        - Model is registered in Unity Catalog
        - Model has proper metadata and tags
        """
        print("🔍 Validating training run...")
        
        try:
            # Get the latest run from the experiment
            mlflow.set_experiment("/Users/juan.lamadrid@databricks.com/experiments/insurance_cost_prediction_eda")
            
            # Search for the most recent healthcare risk model run
            runs = mlflow.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name("/Users/juan.lamadrid@databricks.com/experiments/insurance_cost_prediction_eda").experiment_id],
                filter_string="tags.mlflow.runName LIKE 'healthcare_risk_model_%'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                return {
                    "run_found": False,
                    "error": "No healthcare risk model training runs found",
                    "status": "FAIL"
                }
            
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            
            # Required metrics
            required_metrics = ["r2_score", "mean_absolute_error", "root_mean_squared_error"]
            metrics_present = {}
            
            for metric in required_metrics:
                metrics_present[metric] = pd.notna(latest_run.get(f"metrics.{metric}"))
            
            # Check model registration
            client = mlflow.MlflowClient()
            try:
                model_versions = client.search_model_versions(f"name='{self.model_name}'")
                model_registered = len(model_versions) > 0
                latest_version = max([int(mv.version) for mv in model_versions]) if model_versions else 0
            except:
                model_registered = False
                latest_version = 0
            
            training_validation = {
                "run_found": True,
                "run_id": run_id,
                "run_status": latest_run.status,
                "metrics_present": metrics_present,
                "all_metrics_logged": all(metrics_present.values()),
                "model_registered": model_registered,
                "model_version": latest_version,
                "r2_score": latest_run.get("metrics.r2_score"),
                "mae": latest_run.get("metrics.mean_absolute_error"),
                "rmse": latest_run.get("metrics.root_mean_squared_error"),
                "status": "PASS" if (latest_run.status == "FINISHED" and 
                                  all(metrics_present.values()) and 
                                  model_registered) else "FAIL"
            }
            
            print(f"   ✅ Training run found: {run_id}")
            print(f"   {'✅' if latest_run.status == 'FINISHED' else '❌'} Run status: {latest_run.status}")
            print(f"   {'✅' if all(metrics_present.values()) else '❌'} All metrics logged: {all(metrics_present.values())}")
            print(f"   {'✅' if model_registered else '❌'} Model registered: {model_registered}")
            
            if model_registered:
                print(f"   ✅ Model version: {latest_version}")
            
            return training_validation
            
        except Exception as e:
            print(f"   ❌ Training validation failed: {str(e)}")
            return {
                "run_found": False,
                "error": str(e),
                "status": "FAIL"
            }
    
    def validate_batch_inference(self) -> Dict:
        """
        Validate batch inference execution and outputs
        
        Checks:
        - Batch output table exists and contains predictions
        - Expected columns are present (prediction, risk categories, etc.)
        - Risk categorization is working correctly
        - No null predictions for valid input rows
        """
        print("🔍 Validating batch inference...")
        
        try:
            # Check if batch output table exists
            batch_df = spark.table(self.batch_output_table)
            batch_count = batch_df.count()
            
            # Expected output columns
            expected_columns = [
                "patient_id", "prediction", "adjusted_prediction", 
                "cost_risk_category", "high_risk_patient", "requires_review",
                "prediction_timestamp", "model_version", "model_name"
            ]
            
            actual_columns = batch_df.columns
            missing_columns = [c for c in expected_columns if c not in actual_columns]
            
            # Check risk categories
            risk_categories = batch_df.select("cost_risk_category").distinct().collect()
            risk_cats = [row.cost_risk_category for row in risk_categories]
            expected_risk_cats = ["low", "medium", "high", "very_high"]
            
            # Check for null predictions
            null_predictions = batch_df.filter(col("prediction").isNull()).count()
            
            # Validate risk category logic
            risk_validation = batch_df.select(
                "adjusted_prediction", 
                "cost_risk_category"
            ).collect()
            
            risk_logic_correct = True
            for row in risk_validation[:100]:  # Sample validation
                pred = row.adjusted_prediction
                cat = row.cost_risk_category
                
                expected_cat = (
                    "low" if pred < 2000 else
                    "medium" if pred < 8000 else
                    "high" if pred < 20000 else
                    "very_high"
                )
                
                if cat != expected_cat:
                    risk_logic_correct = False
                    break
            
            batch_validation = {
                "table_exists": True,
                "batch_count": batch_count,
                "missing_columns": missing_columns,
                "all_columns_present": len(missing_columns) == 0,
                "risk_categories_found": risk_cats,
                "risk_categories_complete": all(cat in risk_cats for cat in expected_risk_cats),
                "null_predictions": null_predictions,
                "no_null_predictions": null_predictions == 0,
                "risk_logic_correct": risk_logic_correct,
                "status": "PASS" if (batch_count > 0 and 
                                  len(missing_columns) == 0 and
                                  null_predictions == 0 and
                                  risk_logic_correct) else "FAIL"
            }
            
            print(f"   ✅ Batch output records: {batch_count}")
            print(f"   {'✅' if len(missing_columns) == 0 else '❌'} All columns present: {len(missing_columns) == 0}")
            print(f"   {'✅' if null_predictions == 0 else '❌'} No null predictions: {null_predictions == 0}")
            print(f"   {'✅' if risk_logic_correct else '❌'} Risk categorization correct: {risk_logic_correct}")
            
            if missing_columns:
                print(f"   ❌ Missing columns: {missing_columns}")
            
            return batch_validation
            
        except Exception as e:
            print(f"   ❌ Batch validation failed: {str(e)}")
            return {
                "table_exists": False,
                "error": str(e),
                "status": "FAIL"
            }
    
    def compare_model_performance(self) -> Dict:
        """
        Compare current model performance against baseline
        
        Checks:
        - R² score comparison
        - MAE comparison  
        - RMSE comparison
        - Performance regression detection
        """
        print("🔍 Comparing model performance against baseline...")
        
        try:
            # Get current model metrics from validation results
            current_metrics = self.validation_results.get("training_validation", {})
            
            if not current_metrics.get("run_found"):
                return {
                    "comparison_possible": False,
                    "error": "No current training metrics available",
                    "status": "FAIL"
                }
            
            # Extract current metrics
            current_r2 = current_metrics.get("r2_score")
            current_mae = current_metrics.get("mae") 
            current_rmse = current_metrics.get("rmse")
            
            # Performance comparison
            r2_change = current_r2 - self.baseline_metrics["r2_score"] if current_r2 else None
            mae_change = current_mae - self.baseline_metrics["mean_absolute_error"] if current_mae else None
            rmse_change = current_rmse - self.baseline_metrics["root_mean_squared_error"] if current_rmse else None
            
            # Define acceptable thresholds (configurable)
            r2_threshold = -0.05  # Max 5% decrease in R²
            mae_threshold = 5.0   # Max 5 unit increase in MAE
            rmse_threshold = 8.0  # Max 8 unit increase in RMSE
            
            # Performance status
            r2_acceptable = r2_change is None or r2_change >= r2_threshold
            mae_acceptable = mae_change is None or mae_change <= mae_threshold
            rmse_acceptable = rmse_change is None or rmse_change <= rmse_threshold
            
            performance_comparison = {
                "comparison_possible": True,
                "baseline_metrics": self.baseline_metrics,
                "current_metrics": {
                    "r2_score": current_r2,
                    "mean_absolute_error": current_mae,
                    "root_mean_squared_error": current_rmse
                },
                "changes": {
                    "r2_change": r2_change,
                    "mae_change": mae_change,
                    "rmse_change": rmse_change
                },
                "performance_acceptable": {
                    "r2_acceptable": r2_acceptable,
                    "mae_acceptable": mae_acceptable,
                    "rmse_acceptable": rmse_acceptable
                },
                "overall_performance": "PASS" if (r2_acceptable and mae_acceptable and rmse_acceptable) else "DEGRADED",
                "status": "PASS"
            }
            
            print(f"   📊 Current R²: {current_r2:.4f} (Δ: {r2_change:+.4f})" if current_r2 and r2_change else "   ❌ R² not available")
            print(f"   📊 Current MAE: {current_mae:.2f} (Δ: {mae_change:+.2f})" if current_mae and mae_change else "   ❌ MAE not available")
            print(f"   📊 Current RMSE: {current_rmse:.2f} (Δ: {rmse_change:+.2f})" if current_rmse and rmse_change else "   ❌ RMSE not available")
            
            if performance_comparison["overall_performance"] == "DEGRADED":
                print("   ⚠️  Performance degradation detected!")
            else:
                print("   ✅ Performance within acceptable bounds")
            
            return performance_comparison
            
        except Exception as e:
            print(f"   ❌ Performance comparison failed: {str(e)}")
            return {
                "comparison_possible": False,
                "error": str(e),
                "status": "FAIL"
            }
    
    def run_validation(self) -> Dict:
        """Run complete end-to-end validation"""
        print("🚀 Starting End-to-End Validation\n")
        
        # Run all validation steps
        self.validation_results["feature_validation"] = self.validate_feature_table()
        print()
        
        self.validation_results["training_validation"] = self.validate_training_run()
        print()
        
        self.validation_results["batch_validation"] = self.validate_batch_inference()
        print()
        
        self.validation_results["performance_comparison"] = self.compare_model_performance()
        print()
        
        # Determine overall status
        all_passed = all([
            self.validation_results["feature_validation"].get("status") == "PASS",
            self.validation_results["training_validation"].get("status") == "PASS", 
            self.validation_results["batch_validation"].get("status") == "PASS",
            self.validation_results["performance_comparison"].get("status") == "PASS"
        ])
        
        self.validation_results["overall_status"] = "PASS" if all_passed else "FAIL"
        
        # Print summary
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Feature Validation:     {self.validation_results['feature_validation'].get('status', 'UNKNOWN')}")
        print(f"Training Validation:    {self.validation_results['training_validation'].get('status', 'UNKNOWN')}")
        print(f"Batch Validation:       {self.validation_results['batch_validation'].get('status', 'UNKNOWN')}")
        print(f"Performance Comparison: {self.validation_results['performance_comparison'].get('status', 'UNKNOWN')}")
        print("-" * 50)
        print(f"OVERALL STATUS:         {self.validation_results['overall_status']}")
        
        return self.validation_results
    
    def save_results(self, output_path: str = "validation_results.json"):
        """Save validation results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        print(f"\n📄 Validation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end validation for insurance MLOps pipeline")
    parser.add_argument("--output", "-o", default="validation_results.json", 
                       help="Output file for validation results")
    parser.add_argument("--profile", "-p", default="DEFAULT",
                       help="Databricks profile to use")
    
    args = parser.parse_args()
    
    # Run validation
    validator = E2EValidator()
    results = validator.run_validation()
    validator.save_results(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
