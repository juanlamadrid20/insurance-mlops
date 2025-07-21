# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
import mlflow

# Start MLflow experiment for EDA tracking
mlflow.set_experiment("/Users/juan.lamadrid@databricks.com/experiments/insurance_cost_prediction_eda")

with mlflow.start_run(run_name="healthcare_insurance_eda"):
    # Load cleaned data
    df = spark.table("juan_dev.ml.insurance_silver").toPandas()
    
    # Healthcare-specific data profiling
    eda_results = {
        "total_patients": len(df),
        "avg_age": df['age'].mean(),
        "smoker_percentage": (df['smoker'].sum() / len(df)) * 100,
        "high_cost_threshold": df['charges'].quantile(0.95),
        "missing_data_percentage": (df.isnull().sum() / len(df)) * 100
    }
    
    # Log healthcare compliance metrics
    # mlflow.log_metrics(eda_results)
    # Log healthcare compliance metrics
    # mlflow.log_metrics({k: float(v) for k, v in eda_results.items()})
    
    # Risk factor analysis
    risk_analysis = df.groupby(['smoker', 'age_group']).agg({
        'charges': ['mean', 'median', 'std'],
        'bmi': 'mean'
    }).round(2)
    
    # Log visualizations
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='region', y='charges', hue='smoker')
    plt.title('Healthcare Costs by Region and Smoking Status')
    plt.xticks(rotation=45)
    mlflow.log_figure(plt.gcf(), "cost_distribution_by_region_smoking.png")
    
    # Feature correlation analysis
    correlation_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Healthcare Feature Correlations')
    mlflow.log_figure(plt.gcf(), "feature_correlations.png")
    
    # Healthcare risk insights
    high_risk_patients = df[
        (df['smoker'] == True) & 
        (df['bmi'] > 30) & 
        (df['age'] > 50)
    ]
    
    mlflow.log_metrics({
        "high_risk_patients_count": len(high_risk_patients),
        "high_risk_avg_cost": high_risk_patients['charges'].mean()
    })

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
import mlflow
import mlflow.sklearn

fe = FeatureEngineeringClient()

# Healthcare insurance feature engineering
def create_healthcare_features():
    # Load silver data
    df = spark.table("juan_dev.ml.insurance_silver")
    
    # Advanced healthcare risk features
    healthcare_features = (
        df
        # BMI categorization (clinical standard)
        .withColumn("bmi_category",
                   expr("CASE WHEN bmi < 18.5 THEN 'underweight' " +
                        "WHEN bmi < 25 THEN 'normal' " +
                        "WHEN bmi < 30 THEN 'overweight' " +
                        "ELSE 'obese' END"))
        
        # Age risk scoring
        .withColumn("age_risk_score",
                   expr("CASE WHEN age < 25 THEN 1 " +
                        "WHEN age < 35 THEN 2 " +
                        "WHEN age < 50 THEN 3 " +
                        "WHEN age < 65 THEN 4 " +
                        "ELSE 5 END"))
        
        # Smoking impact factor
        .withColumn("smoking_impact", 
                   expr("CASE WHEN smoker THEN age * 2.5 ELSE age * 1.0 END"))
        
        # Family size risk adjustment
        .withColumn("family_size_factor", 
                   expr("1 + (children * 0.15)"))
        
        # Regional cost adjustment (based on healthcare infrastructure)
        .withColumn("regional_multiplier",
                   expr("CASE WHEN region = 'NORTHEAST' THEN 1.2 " +
                        "WHEN region = 'NORTHWEST' THEN 1.1 " +
                        "WHEN region = 'SOUTHEAST' THEN 1.0 " +
                        "ELSE 0.95 END"))
        
        # Health risk composite score
        .withColumn("health_risk_composite",
                   expr("(age_risk_score * 20) + " +
                        "(CASE WHEN smoker THEN 50 ELSE 0 END) + " +
                        "(CASE WHEN bmi > 30 THEN 30 ELSE 0 END)"))
        
        # Policy interaction features
        # .withColumn("coverage_to_deductible_ratio",
        #            expr("coverage_amount / GREATEST(deductible, 1)"))
    )
    
    return healthcare_features

# Create feature table in Unity Catalog
healthcare_features_df = create_healthcare_features()

fe.create_table(
    name="juan_dev.ml.healthcare_features",
    primary_keys=["customer_id"],
    df=healthcare_features_df,
    description="Healthcare-specific features for insurance cost prediction"
)

# COMMAND ----------

