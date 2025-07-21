# Insurance MLOps Project

This project implements a comprehensive machine learning pipeline for predicting healthcare insurance costs using MLOps best practices on Databricks.

## Architecture & Components

### 1. Data Pipeline (Delta Live Tables)
- **Bronze Layer** (`insurance_bronze.py`): Raw data ingestion from cloud files with streaming
- **Silver Layer** (`insurance_silver.py`): Data quality validation with healthcare compliance checks
- **Data Generation** (`insurance-model-datagen.py`): Synthetic HIPAA-compliant patient data generator

### 2. Feature Engineering
- **Feature Store** (`insurance-model-feature.py`): Advanced healthcare risk features including:
  - BMI categorization (clinical standards)
  - Age risk scoring (1-5 scale)
  - Smoking impact factors
  - Regional cost adjustments
  - Health risk composite scores
- **Unity Catalog Integration**: Features stored in `juan_dev.ml.healthcare_features`

### 3. Model Training & MLflow Integration
- **Training Pipeline** (`insurance-model-train.py`): 
  - Random Forest and Gradient Boosting models
  - Healthcare-specific metrics and validation
  - Cross-validation with business metrics
  - MLflow experiment tracking
- **Registry**: Models stored in Unity Catalog (`juan_dev.ml.healthcare_insurance_model`)

### 4. Model Governance
- **Governance Framework** (`insurance-model-governance.py`):
  - Healthcare compliance validation
  - Performance thresholds (R² > 0.80, MAE < $3,000)
  - Automated model promotion with aliases
  - HIPAA compliance tags

### 5. Batch Inference
- **Batch Scoring** (`insurance-model-batch.py`):
  - Feature Engineering Client integration
  - Automated feature lookup
  - Spark optimization for large-scale processing
  - Risk categorization (low/medium/high/very_high)

## Key Features

### Healthcare-Specific Capabilities:
- **HIPAA Compliance**: Synthetic data generation and privacy controls
- **Clinical Validation**: BMI categories, age risk scores
- **Business Logic**: High-cost patient identification (95th percentile)
- **Risk Assessment**: Composite health risk scoring

### Technical Features:
- **Unity Catalog**: Centralized governance for data, features, and models
- **Feature Store**: Automated feature lookup and serving
- **Delta Live Tables**: Real-time data quality monitoring
- **MLflow 3.0**: Model versioning with metadata and governance
- **Streaming**: Real-time data ingestion and processing

## Data Schema

The project works with insurance data containing:
- **Demographics**: Age, sex, region
- **Health Indicators**: BMI, smoking status, children count
- **Target**: Insurance charges (healthcare costs)
- **Derived Features**: Risk scores, regional multipliers, composite health metrics

## Business Use Cases

1. **Cost Prediction**: Predict healthcare insurance costs for new customers
2. **Risk Assessment**: Identify high-risk patients requiring intervention
3. **Pricing Strategy**: Regional and demographic-based pricing models
4. **Healthcare Analytics**: Population health insights and trends

## Model Performance Targets

- **R² Score**: > 0.80
- **Mean Absolute Error**: < $3,000
- **High-Cost Accuracy**: > 75% for patients above 95th percentile
- **Business Impact**: Improved pricing accuracy and risk assessment

This is a production-ready MLOps system demonstrating enterprise-grade machine learning practices in the healthcare insurance domain, with governance, compliance, and scalability considerations.