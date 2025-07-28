# End-to-End Validation & Regression Tests

This document outlines the comprehensive validation framework for the Insurance MLOps pipeline, covering the execution sequence and validation checks for the three main notebooks.

## 📋 Execution Sequence

Execute the notebooks in this specific order:

1. **Feature Engineering**: `model/training/insurance-model-feature-updated.ipynb`
2. **Model Training**: `model/training/insurance-model-train-updated.ipynb`  
3. **Batch Inference**: `model/batch/insurance-model-batch.ipynb`

## 🔍 Validation Checks

### 1. Feature Table Validation

**Notebook**: `insurance-model-feature-updated.ipynb`

**Validates**:
- ✅ Feature table `juan_dev.ml.insurance_features_v2` is created successfully
- ✅ Row count matches `juan_dev.healthcare_data.dim_patients` (current records only)
- ✅ All required feature columns are present:
  - `customer_id` (primary key)
  - `age_risk_score`
  - `smoking_impact`
  - `family_size_factor`
  - `regional_multiplier`
  - `health_risk_composite`
  - `data_quality_score`
- ✅ No null values in critical features
- ✅ Data types are correct and consistent

**Expected Outcome**:
```
Feature table row count: ~1,000-10,000 (matches source)
Source table row count: ~1,000-10,000 
Row count match: TRUE
All features present: TRUE
```

### 2. Training Run Validation

**Notebook**: `insurance-model-train-updated.ipynb`

**Validates**:
- ✅ Training run completes successfully with status `FINISHED`
- ✅ All required metrics are logged to MLflow:
  - `r2_score` (R-squared)
  - `mean_absolute_error` (MAE)
  - `root_mean_squared_error` (RMSE)
  - `cv_r2_mean` (Cross-validation R²)
  - `high_risk_accuracy` (Healthcare-specific metric)
- ✅ Model is registered in Unity Catalog as `juan_dev.ml.healthcare_risk_model_v2`
- ✅ Model metadata includes:
  - Algorithm type (Random Forest/Gradient Boosting)
  - Feature count and names
  - Healthcare compliance flags
  - Schema version information

**Expected Outcome**:
```
Training run found: <run_id>
Run status: FINISHED
All metrics logged: TRUE
Model registered: TRUE
Model version: 1 (or latest)
R² Score: 0.70-0.85
MAE: 10-20
RMSE: 15-30
```

### 3. Batch Inference Validation

**Notebook**: `insurance-model-batch.ipynb`

**Validates**:
- ✅ Batch output table `juan_dev.healthcare_ml.patient_predictions` is created
- ✅ All expected columns are present:
  - `patient_id`
  - `prediction` (raw model output)
  - `adjusted_prediction` (business rule applied)
  - `cost_risk_category` (low/medium/high/very_high)
  - `high_risk_patient` (boolean flag)
  - `requires_review` (boolean flag)
  - `prediction_timestamp`
  - `model_version`
  - `model_name`
- ✅ Risk categorization logic is correct:
  - Low: < $2,000
  - Medium: $2,000 - $8,000  
  - High: $8,000 - $20,000
  - Very High: > $20,000
- ✅ No null predictions for valid input rows
- ✅ Business rules are applied correctly (minimum $500 threshold)

**Expected Outcome**:
```
Batch output records: ~1,000-10,000 (matches input)
All columns present: TRUE
No null predictions: TRUE
Risk categorization correct: TRUE
Risk distribution: ~60% low, ~25% medium, ~12% high, ~3% very_high
```

### 4. Model Performance Comparison

**Validates**:
- ✅ Current model performance vs. baseline metrics
- ✅ Performance regression detection
- ✅ Acceptable performance thresholds:
  - R² decrease ≤ 5% (threshold: -0.05)
  - MAE increase ≤ 5 units (threshold: +5.0)
  - RMSE increase ≤ 8 units (threshold: +8.0)

**Expected Outcome**:
```
Current R²: 0.7500 (Δ: +0.0000)
Current MAE: 15.00 (Δ: +0.00)
Current RMSE: 25.00 (Δ: +0.00)
Performance within acceptable bounds: TRUE
```

## 🚀 Running Validation

### Prerequisites
- All three notebooks executed successfully in sequence
- Databricks workspace access configured
- MLflow experiment permissions
- Unity Catalog read permissions for relevant schemas

### Execute Validation Script

```bash
cd model/validation
python end_to_end_validation.py --profile DEFAULT --output validation_results.json
```

### Validation Output

The script produces:
- **Console output**: Real-time validation progress and results
- **JSON report**: Detailed validation results saved to `validation_results.json`
- **Exit code**: 0 for success, 1 for failure

## 📊 Sample Validation Report

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "feature_validation": {
    "status": "PASS",
    "feature_count": 5000,
    "source_count": 5000,
    "count_match": true,
    "all_features_present": true
  },
  "training_validation": {
    "status": "PASS",
    "run_found": true,
    "model_registered": true,
    "r2_score": 0.7623,
    "mae": 14.52,
    "rmse": 23.89
  },
  "batch_validation": {
    "status": "PASS",
    "batch_count": 5000,
    "all_columns_present": true,
    "no_null_predictions": true,
    "risk_logic_correct": true
  },
  "performance_comparison": {
    "status": "PASS",
    "overall_performance": "PASS",
    "changes": {
      "r2_change": 0.0123,
      "mae_change": -0.48,
      "rmse_change": -1.11
    }
  },
  "overall_status": "PASS"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Feature table not found**
   - Ensure feature notebook ran successfully
   - Check Unity Catalog permissions
   - Verify table name: `juan_dev.ml.insurance_features_v2`

2. **Training run not found** 
   - Check MLflow experiment exists
   - Verify run name contains `healthcare_risk_model_`
   - Ensure training notebook completed

3. **Batch output missing**
   - Verify model registration and aliasing
   - Check batch notebook execution logs
   - Ensure input table `juan_dev.healthcare_data.silver_patients` exists

4. **Performance degradation**
   - Review training data quality
   - Check for feature drift
   - Validate model hyperparameters
   - Consider retraining with fresh data

### Updating Baselines

To update performance baselines after validating improved model:

1. Update `baseline_metrics` in `end_to_end_validation.py`:
```python
self.baseline_metrics = {
    "r2_score": 0.78,  # New baseline
    "mean_absolute_error": 13.5,  # New baseline  
    "root_mean_squared_error": 22.0  # New baseline
}
```

2. Document the change in version control commit message
3. Re-run validation to confirm new baseline

## 📝 Version Control Integration

### Pre-commit Steps
1. Execute all notebooks in sequence
2. Run validation script
3. Ensure all validations pass
4. Commit notebooks + validation results

### Git Commands
```bash
# Add all notebook changes
git add model/training/*.ipynb model/batch/*.ipynb

# Add validation results 
git add model/validation/validation_results.json

# Commit with descriptive message
git commit -m "feat: end-to-end pipeline validation - all checks pass

- Feature table: 5,000 records aligned with source
- Training: R²=0.76, MAE=14.5, RMSE=23.9  
- Batch: 5,000 predictions with correct risk categories
- Performance: within acceptable bounds vs baseline"

# Push to remote
git push origin main
```

## 🎯 Success Criteria

✅ **All validations must pass**:
- Feature table row count = source table count
- Training metrics logged and model registered  
- Batch inference produces expected outputs with correct schema
- Model performance within acceptable bounds vs baseline

✅ **Documentation updated**:
- Any performance changes documented
- Validation results committed to version control
- README updated if process changes

This completes the end-to-end validation framework for the insurance MLOps pipeline.
