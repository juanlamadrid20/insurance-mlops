# Quick Execution Guide - End-to-End Validation

## 🎯 Step-by-Step Execution

### 1. Execute Notebooks (In Order)

**Execute these notebooks in your Databricks workspace in this exact sequence:**

1. **Feature Engineering**
   ```
   📁 model/training/insurance-model-feature-updated.ipynb
   ```
   - Creates feature table: `juan_dev.ml.insurance_features_v2`
   - Expected runtime: 2-5 minutes
   - ✅ **Success indicator**: Feature table created with row count matching source

2. **Model Training**  
   ```
   📁 model/training/insurance-model-train-updated.ipynb
   ```
   - Trains and registers model: `juan_dev.ml.healthcare_risk_model_v2`
   - Expected runtime: 5-10 minutes
   - ✅ **Success indicator**: Model registered, metrics logged (R², MAE, RMSE)

3. **Batch Inference**
   ```
   📁 model/batch/insurance-model-batch.ipynb
   ```
   - Creates predictions table: `juan_dev.healthcare_ml.patient_predictions`
   - Expected runtime: 3-7 minutes
   - ✅ **Success indicator**: Predictions table with risk categories

### 2. Run Validation Script

After all notebooks complete successfully:

```bash
cd /Users/juan.lamadrid/dev/databricks-projects/ml/mlops-stacks-projects/insurance-mlops/model/validation

python end_to_end_validation.py --profile DEFAULT --output validation_results.json
```

### 3. Expected Validation Results

**Console Output Should Show:**
```
🚀 Starting End-to-End Validation

🔍 Validating feature table...
   ✅ Feature table row count: 5000
   ✅ Source table row count: 5000
   ✅ Row count match: True
   ✅ All features present: True

🔍 Validating training run...
   ✅ Training run found: <run_id>
   ✅ Run status: FINISHED
   ✅ All metrics logged: True
   ✅ Model registered: True
   ✅ Model version: 1

🔍 Validating batch inference...
   ✅ Batch output records: 5000
   ✅ All columns present: True
   ✅ No null predictions: True
   ✅ Risk categorization correct: True

🔍 Comparing model performance against baseline...
   📊 Current R²: 0.7623 (Δ: +0.0123)
   📊 Current MAE: 14.52 (Δ: -0.48)
   📊 Current RMSE: 23.89 (Δ: -1.11)
   ✅ Performance within acceptable bounds

📋 VALIDATION SUMMARY
==================================================
Feature Validation:     PASS
Training Validation:    PASS
Batch Validation:       PASS
Performance Comparison: PASS
--------------------------------------------------
OVERALL STATUS:         PASS

📄 Validation results saved to: validation_results.json
```

## 🔧 If Issues Occur

### Common Problems & Solutions

1. **Feature table creation fails**
   - Check Unity Catalog permissions for `juan_dev.ml` schema
   - Verify source table `juan_dev.healthcare_data.dim_patients` exists
   - Ensure current records filter works: `is_current_record = True`

2. **Training run fails** 
   - Check MLflow experiment permissions
   - Verify feature table was created successfully
   - Check Unity Catalog model registry permissions

3. **Batch inference fails**
   - Ensure model is registered and has correct alias
   - Check input table `juan_dev.healthcare_data.silver_patients` exists
   - Verify feature engineering integration works

4. **Validation script fails**
   - Check Python environment has required packages:
     ```bash
     pip install databricks-feature-engineering mlflow pandas pyspark
     ```
   - Ensure Databricks CLI is configured with correct profile

## 📊 What to Report Back

Please share:

1. **Execution Status**: Which notebooks succeeded/failed
2. **Runtime**: How long each notebook took
3. **Validation Results**: Full console output from validation script
4. **Any Errors**: Complete error messages if failures occur
5. **Data Counts**: Actual row counts from feature table, training data, batch output

### Sample Report Format:
```
✅ Feature notebook: SUCCESS (3 min runtime, 5,247 features created)
✅ Training notebook: SUCCESS (7 min runtime, R²=0.76, model v1 registered)  
✅ Batch notebook: SUCCESS (4 min runtime, 5,247 predictions generated)
✅ Validation: OVERALL PASS (all checks passed)

Row counts:
- Source table: 5,247
- Feature table: 5,247  
- Batch output: 5,247
```

## 🎉 Success Criteria

**All checks must pass for complete validation:**
- [x] Feature table row count = source table count
- [x] All required feature columns present
- [x] Training metrics logged (R², MAE, RMSE)
- [x] Model registered in Unity Catalog
- [x] Batch output contains all expected columns
- [x] Risk categorization logic works correctly
- [x] No null predictions
- [x] Performance within acceptable bounds vs baseline

Once you've tested and reported back, we'll proceed with any necessary adjustments and finalize the pipeline!
