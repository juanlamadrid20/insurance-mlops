# Feature/Label Mapping Matrix: Old Schema → New Healthcare Schema

Based on analysis of the current notebooks and the new healthcare data schema, here's the comprehensive mapping:

## Core Feature Mappings

| Old Field | New Table | New Column | Notes | Transformation Required |
|-----------|-----------|------------|-------|-------------------------|
| age | dim_patients | patient_age_category → age | Derive numeric age from category | Yes - Category to numeric mapping |
| sex | dim_patients | patient_sex | Direct mapping | No - Already categorical |
| bmi | dim_patients | patient_bmi_category → bmi | Derive numeric BMI from category | Yes - Category to numeric mapping |
| children | dim_patients | patient_family_size_category → children | Derive children count from family size | Yes - Category to numeric mapping |
| smoker | dim_patients | patient_smoking_status → smoker | Convert to boolean | Yes - String to boolean |
| region | dim_patients | patient_region | Direct mapping | No - Already categorical |
| charges | dim_patients | health_risk_score | Use as label/target variable | Yes - Different metric concept |

## Engineered Features Mappings

| Old Engineered Field | New Table | New Column/Logic | Notes | Transformation Required |
|---------------------|-----------|------------------|-------|-------------------------|
| age_group | dim_patients | patient_age_category | Direct use of existing categorization | No |
| age_risk_score | dim_patients | health_risk_score or derive from patient_age_category | Use existing or derive | Conditional |
| smoking_impact | dim_patients | Derive from patient_smoking_status + age | Recreate logic | Yes |
| family_size_factor | dim_patients | Derive from patient_family_size_category | Recreate logic | Yes |
| regional_multiplier | dim_patients | Derive from patient_region | Recreate logic | Yes |
| health_risk_composite | dim_patients | health_risk_score | Use existing composite score | No |
| bmi_category | dim_patients | patient_bmi_category | Direct mapping | No |

## Label Column Decision

**New Label Column:** `health_risk_score` (instead of `charges`)
- **Rationale:** The new schema focuses on health risk assessment rather than direct cost prediction
- **Type:** Numeric (int)
- **Range:** Based on patient health factors and risk assessment

## Data Type Conversions Required

| Field | Old Type | New Type | Conversion Logic |
|-------|----------|----------|------------------|
| age | int | string → int | Map categories: YOUNG→25, ADULT→35, MIDDLE_AGED→45, SENIOR→60, ELDERLY→70 |
| bmi | float | string → float | Map categories: UNDERWEIGHT→17.5, NORMAL→22.5, OVERWEIGHT→27.5, OBESE→32.5 |
| smoker | boolean | string → boolean | patient_smoking_status == 'SMOKER' |
| children | int | string → int | Map: SINGLE→0, COUPLE→0, SMALL_FAMILY→1, MEDIUM_FAMILY→2, LARGE_FAMILY→4 |
| sex | string | string | Direct mapping (patient_sex) |
| region | string | string | Direct mapping (patient_region) |

## Additional New Features Available

| New Feature | Source Column | Description | Usage Recommendation |
|-------------|---------------|-------------|----------------------|
| data_quality_score | patient_data_quality_score | Data completeness metric | Use as feature weight |
| hipaa_compliance | hipaa_deidentification_applied | Privacy compliance flag | Use for filtering |
| risk_category | health_risk_category | Categorical risk level | Use as additional feature |
| premium_category | patient_premium_category | Insurance premium tier | Use as feature |

## Implementation Notes

1. **Primary Key Mapping:** `customer_id` → `patient_natural_key` (use for joins)
2. **Time Filtering:** Always filter `is_current_record = True` for current patient state
3. **Data Quality:** Consider using `patient_data_quality_score` as a feature or filter
4. **Privacy:** All data in new schema is HIPAA-compliant by design

## Suggested Updates to Notebooks

### insurance-model-feature.ipynb
- Replace `juan_dev.ml.insurance_silver` with `juan_dev.healthcare_data.dim_patients`
- Add `filter(col("is_current_record") == True)`
- Update all column references to use new schema
- Replace `charges` visualizations with `health_risk_score`

### insurance-model-train.ipynb
- Update data loading to use new schema
- Modify feature engineering to work with categorical → numeric conversions
- Change target variable from `charges` to `health_risk_score`
- Update model evaluation metrics for risk scoring vs cost prediction
