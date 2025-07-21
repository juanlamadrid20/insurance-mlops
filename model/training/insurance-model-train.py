# Databricks notebook source
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering import FeatureLookup


# Configure Unity Catalog integration
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/juan.lamadrid@databricks.com/experiments/insurance_cost_prediction_eda")

# Healthcare model training class
class HealthcareInsuranceModel:
    def __init__(self):
        self.fe = FeatureEngineeringClient()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_training_data(self):
        """Prepare training dataset with optimized feature lookups"""

        # Load base training data
        base_df = spark.table("juan_dev.ml.insurance_silver")

        # Define the feature table once
        feature_table = "juan_dev.ml.healthcare_features"
        lookup_key = "customer_id"

        # Use list comprehension to simplify feature lookups
        feature_lookups = [
            FeatureLookup(
                table_name=feature_table,
                lookup_key=lookup_key,
                feature_name="charges",
                output_name="healthcare_charges"  # Disambiguate from label
            )
        ] + [
            FeatureLookup(
                table_name=feature_table,
                lookup_key=lookup_key,
                feature_name=feature
            )
            for feature in [
                "age_risk_score",
                "smoking_impact",
                "family_size_factor",
                "regional_multiplier",
                "health_risk_composite"
            ]
        ]

        # Use create_training_set to join features
        training_set = self.fe.create_training_set(
            df=base_df,
            feature_lookups=feature_lookups,
            label="charges",
            exclude_columns=["timestamp", "ingestion_timestamp"]
        )

        # Materialize training set for downstream use
        training_df = training_set.load_df()
        display(training_df)

        return training_set
    
    def train_model(self, training_set, model_type="random_forest"):
        """Train healthcare insurance cost prediction model"""
        with mlflow.start_run(run_name=f"healthcare_model_{model_type}"):
            # Load training data
            training_df = training_set.load_df().toPandas()
            
            # Prepare features
            categorical_features = ['sex', 'region'] # , 'policy_type' , 'bmi_category'
            
            numerical_features = ['age', 'bmi', 'children', 'age_risk_score', 
                                'smoking_impact', 'family_size_factor', 
                                'health_risk_composite'] # 'coverage_to_deductible_ratio'
            
            # Encode categorical variables
            for feature in categorical_features:
                le = LabelEncoder()
                training_df[f"{feature}_encoded"] = le.fit_transform(training_df[feature])
                self.label_encoders[feature] = le
            
            # Select features for training
            feature_columns = numerical_features + [f"{f}_encoded" for f in categorical_features]
            X = training_df[feature_columns]
            y = training_df['charges']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Model selection based on healthcare requirements
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:  # gradient_boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            
            # Train model
            model.fit(X_train, y_train)
            self.model = model
            
            # Cross-validation for robust evaluation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # Predictions and evaluation
            y_pred = model.predict(X_test)
            
            # Healthcare-specific metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            
            # Business metrics for healthcare
            high_cost_threshold = training_df['charges'].quantile(0.95)
            high_cost_accuracy = self._evaluate_high_cost_predictions(
                y_test, y_pred, high_cost_threshold
            )
            
            # Log parameters and metrics
            mlflow.log_params({
                "model_type": model_type,
                "n_features": len(feature_columns),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            })
            
            mlflow.log_metrics({
                "r2_score": r2,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse,
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
                "high_cost_accuracy": high_cost_accuracy
            })
            
            # Enhanced MLflow 3.0 model logging with comprehensive metadata
            model_info = self.fe.log_model(
                model=model,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name="juan_dev.ml.healthcare_insurance_model",
                metadata={
                    "algorithm": model_type,
                    "healthcare_compliance": "HIPAA_ready",
                    "model_purpose": "insurance_cost_prediction",
                    "feature_count": len(feature_columns),
                    "training_data_size": len(training_df)
                }
            )
            
            return model_info
    
    def _evaluate_high_cost_predictions(self, y_true, y_pred, threshold):
        """Evaluate model performance on high-cost patients"""
        high_cost_true = y_true >= threshold
        high_cost_pred = y_pred >= threshold
        return (high_cost_true == high_cost_pred).mean()

# Train multiple models for comparison
trainer = HealthcareInsuranceModel()
training_set = trainer.prepare_training_data()

# Train Random Forest model
rf_model_info = trainer.train_model(training_set, "random_forest")

# Train Gradient Boosting model  
# gb_model_info = trainer.train_model(training_set, "gradient_boosting")

# COMMAND ----------

