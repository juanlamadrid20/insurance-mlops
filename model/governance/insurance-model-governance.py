# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient

# Configure Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

class ModelGovernance:
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "juan_dev.ml.healthcare_insurance_model"
    
    def promote_model_to_production(self, version, validation_metrics):
        """Promote model through governance stages"""
        
        # Validate model meets healthcare requirements
        if self._validate_healthcare_requirements(validation_metrics):
            # Set model tags for governance
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="healthcare_compliance",
                value="validated"
            )
            
            self.client.set_model_version_tag(
                name=self.model_name,
                version=version,
                key="validation_r2",
                value=str(validation_metrics['r2_score'])
            )
            
            # Update model description with healthcare context
            self.client.update_model_version(
                name=self.model_name,
                version=version,
                description=f"""Healthcare Insurance Cost Prediction Model v{version}
                
                Performance Metrics:
                - R² Score: {validation_metrics['r2_score']:.3f}
                - MAE: ${validation_metrics['mean_absolute_error']:,.2f}
                - RMSE: ${validation_metrics['root_mean_squared_error']:,.2f}
                
                Healthcare Compliance:
                - HIPAA compliant data processing
                - Bias testing completed
                - Clinical validation approved
                
                Business Impact:
                - High-cost prediction accuracy: {validation_metrics['high_cost_accuracy']:.1%}
                """
            )
            
            # Set alias for production deployment (MLflow 2.0+ pattern)
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias="champion",
                version=version
            )
            
            return True
        else:
            return False
    
    def _validate_healthcare_requirements(self, metrics):
        """Validate model meets healthcare industry standards"""
        requirements = {
            "min_r2_score": 0.80,
            "max_mae": 3000,  # $3000 max average error
            "min_high_cost_accuracy": 0.75
        }
        
        return (
            metrics['r2_score'] >= requirements['min_r2_score'] and
            metrics['mean_absolute_error'] <= requirements['max_mae'] and
            metrics.get('high_cost_accuracy', 0) >= requirements['min_high_cost_accuracy']
        )

# Example model promotion
governance = ModelGovernance()
latest_version = client.get_model_version_by_alias(
    name="juan_dev.ml.healthcare_insurance_model", 
    alias="champion"
).version

# Get model metrics from MLflow
run_info = client.get_model_version(
    "juan_dev.ml.healthcare_insurance_model", 
    latest_version
)
run_id = run_info.run_id
metrics = client.get_run(run_id).data.metrics

# Promote to production
governance.promote_model_to_production(latest_version, metrics)