# Healthcare Insurance MLOps - Databricks Asset Bundles Deployment Guide

This guide explains how to deploy the Healthcare Insurance MLOps project using Databricks Asset Bundles (DABs) following MLOps best practices.

## Overview

The project has been configured with Databricks Asset Bundles to provide:
- **Environment Management**: Dev, Staging, and Production environments
- **Resource Orchestration**: Jobs, workflows, clusters, and model endpoints
- **Dependency Management**: Proper task sequencing and dependencies
- **Monitoring & Alerting**: Automated drift detection and performance monitoring
- **Governance**: Model validation and promotion workflows

## Project Structure

```
healthcare-insurance-mlops/
├── databricks.yml                 # Main bundle configuration
├── resources/                     # Resource definitions
│   ├── clusters.yml              # Cluster configurations
│   ├── experiments.yml           # MLflow experiments
│   └── alerts.yml                # Monitoring alerts
├── deployment/
│   └── deploy.sh                 # Deployment script
├── src/healthcare_mlops/         # Python package
│   ├── __init__.py
│   └── governance.py             # Model governance module
├── 00-training/                  # Training notebooks
├── 01-governance/                # Governance scripts
├── 02-batch/                     # Batch inference
├── 03-monitoring/                # Model monitoring
└── setup.py                     # Python package setup
```

## Resource Architecture

### Jobs and Workflows
1. **Feature Engineering Job**: Extracts and engineers features for the healthcare model
2. **Model Training Job**: Trains Random Forest/Gradient Boosting models 
3. **Model Governance Job**: Validates and promotes models using healthcare-specific criteria
4. **Batch Inference Job**: Daily batch scoring with business logic
5. **Model Monitoring Job**: Drift detection and performance monitoring
6. **ML Training Pipeline**: End-to-end workflow orchestrating all training stages

### Environment Configuration
- **Dev**: Development environment with single-node clusters
- **Staging**: Pre-production with scaled resources and validation
- **Prod**: Production environment with high availability and monitoring

### Resource Definitions
- **Clusters**: Optimized for different workloads (development, training, inference)
- **Experiments**: MLflow experiment organization by component
- **Model Serving**: Endpoints for real-time inference
- **Alerts**: Automated monitoring for performance degradation and drift

## Prerequisites

### 1. Databricks CLI Setup
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication (choose one method)
# Option A: OAuth (recommended for interactive use)
databricks configure --token

# Option B: Service Principal (recommended for CI/CD)
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### 2. Unity Catalog Access
Ensure you have access to:
- Catalog: `juan_dev` (or update variables in databricks.yml)
- Schemas: `healthcare_data`, `ml`
- Permissions: `CREATE_SCHEMA`, `CREATE_TABLE`, `CREATE_MODEL`

### 3. Data Prerequisites
Verify these tables exist with data:
```sql
SELECT * FROM juan_dev.healthcare_data.silver_patients LIMIT 5;
SELECT * FROM juan_dev.healthcare_data.dim_patients WHERE is_current_record = True LIMIT 5;
```

## Deployment Steps

### 1. Update Configuration
Edit `databricks.yml` to customize for your environment:

```yaml
# Update workspace URLs
targets:
  dev:
    workspace:
      host: https://your-workspace.cloud.databricks.com
      
# Update email notifications
email_notifications:
  on_failure: ["your-email@company.com"]
```

### 2. Validate Bundle
```bash
# Validate configuration
databricks bundle validate --target dev
```

### 3. Deploy to Development
```bash
# Deploy using the deployment script
./deployment/deploy.sh -e dev

# Or deploy directly with Databricks CLI
databricks bundle deploy --target dev
```

### 4. Run Initial Workflow
```bash
# Start the ML training pipeline
databricks jobs run-now --job-name "[dev] Healthcare ML Training Pipeline"
```

### 5. Deploy to Higher Environments
```bash
# Deploy to staging
./deployment/deploy.sh -e staging

# Deploy to production
./deployment/deploy.sh -e prod
```

## Job Execution Flow

### ML Training Pipeline
1. **Feature Engineering** → Creates features in feature store
2. **Model Training** → Trains and logs model to MLflow
3. **Model Governance** → Validates model against healthcare criteria
4. **Batch Inference Validation** → Tests model on sample data
5. **Setup Monitoring** → Initializes drift detection and alerts

### Daily Operations
1. **Batch Inference** (2 AM daily) → Scores new patients
2. **Model Monitoring** (6 AM daily) → Checks for drift and performance issues
3. **Automated Retraining** → Triggered if monitoring detects issues

## Monitoring and Alerting

### Performance Metrics
- **R² Score**: Must be ≥ 0.70 for promotion
- **MAE**: Must be ≤ 15.0 for risk scores
- **High-Risk Accuracy**: Must be ≥ 60%

### Drift Detection
- **Statistical Drift**: Changes in prediction distributions
- **Demographic Drift**: Shifts in patient population characteristics
- **Performance Drift**: Model accuracy degradation

### Alert Conditions
- **High MAE**: MAE > 15.0 triggers retraining alert
- **Low Volume**: < 50 daily predictions triggers pipeline check
- **Drift Alert**: Composite drift score > 0.3 triggers investigation

## Healthcare Compliance

### HIPAA Compliance
- All data processing follows deidentification standards
- Models tagged with compliance status
- Audit trails maintained through Unity Catalog

### Clinical Validation
- BMI categorization uses clinical standards
- Age risk scoring follows actuarial practices
- Smoking impact factors based on healthcare evidence

### Governance Requirements
- Automated bias testing for demographic groups
- Performance validation against industry standards
- Champion/challenger pattern for safe model updates

## Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Set up authentication
databricks configure --token
# Or set environment variables
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
```

#### 2. Missing Tables
```sql
-- Verify prerequisite tables exist
SHOW TABLES IN juan_dev.healthcare_data LIKE '*patient*';
```

#### 3. Permission Issues
- Ensure Unity Catalog permissions for schemas
- Verify service principal has necessary privileges
- Check workspace admin permissions for job creation

#### 4. Job Failures
```bash
# Check job run details
databricks jobs list
databricks jobs get-run --run-id <run-id>
```

### Validation Commands
```bash
# Validate bundle without deploying
databricks bundle validate --target dev

# List deployed resources
databricks jobs list --output table

# Check model registry
databricks models list --output table
```

## Best Practices

### Development Workflow
1. **Feature Development**: Test in development environment first
2. **Code Review**: Use pull requests for all changes
3. **Environment Promotion**: Dev → Staging → Production
4. **Monitoring**: Set up alerts before production deployment

### Resource Management
- Use appropriate cluster sizes for workloads
- Enable autotermination to control costs
- Tag resources for cost tracking and management
- Use spot instances for development clusters

### Security
- Use service principals for production workloads
- Implement least-privilege access controls
- Regularly rotate access tokens and credentials
- Monitor access logs and usage patterns

### Model Lifecycle
- Maintain model versioning through MLflow
- Document model changes and performance impacts
- Test governance checks in lower environments
- Plan rollback procedures for model updates

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy Healthcare MLOps
on:
  push:
    branches: [main]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: databricks/setup-cli@main
    - name: Deploy to staging
      run: databricks bundle deploy --target staging
      env:
        DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
```

### Azure DevOps Pipeline
```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.9'
    
- script: |
    pip install databricks-cli
    databricks bundle deploy --target staging
  env:
    DATABRICKS_TOKEN: $(DATABRICKS_TOKEN)
    DATABRICKS_HOST: $(DATABRICKS_HOST)
```

## Support and Maintenance

### Regular Tasks
- Review job execution logs weekly
- Monitor model performance metrics daily  
- Update dependencies quarterly
- Refresh training data monthly
- Review and update governance thresholds as needed

### Scaling Considerations
- Monitor cluster utilization and adjust sizes
- Consider multi-region deployment for high availability
- Implement data archiving for historical predictions
- Plan for increased data volumes and model complexity

For additional support, refer to:
- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Unity Catalog Best Practices](https://docs.databricks.com/data-governance/unity-catalog/best-practices.html)