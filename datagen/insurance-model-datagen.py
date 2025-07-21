# Databricks notebook source
# Install the faker library
%pip install faker

# COMMAND ----------

from faker import Faker
import numpy as np
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType, TimestampType

class HealthcareSyntheticDataGenerator:
    def __init__(self):
        self.fake = Faker()
        
    def generate_synthetic_patients(self, n_patients=10000):
        """Generate HIPAA-compliant synthetic patient data"""
        
        np.random.seed(42)  # Reproducible synthetic data
        synthetic_data = []
        
        for i in range(n_patients):
            # Basic demographics
            age = int(np.random.normal(45, 15))
            age = max(18, min(85, age))  # Constrain age range
            
            sex = str(np.random.choice(['MALE', 'FEMALE']))
            region = str(np.random.choice(['NORTHEAST', 'NORTHWEST', 'SOUTHEAST', 'SOUTHWEST']))
            
            # Health indicators with realistic distributions
            bmi_base = np.random.normal(28, 6)
            bmi = float(max(16, min(50, bmi_base)))  # Ensure bmi is a float
            
            # Smoking correlated with age and region
            smoking_prob = 0.15 + (age - 30) * 0.001
            smoker = bool(np.random.random() < smoking_prob)
            
            children = int(max(0, np.random.poisson(1.2)))
            
            # Generate realistic cost based on risk factors
            base_cost = 3000
            age_factor = (age / 40) ** 1.5
            bmi_factor = 1 + max(0, (bmi - 25) / 25)
            smoking_factor = 2.5 if smoker else 1.0
            children_factor = 1 + (children * 0.1)
            
            noise = float(np.random.lognormal(0, 0.3))
            charges = base_cost * age_factor * bmi_factor * smoking_factor * children_factor * noise
            
            synthetic_patient = {
                'customer_id': int(1000000 + i),
                'age': int(age),
                'sex': str(sex),
                'bmi': float(round(bmi, 1)),
                'children': int(children),
                'smoker': bool(smoker),
                'region': str(region),
                'charges': float(round(charges, 2)),
                'timestamp': self.fake.date_time_this_year()
            }
            
            synthetic_data.append(synthetic_patient)
        
        # Define schema
        schema = StructType([
            StructField('customer_id', IntegerType(), False),
            StructField('age', IntegerType(), False),
            StructField('sex', StringType(), False),
            StructField('bmi', FloatType(), False),
            StructField('children', IntegerType(), False),
            StructField('smoker', BooleanType(), False),
            StructField('region', StringType(), False),
            StructField('charges', FloatType(), False),
            StructField('timestamp', TimestampType(), False)
        ])
        
        # Convert to Spark DataFrame
        synthetic_df = spark.createDataFrame(synthetic_data, schema=schema)
        
        # Save to Unity Catalog (replace with your catalog & schema if needed)
        synthetic_df.write.mode("overwrite").saveAsTable("juan_dev.ml.synthetic_test_data")
        
        return synthetic_df

# Generate synthetic test data
generator = HealthcareSyntheticDataGenerator()
synthetic_data = generator.generate_synthetic_patients(5000)
