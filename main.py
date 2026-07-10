import os
import pandas as pd

from config import DATA_PATH

print("=" * 60)
print("Customer Churn Prediction Project")
print("=" * 60)

print("\nChecking Dataset...\n")

if not os.path.exists(DATA_PATH):

    print("Dataset not found.")

else:

    df = pd.read_csv(DATA_PATH)

    print("Dataset Loaded Successfully\n")

    print("=" * 60)
    print("Dataset Shape")
    print("=" * 60)

    print(df.shape)

    print("\n")

    print("=" * 60)
    print("Columns")
    print("=" * 60)

    print(df.columns.tolist())

    print("\n")

    print("=" * 60)
    print("First Five Rows")
    print("=" * 60)

    print(df.head())

    print("\n")

    print("=" * 60)
    print("Missing Values")
    print("=" * 60)

    print(df.isnull().sum())

    print("\n")

    print("=" * 60)
    print("Data Types")
    print("=" * 60)

    print(df.dtypes)

    print("\n")

    print("=" * 60)
    print("Duplicate Rows")
    print("=" * 60)

    print(df.duplicated().sum())

    print("\n")

    print("=" * 60)
    print("Dataset Imported Successfully")
    print("=" * 60)

from config import DATA_PATH
from src.preproessing import DataPreprocessor
from src.eda import EDA

print("="*70)
print("CUSTOMER CHURN PREDICTION")
print("="*70)

processor = DataPreprocessor(DATA_PATH)
df = processor.preprocess()

print("\nDataset Ready For Machine Learning")

eda = EDA(df)
eda.run()

print("\nReady for Feature Engineering")

from src.feature_engineering import FeatureEngineering

feature = FeatureEngineering(df)

X_train, X_test, y_train, y_test = feature.run()

print("Feature Engineering Completed")

from src.model_training import ModelTrainer

trainer=ModelTrainer(

    X_train,

    X_test,

    y_train,

    y_test

)

trainer.run()

import joblib

from src.explainability import Explainability

best_model = joblib.load(

    "models/best_model.pkl"

)

explainer = Explainability(

    best_model,

    X_train,

    X_test

)

explainer.run()