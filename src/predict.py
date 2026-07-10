import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


class PredictionPreprocessor:
    
    def __init__(self, scaler_path="models/scaler.pkl"):
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = LabelEncoder()
        
    def preprocess_new_data(self, df):
        """
        Preprocess new customer data for prediction
        Mirrors the training preprocessing pipeline
        """
        df = df.copy()
        
        # Handle TotalCharges - convert to numeric and fill missing
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Fill with median if there are missing values
            if df["TotalCharges"].isnull().any():
                median_value = df["TotalCharges"].median()
                df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
        
        # Remove customerID if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        
        # Scale numerical features FIRST (before encoding, matching training pipeline)
        # The scaler was fitted on original numerical columns only
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        numerical_cols_present = [col for col in numerical_cols if col in df.columns]
        
        if numerical_cols_present:
            df[numerical_cols_present] = self.scaler.transform(df[numerical_cols_present])
        
        # Label encode binary columns
        binary_columns = []
        for column in df.columns:
            if df[column].dtype == object:
                if df[column].nunique() == 2:
                    binary_columns.append(column)
        
        for column in binary_columns:
            # Fit on the data and transform
            df[column] = self.label_encoder.fit_transform(df[column])
        
        # One-hot encode remaining categorical columns
        categorical_columns = []
        for column in df.columns:
            if df[column].dtype == object:
                categorical_columns.append(column)
        
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Ensure all expected columns are present (match training data)
        # Load the training data to get expected columns
        try:
            train_data = pd.read_csv("data/clean_telco.csv")
            expected_columns = train_data.drop("Churn", axis=1).columns.tolist()
            
            # Add missing columns with 0
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Remove extra columns not in training data
            df = df[expected_columns]
            
        except Exception as e:
            print(f"Warning: Could not load training data for column matching: {e}")
        
        return df
    
    def predict(self, df, model_path="models/best_model.pkl"):
        """
        Make predictions on preprocessed data
        """
        # Load model
        model = joblib.load(model_path)
        
        # Preprocess
        df_processed = self.preprocess_new_data(df)
        
        # Make predictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            "Prediction": predictions,
            "Probability": probabilities,
            "Churn_Risk": ["High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low" for prob in probabilities]
        })
        
        return results


def predict_from_csv(csv_path, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"):
    """
    Convenience function to predict from a CSV file
    """
    df = pd.read_csv(csv_path)
    preprocessor = PredictionPreprocessor(scaler_path)
    results = preprocessor.predict(df, model_path)
    return results


def predict_from_dict(customer_data, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"):
    """
    Convenience function to predict from a dictionary of customer data
    """
    df = pd.DataFrame([customer_data])
    preprocessor = PredictionPreprocessor(scaler_path)
    results = preprocessor.predict(df, model_path)
    return results
