
import pandas as pd
import numpy as np
import joblib


class PredictionPreprocessor:

    def __init__(
        self,
        scaler_path="models/scaler.pkl",
        model_path="models/best_model.pkl"
    ):
        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Load model
        self.model = joblib.load(model_path)

        # Get the exact columns used when the scaler was fitted
        if hasattr(self.scaler, "feature_names_in_"):
            self.scaler_columns = list(self.scaler.feature_names_in_)
        else:
            self.scaler_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "PaperlessBilling",
                "MonthlyCharges",
                "TotalCharges"
            ]

        # Get the exact columns expected by the trained model
        if hasattr(self.model, "feature_names_in_"):
            self.model_columns = list(self.model.feature_names_in_)
        else:
            # Fall back to the cleaned training dataset
            try:
                train_data = pd.read_csv("data/clean_telco.csv")

                if "Churn" in train_data.columns:
                    train_data = train_data.drop(columns=["Churn"])

                self.model_columns = train_data.columns.tolist()

            except Exception:
                self.model_columns = None

    # ---------------------------------------------------------
    # Preprocess new customer data
    # ---------------------------------------------------------
    def preprocess_new_data(self, df):

        df = df.copy()

        # -----------------------------------------------------
        # 1. Clean column names
        # -----------------------------------------------------
        df.columns = df.columns.astype(str).str.strip()

        # -----------------------------------------------------
        # 2. Handle TotalCharges
        # -----------------------------------------------------
        if "TotalCharges" in df.columns:

            df["TotalCharges"] = df["TotalCharges"].replace(
                " ",
                pd.NA
            )

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

            # Use training median where possible
            if df["TotalCharges"].isnull().any():

                median_value = df["TotalCharges"].median()

                if pd.isna(median_value):
                    median_value = 0

                df["TotalCharges"] = df[
                    "TotalCharges"
                ].fillna(median_value)

        # -----------------------------------------------------
        # 3. Remove customerID
        # -----------------------------------------------------
        if "customerID" in df.columns:

            df = df.drop(
                columns=["customerID"]
            )

        # -----------------------------------------------------
        # 4. Encode binary columns
        #
        # This MUST happen BEFORE scaling because the scaler
        # was fitted using these columns as numeric features.
        # -----------------------------------------------------
        binary_mappings = {

            "gender": {
                "Female": 0,
                "Male": 1
            },

            "Partner": {
                "No": 0,
                "Yes": 1
            },

            "Dependents": {
                "No": 0,
                "Yes": 1
            },

            "PhoneService": {
                "No": 0,
                "Yes": 1
            },

            "PaperlessBilling": {
                "No": 0,
                "Yes": 1
            },

            "Churn": {
                "No": 0,
                "Yes": 1
            }
        }

        for column, mapping in binary_mappings.items():

            if column in df.columns:

                df[column] = df[column].map(mapping)

        # -----------------------------------------------------
        # 5. Convert numerical columns to numeric
        # -----------------------------------------------------
        numerical_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for column in numerical_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # -----------------------------------------------------
        # 6. One-hot encode remaining categorical columns
        # -----------------------------------------------------
        categorical_columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        if categorical_columns:

            df = pd.get_dummies(
                df,
                columns=categorical_columns,
                drop_first=True
            )

        # -----------------------------------------------------
        # 7. Convert boolean columns to integers
        # -----------------------------------------------------
        bool_columns = df.select_dtypes(
            include=["bool"]
        ).columns.tolist()

        for column in bool_columns:

            df[column] = df[column].astype(int)

        # -----------------------------------------------------
        # 8. Make sure scaler columns exist
        #
        # The scaler expects EXACTLY the same numerical
        # features it saw during training.
        # -----------------------------------------------------
        for column in self.scaler_columns:

            if column not in df.columns:

                df[column] = 0

        # -----------------------------------------------------
        # 9. Scale using the SAME columns and SAME scaler
        # -----------------------------------------------------
        df[self.scaler_columns] = self.scaler.transform(
            df[self.scaler_columns]
        )

        # -----------------------------------------------------
        # 10. Match model feature columns
        # -----------------------------------------------------
        if self.model_columns is not None:

            # Add missing model columns
            for column in self.model_columns:

                if column not in df.columns:

                    df[column] = 0

            # Remove unexpected columns
            df = df[
                self.model_columns
            ]

        # -----------------------------------------------------
        # 11. Make everything numeric
        # -----------------------------------------------------
        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        df = df.fillna(0)

        return df

    # ---------------------------------------------------------
    # Make prediction
    # ---------------------------------------------------------
    def predict(self, df):

        # Preprocess new data
        df_processed = self.preprocess_new_data(df)

        # Prediction
        predictions = self.model.predict(
            df_processed
        )

        # Probability
        probabilities = self.model.predict_proba(
            df_processed
        )[:, 1]

        # Churn risk
        churn_risk = []

        for probability in probabilities:

            if probability > 0.7:

                churn_risk.append("High")

            elif probability > 0.4:

                churn_risk.append("Medium")

            else:

                churn_risk.append("Low")

        # Results
        results = pd.DataFrame({

            "Prediction": predictions,

            "Probability": probabilities,

            "Churn_Risk": churn_risk

        })

        return results


# =============================================================
# Predict from CSV
# =============================================================

def predict_from_csv(
    csv_path,
    model_path="models/best_model.pkl",
    scaler_path="models/scaler.pkl"
):

    df = pd.read_csv(csv_path)

    preprocessor = PredictionPreprocessor(
        scaler_path=scaler_path,
        model_path=model_path
    )

    results = preprocessor.predict(df)

    return results


# =============================================================
# Predict from dictionary
# =============================================================

def predict_from_dict(
    customer_data,
    model_path="models/best_model.pkl",
    scaler_path="models/scaler.pkl"
):

    df = pd.DataFrame(
        [customer_data]
    )

    preprocessor = PredictionPreprocessor(
        scaler_path=scaler_path,
        model_path=model_path
    )

    results = preprocessor.predict(df)

    return results
