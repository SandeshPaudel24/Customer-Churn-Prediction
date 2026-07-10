import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    # -----------------------------
    # Load Dataset
    # -----------------------------
    def load_data(self):

        print("\nLoading dataset...")

        self.df = pd.read_csv(self.filepath)

        print("Dataset loaded successfully.")
        print(f"Shape : {self.df.shape}")

    # -----------------------------
    # Dataset Information
    # -----------------------------
    def dataset_info(self):

        print("\nDataset Information")
        print("-" * 60)

        print(self.df.info())

        print("\nData Types")
        print(self.df.dtypes)

    # -----------------------------
    # Missing Values
    # -----------------------------
    def check_missing_values(self):

        print("\nChecking Missing Values")

        print(self.df.isnull().sum())

    # -----------------------------
    # Duplicate Values
    # -----------------------------
    def check_duplicates(self):

        duplicates = self.df.duplicated().sum()

        print(f"\nDuplicate Rows : {duplicates}")

    # -----------------------------
    # Handle TotalCharges
    # -----------------------------
    def handle_totalcharges(self):

        print("\nCleaning TotalCharges...")

        self.df["TotalCharges"] = self.df["TotalCharges"].replace(" ", pd.NA)

        self.df["TotalCharges"] = pd.to_numeric(
            self.df["TotalCharges"],
            errors="coerce"
        )

        median_value = self.df["TotalCharges"].median()

        self.df["TotalCharges"] = self.df["TotalCharges"].fillna(
            median_value
        )

        print("TotalCharges cleaned.")

    # -----------------------------
    # Remove customerID
    # -----------------------------
    def remove_customer_id(self):

        if "customerID" in self.df.columns:

            self.df.drop(
                columns=["customerID"],
                inplace=True
            )

            print("customerID removed.")

    # -----------------------------
    # Label Encoding
    # -----------------------------
    def label_encode(self):

        print("\nEncoding Binary Columns...")

        encoder = LabelEncoder()

        binary_columns = []

        for column in self.df.columns:

            if self.df[column].dtype == object:

                if self.df[column].nunique() == 2:

                    binary_columns.append(column)

        for column in binary_columns:

            self.df[column] = encoder.fit_transform(
                self.df[column]
            )

        print(binary_columns)

    # -----------------------------
    # One Hot Encoding
    # -----------------------------
    def one_hot_encoding(self):

        print("\nOne Hot Encoding...")

        categorical_columns = []

        for column in self.df.columns:

            if self.df[column].dtype == object:

                categorical_columns.append(column)

        self.df = pd.get_dummies(

            self.df,

            columns=categorical_columns,

            drop_first=True

        )

        print("Encoding completed.")

    # -----------------------------
    # Final Shape
    # -----------------------------
    def final_shape(self):

        print("\nFinal Dataset Shape")

        print(self.df.shape)

    # -----------------------------
    # Save Dataset
    # -----------------------------
    def save_dataset(self):

        output = "data/clean_telco.csv"

        self.df.to_csv(

            output,

            index=False

        )

        print(f"\nClean dataset saved : {output}")

    # -----------------------------
    # Run All
    # -----------------------------
    def preprocess(self):

        self.load_data()

        self.dataset_info()

        self.check_missing_values()

        self.check_duplicates()

        self.handle_totalcharges()

        self.remove_customer_id()

        self.label_encode()

        self.one_hot_encoding()

        self.final_shape()

        self.save_dataset()

        return self.df