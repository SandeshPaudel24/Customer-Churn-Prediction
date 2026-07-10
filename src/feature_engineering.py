import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


class FeatureEngineering:

    def __init__(self, df):

        self.df = df

        self.scaler = StandardScaler()

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

    #######################################################

    def split_data(self):

        print("\nSplitting Dataset...")

        X = self.df.drop("Churn", axis=1)

        y = self.df["Churn"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        print("Training :", self.X_train.shape)

        print("Testing :", self.X_test.shape)

    #######################################################

    def apply_smote(self):

        print("\nApplying SMOTE...")

        smote = SMOTE(

            random_state=42

        )

        self.X_train, self.y_train = smote.fit_resample(

            self.X_train,

            self.y_train

        )

        print("After SMOTE")

        print(self.X_train.shape)

    #######################################################

    def scale_data(self):

        print("\nScaling Features...")

        numeric = self.X_train.select_dtypes(

            include=["int64", "float64"]

        ).columns

        self.X_train[numeric] = self.scaler.fit_transform(

            self.X_train[numeric]

        )

        self.X_test[numeric] = self.scaler.transform(

            self.X_test[numeric]

        )

        print("Scaling Complete.")

    #######################################################

    def save_objects(self):

        os.makedirs("models", exist_ok=True)

        joblib.dump(

            self.scaler,

            "models/scaler.pkl"

        )

        print("\nScaler Saved")

    #######################################################

    def save_dataset(self):

        train = self.X_train.copy()

        train["Churn"] = self.y_train

        train.to_csv(

            "data/train.csv",

            index=False

        )

        test = self.X_test.copy()

        test["Churn"] = self.y_test

        test.to_csv(

            "data/test.csv",

            index=False

        )

        print("Train/Test Dataset Saved")

    #######################################################

    def run(self):

        self.split_data()

        self.apply_smote()

        self.scale_data()

        self.save_objects()

        self.save_dataset()

        return (

            self.X_train,

            self.X_test,

            self.y_train,

            self.y_test

        )