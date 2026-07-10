import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

os.makedirs("models",exist_ok=True)
os.makedirs("outputs/metrics",exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)

class ModelTrainer:

    def __init__(self,
                 X_train,
                 X_test,
                 y_train,
                 y_test):

        self.X_train=X_train
        self.X_test=X_test

        self.y_train=y_train
        self.y_test=y_test

        self.models={}
        self.results=[]

    ####################################################

    def logistic_regression(self):

        print("\nTraining Logistic Regression...\n")

        model=LogisticRegression(

            max_iter=1000,

            random_state=42

        )

        model.fit(

            self.X_train,

            self.y_train

        )

        self.models["Logistic Regression"]=model

        print("Completed.")

    ####################################################

    def decision_tree(self):

        print("\nTraining Decision Tree...\n")

        model=DecisionTreeClassifier(

            random_state=42

        )

        model.fit(

            self.X_train,

            self.y_train

        )

        self.models["Decision Tree"]=model

        print("Completed.")

    ####################################################

    def evaluate(self,name,model):

        prediction=model.predict(

            self.X_test

        )

        probability=model.predict_proba(

            self.X_test

        )[:,1]

        accuracy=accuracy_score(

            self.y_test,

            prediction

        )

        precision=precision_score(

            self.y_test,

            prediction

        )

        recall=recall_score(

            self.y_test,

            prediction

        )

        f1=f1_score(

            self.y_test,

            prediction

        )

        roc=roc_auc_score(

            self.y_test,

            probability

        )

        cv=StratifiedKFold(

            n_splits=5,

            shuffle=True,

            random_state=42

        )

        cv_score=np.mean(

            cross_val_score(

                model,

                self.X_train,

                self.y_train,

                cv=cv,

                scoring="accuracy"

            )

        )

        self.results.append({

            "Model":name,

            "Accuracy":accuracy,

            "Precision":precision,

            "Recall":recall,

            "F1":f1,

            "ROC":roc,

            "CrossValidation":cv_score

        })

        print("\n-----------------------")

        print(name)

        print("-----------------------")

        print(f"Accuracy : {accuracy:.4f}")

        print(f"Precision : {precision:.4f}")

        print(f"Recall : {recall:.4f}")

        print(f"F1 Score : {f1:.4f}")

        print(f"ROC AUC : {roc:.4f}")

        print(f"CV Score : {cv_score:.4f}")

    ####################################################

    # def evaluate_all(self):

    #     for name,model in self.models.items():

    #         self.evaluate(

    #             name,

    #             model

    #         )

    ####################################################

    def save_models(self):

        for name,model in self.models.items():

            filename=name.lower().replace(" ","_")+".pkl"

            joblib.dump(

                model,

                os.path.join(

                    "models",

                    filename

                )

            )

        print("\nModels Saved")

    ####################################################

    def save_metrics(self):

        result=pd.DataFrame(

            self.results

        )

        result.to_csv(

            "outputs/metrics/model_metrics.csv",

            index=False

        )

        print("\nMetrics Saved")

        print(result)

    ####################################################

    # def run(self):

    #     self.logistic_regression()

    #     self.decision_tree()

    #     self.evaluate_all()

    #     self.save_models()

    #     self.save_metrics()

####################################################

    def random_forest(self):

        print("\nTraining Random Forest...\n")

        model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=10
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        self.models["Random Forest"] = model

        print("Random Forest Completed.")

    ####################################################

    def xgboost(self):

        print("\nTraining XGBoost...\n")

        model = XGBClassifier(

            random_state=42,

            eval_metric="logloss",

            n_estimators=200,

            learning_rate=0.05,

            max_depth=6

        )

        model.fit(

            self.X_train,

            self.y_train

        )

        self.models["XGBoost"] = model

        print("XGBoost Completed.")

    ####################################################

    def tune_random_forest(self):

        print("\nGridSearchCV on Random Forest...\n")

        parameters = {

            "n_estimators":[100,200],

            "max_depth":[5,10,15],

            "min_samples_split":[2,5]

        }

        grid = GridSearchCV(

            RandomForestClassifier(random_state=42),

            parameters,

            cv=5,

            scoring="accuracy",

            n_jobs=-1

        )

        grid.fit(

            self.X_train,

            self.y_train

        )

        best_model = grid.best_estimator_

        self.models["Random Forest Tuned"] = best_model

        print("\nBest Parameters")

        print(grid.best_params_)

    ####################################################

    def tune_xgboost(self):

        print("\nGridSearchCV on XGBoost...\n")

        parameters = {

            "n_estimators":[100,200],

            "max_depth":[4,6],

            "learning_rate":[0.05,0.1]

        }

        grid = GridSearchCV(

            XGBClassifier(

                random_state=42,

                eval_metric="logloss"

            ),

            parameters,

            cv=5,

            scoring="accuracy",

            n_jobs=-1

        )

        grid.fit(

            self.X_train,

            self.y_train

        )

        best_model = grid.best_estimator_

        self.models["XGBoost Tuned"] = best_model

        print(grid.best_params_)


####################################################

    def run(self):

        self.logistic_regression()

        self.decision_tree()

        self.random_forest()

        self.xgboost()

        self.tune_random_forest()

        self.tune_xgboost()

        self.evaluate_all()

        self.save_models()

        self.save_metrics()

############################################################

    def plot_confusion_matrix(self, name, model):

        predictions = model.predict(self.X_test)

        cm = confusion_matrix(self.y_test, predictions)

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            f"outputs/plots/{name}_confusion_matrix.png".replace(" ","_")
        )

        plt.close()

    ############################################################

    def plot_roc_curve(self, name, model):

        RocCurveDisplay.from_estimator(
            model,
            self.X_test,
            self.y_test
        )

        plt.title(f"{name} ROC Curve")

        plt.tight_layout()

        plt.savefig(
            f"outputs/plots/{name}_roc_curve.png".replace(" ","_")
        )

        plt.close()

    ############################################################

    def plot_precision_recall(self, name, model):

        PrecisionRecallDisplay.from_estimator(
            model,
            self.X_test,
            self.y_test
        )

        plt.title(f"{name} Precision Recall")

        plt.tight_layout()

        plt.savefig(
            f"outputs/plots/{name}_precision_recall.png".replace(" ","_")
        )

        plt.close()

    ############################################################

    def save_predictions(self, name, model):

        prediction = model.predict(self.X_test)

        probability = model.predict_proba(self.X_test)[:,1]

        result = pd.DataFrame({

            "Actual":self.y_test,

            "Prediction":prediction,

            "Probability":probability

        })

        result.to_csv(

            f"outputs/predictions/{name}.csv".replace(" ","_"),

            index=False

        )

    ############################################################

    def compare_models(self):

        df = pd.DataFrame(self.results)

        plt.figure(figsize=(10,6))

        plt.bar(

            df["Model"],

            df["Accuracy"]

        )

        plt.xticks(rotation=25)

        plt.ylabel("Accuracy")

        plt.title("Model Comparison")

        plt.tight_layout()

        plt.savefig(

            "outputs/plots/model_accuracy_comparison.png"

        )

        plt.close()

    ############################################################

    def save_best_model(self):

        df = pd.DataFrame(self.results)

        best = df.sort_values(

            by="Accuracy",

            ascending=False

        ).iloc[0]

        best_name = best["Model"]

        best_model = self.models[best_name]

        joblib.dump(

            best_model,

            "models/best_model.pkl"

        )

        print("\nBest Model")

        print(best_name)

        print(best["Accuracy"])

    ############################################################

    def evaluate_all(self):

        for name,model in self.models.items():

            self.evaluate(name,model)

            self.plot_confusion_matrix(name,model)

            self.plot_roc_curve(name,model)

            self.plot_precision_recall(name,model)

            self.save_predictions(name,model)

        self.compare_models()

        self.save_best_model()