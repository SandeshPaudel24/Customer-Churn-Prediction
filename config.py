import os

# ====================================================
# PATHS
# ====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs"
)

EDA_PATH = os.path.join(
    OUTPUT_PATH,
    "eda"
)

ROC_PATH = os.path.join(
    OUTPUT_PATH,
    "roc_curve"
)

CONFUSION_PATH = os.path.join(
    OUTPUT_PATH,
    "confusion_matrix"
)

FEATURE_PATH = os.path.join(
    OUTPUT_PATH,
    "feature_importance"
)

SHAP_PATH = os.path.join(
    OUTPUT_PATH,
    "shap"
)

METRIC_PATH = os.path.join(
    OUTPUT_PATH,
    "metrics"
)

REPORT_PATH = os.path.join(
    OUTPUT_PATH,
    "reports"
)



folders = [
    MODEL_PATH,
    OUTPUT_PATH,
    EDA_PATH,
    ROC_PATH,
    CONFUSION_PATH,
    FEATURE_PATH,
    SHAP_PATH,
    METRIC_PATH,
    REPORT_PATH
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)