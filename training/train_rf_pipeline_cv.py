
# train_rf_pipeline_cv.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ml_utils import make_numeric_preprocessor, evaluate_with_oof, persist_model, format_report, save_text, save_oof

DATA_CSV = "preprocessed_data_clean.csv"
REPORT_FILE = "rf_report.txt"
OOF_FILE = "rf_oof.csv"
MODEL_PATH = "rf_pipeline.joblib"
META_PATH = "rf_meta.json"
N_SPLITS = 10
RANDOM_STATE = 42

# Load data
df = pd.read_csv(DATA_CSV)
y = df['label'].replace('NA', np.nan).astype(float)
X = df.drop(columns=['label']).copy()

mask = y.notna()
X = X[mask]
y = y[mask].astype(int).values

# Leakage-safe pipeline
pre = make_numeric_preprocessor(impute_strategy="median")
est = Pipeline([
    ("num", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# OOF evaluation
metrics, oof_pred, oof_score = evaluate_with_oof(est, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE)

# Save report and OOF
report = format_report("RandomForest", metrics, N_SPLITS)
print(report)
save_text(report, REPORT_FILE)
save_oof(oof_pred, oof_score, OOF_FILE)

# Persist full model
persist_model(est, X, y, MODEL_PATH, META_PATH, meta_extra={
    "model_name": "RandomForest (pipeline)",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE
})
print(f"Saved model to {MODEL_PATH} and metadata to {META_PATH}.")
