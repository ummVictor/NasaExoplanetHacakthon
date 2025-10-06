
# train_ada_pipeline_cv.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from ml_utils import make_numeric_preprocessor, evaluate_with_oof, persist_model, format_report, save_text, save_oof

DATA_CSV = "preprocessed_data_clean.csv"
REPORT_FILE = "ada_report.txt"
OOF_FILE = "ada_oof.csv"
MODEL_PATH = "ada_pipeline.joblib"
META_PATH = "ada_meta.json"
N_SPLITS = 10
RANDOM_STATE = 42

# Load data
df = pd.read_csv(DATA_CSV)
y = df['label'].replace('NA', np.nan).astype(float)
X = df.drop(columns=['label']).copy()

mask = y.notna()
X = X[mask]
y = y[mask].astype(int).values

# Leakage-safe pipeline (median impute inside CV)
pre = make_numeric_preprocessor(impute_strategy="median")
base = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
est = Pipeline([
    ("num", pre),
    ("clf", AdaBoostClassifier(estimator=base, n_estimators=200, random_state=RANDOM_STATE))
])

# OOF evaluation
metrics, oof_pred, oof_score = evaluate_with_oof(est, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE)

# Save report and OOF
report = format_report("AdaBoost", metrics, N_SPLITS)
print(report)
save_text(report, REPORT_FILE)
save_oof(oof_pred, oof_score, OOF_FILE)

# Persist full model
persist_model(est, X, y, MODEL_PATH, META_PATH, meta_extra={
    "model_name": "AdaBoost (DecisionTree base, pipeline)",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE
})
print(f"Saved model to {MODEL_PATH} and metadata to {META_PATH}.")
