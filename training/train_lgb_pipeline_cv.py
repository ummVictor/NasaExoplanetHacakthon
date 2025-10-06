
# train_lgb_pipeline_cv.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from ml_utils import make_numeric_preprocessor, evaluate_with_oof, persist_model, format_report, save_text, save_oof

DATA_CSV = "preprocessed_data_clean.csv"
REPORT_FILE = "lgb_report.txt"
OOF_FILE = "lgb_oof.csv"
MODEL_PATH = "lgb_pipeline.joblib"
META_PATH = "lgb_meta.json"
N_SPLITS = 10
RANDOM_STATE = 42

# Load data
df = pd.read_csv(DATA_CSV)
y = df['label'].replace('NA', np.nan).astype(float)
X = df.drop(columns=['label']).copy()

mask = y.notna()
X = X[mask]
y = y[mask].astype(int).values  # numpy for speed

# Build leakage-safe pipeline
pre = make_numeric_preprocessor(impute_strategy="median")  # LGBM can handle NaNs, but median keeps parity with others
est = Pipeline([
    ("num", pre),
    ("clf", lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    ))
])

# OOF evaluation with per-fold scale_pos_weight
metrics, oof_pred, oof_score = evaluate_with_oof(
    est, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE, set_lgb_pos_weight=True
)

# Save report and OOF
report = format_report("LightGBM", metrics, N_SPLITS)
print(report)
save_text(report, REPORT_FILE)
save_oof(oof_pred, oof_score, OOF_FILE)

# Persist full model
persist_model(est, X, y, MODEL_PATH, META_PATH, meta_extra={
    "model_name": "LGBMClassifier (pipeline)",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE
})
print(f"Saved model to {MODEL_PATH} and metadata to {META_PATH}.")
