
# train_stack_pipeline_cv.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from ml_utils import make_numeric_preprocessor, evaluate_with_oof, persist_model, format_report, save_text, save_oof

DATA_CSV = "preprocessed_data_clean.csv"
REPORT_FILE = "stack_report.txt"
OOF_FILE = "stack_oof.csv"
MODEL_PATH = "stack_model.joblib"
META_PATH = "stack_meta.json"
N_SPLITS = 5
RANDOM_STATE = 42

# Load data
df = pd.read_csv(DATA_CSV)
y = df['label'].replace('NA', np.nan).astype(float)
X = df.drop(columns=['label']).copy()

mask = y.notna()
X = X[mask]
y = y[mask].astype(int).values

# Each base learner gets its OWN leakage-safe numeric preprocessor inside
pre = make_numeric_preprocessor(impute_strategy="median")

lgbm = Pipeline([
    ("num", make_numeric_preprocessor("median")),
    ("clf", lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
    ))
])

rf = Pipeline([
    ("num", make_numeric_preprocessor("median")),
    ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
])

ada = Pipeline([
    ("num", make_numeric_preprocessor("median")),
    ("clf", AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
                               n_estimators=200, random_state=RANDOM_STATE))
])

et = Pipeline([
    ("num", make_numeric_preprocessor("median")),
    ("clf", ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
])

estimators = [('lgb', lgbm), ('rf', rf), ('ada', ada), ('et', et)]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=N_SPLITS,
    n_jobs=-1,
    passthrough=False
)

# Note: preprocessing lives INSIDE each base estimator -> no leakage
est = stack

# OOF evaluation (external CV separate from the internal CV used by Stacking)
metrics, oof_pred, oof_score = evaluate_with_oof(est, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE)

# Save report and OOF
report = format_report("StackingClassifier", metrics, N_SPLITS)
print(report)
save_text(report, REPORT_FILE)
save_oof(oof_pred, oof_score, OOF_FILE)

# Persist full model
persist_model(est, X, y, MODEL_PATH, META_PATH, meta_extra={
    "model_name": "StackingClassifier (pipelines for bases)",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE
})
print(f"Saved model to {MODEL_PATH} and metadata to {META_PATH}.")
