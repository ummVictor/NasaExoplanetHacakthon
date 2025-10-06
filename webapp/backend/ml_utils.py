
# ml_utils.py (patched)
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import joblib

# ---- Small, safe transformers ----

class SelectNumeric(BaseEstimator, TransformerMixin):
    """Selects numeric columns from a pandas DataFrame.
    Stores the selected columns during fit and ensures they exist at transform time.
    """
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SelectNumeric expects a pandas DataFrame.")
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SelectNumeric expects a pandas DataFrame.")
        # Ensure all learned columns exist; create missing ones filled with NaN
        missing = [c for c in self.numeric_columns_ if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = np.nan
        return X[self.numeric_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.numeric_columns_, dtype=object)

class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drops columns that are entirely NaN in the CURRENT data split.
    This is fitted inside CV, so it's leakage-safe.
    """
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DropAllNaNColumns expects a pandas DataFrame.")
        self.kept_columns_ = X.columns[~X.isna().all()].tolist()
        return self

    def transform(self, X: pd.DataFrame):
        # Keep only columns learned during fit (order preserved)
        return X.loc[:, self.kept_columns_]

# ---- Preprocessor builder ----

def make_numeric_preprocessor(impute_strategy: str = "median") -> Pipeline:
    """Returns a leakage-safe numeric preprocessor:
       1) select numeric columns,
       2) drop all-NaN columns,
       3) impute remaining NaNs with chosen strategy.
    """
    return Pipeline(steps=[
        ("select_num", SelectNumeric()),
        ("drop_all_nan", DropAllNaNColumns()),
        ("imputer", SimpleImputer(strategy=impute_strategy)),
    ])

# ---- Metrics / evaluation ----

@dataclass
class EvalResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    support_pos: int
    support_neg: int
    confusion: Tuple[int, int, int, int]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray]) -> 'EvalResult':
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc = None
    pr = None
    if y_score is not None:
        roc = roc_auc_score(y_true, y_score)
        precs, recs, _ = precision_recall_curve(y_true, y_score)
        pr  = auc(recs, precs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return EvalResult(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        roc_auc=roc, pr_auc=pr,
        support_pos=int((y_true == 1).sum()),
        support_neg=int((y_true == 0).sum()),
        confusion=(int(tn), int(fp), int(fn), int(tp))
    )

def _try_set_lgb_pos_weight(estimator, pos_weight: float) -> None:
    """If estimator is a Pipeline with final step named 'clf' that is an LGBMClassifier,
       set scale_pos_weight on that step for this fold.
    """
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None

    try:
        if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
            clf = estimator.named_steps["clf"]
            if LGBMClassifier is not None and isinstance(clf, LGBMClassifier):
                estimator.set_params(clf__scale_pos_weight=pos_weight)
    except Exception:
        # Be forgiving: simply skip adjustment if structure differs
        pass

def evaluate_with_oof(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
    set_lgb_pos_weight: bool = False
):
    """Compute OOF predictions using StratifiedKFold and return metrics + OOF arrays.
       The estimator should support predict_proba or decision_function for scores.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=int)

    # Try to produce a score for AUCs
    oof_score = np.zeros(len(y), dtype=float)
    have_scores = True

    for tr_idx, val_idx in skf.split(X, y):
        model = clone(estimator)
        y_tr = y[tr_idx]

        if set_lgb_pos_weight:
            neg = int((y_tr == 0).sum())
            pos = int((y_tr == 1).sum())
            pos_weight = (neg / (pos + 1e-9))
            _try_set_lgb_pos_weight(model, pos_weight)

        # Fit on training fold
        model.fit(X.iloc[tr_idx], y_tr)

        # Predict on validation fold
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])

        # Scores for AUCs
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.iloc[val_idx])
            if proba.ndim == 2 and proba.shape[1] > 1:
                score = proba[:, 1]
        if score is None and hasattr(model, "decision_function"):
            score = model.decision_function(X.iloc[val_idx])
        if score is None:
            have_scores = False
        else:
            oof_score[val_idx] = score

    metrics = compute_metrics(y, oof_pred, oof_score if have_scores else None)
    return metrics, oof_pred, (oof_score if have_scores else None)

# ---- Persistence helpers ----

def persist_model(estimator, X: pd.DataFrame, y: np.ndarray, model_path: str, metadata_path: str, meta_extra: Optional[Dict[str, Any]] = None):
    """Refit estimator on the full data and persist with metadata."""
    estimator.fit(X, y)
    joblib.dump(estimator, model_path)

    try:
        import sklearn  # noqa: F401
        skver = sklearn.__version__
    except Exception:
        skver = None
    try:
        import lightgbm
        lgbver = lightgbm.__version__
    except Exception:
        lgbver = None

    meta = {
        "saved_at": pd.Timestamp.utcnow().isoformat(),
        "model_path": model_path,
        "sklearn_version": skver,
        "lightgbm_version": lgbver,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "n_rows": int(len(y)),
        "n_features_after_preprocess": None,  # can be filled by caller if needed
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

def save_oof(oof_pred: np.ndarray, oof_score: Optional[np.ndarray], path_csv: str):
    df = {"oof_pred": oof_pred.tolist()}
    if oof_score is not None:
        df["oof_score"] = oof_score.tolist()
    pd.DataFrame(df).to_csv(path_csv, index=False)

def format_report(model_name: str, metrics: EvalResult, n_splits: int) -> str:
    parts = [
        f"{model_name} OOF metrics (StratifiedKFold, K={n_splits})",
        f"Accuracy: {metrics.accuracy:.4f}",
        f"Precision: {metrics.precision:.4f}",
        f"Recall: {metrics.recall:.4f}",
        f"F1 Score: {metrics.f1:.4f}",
    ]
    if metrics.roc_auc is not None:
        parts.append(f"ROC AUC: {metrics.roc_auc:.4f}")
    if metrics.pr_auc is not None:
        parts.append(f"PR AUC: {metrics.pr_auc:.4f}")
    tn, fp, fn, tp = metrics.confusion
    parts.append(f"Confusion Matrix [TN, FP, FN, TP]: [{tn}, {fp}, {fn}, {tp}]")
    parts.append(f"Positives: {metrics.support_pos}  Negatives: {metrics.support_neg}")
    return "\n".join(parts) + "\n"

def save_text(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)
