"""
Unified exoplanet classifier: (optional) TSFRESH -> clean -> CV -> stacking -> export -> web UI.

Usage:
  # plain tabular
  python train_stack_and_app.py --csv unified_exoplanets.csv --target label --idcols "mission object_id source_id_raw"

  # with TSFRESH (comprehensive ~700+ features)
  python train_stack_and_app.py ^
    --csv unified_exoplanets.csv ^
    --target label ^
    --idcols "mission object_id source_id_raw" ^
    --tsfresh_dir lightcurves/ ^
    --tsfresh_id_col object_id --tsfresh_time_col time --tsfresh_flux_col flux ^
    --tsfresh_preset comprehensive ^
    --tsfresh_join_on object_id ^
    --tsfresh_cache tsfresh_features.parquet
"""

import argparse, json, os, joblib, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_recall_curve

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# ---- Optional LightGBM (fallback to HistGradientBoosting) ----
try:
    import lightgbm as lgb
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# ----------------------------- utils -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV  = PROJECT_ROOT / "unified_exoplanets.csv"
OLD_NAME     = PROJECT_ROOT / "unified_exoplanets_uncleaned_combined.csv"

def _safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def load_data(path: Path | str | None) -> pd.DataFrame:
    """
    Load CSV; if the exact path is missing, try sensible fallbacks:
    - unified_exoplanets.csv (from ETL)
    - unified_exoplanets_uncleaned_combined.csv (older name)
    - newest match of 'unified_exoplanets*.csv' anywhere under the repo
    """
    candidates: List[Path] = []
    if path:
        p = Path(path)
        candidates.append(p if p.is_absolute() else (Path.cwd() / p))
    candidates += [DEFAULT_CSV, OLD_NAME]

    for cand in candidates:
        if _safe_exists(cand):
            print(f"[load_data] Using CSV: {cand}")
            return pd.read_csv(cand, comment="#", low_memory=False)

    matches = list(PROJECT_ROOT.rglob("unified_exoplanets*.csv"))
    if matches:
        newest = max(matches, key=lambda q: q.stat().st_mtime)
        print(f"[load_data] Using newest match: {newest}")
        return pd.read_csv(newest, comment="#", low_memory=False)

    raise FileNotFoundError(
        "Could not find a training CSV.\n"
        f"Looked for:\n  - {path}\n  - {DEFAULT_CSV}\n  - {OLD_NAME}\n"
        "Tip: run the ETL first or pass --csv <full-path-to-unified_exoplanets.csv>"
    )

def prune_columns(df: pd.DataFrame, target: str, idcols: List[str], min_nonnull_frac: float = 0.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns that are entirely NaN, and optionally columns with non-null coverage < min_nonnull_frac.
    Returns (df_new, dropped_columns_list).
    """
    n = len(df)
    feature_cols = [c for c in df.columns if c not in idcols + [target]]
    dropped: List[str] = []

    # Drop columns that are 100% NaN
    all_nan = [c for c in feature_cols if df[c].notna().sum() == 0]
    if all_nan:
        df = df.drop(columns=all_nan)
        dropped.extend(all_nan)

    # Optionally drop ultra-sparse columns
    if min_nonnull_frac > 0.0:
        still_features = [c for c in df.columns if c not in idcols + [target]]
        sparse = [c for c in still_features if (df[c].notna().sum() / max(n, 1)) < min_nonnull_frac]
        if sparse:
            df = df.drop(columns=sparse)
            dropped.extend(sparse)

    return df, dropped

def split_columns(df: pd.DataFrame, target: str, idcols: List[str]) -> Tuple[List[str], List[str]]:
    # treat anything object-like as categorical
    feats = [c for c in df.columns if c not in idcols + [target]]
    cat_cols = [c for c in feats if str(df[c].dtype) == 'object']
    num_cols = [c for c in feats if c not in cat_cols]
    return num_cols, cat_cols

def make_preprocessor(num_cols: List[str], cat_cols: List[str], scale_for_linear: bool = False) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_for_linear:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop",
        n_jobs=None
    )

def tune_threshold_for_recall(y_true: np.ndarray, proba: np.ndarray, min_precision: float = 0.6) -> Tuple[float, Dict[str, float]]:
    """Pick a threshold that maximizes recall while keeping precision >= min_precision."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    best_r, best_t, best_p = 0.0, 0.5, 0.5
    for p, r, t in zip(prec, rec, np.append(thr, 1.0)):
        if p >= min_precision and r >= best_r:
            best_r, best_t, best_p = r, t, p
    return best_t, {"precision": best_p, "recall": best_r}

def kfold_eval(clf: Pipeline, X: pd.DataFrame, y: np.ndarray, folds: int = 10, min_precision: float = 0.6, random_state: int = 42) -> Dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    aucs, precs, recs, f1s, accs, thrs = [], [], [], [], [], []

    for tr, va in skf.split(X, y):
        clf.fit(X.iloc[tr], y[tr])
        proba = clf.predict_proba(X.iloc[va])[:, 1]
        thr, pr = tune_threshold_for_recall(y[va], proba, min_precision=min_precision)
        yhat = (proba >= thr).astype(int)

        aucs.append(roc_auc_score(y[va], proba))
        p, r, f1, _ = precision_recall_fscore_support(y[va], yhat, average="binary", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1); accs.append(accuracy_score(y[va], yhat)); thrs.append(thr)

    return {
        "AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
        "Precision_mean": float(np.mean(precs)), "Recall_mean": float(np.mean(recs)),
        "F1_mean": float(np.mean(f1s)), "Accuracy_mean": float(np.mean(accs)),
        "Threshold_mean": float(np.mean(thrs))
    }

# ----------------------------- TSFRESH branch -----------------------------
def _get_tsfresh_fc_params(preset: str):
    try:
        from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
    except ImportError as e:
        raise ImportError("tsfresh is not installed. Install with: pip install tsfresh dask") from e
    preset = preset.lower()
    if preset == "minimal":
        return MinimalFCParameters()
    if preset == "efficient":
        return EfficientFCParameters()
    # default: comprehensive (~700+ features)
    return ComprehensiveFCParameters()

def _read_lightcurves(ts_dir: Path, id_col: str, time_col: str, flux_col: str) -> pd.DataFrame:
    files = sorted([p for p in ts_dir.glob("**/*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV lightcurves found in: {ts_dir}")
    dfs = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
            # keep only necessary columns; enforce names
            if not all(c in d.columns for c in [id_col, time_col, flux_col]):
                # try case-insensitive fallback
                cols_lower = {c.lower(): c for c in d.columns}
                idc  = cols_lower.get(id_col.lower(), id_col)
                timc = cols_lower.get(time_col.lower(), time_col)
                flxc = cols_lower.get(flux_col.lower(), flux_col)
                d = d.rename(columns={idc:id_col, timc:time_col, flxc:flux_col})
            d = d[[id_col, time_col, flux_col]].dropna()
            dfs.append(d)
        except Exception:
            # skip broken file but continue
            continue
    if not dfs:
        raise RuntimeError("No usable lightcurve CSVs after parsing.")
    ts = pd.concat(dfs, ignore_index=True)
    # basic cleaning
    ts = ts.dropna().sort_values([id_col, time_col])
    return ts

def compute_tsfresh_features(ts_dir: Path,
                             id_col: str,
                             time_col: str,
                             flux_col: str,
                             join_on: str,
                             main_df: pd.DataFrame,
                             preset: str = "comprehensive",
                             cache_path: Optional[Path] = None,
                             n_jobs: int = 0,
                             do_supervised_select: bool = False) -> pd.DataFrame:
    """
    Extract TSFRESH features and (optionally) select relevant ones via labels.
    Returns a DataFrame indexed by join_on with TSFRESH_* columns ready to merge.
    """
    # Use cache if available
    if cache_path and cache_path.exists():
        feats = pd.read_parquet(cache_path)
        return feats

    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute

    ts = _read_lightcurves(ts_dir, id_col, time_col, flux_col)
    fc_params = _get_tsfresh_fc_params(preset)

    # tsfresh expects column_id, column_sort, column_value
    features = extract_features(
        ts,
        column_id=id_col,
        column_sort=time_col,
        column_value=flux_col,
        default_fc_parameters=fc_params,
        disable_progressbar=False,
        n_jobs=n_jobs
    )

    # impute per tsfresh’s recommended strategy
    impute(features)

    # Optional supervised selection (drop features not predictive of label)
    if do_supervised_select:
        lab = main_df[[join_on, "label"]].dropna().drop_duplicates(subset=[join_on]).set_index(join_on)["label"]
        common_idx = features.index.intersection(lab.index)
        selected = select_features(features.loc[common_idx], lab.loc[common_idx])
        feats = selected.add_prefix("TSFRESH_")
    else:
        feats = features.add_prefix("TSFRESH_")

    # Persist cache if requested
    if cache_path:
        feats.to_parquet(cache_path, index=True)

    return feats

def maybe_merge_tsfresh(df: pd.DataFrame, args) -> pd.DataFrame:
    if not args.tsfresh_dir:
        return df  # no-op

    ts_dir = Path(args.tsfresh_dir)
    cache_path = Path(args.tsfresh_cache) if args.tsfresh_cache else None
    join_on = args.tsfresh_join_on or "object_id"

    print(f"\n[TSFRESH] Extracting features from {ts_dir} (preset: {args.tsfresh_preset})...")
    feats = compute_tsfresh_features(
        ts_dir=ts_dir,
        id_col=args.tsfresh_id_col,
        time_col=args.tsfresh_time_col,
        flux_col=args.tsfresh_flux_col,
        join_on=join_on,
        main_df=df,
        preset=args.tsfresh_preset,
        cache_path=cache_path,
        n_jobs=args.tsfresh_jobs,
        do_supervised_select=args.tsfresh_select
    )

    # Merge into main table
    if join_on not in df.columns:
        raise KeyError(f"'--tsfresh_join_on {join_on}' not found in main CSV columns.")
    merged = df.merge(feats, how="left", left_on=join_on, right_index=True)
    print(f"[TSFRESH] Merged {feats.shape[1]} features on '{join_on}'. New shape: {merged.shape}")
    return merged

# ----------------------------- main -----------------------------
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv).copy()
    # Basic label cleanup
    if args.target not in df.columns:
        raise KeyError(f"Target column '{args.target}' not found in CSV.")
    df = df[df[args.target].notna()]
    if df[args.target].dtype != int:
        df[args.target] = df[args.target].astype(int)

    # Optional: append TSFRESH features from raw lightcurves
    df = maybe_merge_tsfresh(df, args)

    idcols = args.idcols.split() if args.idcols else []

    # Drop dead/ultra-sparse columns to avoid imputer warnings and dead features
    df, dropped_cols = prune_columns(df, target=args.target, idcols=idcols, min_nonnull_frac=args.min_nonnull_frac)
    if dropped_cols:
        print(f"[prune] Dropped {len(dropped_cols)} columns (all-NaN or too sparse), e.g.: {dropped_cols[:10]}")

    num_cols, cat_cols = split_columns(df, args.target, idcols)

    # Two preprocessors: tree-friendly (no scaling), linear-friendly (scaling)
    prep_tree = make_preprocessor(num_cols, cat_cols, scale_for_linear=False)
    prep_lin  = make_preprocessor(num_cols, cat_cols, scale_for_linear=True)  # kept for future linear/meta models

    # Base learners
    if _HAVE_LGBM:
        lgbm = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=64,
            colsample_bytree=0.8, subsample=0.8, reg_alpha=0.0, reg_lambda=0.2,
            objective="binary", class_weight="balanced", random_state=42
        )
        print("[Info] Using LightGBM.")
    else:
        lgbm = HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.05, max_iter=800, random_state=42
        )
        print("[Info] LightGBM not installed; using HistGradientBoosting fallback.")

    rf   = RandomForestClassifier(n_estimators=600, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    adb  = AdaBoostClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
    et   = ExtraTreesClassifier(n_estimators=600, max_depth=None, random_state=42, n_jobs=-1)

    # Pipelines
    pipe_lgbm = Pipeline([("prep", prep_tree), ("clf", lgbm)])
    pipe_rf   = Pipeline([("prep", prep_tree), ("clf", rf)])
    pipe_adb  = Pipeline([("prep", prep_tree), ("clf", adb)])
    pipe_et   = Pipeline([("prep", prep_tree), ("clf", et)])

    # Split X/y (drop ids)
    X = df.drop(columns=[args.target] + idcols)
    y = df[args.target].values

    print("\nEvaluating base models with CV (threshold tuned for high recall)...")
    for name, pipe in [("GBM", pipe_lgbm), ("RandomForest", pipe_rf), ("AdaBoost", pipe_adb), ("ExtraTrees", pipe_et)]:
        res = kfold_eval(pipe, X, y, folds=args.folds, min_precision=args.min_precision)
        print(f"{name}: {json.dumps(res, indent=2)}")

    # Stacking (meta-learner is logistic regression)
    estimators = [
        ("gbm", pipe_lgbm),
        ("rf",  pipe_rf),
        ("adb", pipe_adb),
        ("et",  pipe_et)
    ]
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False, n_jobs=-1
    )
    stack_pipe = stack

    print("\nEvaluating STACKED ensemble with CV...")
    res_stack = kfold_eval(stack_pipe, X, y, folds=args.folds, min_precision=args.min_precision)
    print("Stacked:", json.dumps(res_stack, indent=2))

    # Fit stacked on all data and save
    stack_pipe.fit(X, y)
    model_path = Path(args.outdir) / "exo_stacked_model.joblib"
    joblib.dump({
        "model": stack_pipe,
        "feature_columns": X.columns.tolist(),
        "id_columns": idcols,
        "target": args.target,
        "cv_report": {"base": "see console", "stacked": res_stack}
    }, model_path)
    print(f"\nSaved model → {model_path.resolve()}")

    if args.serve:
        launch_app(model_path)

# ----------------------------- web app -----------------------------
def launch_app(model_path: Path):
    # Defer gradio import so training runs without it
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit(
            "Gradio is not installed. To use --serve, run:\n"
            "  pip install -U gradio"
        )

    blob = joblib.load(model_path)
    model = blob["model"]
    feat_cols = blob["feature_columns"]

    def predict_one(record: dict):
        row = pd.DataFrame([{k: record.get(k, None) for k in feat_cols}])
        proba = model.predict_proba(row)[:, 1][0]
        return float(proba)

    def predict_csv(file):
        df = pd.read_csv(file.name)
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise gr.Error(f"Your CSV is missing {len(missing)} required feature columns, e.g. {missing[:10]}")
        proba = model.predict_proba(df[feat_cols])[:, 1]
        out = df.copy()
        out["planet_prob"] = proba
        out["pred_label@0.5"] = (proba >= 0.5).astype(int)
        save_path = Path(file.name).with_suffix(".predictions.csv")
        out.to_csv(save_path, index=False)
        return save_path.name

    with gr.Blocks(title="Exoplanet Classifier") as demo:
        gr.Markdown("# Exoplanet Vetting (Stacked ML)\nPredict planet probability from tabular + (optional) TSFRESH features.")
        with gr.Tab("Single Entry"):
            inputs = [gr.Textbox(label=f, placeholder="value (leave blank if unknown)") for f in feat_cols]
            btn = gr.Button("Predict")
            out = gr.Number(label="Planet probability")
            btn.click(fn=lambda *vals: predict_one(dict(zip(feat_cols, vals))), inputs=inputs, outputs=out)

        with gr.Tab("Batch CSV"):
            up = gr.File(label="Upload CSV with same feature columns", file_types=[".csv"])
            btn2 = gr.Button("Run batch predictions")
            out2 = gr.File(label="Download predictions")
            btn2.click(fn=predict_csv, inputs=up, outputs=out2)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# ----------------------------- entry -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    ap.add_argument("--target", type=str, default="label")
    ap.add_argument("--idcols", type=str, default="mission object_id source_id_raw")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--min_precision", type=float, default=0.6, help="Tune threshold for highest recall with precision >= this value")
    ap.add_argument("--min_nonnull_frac", type=float, default=0.0, help="Drop feature columns with non-null coverage below this fraction (0.0 disables)")
    ap.add_argument("--outdir", type=str, default="models_out")
    ap.add_argument("--serve", action="store_true", help="Launch Gradio app after training")

    # TSFRESH options
    ap.add_argument("--tsfresh_dir", type=str, default=None, help="Directory containing raw light curves (.csv)")
    ap.add_argument("--tsfresh_id_col", type=str, default="object_id")
    ap.add_argument("--tsfresh_time_col", type=str, default="time")
    ap.add_argument("--tsfresh_flux_col", type=str, default="flux")
    ap.add_argument("--tsfresh_preset", type=str, default="comprehensive", choices=["minimal","efficient","comprehensive"])
    ap.add_argument("--tsfresh_join_on", type=str, default="object_id", help="Column in main CSV to join TSFRESH features on")
    ap.add_argument("--tsfresh_cache", type=str, default=None, help="Parquet path to cache extracted features")
    ap.add_argument("--tsfresh_jobs", type=int, default=0, help="n_jobs for tsfresh (0=auto)")
    ap.add_argument("--tsfresh_select", action="store_true", help="Use supervised feature selection (requires labels)")

    args = ap.parse_args()

    # Silence known harmless sklearn/LightGBM warning when pipeline passes arrays
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
    )

    main(args)
