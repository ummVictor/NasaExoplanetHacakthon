# clean_unified_exoplanets_v2.py
# - Per-mission outlier removal (IQR)
# - Per-mission median imputation (numeric), most-freq (categorical)
# - Optional per-mission z-score normalization (replace numeric columns)
# - Export two files: modeling CSV (numeric-friendly) + UI CSV (NaN -> "N/A")
# - Write a concise cleaning report

import argparse
from pathlib import Path
from textwrap import indent
import numpy as np
import pandas as pd

DEFAULTS = dict(
    # FIX: use raw string OR forward slashes
    path=r"C:\Users\victo\OneDrive\Desktop\Git\NasaExoplanetHacakthon\unification\unified_exoplanets_uncleaned_combined.csv",
    out_model_csv="unified_exoplanets_cleaned.csv",
    out_ui_csv="unified_exoplanets_cleaned_ui.csv",
    report_txt="cleaning_report.txt",
    mission_col="mission",
    label_col="label",
    iqr_k=1.5,
    normalize=True,                # default behavior; see paired CLI flags below
    keep_cols_always=["mission", "label", "object_id", "source_id_raw"]
)

# ---------- helpers ----------
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)

def summarize_missingness(df: pd.DataFrame, title: str, topn: int = 20) -> str:
    miss = df.isna().mean().sort_values(ascending=False)
    head = miss.head(topn)
    return f"{title}\n(showing top {len(head)}):\n{head}"

def per_mission_iqr_mask(df: pd.DataFrame, mission_col: str, cols: list, k: float = 1.5) -> pd.Series:
    """Return a boolean mask of rows to KEEP after IQR trimming per mission for each numeric column."""
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby(mission_col):
        idx = g.index
        for c in cols:
            if c not in g.columns:
                continue
            s_num = pd.to_numeric(g[c], errors="coerce")
            q1, q3 = s_num.quantile(0.25), s_num.quantile(0.75)
            if pd.isna(q1) or pd.isna(q3):
                continue
            iqr = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            in_fence = (s_num >= lo) & (s_num <= hi)
            keep.loc[idx] &= (in_fence | s_num.isna())  # keep NaNs; they’ll be imputed later
    return keep

def per_mission_impute(df: pd.DataFrame, mission_col: str):
    """Median-impute numeric; most-frequent for categoricals, per mission."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    for _, g in df.groupby(mission_col):
        idx = g.index
        if num_cols:
            med = g[num_cols].median(numeric_only=True)
            df.loc[idx, num_cols] = g[num_cols].fillna(med)
        if cat_cols:
            mode_vals = {}
            for c in cat_cols:
                mode = g[c].mode(dropna=True)
                if len(mode) > 0:
                    mode_vals[c] = mode.iloc[0]
            if mode_vals:
                df.loc[idx, list(mode_vals.keys())] = g[list(mode_vals.keys())].fillna(pd.Series(mode_vals))
    return df

def per_mission_zscore(df: pd.DataFrame, mission_col: str, numeric_cols: list) -> pd.DataFrame:
    """Replace numeric columns with per-mission z-scores (stable if std≈0)."""
    out = df.copy()
    for _, g in df.groupby(mission_col):
        idx = g.index
        mu = g[numeric_cols].mean(numeric_only=True)
        sd = g[numeric_cols].std(numeric_only=True).replace(0, np.nan)
        z = (g[numeric_cols] - mu) / sd
        out.loc[idx, numeric_cols] = z.fillna(0.0)  # if std was 0, set z=0
    return out

def fill_ui_na(df: pd.DataFrame) -> pd.DataFrame:
    """Return a UI-friendly copy with NaNs replaced by string 'N/A'."""
    return df.where(~df.isna(), other="N/A")

# ---------- main ----------
def main(args):
    # ✅ Using Path handles forward/back slashes cross-platform
    src = Path(args.path)
    out_model = Path(args.out_model_csv)
    out_ui = Path(args.out_ui_csv)
    rpt = Path(args.report_txt)

    df0 = read_csv(src)
    rep = []
    rep.append("=== DATA CLEANING (v2) ===")
    rep.append(f"Input: {src}")
    rep.append(f"Rows x Cols (raw): {df0.shape[0]} x {df0.shape[1]}")

    # Basic checks
    if args.label_col not in df0.columns:
        raise KeyError(f"Target '{args.label_col}' not found.")
    if args.mission_col not in df0.columns:
        raise KeyError(f"Mission column '{args.mission_col}' not found.")

    # Drop rows with missing label; cast
    df = df0[df0[args.label_col].notna()].copy()
    try:
        df[args.label_col] = df[args.label_col].astype(int)
    except Exception:
        pass

    rep.append("\nMission counts (before):\n" + str(df[args.mission_col].value_counts()))
    rep.append("\nClass counts (before):\n" + str(df[args.label_col].value_counts()))
    rep.append("\n" + summarize_missingness(df, "Missingness (before)"))

    # --- 1) Outlier removal (IQR) per mission on numeric columns
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols_all if c not in (args.keep_cols_always or [])]
    keep_mask = per_mission_iqr_mask(df, args.mission_col, num_cols, k=args.iqr_k)
    dropped_outliers = int((~keep_mask & df[num_cols].notna().any(axis=1)).sum())
    df = df[keep_mask].copy()
    rep.append(f"\nOutlier removal (IQR, k={args.iqr_k:.2f}) → dropped rows: {dropped_outliers}")

    # --- 2) Imputation per mission (median numeric, most-freq categorical)
    df = per_mission_impute(df, args.mission_col)
    rep.append("\n" + summarize_missingness(df, "Missingness (after imputation)"))

    # --- 3) Normalization (z-score) per mission (optional)
    if args.normalize:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in (args.keep_cols_always or [])]
        df = per_mission_zscore(df, args.mission_col, num_cols)
        rep.append(f"\nNormalization: per-mission z-score applied to {len(num_cols)} numeric columns")

    # --- 4) Final sanity
    if df[args.mission_col].isna().any():
        raise ValueError("Mission missing values remain after cleaning.")
    if df[args.label_col].isna().any():
        raise ValueError("Label missing values remain after cleaning.")

    # --- 5) Save outputs
    df.to_csv(out_model, index=False)           # model-friendly
    ui = fill_ui_na(df)                         # UI-friendly (NaN -> "N/A")
    ui.to_csv(out_ui, index=False)

    rep.append(f"\nRows x Cols (after): {df.shape[0]} x {df.shape[1]}")
    rep.append(f"Saved model CSV → {out_model}")
    rep.append(f"Saved UI CSV    → {out_ui}")

    rpt.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))

# ---------- cli ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULTS["path"])
    ap.add_argument("--out_model_csv", default=DEFAULTS["out_model_csv"])
    ap.add_argument("--out_ui_csv", default=DEFAULTS["out_ui_csv"])
    ap.add_argument("--report_txt", default=DEFAULTS["report_txt"])
    ap.add_argument("--mission_col", default=DEFAULTS["mission_col"])
    ap.add_argument("--label_col", default=DEFAULTS["label_col"])
    ap.add_argument("--iqr_k", type=float, default=DEFAULTS["iqr_k"], help="IQR multiplier for outlier fences (1.5 typical; 3.0 conservative)")

    # ✅ Paired flags for normalization so you can turn it off:
    norm_group = ap.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", dest="normalize", action="store_true", help="Enable per-mission z-score")
    norm_group.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable per-mission z-score")
    ap.set_defaults(normalize=DEFAULTS["normalize"])

    ap.add_argument("--keep_cols_always", nargs="*", default=DEFAULTS["keep_cols_always"],
                    help="Columns never outlier-trimmed/normalized")

    args = ap.parse_args()
    main(args)
