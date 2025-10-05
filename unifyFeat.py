# etl_unify_exoplanets.py
# KOI (Kepler), K2, TOI → unified modeling table + quick EDA + feature rationale
# Upgrades:
# - Dynamic transit-depth unit detection (fraction/percent/ppm)
# - Track chosen source column per unified feature (CHOSEN)
# - Convert Kepler BKJD → BJD (epoch_bjd)
# - Cleaner dropped-column categorization
# - Rationale includes chosen sources and transforms
# - Label-leakage guard, optional Parquet output

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

# ---------- INPUT / OUTPUT PATHS ----------
BASE = Path(r"C:\Users\victo\OneDrive\Desktop\Git\NasaExoplanetHacakthon")
KEPLER_PATH = BASE / "cumulative_2025.10.04_12.55.45.csv"
K2_PATH     = BASE / "k2pandc_2025.10.04_12.56.09.csv"
TOI_PATH    = BASE / "TOI_2025.10.04_12.55.52.csv"

OUT_DIR = BASE
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_CSV   = OUT_DIR / "unified_exoplanets.csv"
UNIFIED_PQ    = OUT_DIR / "unified_exoplanets.parquet"
EDA_REPORT    = OUT_DIR / "unified_eda_report.txt"
RATIONALE_MD  = OUT_DIR / "unified_feature_rationale.md"
DROP_KEPLER   = OUT_DIR / "dropped_columns_KEPLER.txt"
DROP_K2       = OUT_DIR / "dropped_columns_K2.txt"
DROP_TESS     = OUT_DIR / "dropped_columns_TESS.txt"

# ---------- LOAD (handle NASA comment headers) ----------
def read_nasa_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, comment="#", low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", sep=None, comment="#", low_memory=False)

kpl = read_nasa_csv(KEPLER_PATH)
k2  = read_nasa_csv(K2_PATH)
toi = read_nasa_csv(TOI_PATH)

# ---------- TRACKING / INSTRUMENTATION ----------
USED = {"KEPLER": set(), "K2": set(), "TESS": set()}                     # chosen source cols used as features
LABEL_USED = {"KEPLER": set(), "K2": set(), "TESS": set()}               # label-source columns (never features)
ALIASES = {"KEPLER": defaultdict(list), "K2": defaultdict(list), "TESS": defaultdict(list)}  # present-not-chosen
TRANSFORMS = []  # (dataset, unified_target, note, source_col)
CHOSEN = {"KEPLER": {}, "K2": {}, "TESS": {}}                            # unified_target -> chosen source col

def _present(df, cols):
    return [c for c in cols if c in df.columns]

def _choose_first(df, cols):
    pres = _present(df, cols)
    return (pres[0], pres) if pres else (None, [])

def num_pick(df, candidates, ds, target):
    chosen, pres = _choose_first(df, candidates)
    if pres:
        ALIASES[ds][target].extend([c for c in pres if c != chosen])
    if chosen:
        USED[ds].add(chosen)
        CHOSEN[ds][target] = chosen
        return pd.to_numeric(df[chosen], errors="coerce")
    return pd.Series([np.nan]*len(df), index=df.index, dtype="float64")

def txt_pick(df, candidates, ds, target):
    chosen, pres = _choose_first(df, candidates)
    if pres:
        ALIASES[ds][target].extend([c for c in pres if c != chosen])
    if chosen:
        USED[ds].add(chosen)
        CHOSEN[ds][target] = chosen
        return df[chosen].astype(str)
    return pd.Series([np.nan]*len(df), index=df.index, dtype="object")

def _detect_depth_unit(series: pd.Series):
    """Heuristic: return 'ppm' | 'percent' | 'fraction' | None"""
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return None
    mx, md = s.max(), s.median()
    if mx <= 1.0 and md <= 0.2:
        return "fraction"
    if mx <= 100.0 and md <= 10.0:
        return "percent"
    return "ppm"

def depth_ppm_pick(df, ds, target, frac_cols=None, percent_cols=None, ppm_cols=None):
    frac_cols = frac_cols or []
    percent_cols = percent_cols or []
    ppm_cols = ppm_cols or []
    for group in (ppm_cols, percent_cols, frac_cols):
        chosen, pres = _choose_first(df, group)
        if chosen:
            if pres:
                ALIASES[ds][target].extend([c for c in pres if c != chosen])
            USED[ds].add(chosen)
            CHOSEN[ds][target] = chosen
            s = pd.to_numeric(df[chosen], errors="coerce")
            unit = _detect_depth_unit(s)
            if unit == "fraction":
                TRANSFORMS.append((ds, target, "fraction → ppm (×1,000,000)", chosen))
                return s * 1_000_000.0
            if unit == "percent":
                TRANSFORMS.append((ds, target, "percent → ppm (×10,000)", chosen))
                return s * 10_000.0
            TRANSFORMS.append((ds, target, "depth already in ppm", chosen))
            return s
    return pd.Series([np.nan]*len(df), index=df.index, dtype="float64")

def build_label(series: pd.Series, mission: str) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    if mission == "KEPLER":
        pos, neg = {"CONFIRMED","CANDIDATE","KOI","PC"}, {"FALSE POSITIVE","FP","REFUTED"}
    elif mission == "K2":
        pos, neg = {"CONFIRMED","CANDIDATE","PC","KP","KNOWN PLANET"}, {"FALSE POSITIVE","FP","REFUTED","FA","FALSE ALARM"}
    else:  # TESS
        pos, neg = {"CP","KP","PC","CONFIRMED","CANDIDATE","KNOWN PLANET"}, {"FP","FA","FALSE POSITIVE","FALSE ALARM","REFUTED"}
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin(pos)] = 1.0
    out[s.isin(neg)] = 0.0
    return out

def label_from(df, cand_cols, ds, mission):
    chosen, pres = _choose_first(df, cand_cols)
    if pres:
        LABEL_USED[ds].update(pres)  # label-source columns are never model features
    if chosen:
        return build_label(df[chosen], mission)
    return pd.Series([np.nan]*len(df), index=df.index, dtype="float")

# ---------- UNIFIED FEATURE DOC (meaning + rationale) ----------
UNIFIED_DOC = {
    "mission":       ("Mission name (KEPLER/K2/TESS)",
                      "Provenance for stratified evaluation and to learn mission-specific systematics."),
    "object_id":     ("Human-readable object identifier",
                      "For traceability/joins; not a model feature."),
    "source_id_raw": ("Source’s original ID string",
                      "Extra identifier for robust de-duplication across catalogs."),
    "label":         ("Supervised target: 1=planet/PC, 0=false positive",
                      "Built from disposition; NEVER used as a feature."),
    "period_d":      ("Orbital period (days)",
                      "Core discriminant; true planets show coherent periodicity."),
    "dur_h":         ("Transit duration (hours)",
                      "Constrained by a/R* and impact parameter; flags improbable geometries."),
    "depth_ppm":     ("Transit depth (ppm)",
                      "Proxy for (Rp/R*)²; normalized across datasets."),
    "impact":        ("Impact parameter b",
                      "High b + deep/sharp events can indicate grazing EBs."),
    "ror":           ("Rp/R* (radius ratio)",
                      "Geometric parameter; complements depth with limb-darkening."),
    "prad_re":       ("Planet radius (Earth radii)",
                      "Physical size; helps reject astrophysical FPs/dilution."),
    "a_au":          ("Semi-major axis (AU)",
                      "With stellar params informs irradiation & transit probability."),
    "a_over_rstar":  ("Scaled separation a/R*",
                      "Key to transit geometry and duration expectations."),
    "insol_earth":   ("Incident flux (Earth=1)",
                      "Irradiation context; separates EB vs planet edge cases."),
    "teq_k":         ("Equilibrium temperature (K)",
                      "Irradiation proxy; correlations with depth/period can flag FPs."),
    "teff_k":        ("Stellar effective temperature (K)",
                      "Host context for priors (e.g., radius expectations)."),
    "logg_cgs":      ("Stellar surface gravity (cgs)",
                      "Dwarfs vs giants; many FPs around giants due to dilution."),
    "radius_rsun":   ("Stellar radius (R☉)",
                      "Needed to convert depth to radius; flags giant hosts."),
    "mass_msun":     ("Stellar mass (M☉)",
                      "Sets dynamics and a/R* expectations."),
    "feh_dex":       ("Stellar metallicity [Fe/H]",
                      "Weak prior on occurrence; sometimes predictive for FPs."),
    "mes":           ("Multiple Event Statistic",
                      "Detection statistic; higher → more reliable."),
    "snr":           ("Model/fit signal-to-noise",
                      "General detection quality metric."),
    "fpflag_nt":     ("Not transit-like flag",
                      "High-signal FP indicator."),
    "fpflag_ss":     ("Significant secondary flag",
                      "EB-like secondary eclipse signature."),
    "fpflag_co":     ("Centroid offset flag",
                      "Likely blend/background EB."),
    "fpflag_ec":     ("Ephemeris match/confusion flag",
                      "Likely contamination from known variables."),
    "epoch_bjd_raw": ("Transit mid-time (as provided)",
                      "Provenance; Kepler is BKJD."),
    "epoch_bjd":     ("Transit mid-time (BJD)",
                      "Unified BJD epoch for all missions."),
}

# ---------- MAPPERS (→ golden schema) ----------
def map_kepler(df):
    ds = "KEPLER"
    out = pd.DataFrame(index=df.index)
    out["mission"]        = "KEPLER"
    out["object_id"]      = txt_pick(df, ["kepid","kepoi_name","kepler_name"], ds, "object_id")
    out["source_id_raw"]  = txt_pick(df, ["kepid","kepoi_name","kepler_name"], ds, "source_id_raw")
    out["label"]          = label_from(df, ["koi_disposition","koi_pdisposition"], ds, "KEPLER")
    out["period_d"]       = num_pick(df, ["koi_period"], ds, "period_d")
    out["dur_h"]          = num_pick(df, ["koi_duration"], ds, "dur_h")
    out["depth_ppm"]      = depth_ppm_pick(df, ds, "depth_ppm", ppm_cols=["koi_depth"])
    out["impact"]         = num_pick(df, ["koi_impact"], ds, "impact")
    out["ror"]            = num_pick(df, ["koi_ror"], ds, "ror")
    out["prad_re"]        = num_pick(df, ["koi_prad"], ds, "prad_re")
    out["a_au"]           = num_pick(df, ["koi_sma"], ds, "a_au")
    out["a_over_rstar"]   = num_pick(df, ["koi_dor"], ds, "a_over_rstar")
    out["insol_earth"]    = num_pick(df, ["koi_insol"], ds, "insol_earth")
    out["teq_k"]          = num_pick(df, ["koi_teq"], ds, "teq_k")
    out["teff_k"]         = num_pick(df, ["koi_steff"], ds, "teff_k")
    out["logg_cgs"]       = num_pick(df, ["koi_slogg"], ds, "logg_cgs")
    out["radius_rsun"]    = num_pick(df, ["koi_srad"], ds, "radius_rsun")
    out["mass_msun"]      = num_pick(df, ["koi_smass"], ds, "mass_msun")
    out["feh_dex"]        = num_pick(df, ["koi_smet"], ds, "feh_dex")
    out["mes"]            = num_pick(df, ["koi_max_mult_ev"], ds, "mes")
    out["snr"]            = num_pick(df, ["koi_model_snr"], ds, "snr")
    out["fpflag_nt"]      = num_pick(df, ["koi_fpflag_nt"], ds, "fpflag_nt")
    out["fpflag_ss"]      = num_pick(df, ["koi_fpflag_ss"], ds, "fpflag_ss")
    out["fpflag_co"]      = num_pick(df, ["koi_fpflag_co"], ds, "fpflag_co")
    out["fpflag_ec"]      = num_pick(df, ["koi_fpflag_ec"], ds, "fpflag_ec")
    out["epoch_bjd_raw"]  = num_pick(df, ["koi_time0bk"], ds, "epoch_bjd_raw")  # BKJD
    return out

def map_k2(df):
    ds = "K2"
    out = pd.DataFrame(index=df.index)
    out["mission"]        = "K2"
    out["object_id"]      = txt_pick(df, ["pl_name","hostname"], ds, "object_id")
    out["source_id_raw"]  = txt_pick(df, ["pl_name","hostname"], ds, "source_id_raw")
    out["label"]          = label_from(df, ["disposition"], ds, "K2")
    out["period_d"]       = num_pick(df, ["pl_orbper"], ds, "period_d")
    out["dur_h"]          = num_pick(df, ["pl_trandur","pl_trandurh"], ds, "dur_h")
    out["depth_ppm"]      = depth_ppm_pick(df, ds, "depth_ppm", frac_cols=["pl_trandep"], ppm_cols=["pl_trandep"])
    out["impact"]         = num_pick(df, ["pl_imppar"], ds, "impact")
    out["ror"]            = num_pick(df, ["pl_ratror"], ds, "ror")
    out["prad_re"]        = num_pick(df, ["pl_rade"], ds, "prad_re")
    out["a_au"]           = num_pick(df, ["pl_orbsmax"], ds, "a_au")
    out["a_over_rstar"]   = num_pick(df, ["pl_ratdor"], ds, "a_over_rstar")
    out["insol_earth"]    = num_pick(df, ["pl_insol"], ds, "insol_earth")
    out["teq_k"]          = num_pick(df, ["pl_eqt"], ds, "teq_k")
    out["teff_k"]         = num_pick(df, ["st_teff"], ds, "teff_k")
    out["logg_cgs"]       = num_pick(df, ["st_logg"], ds, "logg_cgs")
    out["radius_rsun"]    = num_pick(df, ["st_rad"], ds, "radius_rsun")
    out["mass_msun"]      = num_pick(df, ["st_mass"], ds, "mass_msun")
    out["feh_dex"]        = num_pick(df, ["st_met"], ds, "feh_dex")
    out["mes"]            = num_pick(df, ["mes"], ds, "mes")
    out["snr"]            = num_pick(df, ["snr"], ds, "snr")
    out["fpflag_nt"]      = num_pick(df, ["fpflag_nt"], ds, "fpflag_nt")
    out["fpflag_ss"]      = num_pick(df, ["fpflag_ss"], ds, "fpflag_ss")
    out["fpflag_co"]      = num_pick(df, ["fpflag_co"], ds, "fpflag_co")
    out["fpflag_ec"]      = num_pick(df, ["fpflag_ec"], ds, "fpflag_ec")
    out["epoch_bjd_raw"]  = num_pick(df, ["pl_tranmid"], ds, "epoch_bjd_raw")
    return out

def map_toi(df):
    ds = "TESS"
    out = pd.DataFrame(index=df.index)
    out["mission"]        = "TESS"
    out["object_id"]      = txt_pick(df, ["toi","tid"], ds, "object_id")
    out["source_id_raw"]  = txt_pick(df, ["toi","tid"], ds, "source_id_raw")
    out["label"]          = label_from(df, ["tfopwg_disp"], ds, "TESS")
    out["period_d"]       = num_pick(df, ["pl_orbper"], ds, "period_d")
    out["dur_h"]          = num_pick(df, ["pl_trandurh","pl_trandur"], ds, "dur_h")
    out["depth_ppm"]      = depth_ppm_pick(df, ds, "depth_ppm", ppm_cols=["pl_trandep"])
    out["impact"]         = num_pick(df, ["pl_imppar"], ds, "impact")
    out["ror"]            = num_pick(df, ["pl_ratror","pl_ratdor"], ds, "ror")
    out["prad_re"]        = num_pick(df, ["pl_rade"], ds, "prad_re")
    out["a_au"]           = num_pick(df, ["pl_orbsmax"], ds, "a_au")
    out["a_over_rstar"]   = num_pick(df, ["pl_ratdor"], ds, "a_over_rstar")
    out["insol_earth"]    = num_pick(df, ["pl_insol"], ds, "insol_earth")
    out["teq_k"]          = num_pick(df, ["pl_eqt"], ds, "teq_k")
    out["teff_k"]         = num_pick(df, ["st_teff"], ds, "teff_k")
    out["logg_cgs"]       = num_pick(df, ["st_logg"], ds, "logg_cgs")
    out["radius_rsun"]    = num_pick(df, ["st_rad"], ds, "radius_rsun")
    out["mass_msun"]      = num_pick(df, ["st_mass"], ds, "mass_msun")
    out["feh_dex"]        = num_pick(df, ["st_met"], ds, "feh_dex")
    out["mes"]            = num_pick(df, ["mes"], ds, "mes")
    out["snr"]            = num_pick(df, ["snr"], ds, "snr")
    out["fpflag_nt"]      = num_pick(df, ["fpflag_nt"], ds, "fpflag_nt")
    out["fpflag_ss"]      = num_pick(df, ["fpflag_ss"], ds, "fpflag_ss")
    out["fpflag_co"]      = num_pick(df, ["fpflag_co"], ds, "fpflag_co")
    out["fpflag_ec"]      = num_pick(df, ["fpflag_ec"], ds, "fpflag_ec")
    out["epoch_bjd_raw"]  = num_pick(df, ["pl_tranmid"], ds, "epoch_bjd_raw")
    return out

# ---------- UNIFY ----------
kpl_u, k2_u, toi_u = map_kepler(kpl), map_k2(k2), map_toi(toi)
unified = pd.concat([kpl_u, k2_u, toi_u], ignore_index=True)

# Add unified epoch in BJD (convert Kepler BKJD → BJD)
unified["epoch_bjd"] = np.nan
kepler_mask = unified["mission"] == "KEPLER"
unified.loc[kepler_mask, "epoch_bjd"] = unified.loc[kepler_mask, "epoch_bjd_raw"] + 2454833.0
non_kepler = ~kepler_mask
unified.loc[non_kepler, "epoch_bjd"] = unified.loc[non_kepler, "epoch_bjd_raw"]

# Training set: drop unknown labels; cast to int
unified_train = unified.dropna(subset=["label"]).copy()
unified_train["label"] = unified_train["label"].astype(int)

# Guard against label leakage (no disposition/tfopwg fields in features)
leak_cols = [c for c in unified_train.columns if ("disposition" in c.lower()) or ("tfopwg" in c.lower())]
if leak_cols:
    raise RuntimeError(f"Label leakage risk: {leak_cols} present in unified features.")

# Deduplicate (optional but recommended)
unified_train = unified_train.drop_duplicates(subset=["mission","object_id"])

# ---------- SAVE MAIN DATA ----------
unified_train.to_csv(UNIFIED_CSV, index=False, encoding="utf-8")
try:
    unified_train.to_parquet(UNIFIED_PQ, index=False)
except Exception:
    pass  # parquet optional

# ---------- QUICK EDA ----------
mission_counts = unified_train["mission"].value_counts()
class_counts   = unified_train["label"].value_counts()
miss_rate      = unified_train.isna().mean().sort_values(ascending=False)

num_df = unified_train.select_dtypes(include=[np.number])
pairs = []
if num_df.shape[1] >= 2:
    corr = num_df.corr().abs()
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if np.isfinite(val) and val >= 0.90:
                pairs.append((cols[i], cols[j], float(val)))
    pairs.sort(key=lambda x: -x[2])

# ---------- WHAT WE KEPT vs DROPPED (by dataset) ----------
def categorize_column(name: str) -> str:
    n = name.lower()
    # uncertainty / limits
    if re.search(r'(?:^|[_])(err|err1|err2|unc|ulim|llim|sigma|errp|errm)(?:$|[_])', n):
        return "uncertainty/limits (excluded for v1)"
    # coordinates & astrometry (avoid false positives)
    coord_keys = ("ra", "dec", "glat", "glon", "elon", "elat", "pmra", "pmdec", "par", "parallax", "dist")
    if n in {"ra","dec"} or n.startswith(coord_keys):
        return "sky coordinates/astrometry"
    # identifiers / metadata
    meta_keys = ("_id","id_","name","tic","kic","kepoi","kepid","hostname","url","source",
                 "facility","ref","bib","disc","fov","sector","camera","ccd")
    if any(k in n for k in meta_keys):
        return "metadata/IDs/provenance"
    # timestamps (non-ephemeris)
    if any(k in n for k in ["date","mjd","jd","bjd","time"]) and "tran" not in n:
        return "timestamps/metadata"
    # label/vetting fields (not counting fpflag_* which are features)
    if ("disposition" in n) or ("tfopwg" in n) or ("vet" in n and "fpflag" not in n):
        return "label/vetting-only"
    # duplicates/alternates normalized away
    if any(k in n for k in ["depth","trandep","tran","ratror","ratdor"]):
        return "redundant/alternate encoding"
    return "other/unmapped for v1"

def summarize_drops(df, ds, drop_path):
    all_cols = set(df.columns)
    used = set(USED[ds])
    label_only = set(LABEL_USED[ds])
    alias_present = set()
    for v in ALIASES[ds].values():
        alias_present.update(v)
    dropped = sorted(all_cols - used - label_only - alias_present)
    cats = defaultdict(list)
    for c in dropped:
        cats[categorize_column(c)].append(c)
    lines = [f"{ds}: dropped columns (not used as features)\n"]
    for cat, cols in sorted(cats.items(), key=lambda x: x[0]):
        lines.append(f"\n[{cat}] ({len(cols)}):")
        for c in sorted(cols):
            lines.append(f"  - {c}")
    lines.append("\nPresent but not chosen (redundant aliases):")
    alias_lines = []
    for tgt, alist in sorted(ALIASES[ds].items()):
        if alist:
            alias_lines.append(f"  - {tgt}: " + ", ".join(sorted(set(alist))))
    lines.extend(alias_lines or ["  (none)"])
    drop_path.write_text("\n".join(lines), encoding="utf-8")

summarize_drops(kpl, "KEPLER", DROP_KEPLER)
summarize_drops(k2, "K2", DROP_K2)
summarize_drops(toi, "TESS", DROP_TESS)

# ---------- RATIONALE MARKDOWN ----------
def rationale_markdown() -> str:
    lines = ["# Unified Feature Rationale\n"]
    lines.append("This document explains **which features were kept**, **why**, how they were **normalized**, and the **source columns** per mission.\n")
    lines.append("## Kept (Unified) Features\n")
    for feat, (desc, why) in UNIFIED_DOC.items():
        lines.append(f"### `{feat}`")
        lines.append(f"- **What it is:** {desc}")
        lines.append(f"- **Why keep it:** {why}")
        # per-mission chosen source
        lines.append("- **Chosen source columns:**")
        for ds in ["KEPLER","K2","TESS"]:
            src = CHOSEN[ds].get(feat)
            if src:
                lines.append(f"  - {ds}: `{src}`")
        # transforms (unit conversions etc.)
        tnotes = [f"- {d}/{t}: {note} (source `{src}`)" for (d,t,note,src) in TRANSFORMS if t == feat]
        if tnotes:
            lines.append("- **Normalization/transform:**")
            lines.extend(tnotes)
        # aliases present but not chosen
        alias_notes = []
        for ds in ["KEPLER","K2","TESS"]:
            al = sorted(set(ALIASES[ds].get(feat, [])))
            if al:
                alias_notes.append(f"- {ds} aliases not chosen: " + ", ".join(al))
        if alias_notes:
            lines.append("- **Redundant aliases encountered:**")
            lines.extend(alias_notes)
        lines.append("")  # spacer
    # Label-only sources
    lines.append("## Label-Only Sources (not used as features)\n")
    for ds in ["KEPLER","K2","TESS"]:
        if LABEL_USED[ds]:
            lines.append(f"- **{ds}**: " + ", ".join(sorted(LABEL_USED[ds])))
    # High-level drop rationale
    lines.append("\n## What We Dropped and Why (high level)\n")
    lines.append("- **Metadata/IDs/Provenance:** kept minimal IDs for joining; other IDs/URLs/facility fields are non-predictive.")
    lines.append("- **Uncertainties/Limits:** excluded in v1; consider later for calibrated models.")
    lines.append("- **Coordinates/Astrometry:** sky position not causally informative for planet vs FP and complicates cross-mission mixing.")
    lines.append("- **Redundant encodings:** normalized multiple encodings (fraction/percent/ppm) into single units.")
    lines.append("- **Vetting/label fields:** used only to build the target (avoid leakage).")
    return "\n".join(lines)

RATIONALE_MD.write_text(rationale_markdown(), encoding="utf-8")

# ---------- BUILD EDA REPORT ----------
eda_lines = []
eda_lines.append("Mission counts:\n" + str(mission_counts))
eda_lines.append("\nClass counts (1=planet/PC, 0=FP):\n" + str(class_counts))
eda_lines.append("\nTop 20 columns by missingness:\n" + str(miss_rate.head(20)))
eda_lines.append(f"\nHighly correlated pairs (|r|>=0.90): {len(pairs)} total. Top 30:")
for c1, c2, r in pairs[:30]:
    eda_lines.append(f"{c1} ↔ {c2}: r={r:.3f}")
# Per-mission class balance
eda_lines.append("\nClass balance by mission:")
eda_lines.append(str(unified_train.groupby("mission")["label"].value_counts()))
# Concise feature rationale summary
eda_lines.append("\n---\nFEATURE SELECTION & RATIONALE (summary)\n")
eda_lines.append("Kept unified features:\n" + ", ".join(list(UNIFIED_DOC.keys())))
for ds, df in [("KEPLER", kpl), ("K2", k2), ("TESS", toi)]:
    total = len(df.columns)
    kept = len(USED[ds])
    label_only = len(LABEL_USED[ds])
    alias_ct = sum(len(v) for v in ALIASES[ds].values())
    eda_lines.append(f"{ds}: used {kept} source cols for features, {label_only} for label-only, {alias_ct} aliases, out of {total} total columns.")
eda_lines.append("\nSee detailed explanations in: " + str(RATIONALE_MD.name))
eda_lines.append("Full dropped-column lists per dataset are in the corresponding 'dropped_columns_*.txt' files.")

EDA_REPORT.write_text("\n".join(eda_lines), encoding="utf-8")

print("Saved CSV:         ", UNIFIED_CSV.resolve())
print("Saved Parquet:     ", UNIFIED_PQ.resolve())
print("Saved EDA:         ", EDA_REPORT.resolve())
print("Saved Rationale:   ", RATIONALE_MD.resolve())
print("Dropped (Kepler):  ", DROP_KEPLER.resolve())
print("Dropped (K2):      ", DROP_K2.resolve())
print("Dropped (TESS):    ", DROP_TESS.resolve())

