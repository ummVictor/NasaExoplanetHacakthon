# etl_unify_exoplanets.py
# KOI (Kepler), K2, TOI → unified modeling table + quick EDA + feature rationale

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

# ---------- MAPPING INSTRUMENTATION (track what we used and why) ----------
USED = {"KEPLER": set(), "K2": set(), "TESS": set()}                  # chosen source cols used as features
LABEL_USED = {"KEPLER": set(), "K2": set(), "TESS": set()}            # source cols used to build labels only
ALIASES = {"KEPLER": defaultdict(list), "K2": defaultdict(list), "TESS": defaultdict(list)}  # present-but-not-chosen
TRANSFORMS = []  # records unit conversions, etc. (dataset, target, note, source)

def _present(df, cols):
    pres = [c for c in cols if c in df.columns]
    return pres

def _choose_first(df, cols):
    pres = _present(df, cols)
    return (pres[0], pres) if pres else (None, [])

def num_pick(df, candidates, ds, target):
    chosen, pres = _choose_first(df, candidates)
    if pres:
        ALIASES[ds][target].extend([c for c in pres if c != chosen])
    if chosen:
        USED[ds].add(chosen)
        return pd.to_numeric(df[chosen], errors="coerce")
    return pd.Series([np.nan]*len(df), index=df.index, dtype="float64")

def txt_pick(df, candidates, ds, target):
    chosen, pres = _choose_first(df, candidates)
    if pres:
        ALIASES[ds][target].extend([c for c in pres if c != chosen])
    if chosen:
        USED[ds].add(chosen)
        return df[chosen].astype(str)
    return pd.Series([np.nan]*len(df), index=df.index, dtype="object")

def depth_ppm_pick(df, ds, target, frac_cols=None, percent_cols=None, ppm_cols=None):
    frac_cols = frac_cols or []
    percent_cols = percent_cols or []
    ppm_cols = ppm_cols or []

    # ppm direct
    chosen, pres = _choose_first(df, ppm_cols)
    if chosen:
        USED[ds].add(chosen)
        if pres:
            ALIASES[ds][target].extend([c for c in pres if c != chosen])
        TRANSFORMS.append((ds, target, "depth already in ppm", chosen))
        return pd.to_numeric(df[chosen], errors="coerce")

    # percent → ppm
    chosen, pres = _choose_first(df, percent_cols)
    if chosen:
        USED[ds].add(chosen)
        if pres:
            ALIASES[ds][target].extend([c for c in pres if c != chosen])
        TRANSFORMS.append((ds, target, "percent → ppm (×10,000)", chosen))
        return pd.to_numeric(df[chosen], errors="coerce") * 10000.0

    # fraction → ppm
    chosen, pres = _choose_first(df, frac_cols)
    if chosen:
        USED[ds].add(chosen)
        if pres:
            ALIASES[ds][target].extend([c for c in pres if c != chosen])
        TRANSFORMS.append((ds, target, "fraction → ppm (×1,000,000)", chosen))
        return pd.to_numeric(df[chosen], errors="coerce") * 1_000_000.0

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
        # Label source columns are never model features
        LABEL_USED[ds].update(pres)
    if chosen:
        return build_label(df[chosen], mission)
    return pd.Series([np.nan]*len(df), index=df.index, dtype="float")

# ---------- UNIFIED FEATURE DOC (meaning + rationale) ----------
UNIFIED_DOC = {
    "mission":       ("Mission name (KEPLER/K2/TESS)",
                      "Keeps provenance so models can learn mission-specific systematics or allow stratified evaluation."),
    "object_id":     ("Human-readable object identifier",
                      "For traceability and merging back to source; not used by models but kept in table."),
    "source_id_raw": ("Source’s original ID string",
                      "A second identifier for robust de-duplication across catalogs."),
    "label":         ("Supervised target: 1=planet/PC, 0=false positive",
                      "Built from mission disposition fields; NEVER used as a feature."),
    "period_d":      ("Orbital period (days)",
                      "Core discriminant; real planets exhibit coherent periodicity."),
    "dur_h":         ("Transit duration (hours)",
                      "Constrained by a/R* and impact parameter; helps reject improbable geometries."),
    "depth_ppm":     ("Transit depth (ppm)",
                      "Proxy for (Rp/R*)²; normalized across datasets to the same unit."),
    "impact":        ("Impact parameter b",
                      "High b with deep/sharp events can indicate grazing/EBs; useful for FP control."),
    "ror":           ("Rp/R* (radius ratio)",
                      "Direct geometric parameter; complements depth with limb-darkening effects."),
    "prad_re":       ("Planet radius (Earth radii)",
                      "Physical size to catch astrophysical false positives or stellar blends."),
    "a_au":          ("Semi-major axis (AU)",
                      "Along with stellar params informs irradiation and transit probability."),
    "a_over_rstar":  ("Scaled separation a/R*",
                      "Key to transit shape physics and duration expectations."),
    "insol_earth":   ("Incident flux (Earth=1)",
                      "Derived context of irradiation; aids separating EBs vs planets in edge cases."),
    "teq_k":         ("Equilibrium temperature (K)",
                      "Another irradiation proxy; correlations with depth/period can flag FPs."),
    "teff_k":        ("Stellar effective temperature (K)",
                      "Host-star context for priors (e.g., radius expectations)."),
    "logg_cgs":      ("Stellar surface gravity (cgs)",
                      "Dwarfs vs giants: many FPs around giants due to diluted EBs."),
    "radius_rsun":   ("Stellar radius (R☉)",
                      "Needed to convert depth to physical radius; flags giant hosts."),
    "mass_msun":     ("Stellar mass (M☉)",
                      "Sets orbital dynamics and a/R* expectations."),
    "feh_dex":       ("Stellar metallicity [Fe/H]",
                      "Weak prior on occurrence; sometimes predictive for FPs."),
    "mes":           ("Multiple Event Statistic",
                      "Kepler/TESS detection statistic; higher tends to be more reliable."),
    "snr":           ("Model/fit signal-to-noise",
                      "General detection quality metric across sets."),
    "fpflag_nt":     ("Not transit-like flag",
                      "Pre-computed vetting flags are high-signal FP indicators."),
    "fpflag_ss":     ("Significant secondary flag",
                      "EB-like secondary eclipse signature."),
    "fpflag_co":     ("Centroid offset flag",
                      "Indicates blend/background EB."),
    "fpflag_ec":     ("Ephemeris match/confusion flag",
                      "Indicates likely contamination from known variables."),
    "epoch_bjd_raw": ("Transit mid-time (BJD-like)",
                      "Useful for phase-folding and cross-matching ephemerides."),
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

# Training set: drop unknown labels; cast to int
unified_train = unified.dropna(subset=["label"]).copy()
unified_train["label"] = unified_train["label"].astype(int)

# ---------- SAVE MAIN CSV ----------
unified_train.to_csv(UNIFIED_CSV, index=False, encoding="utf-8")

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
    if re.search(r'(?:^|[_])(?:err|err1|err2|unc|ulim|llim|sigma|errp|errm)(?:$|[_])', n):
        return "uncertainty/limits (excluded for baseline; optionally add later)"
    # coordinates & astrometry (avoid false matches with 'rade')
    if n in {"ra","dec"} or n.startswith(("ra_", "dec_", "glat", "glon", "elon", "elat", "pmra","pmdec","par","dist")):
        return "sky coordinates/astrometry (not directly predictive; mission-agnostic)"
    # identifiers / metadata
    if any(key in n for key in ["_id","id_", "name","tic","kic","kepoi","kepid","hostname","url","source","facility","ref","bib","disc","fov", "sector","camera","ccd"]):
        return "metadata/IDs/provenance (not model features)"
    # time stamps (non-ephemeris)
    if any(key in n for key in ["date","jd","mjd","bjd","time"]) and "tran" not in n:
        return "timestamps/metadata (not used as features)"
    # label-ish / vetting-only fields
    if "disposition" in n or "tfopwg" in n or "vet" in n and "fpflag" not in n:
        return "label/vetting-only (used to build target; not a feature)"
    # duplicates / alternate encodings that we normalized
    if "depth" in n or "tran" in n or "ratror" in n or "ratdor" in n:
        return "redundant/alternate encoding (normalized into unified columns)"
    # default
    return "other/unmapped for v1 (consider later)"

def summarize_drops(df, ds, drop_path):
    # everything present
    all_cols = set(df.columns)
    used = set(USED[ds])
    label_only = set(LABEL_USED[ds])
    alias_present = set()
    for v in ALIASES[ds].values():
        alias_present.update(v)
    # Aliases are reported separately; treat as 'present but not chosen'
    dropped = sorted(all_cols - used - label_only - alias_present)
    cats = defaultdict(list)
    for c in dropped:
        cats[categorize_column(c)].append(c)

    # write report
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
    if alias_lines:
        lines.extend(alias_lines)
    else:
        lines.append("  (none)")
    drop_path.write_text("\n".join(lines), encoding="utf-8")

# build dropped summaries per dataset
summarize_drops(kpl, "KEPLER", DROP_KEPLER)
summarize_drops(k2, "K2", DROP_K2)
summarize_drops(toi, "TESS", DROP_TESS)

# ---------- RATIONALE MARKDOWN ----------
def rationale_markdown() -> str:
    lines = ["# Unified Feature Rationale\n"]
    # overview
    lines.append("This document explains **which features were kept**, **why they were selected**, how they were **normalized**, and **which source columns** they came from for each mission.\n")
    lines.append("## Kept (Unified) Features\n")
    for feat, (desc, why) in UNIFIED_DOC.items():
        lines.append(f"### `{feat}`")
        lines.append(f"- **What it is:** {desc}")
        lines.append(f"- **Why keep it:** {why}")
        # sources used per mission
        src_by_ds = []
        for ds in ["KEPLER","K2","TESS"]:
            used_here = [c for c in USED[ds] if c in _present({"KEPLER":kpl,"K2":k2,"TESS":toi}[ds], [c])]
            # Collect the specific chosen source for this target from ALIASES+USED signal
            # We approximate: chosen = first of (ALIASES+tgt) union USED intersect candidates recorded during mapping.
            # For readability, show both chosen and present-alternatives if recorded.
        # Add a short transform summary:
        tnotes = [f"- {d}/{t}: {note} (source `{src}`)" for (d,t,note,src) in TRANSFORMS if t == feat]
        if tnotes:
            lines.append("- **Normalization/transform:**")
            lines.extend(tnotes)
        # list present-but-not-chosen aliases
        alias_notes = []
        for ds in ["KEPLER","K2","TESS"]:
            al = sorted(set(ALIASES[ds].get(feat, [])))
            if al:
                alias_notes.append(f"- {ds} aliases not chosen: " + ", ".join(al))
        if alias_notes:
            lines.append("- **Redundant aliases encountered:**")
            lines.extend(alias_notes)
        lines.append("")  # blank line

    # Label-only fields
    lines.append("## Label-Only Sources (not used as features)\n")
    for ds in ["KEPLER","K2","TESS"]:
        if LABEL_USED[ds]:
            lines.append(f"- **{ds}**: " + ", ".join(sorted(LABEL_USED[ds])))
    lines.append("")

    # High-level drop rationale
    lines.append("## What We Dropped and Why (high level)\n")
    lines.append("- **Metadata/IDs/Provenance:** kept only a minimal set of IDs for joining; other IDs/URLs/facility fields are non-predictive.")
    lines.append("- **Uncertainties / limits:** excluded in v1 to keep the baseline deterministic; can be included later for calibrated models.")
    lines.append("- **Coordinates/Astrometry:** sky position does not causally inform planet vs. FP classification and complicates cross-mission mixing.")
    lines.append("- **Redundant encodings:** where multiple columns encode the same concept (e.g., fraction vs. ppm depth), we normalized to a single unit.")
    lines.append("- **Vetting/label fields:** anything that encodes the final human disposition is used only to build the label (to avoid leakage).")
    return "\n".join(lines)

RATIONALE_MD.write_text(rationale_markdown(), encoding="utf-8")

# ---------- BUILD EDA REPORT (and embed a concise rationale summary) ----------
eda_lines = []
eda_lines.append("Mission counts:\n" + str(mission_counts))
eda_lines.append("\nClass counts (1=planet/PC, 0=FP):\n" + str(class_counts))
eda_lines.append("\nTop 20 columns by missingness:\n" + str(miss_rate.head(20)))
eda_lines.append(f"\nHighly correlated pairs (|r|>=0.90): {len(pairs)} total. Top 30:")
for c1, c2, r in pairs[:30]:
    eda_lines.append(f"{c1} ↔ {c2}: r={r:.3f}")

# concise rationale summary
def short_rationale_summary():
    lines = []
    lines.append("\n---\nFEATURE SELECTION & RATIONALE (summary)\n")
    lines.append("Kept unified features:")
    lines.append(", ".join(list(UNIFIED_DOC.keys())))
    # counts by dataset
    for ds, df in [("KEPLER", kpl), ("K2", k2), ("TESS", toi)]:
        total = len(df.columns)
        kept = len(USED[ds])
        label_only = len(LABEL_USED[ds])
        alias_ct = sum(len(v) for v in ALIASES[ds].values())
        lines.append(f"{ds}: used {kept} source cols for features, {label_only} for label-only, {alias_ct} present-but-not-chosen aliases, out of {total} total columns.")
    lines.append("\nSee detailed explanations in: " + str(RATIONALE_MD.name))
    lines.append("Full dropped-column lists per dataset are in the corresponding 'dropped_columns_*.txt' files.")
    return "\n".join(lines)

eda_lines.append(short_rationale_summary())

# Always write the EDA report in UTF-8
EDA_REPORT.write_text("\n".join(eda_lines), encoding="utf-8")

print("Saved CSV:         ", UNIFIED_CSV.resolve())
print("Saved EDA:         ", EDA_REPORT.resolve())
print("Saved Rationale:   ", RATIONALE_MD.resolve())
print("Dropped (Kepler):  ", DROP_KEPLER.resolve())
print("Dropped (K2):      ", DROP_K2.resolve())
print("Dropped (TESS):    ", DROP_TESS.resolve())
