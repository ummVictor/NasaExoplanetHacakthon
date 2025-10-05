# Unified Feature Rationale

This document explains **which features were kept**, **why they were selected**, how they were **normalized**, and **which source columns** they came from for each mission.

## Kept (Unified) Features

### `mission`
- **What it is:** Mission name (KEPLER/K2/TESS)
- **Why keep it:** Keeps provenance so models can learn mission-specific systematics or allow stratified evaluation.

### `object_id`
- **What it is:** Human-readable object identifier
- **Why keep it:** For traceability and merging back to source; not used by models but kept in table.
- **Redundant aliases encountered:**
- KEPLER aliases not chosen: kepler_name, kepoi_name
- K2 aliases not chosen: hostname
- TESS aliases not chosen: tid

### `source_id_raw`
- **What it is:** Source’s original ID string
- **Why keep it:** A second identifier for robust de-duplication across catalogs.
- **Redundant aliases encountered:**
- KEPLER aliases not chosen: kepler_name, kepoi_name
- K2 aliases not chosen: hostname
- TESS aliases not chosen: tid

### `label`
- **What it is:** Supervised target: 1=planet/PC, 0=false positive
- **Why keep it:** Built from mission disposition fields; NEVER used as a feature.

### `period_d`
- **What it is:** Orbital period (days)
- **Why keep it:** Core discriminant; real planets exhibit coherent periodicity.

### `dur_h`
- **What it is:** Transit duration (hours)
- **Why keep it:** Constrained by a/R* and impact parameter; helps reject improbable geometries.

### `depth_ppm`
- **What it is:** Transit depth (ppm)
- **Why keep it:** Proxy for (Rp/R*)²; normalized across datasets to the same unit.
- **Normalization/transform:**
- KEPLER/depth_ppm: depth already in ppm (source `koi_depth`)
- TESS/depth_ppm: depth already in ppm (source `pl_trandep`)

### `impact`
- **What it is:** Impact parameter b
- **Why keep it:** High b with deep/sharp events can indicate grazing/EBs; useful for FP control.

### `ror`
- **What it is:** Rp/R* (radius ratio)
- **Why keep it:** Direct geometric parameter; complements depth with limb-darkening effects.

### `prad_re`
- **What it is:** Planet radius (Earth radii)
- **Why keep it:** Physical size to catch astrophysical false positives or stellar blends.

### `a_au`
- **What it is:** Semi-major axis (AU)
- **Why keep it:** Along with stellar params informs irradiation and transit probability.

### `a_over_rstar`
- **What it is:** Scaled separation a/R*
- **Why keep it:** Key to transit shape physics and duration expectations.

### `insol_earth`
- **What it is:** Incident flux (Earth=1)
- **Why keep it:** Derived context of irradiation; aids separating EBs vs planets in edge cases.

### `teq_k`
- **What it is:** Equilibrium temperature (K)
- **Why keep it:** Another irradiation proxy; correlations with depth/period can flag FPs.

### `teff_k`
- **What it is:** Stellar effective temperature (K)
- **Why keep it:** Host-star context for priors (e.g., radius expectations).

### `logg_cgs`
- **What it is:** Stellar surface gravity (cgs)
- **Why keep it:** Dwarfs vs giants: many FPs around giants due to diluted EBs.

### `radius_rsun`
- **What it is:** Stellar radius (R☉)
- **Why keep it:** Needed to convert depth to physical radius; flags giant hosts.

### `mass_msun`
- **What it is:** Stellar mass (M☉)
- **Why keep it:** Sets orbital dynamics and a/R* expectations.

### `feh_dex`
- **What it is:** Stellar metallicity [Fe/H]
- **Why keep it:** Weak prior on occurrence; sometimes predictive for FPs.

### `mes`
- **What it is:** Multiple Event Statistic
- **Why keep it:** Kepler/TESS detection statistic; higher tends to be more reliable.

### `snr`
- **What it is:** Model/fit signal-to-noise
- **Why keep it:** General detection quality metric across sets.

### `fpflag_nt`
- **What it is:** Not transit-like flag
- **Why keep it:** Pre-computed vetting flags are high-signal FP indicators.

### `fpflag_ss`
- **What it is:** Significant secondary flag
- **Why keep it:** EB-like secondary eclipse signature.

### `fpflag_co`
- **What it is:** Centroid offset flag
- **Why keep it:** Indicates blend/background EB.

### `fpflag_ec`
- **What it is:** Ephemeris match/confusion flag
- **Why keep it:** Indicates likely contamination from known variables.

### `epoch_bjd_raw`
- **What it is:** Transit mid-time (BJD-like)
- **Why keep it:** Useful for phase-folding and cross-matching ephemerides.

## Label-Only Sources (not used as features)

- **KEPLER**: koi_disposition, koi_pdisposition
- **K2**: disposition
- **TESS**: tfopwg_disp

## What We Dropped and Why (high level)

- **Metadata/IDs/Provenance:** kept only a minimal set of IDs for joining; other IDs/URLs/facility fields are non-predictive.
- **Uncertainties / limits:** excluded in v1 to keep the baseline deterministic; can be included later for calibrated models.
- **Coordinates/Astrometry:** sky position does not causally inform planet vs. FP classification and complicates cross-mission mixing.
- **Redundant encodings:** where multiple columns encode the same concept (e.g., fraction vs. ppm depth), we normalized to a single unit.
- **Vetting/label fields:** anything that encodes the final human disposition is used only to build the label (to avoid leakage).