# Unified Feature Rationale

This document explains **which features were kept**, **why**, how they were **normalized**, and the **source columns** per mission.

## Kept (Unified) Features

### `mission`
- **What it is:** Mission name (KEPLER/K2/TESS)
- **Why keep it:** Provenance for stratified evaluation and to learn mission-specific systematics.
- **Chosen source columns:**

### `object_id`
- **What it is:** Human-readable object identifier
- **Why keep it:** For traceability/joins; not a model feature.
- **Chosen source columns:**
  - KEPLER: `kepid`
  - K2: `pl_name`
  - TESS: `toi`
- **Redundant aliases encountered:**
- KEPLER aliases not chosen: kepler_name, kepoi_name
- K2 aliases not chosen: hostname
- TESS aliases not chosen: tid

### `source_id_raw`
- **What it is:** Source’s original ID string
- **Why keep it:** Extra identifier for robust de-duplication across catalogs.
- **Chosen source columns:**
  - KEPLER: `kepid`
  - K2: `pl_name`
  - TESS: `toi`
- **Redundant aliases encountered:**
- KEPLER aliases not chosen: kepler_name, kepoi_name
- K2 aliases not chosen: hostname
- TESS aliases not chosen: tid

### `label`
- **What it is:** Supervised target: 1=planet/PC, 0=false positive
- **Why keep it:** Built from disposition; NEVER used as a feature.
- **Chosen source columns:**

### `period_d`
- **What it is:** Orbital period (days)
- **Why keep it:** Core discriminant; true planets show coherent periodicity.
- **Chosen source columns:**
  - KEPLER: `koi_period`
  - K2: `pl_orbper`
  - TESS: `pl_orbper`

### `dur_h`
- **What it is:** Transit duration (hours)
- **Why keep it:** Constrained by a/R* and impact parameter; flags improbable geometries.
- **Chosen source columns:**
  - KEPLER: `koi_duration`
  - TESS: `pl_trandurh`

### `depth_ppm`
- **What it is:** Transit depth (ppm)
- **Why keep it:** Proxy for (Rp/R*)²; normalized across datasets.
- **Chosen source columns:**
  - KEPLER: `koi_depth`
  - TESS: `pl_trandep`
- **Normalization/transform:**
- KEPLER/depth_ppm: depth already in ppm (source `koi_depth`)
- TESS/depth_ppm: depth already in ppm (source `pl_trandep`)

### `impact`
- **What it is:** Impact parameter b
- **Why keep it:** High b + deep/sharp events can indicate grazing EBs.
- **Chosen source columns:**
  - KEPLER: `koi_impact`

### `ror`
- **What it is:** Rp/R* (radius ratio)
- **Why keep it:** Geometric parameter; complements depth with limb-darkening.
- **Chosen source columns:**

### `prad_re`
- **What it is:** Planet radius (Earth radii)
- **Why keep it:** Physical size; helps reject astrophysical FPs/dilution.
- **Chosen source columns:**
  - KEPLER: `koi_prad`
  - K2: `pl_rade`
  - TESS: `pl_rade`

### `a_au`
- **What it is:** Semi-major axis (AU)
- **Why keep it:** With stellar params informs irradiation & transit probability.
- **Chosen source columns:**
  - K2: `pl_orbsmax`

### `a_over_rstar`
- **What it is:** Scaled separation a/R*
- **Why keep it:** Key to transit geometry and duration expectations.
- **Chosen source columns:**

### `insol_earth`
- **What it is:** Incident flux (Earth=1)
- **Why keep it:** Irradiation context; separates EB vs planet edge cases.
- **Chosen source columns:**
  - KEPLER: `koi_insol`
  - K2: `pl_insol`
  - TESS: `pl_insol`

### `teq_k`
- **What it is:** Equilibrium temperature (K)
- **Why keep it:** Irradiation proxy; correlations with depth/period can flag FPs.
- **Chosen source columns:**
  - KEPLER: `koi_teq`
  - K2: `pl_eqt`
  - TESS: `pl_eqt`

### `teff_k`
- **What it is:** Stellar effective temperature (K)
- **Why keep it:** Host context for priors (e.g., radius expectations).
- **Chosen source columns:**
  - KEPLER: `koi_steff`
  - K2: `st_teff`
  - TESS: `st_teff`

### `logg_cgs`
- **What it is:** Stellar surface gravity (cgs)
- **Why keep it:** Dwarfs vs giants; many FPs around giants due to dilution.
- **Chosen source columns:**
  - KEPLER: `koi_slogg`
  - K2: `st_logg`
  - TESS: `st_logg`

### `radius_rsun`
- **What it is:** Stellar radius (R☉)
- **Why keep it:** Needed to convert depth to radius; flags giant hosts.
- **Chosen source columns:**
  - KEPLER: `koi_srad`
  - K2: `st_rad`
  - TESS: `st_rad`

### `mass_msun`
- **What it is:** Stellar mass (M☉)
- **Why keep it:** Sets dynamics and a/R* expectations.
- **Chosen source columns:**
  - K2: `st_mass`

### `feh_dex`
- **What it is:** Stellar metallicity [Fe/H]
- **Why keep it:** Weak prior on occurrence; sometimes predictive for FPs.
- **Chosen source columns:**
  - K2: `st_met`

### `mes`
- **What it is:** Multiple Event Statistic
- **Why keep it:** Detection statistic; higher → more reliable.
- **Chosen source columns:**

### `snr`
- **What it is:** Model/fit signal-to-noise
- **Why keep it:** General detection quality metric.
- **Chosen source columns:**
  - KEPLER: `koi_model_snr`

### `fpflag_nt`
- **What it is:** Not transit-like flag
- **Why keep it:** High-signal FP indicator.
- **Chosen source columns:**
  - KEPLER: `koi_fpflag_nt`

### `fpflag_ss`
- **What it is:** Significant secondary flag
- **Why keep it:** EB-like secondary eclipse signature.
- **Chosen source columns:**
  - KEPLER: `koi_fpflag_ss`

### `fpflag_co`
- **What it is:** Centroid offset flag
- **Why keep it:** Likely blend/background EB.
- **Chosen source columns:**
  - KEPLER: `koi_fpflag_co`

### `fpflag_ec`
- **What it is:** Ephemeris match/confusion flag
- **Why keep it:** Likely contamination from known variables.
- **Chosen source columns:**
  - KEPLER: `koi_fpflag_ec`

### `epoch_bjd_raw`
- **What it is:** Transit mid-time (as provided)
- **Why keep it:** Provenance; Kepler is BKJD.
- **Chosen source columns:**
  - KEPLER: `koi_time0bk`
  - TESS: `pl_tranmid`

### `epoch_bjd`
- **What it is:** Transit mid-time (BJD)
- **Why keep it:** Unified BJD epoch for all missions.
- **Chosen source columns:**

## Label-Only Sources (not used as features)

- **KEPLER**: koi_disposition, koi_pdisposition
- **K2**: disposition
- **TESS**: tfopwg_disp

## What We Dropped and Why (high level)

- **Metadata/IDs/Provenance:** kept minimal IDs for joining; other IDs/URLs/facility fields are non-predictive.
- **Uncertainties/Limits:** excluded in v1; consider later for calibrated models.
- **Coordinates/Astrometry:** sky position not causally informative for planet vs FP and complicates cross-mission mixing.
- **Redundant encodings:** normalized multiple encodings (fraction/percent/ppm) into single units.
- **Vetting/label fields:** used only to build the target (avoid leakage).