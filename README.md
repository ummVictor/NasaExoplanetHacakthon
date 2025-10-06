# 🌌 CODELOCK  
**Machine Learning for Automated Exoplanet Detection**  
*By Martin Ha, Victor Um, and Christina Mourad*  
NASA Space Apps Challenge 2025  

---

## 🚀 Project Overview

**CODELOCK** is a machine learning pipeline designed to accurately identify exoplanet candidates across NASA’s **Kepler**, **K2**, and **TESS** missions.  
By unifying data from multiple space telescopes and combining classical and ensemble learning methods, our system distinguishes real exoplanets from false positives with high precision and recall.  

The project addresses the core challenge of **scalable, automated exoplanet vetting**—helping astronomers prioritize follow-up observations, optimize telescope time, and expand our understanding of planetary systems beyond our own.

---

## 🎯 Goal

Develop a unified, reproducible machine learning framework that:
- Integrates NASA’s open-source Kepler, K2, and TESS datasets  
- Trains and compares multiple models for exoplanet classification  
- Maximizes **recall** (true detections) while minimizing **false positives**  
- Supports deployment to an interactive interface for exploration and visualization  

---

## 🧠 Methodology

### **1️⃣ Preprocessing**
- Cleaned raw mission data by removing empty, duplicated, and non-predictive fields  
- Standardized measurement units (e.g., converting all transit depths to ppm)  
- Normalized and scaled numeric features across all missions  
- Handled missing values through targeted imputation  
- Guarded against label leakage by removing disposition-related columns  

### **2️⃣ Data Integration & Cleaning**
- Merged Kepler, K2, and TESS datasets into a single standardized table (~17,000 entries)  
- Unified equivalent feature names across missions (e.g., `koi_period`, `pl_orbper` → `period_d`)  
- Dropped redundant, astrometric, or uncertainty columns (>95% missing)  
- Converted Kepler epochs (BKJD) into Barycentric Julian Date (BJD) for consistency  
- Created a cross-mission label: `1 = confirmed/candidate`, `0 = false positive`

### **3️⃣ Modeling & Evaluation**
We compared and optimized multiple ensemble-based models:
- **LightGBM (Gradient Boosted Trees)** — fast, interpretable baseline  
- **AdaBoost** — strong recall and adaptive weighting  
- **Random Forest** — robust, low variance classifier  
- **StackingClassifier** — meta-ensemble combining all models for maximum recall  

All models were evaluated using **10-fold cross-validation** with metrics:
**Accuracy, Precision, Recall, F1, and ROC-AUC**.

---

## 📊 Results

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC | Highlights |
|:------|:----------|:--------|:------------|:----|:----------|:------------|
| **LightGBM** | ~0.94 | 0.96 | 0.93 | **0.944** | **0.969** | Best precision–recall balance |
| **AdaBoost** | ~0.93 | **0.972** | 0.91 | 0.942 | 0.967 | Strong recall, few missed positives |
| **Random Forest** | ~0.94 | 0.978 | 0.91 | **0.945** | **0.970** | Excellent recall and generalization |
| **Stacking Classifier** | ~0.92 | **0.995** | 0.88 | 0.939 | 0.977 | Maximizes true positives, tolerates more false alarms |

**All models achieved >91% accuracy and >0.94 F1**, demonstrating strong predictive performance.  
The **StackingClassifier** achieved near-perfect recall, making it ideal when missing a true planet is unacceptable.  
**LightGBM** and **Random Forest** provided the best trade-off for real-world deployment.

---

## 💡 Conclusion

CODELOCK successfully demonstrates that ensemble-based machine learning can:
- Improve exoplanet detection accuracy and recall  
- Scale across missions with unified, standardized preprocessing  
- Serve as a reproducible framework for future space survey pipelines  

This work supports NASA’s broader mission by helping astronomers **automate candidate vetting**, **reduce manual workloads**, and **accelerate discoveries** of worlds beyond our solar system.

---

## 🧩 Tech Stack

- **Python 3.11+**  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`  
- Data: NASA Exoplanet Archive (Kepler, K2, TESS datasets)  
- Cross-validation: `StratifiedKFold (k=10)`  
- Deployment-ready outputs: unified CSV/Parquet dataset + trained models  

---

## 📚 Future Work

- Integrate **deep learning** for feature extraction from light curves  
- Add **explainable AI (XAI)** modules for interpretability  
- Deploy an **interactive web interface** for real-time classification and visualization  

---

## 🪐 Team

**CODELOCK** was created by:  
- **Martin Ha**  
- **Victor Um**  
- **Christina Mourad**

NASA Space Apps Challenge 2025 — *Automated Exoplanet Detection with Machine Learning*  
