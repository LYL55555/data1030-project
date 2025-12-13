# DATA 1030 Final Project — Occupancy Detection

**Author:** Yanle Lyu (Brown University, Class of 2025)  
**Course:** DATA 1030 — Introduction to Data Science  
**Repository:** https://github.com/LYL55555/data1030-project

---

## Project Overview
This project studies whether **indoor occupancy can be accurately predicted using only low-cost environmental sensors and recent time history**. Accurate occupancy detection is critical for building automation systems (HVAC, lighting), where better predictions can reduce energy waste without sacrificing comfort.

Using the **UCI Occupancy Detection Dataset**, I build a **time-aware machine learning pipeline**, compare linear and non-linear models, quantify uncertainty from temporal splitting, and analyze **global and local interpretability**.

---

## Dataset
- **Source:** UCI Machine Learning Repository — Occupancy Detection Dataset  
  http://archive.ics.uci.edu/dataset/357/occupancy+detection
- **Observations:** 17,893 time-stamped samples from a single office room
- **Features:** Temperature, Humidity, Light, CO2, HumidityRatio
- **Target:** Binary occupancy label (0 = empty, 1 = occupied)
- **Class imbalance:** ~79% empty / ~21% occupied
- **No missing values**

Strong daily/weekly patterns and sensor autocorrelation make **time-respecting evaluation mandatory**.

---

## Feature Engineering
- **Temporal features**
  - Hour of day (0–23)
  - One-hot encoded day of week (dow_0–dow_6)
- **Lag features (30 minutes)** for all sensor variables
- **Final feature set**
  - Current sensors + 30-minute lags + time indicators

---

## Data Splitting & Evaluation
- **Splitting strategy:** Time-ordered (no shuffling)  
  60% train / 20% validation / 20% test
- **Cross-validation:** TimeSeriesSplit
- **Metric:** F1 score (robust to class imbalance)
- **Baseline:** Stratified random guesser

---

## Models Compared
- Logistic Regression (L1, L2, Elastic Net)
- KNN
- SVM (RBF)
- Random Forest
- XGBoost

---

## Results
| Model | CV Mean F1 | CV Std | Test F1 |
|------|-----------:|-------:|--------:|
| Baseline | 0.196 | 0.108 | 0.241 |
| LogReg (L1) | 0.741 | 0.374 | 0.991 |
| LogReg (L2) | 0.734 | 0.370 | 0.991 |
| Elastic Net | 0.707 | 0.368 | 0.991 |
| KNN | 0.579 | 0.400 | 0.968 |
| SVM (RBF) | 0.217 | 0.339 | 0.899 |
| Random Forest | 0.500 | 0.400 | 0.955 |
| **XGBoost** | **0.745** | **0.374** | **0.992** |

---

## Final Model
### ✅ Best Model: **XGBoost**
- **Test F1:** **0.992**
- Most predictive under time-series cross-validation
- Interpretable via SHAP
- In the end I retrained the model with `train` and `validation`, and put it on test, I got a test F1 of 0.966, which is not bad as well.

Linear models achieve nearly identical performance and are preferred when simplicity and interpretability are priorities.

---

## Interpretability Highlights
- Dominant features: **Light**, **Light (lag 30)**, **CO2**, **CO2 (lag 30)**
- Errors occur when Light and CO2 signals disagree
- Results align with physical intuition

---

## References
- UCI Occupancy Detection Dataset  
  http://archive.ics.uci.edu/dataset/357/occupancy+detection


---

## Environment
- Standard Data 1030 environment, see data1030.yml for details.