```markdown
# DS675 Mini-Project: Diabetes Risk Prediction Using Machine Learning

## Overview
Supervised machine learning project predicting diabetes risk using the CDC BRFSS 2015 Health Indicators dataset as part of DS675 Machine Learning at NJIT.

**Dataset:** [Diabetes Health Indicators Dataset — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  


---

## Group Members & Contributions

| Member | Contribution |
|--------|-------------|
| Rui Costa | EDA, Random Forest feature selection, Logistic Regression baseline model |
| Nikunj Kantaria | XGBoost model, SHAP explainability analysis, balanced/imbalanced/SMOTE dataset comparison |
| Ayelet Zaidenberg | KNN hyperparameter tuning, feature convergence analysis, SMOTE resampling |

---

## Notebooks

| Notebook | Author | Description |
|----------|--------|-------------|
| `rui_eda_baseline.ipynb` | Rui Costa | EDA, correlation analysis, RF feature selection, Logistic Regression baseline |
| `nikunj_shap_analysis.ipynb` | Nikunj Kantaria | XGBoost training, SHAP feature importance, beeswarm/dependence/waterfall plots, dataset balancing comparison |
| `ayelet_knn_tuning.ipynb` | Ayelet Zaidenberg | KNN GridSearchCV hyperparameter tuning, feature convergence experiment |

---

## Results Summary

| Model | Dataset | Accuracy | F1 Score | AUC-ROC |
|-------|---------|----------|----------|---------|
| Logistic Regression (Baseline) | Balanced 50/50 | 0.7458 | 0.7503 | 0.8232 |
| KNN (k=43, manhattan, uniform) | Balanced 50/50 | 0.7405 | 0.7450 | — |
| XGBoost | Balanced 50/50 | 0.7554 | 0.7658 | 0.8321 |
| XGBoost | Imbalanced (raw) | 0.8656* | 0.2537 | 0.8273 |
| XGBoost | SMOTE | 0.8642* | 0.2988 | 0.8229 |

*\*High accuracy is misleading due to majority class dominance — F1 and AUC are the appropriate metrics.*

---

## Key Findings

- **GenHlth is the #1 predictor** by SHAP analysis — not BMI as Random Forest importance suggests. RF undervalues correlated features; SHAP attributes credit more accurately.
- **HighChol jumps from RF rank #9 to SHAP rank #5** — significantly undervalued by RF.
- **Just 2 features** (GenHlth + HighBP) recover **95% of KNN accuracy** achievable with all 21 features.
- **SMOTE does not improve** over the pre-balanced Kaggle dataset — AUC stays stable across all three balancing strategies, but F1 collapses on imbalanced data.
- **Accuracy is a misleading metric** for imbalanced diabetes prediction — F1 and AUC-ROC are the appropriate evaluation measures.

---

## Methodology

- **Train/test split:** 80/20, `random_state=42`, stratified
- **Scaling:** StandardScaler fitted on training set only (no data leakage)
- **Validation:** 5-fold cross-validation for hyperparameter tuning
- **Baseline:** Logistic Regression (all 21 features, L2 regularization)
- **SHAP:** TreeExplainer on 2,000-sample test subset

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn
```

---

## References

1. Teboul, A. (2021). Diabetes Health Indicators Dataset. Kaggle.
2. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
3. Chen, T., & Guestrin, C. (2016). XGBoost. ACM KDD.
4. Chawla et al. (2002). SMOTE. Journal of AI Research.
```
