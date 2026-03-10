# Predicting 30-Day Hospital Readmissions with Machine Learning

## Overview
Hospital readmissions are costly and often preventable. This project builds and interprets machine learning models to predict whether a diabetic patient will be readmitted within 30 days of discharge using the Diabetes 130-US hospitals dataset.

The project compares an interpretable baseline model (logistic regression) against a nonlinear model (XGBoost), then evaluates risk concentration, calibration, feature importance, and subgroup behavior.

## Business / Clinical Question
Can hospitals identify a manageable subset of high-risk patients for post-discharge intervention programs such as follow-up calls, medication reconciliation, and care coordination?

## Dataset
- Source: Diabetes 130-US hospitals dataset
- Population: Hospital encounters for diabetic patients across 130 U.S. hospitals
- Target: Readmitted within 30 days
- Feature types:
  - demographics
  - diagnoses
  - medication and treatment history
  - prior healthcare utilization

## Methods
### Preprocessing
- handled missing values
- one-hot encoded categorical variables
- scaled numeric variables for logistic regression
- preserved the same preprocessing pipeline across train/test data

### Models
- Logistic Regression (L1-regularized baseline)
- XGBoost (nonlinear model)

### Evaluation Metrics
Because 30-day readmission is an imbalanced outcome, I focused on:
- ROC-AUC
- PR-AUC
- lift and cumulative gains
- calibration
- threshold-based risk targeting

## Results
| Model | ROC-AUC | PR-AUC |
|------|---------|--------|
| Logistic Regression | ~0.653 | ~0.208 |
| XGBoost | **0.676** | **0.228** |

### Key Findings
- XGBoost outperformed logistic regression, suggesting important nonlinear interactions in the data.
- Prior inpatient utilization was the strongest predictor of readmission risk.
- Patients with no prior inpatient admissions had predicted readmission risk around 9%, while patients with 10+ prior admissions exceeded 35%.
- Using a 15% probability threshold flagged about 20% of patients as high risk.
- The top risk decile had about **2.35x** the average readmission rate.
- Targeting the top 20% highest-risk patients captured about **35%** of all readmissions.

## Interpretation
Model interpretation with SHAP and partial dependence plots showed that the most important predictors were:
- number of prior inpatient visits
- number of emergency visits
- time in hospital
- number of medications
- number of diagnoses

These patterns indicate that readmission risk is driven more by prior healthcare utilization and disease burden than by any single diagnosis alone.

## Fairness Check
Average predicted risk was broadly similar across race and gender groups, suggesting no major disparities in average model output.

## Example Risk Profiles
Using the calibrated XGBoost model:
- Low-risk patient: **4.6%** predicted readmission probability
- High-risk patient: **20.8%** predicted readmission probability

This corresponds to approximately **5.5x higher odds** of readmission for the high-risk profile.

## Tools Used
- Python
- Pandas
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- seaborn
- Streamlit

## Next Steps
- add more robust subgroup fairness metrics
- test alternative calibration approaches
- group diagnosis codes into broader clinical categories
- deploy a lightweight risk calculator interface

## Project Structure
```text
hospital-readmission-ml/
├── README.md
├── notebook/
│   └── readmission_prediction.ipynb
├── images/
├── models/
└── app/
    └── streamlit_app.py
