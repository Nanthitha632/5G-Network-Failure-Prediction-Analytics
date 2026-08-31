#  Model Evaluation Results

This folder documents the evaluation results from the predictive modeling stage of the 5G Network Failure Prediction project.

## Logistic Regression Baseline

The Logistic Regression model was trained as an interpretable baseline for predicting whether a network cell would experience a failure within the next 30 minutes.

| Metric | Result |
|---|---:|
| Precision | 0.4726 |
| Recall | 0.9763 |
| F1 Score | 0.6369 |
| ROC-AUC | 0.9975 |
| PR-AUC | 0.8373 |
| Accuracy | 98.72% |

## Confusion Matrix

| | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | 97,118 | 1,242 |
| Actual Failure | 27 | 1,113 |

### Interpretation

The Logistic Regression baseline achieved **97.63% recall**, correctly identifying **1,113 of 1,140 upcoming failures** in the test set.

Only **27 upcoming failures were missed**.

The lower precision of **47.26%** indicates that the high-sensitivity baseline also generated false-positive alerts, making precision-versus-recall an important consideration for operational deployment.

---

## XGBoost Model

XGBoost was developed as the advanced nonlinear model for capturing complex relationships between network telemetry KPIs.

The final XGBoost metrics and direct model comparison are documented alongside the completed model evaluation outputs.

---

## Model Comparison

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4726 | 0.9763 | 0.6369 | 0.9975 | 0.8373 |
| XGBoost | Pending | Pending | Pending | Pending | Pending |

---

## Evaluation Strategy

The project evaluates models using multiple metrics because network-failure prediction is a highly imbalanced classification problem.

- **Precision** — reliability of generated failure alerts
- **Recall** — percentage of actual upcoming failures detected
- **F1 Score** — balance between precision and recall
- **ROC-AUC** — overall class-separation capability
- **PR-AUC** — precision-recall performance under class imbalance
- **Confusion Matrix** — operational view of detected, missed, and false failure alerts
