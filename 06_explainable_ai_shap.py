# Databricks notebook source
# DBTITLE 1,Run prerequisite notebook
# MAGIC %run /Users/vdns10162@gmail.com/05_failure_prediction_model

# COMMAND ----------

# =====================================================================
# TASK 6 - EXPLAINABLE AI (SHAP)
# Explain WHY XGBoost predicts future 5G network failures
# =====================================================================

import sys
import subprocess
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. INSTALL / IMPORT SHAP
# ---------------------------------------------------------

try:
    import shap
except ImportError:
    print("Installing SHAP...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "shap"
    ])
    import shap


print("=" * 75)
print("TASK 6 - EXPLAINABLE AI WITH SHAP")
print("=" * 75)


# ---------------------------------------------------------
# 2. SAMPLE TEST DATA
# SHAP doesn't need all ~100K test records.
# 5,000 gives us meaningful explanations efficiently.
# ---------------------------------------------------------

sample_size = min(5000, len(X_test))

X_shap = X_test.sample(
    n=sample_size,
    random_state=42
)

print("\nSHAP sample size:", f"{len(X_shap):,}")


# ---------------------------------------------------------
# 3. CREATE TREE EXPLAINER
# ---------------------------------------------------------

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer(X_shap)

print("SHAP values successfully generated.")


# ---------------------------------------------------------
# 4. GLOBAL FEATURE IMPORTANCE
# Which KPIs influence predictions most overall?
# ---------------------------------------------------------

mean_abs_shap = np.abs(
    shap_values.values
).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean_ABS_SHAP": mean_abs_shap
}).sort_values(
    "Mean_ABS_SHAP",
    ascending=False
)

print("\n" + "=" * 75)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 75)

print(
    shap_importance
    .round(4)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 5. SHAP SUMMARY PLOT
# Shows feature importance + direction of impact
# ---------------------------------------------------------

print("\nGenerating SHAP Summary Plot...")

shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=X_shap.columns,
    show=True
)


# ---------------------------------------------------------
# 6. SHAP BAR PLOT
# Cleaner executive-level feature importance view
# ---------------------------------------------------------

print("\nGenerating SHAP Feature Importance Plot...")

shap.summary_plot(
    shap_values.values,
    X_shap,
    feature_names=X_shap.columns,
    plot_type="bar",
    show=True
)


# ---------------------------------------------------------
# 7. EXPLAIN ONE HIGH-RISK CELL
# ---------------------------------------------------------

sample_probabilities = xgb_model.predict_proba(
    X_shap
)[:, 1]

highest_risk_position = np.argmax(
    sample_probabilities
)

highest_risk_probability = (
    sample_probabilities[highest_risk_position]
)

high_risk_record = (
    X_shap.iloc[highest_risk_position]
)

high_risk_shap = (
    shap_values.values[highest_risk_position]
)


local_explanation = pd.DataFrame({

    "Feature":
        X_shap.columns,

    "Feature_Value":
        high_risk_record.values,

    "SHAP_Impact":
        high_risk_shap

})

local_explanation["ABS_Impact"] = (
    local_explanation["SHAP_Impact"].abs()
)

local_explanation = (
    local_explanation
    .sort_values(
        "ABS_Impact",
        ascending=False
    )
)


print("\n" + "=" * 75)
print("HIGH-RISK CELL EXPLANATION")
print("=" * 75)

print(
    "Predicted failure probability:",
    f"{highest_risk_probability * 100:.2f}%"
)

print("\nTop contributing KPIs:")

print(
    local_explanation[
        [
            "Feature",
            "Feature_Value",
            "SHAP_Impact"
        ]
    ]
    .head(10)
    .round(4)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 8. IDENTIFY RISK-INCREASING FACTORS
# Positive SHAP values push prediction toward FAILURE.
# ---------------------------------------------------------

risk_drivers = (
    local_explanation[
        local_explanation["SHAP_Impact"] > 0
    ]
    .head(5)
)

print("\n" + "=" * 75)
print("TOP FAILURE RISK DRIVERS")
print("=" * 75)

if len(risk_drivers) > 0:

    for _, row in risk_drivers.iterrows():

        print(
            f"{row['Feature']}: "
            f"value={row['Feature_Value']:.2f}, "
            f"SHAP impact=+{row['SHAP_Impact']:.4f}"
        )

else:

    print(
        "No positive SHAP drivers identified "
        "for this observation."
    )


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("TASK 6 COMPLETE")
print("=" * 75)

print("""
Explainable AI layer successfully added.

XGBoost
   ↓
Failure Probability
   ↓
SHAP Explanation
   ↓
Global KPI Drivers
   ↓
Individual High-Risk Cell Explanation

NEXT:
Anomaly Detection with Isolation Forest
""")

print("=" * 75)