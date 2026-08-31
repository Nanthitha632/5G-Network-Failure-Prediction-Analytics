# Databricks notebook source
# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

# Predictions
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]
log_pred = (log_prob >= 0.50).astype(int)

# Evaluation
log_precision = precision_score(y_test, log_pred, zero_division=0)
log_recall = recall_score(y_test, log_pred, zero_division=0)
log_f1 = f1_score(y_test, log_pred, zero_division=0)
log_auc = roc_auc_score(y_test, log_prob)
log_pr_auc = average_precision_score(y_test, log_prob)

print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print("Precision :", round(log_precision, 4))
print("Recall    :", round(log_recall, 4))
print("F1 Score  :", round(log_f1, 4))
print("ROC-AUC   :", round(log_auc, 4))
print("PR-AUC    :", round(log_pr_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        log_pred,
        digits=4,
        zero_division=0
    )
)

# COMMAND ----------

# =====================================================================
# TASK 5 - 5G FAILURE PREDICTION
# Predict whether a cell will fail in the NEXT 30 MINUTES
# Databricks + PySpark + Scikit-learn + XGBoost
# =====================================================================

# ---------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------

import sys
import subprocess
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Install XGBoost only if the Databricks environment does not have it.
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Installing...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "xgboost"
    ])
    from xgboost import XGBClassifier


print("=" * 75)
print("TASK 5 - 30-MINUTE 5G FAILURE PREDICTION")
print("=" * 75)


# =====================================================================
# 1. LOAD GOLD DELTA DATA
# =====================================================================

gold_table = "workspace.default.network_telemetry_gold"

df = (
    spark.table(gold_table)
    .orderBy("cell_id", "timestamp")
)

print("\nGold Delta table loaded.")
print("Rows:", f"{df.count():,}")


# =====================================================================
# 2. CREATE FUTURE FAILURE TARGET
#
# Each row represents a 5-minute observation.
# Next 6 observations = next 30 minutes.
#
# failure_next_30m = 1 if ANY failure occurs in the next 6 records
# for the SAME CELL.
# =====================================================================

future_window = (
    Window
    .partitionBy("cell_id")
    .orderBy("timestamp")
    .rowsBetween(1, 6)
)

df = df.withColumn(
    "failure_next_30m",
    F.max("failure_event").over(future_window)
)

# Remove trailing observations where a complete future window
# does not exist.
df = df.filter(
    F.col("failure_next_30m").isNotNull()
)

# We don't want to "predict" a failure that is already occurring.
df = df.filter(
    F.col("failure_event") == 0
)


# =====================================================================
# 3. SELECT MODEL FEATURES
#
# No failure_event-derived fields or operational_status are used.
# This prevents target leakage.
# =====================================================================

feature_columns = [
    "traffic_load_pct",
    "connected_users",
    "signal_strength_dbm",
    "sinr_db",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "cpu_utilization_pct",
    "alarm_count",
    "hour_of_day"
]

model_spark = (
    df
    .select(
        "timestamp",
        "cell_id",
        *feature_columns,
        "failure_next_30m"
    )
    .dropna()
)


# =====================================================================
# 4. TARGET VALIDATION
# =====================================================================

target_summary = (
    model_spark
    .groupBy("failure_next_30m")
    .count()
    .orderBy("failure_next_30m")
)

print("\n" + "=" * 75)
print("FUTURE FAILURE TARGET")
print("=" * 75)

target_summary.show()

total_model_rows = model_spark.count()

failure_rows = (
    model_spark
    .filter(F.col("failure_next_30m") == 1)
    .count()
)

future_failure_rate = (
    failure_rows / total_model_rows * 100
)

print("Model-ready rows:", f"{total_model_rows:,}")
print(
    "Future failure rate:",
    f"{future_failure_rate:.2f}%"
)


# =====================================================================
# 5. CONVERT ONLY MODEL COLUMNS TO PANDAS
#
# ~500K rows x a small number of columns is acceptable here.
# Heavy cleaning/transformation stayed in Spark.
# =====================================================================

pdf = (
    model_spark
    .orderBy("timestamp")
    .toPandas()
)

pdf["failure_next_30m"] = (
    pdf["failure_next_30m"]
    .astype(int)
)


# =====================================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
#
# First 80% of TIME = training
# Last 20% of TIME  = testing
#
# This is more realistic than a random split for telemetry.
# =====================================================================

split_index = int(len(pdf) * 0.80)

train_df = pdf.iloc[:split_index].copy()
test_df = pdf.iloc[split_index:].copy()

X_train = train_df[feature_columns]
y_train = train_df["failure_next_30m"]

X_test = test_df[feature_columns]
y_test = test_df["failure_next_30m"]


print("\n" + "=" * 75)
print("TIME-BASED SPLIT")
print("=" * 75)

print("Training rows:", f"{len(train_df):,}")
print("Testing rows :", f"{len(test_df):,}")

print(
    "\nTrain period:",
    train_df["timestamp"].min(),
    "to",
    train_df["timestamp"].max()
)

print(
    "Test period :",
    test_df["timestamp"].min(),
    "to",
    test_df["timestamp"].max()
)

print("\nTraining target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# =====================================================================
# 7. CLASS IMBALANCE
# =====================================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = (
    negative_count / positive_count
)

print("\n" + "=" * 75)
print("CLASS IMBALANCE")
print("=" * 75)

print("Normal records   :", f"{negative_count:,}")
print("Upcoming failures:", f"{positive_count:,}")

print(
    "Normal : Failure ratio =",
    f"{scale_pos_weight:.2f} : 1"
)

print(
    "XGBoost scale_pos_weight:",
    f"{scale_pos_weight:.2f}"
)


# =====================================================================
# 8. LOGISTIC REGRESSION BASELINE
# =====================================================================

print("\n" + "=" * 75)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 75)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_train
)

log_prob = (
    log_model
    .predict_proba(X_test_scaled)[:, 1]
)

log_pred = (
    log_prob >= 0.50
).astype(int)

log_precision = precision_score(
    y_test,
    log_pred,
    zero_division=0
)

log_recall = recall_score(
    y_test,
    log_pred,
    zero_division=0
)

log_f1 = f1_score(
    y_test,
    log_pred,
    zero_division=0
)

log_auc = roc_auc_score(
    y_test,
    log_prob
)

log_pr_auc = average_precision_score(
    y_test,
    log_prob
)

print("\nLOGISTIC REGRESSION RESULTS")

print(
    "Precision:",
    round(log_precision, 4)
)

print(
    "Recall   :",
    round(log_recall, 4)
)

print(
    "F1 Score :",
    round(log_f1, 4)
)

print(
    "ROC-AUC  :",
    round(log_auc, 4)
)

print(
    "PR-AUC   :",
    round(log_pr_auc, 4)
)


# =====================================================================
# 9. XGBOOST MODEL
# =====================================================================

print("\n" + "=" * 75)
print("TRAINING XGBOOST")
print("=" * 75)

xgb_model = XGBClassifier(

    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,

    subsample=0.80,
    colsample_bytree=0.80,

    scale_pos_weight=scale_pos_weight,

    objective="binary:logistic",
    eval_metric="logloss",

    random_state=42,
    n_jobs=4
)

xgb_model.fit(
    X_train,
    y_train
)


# =====================================================================
# 10. XGBOOST PREDICTIONS
# =====================================================================

xgb_prob = (
    xgb_model
    .predict_proba(X_test)[:, 1]
)

xgb_pred = (
    xgb_prob >= 0.50
).astype(int)


# =====================================================================
# 11. XGBOOST EVALUATION
# =====================================================================

xgb_precision = precision_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)

xgb_pr_auc = average_precision_score(
    y_test,
    xgb_prob
)

print("\n" + "=" * 75)
print("XGBOOST RESULTS")
print("=" * 75)

print(
    "Precision:",
    round(xgb_precision, 4)
)

print(
    "Recall   :",
    round(xgb_recall, 4)
)

print(
    "F1 Score :",
    round(xgb_f1, 4)
)

print(
    "ROC-AUC  :",
    round(xgb_auc, 4)
)

print(
    "PR-AUC   :",
    round(xgb_pr_auc, 4)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_pred,
        digits=4,
        zero_division=0
    )
)


# =====================================================================
# 12. MODEL COMPARISON
# =====================================================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "XGBoost"
    ],

    "Precision": [
        log_precision,
        xgb_precision
    ],

    "Recall": [
        log_recall,
        xgb_recall
    ],

    "F1_Score": [
        log_f1,
        xgb_f1
    ],

    "ROC_AUC": [
        log_auc,
        xgb_auc
    ],

    "PR_AUC": [
        log_pr_auc,
        xgb_pr_auc
    ]

})

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    comparison
    .round(4)
    .to_string(index=False)
)


# =====================================================================
# 13. FEATURE IMPORTANCE
# =====================================================================

feature_importance = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        xgb_model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 75)
print("TOP XGBOOST FEATURES")
print("=" * 75)

print(
    feature_importance
    .round(4)
    .to_string(index=False)
)


# =====================================================================
# 14. CREATE TEST RISK OUTPUT
# =====================================================================

risk_output = test_df[
    ["timestamp", "cell_id"]
].copy()

risk_output["actual_failure_next_30m"] = (
    y_test.values
)

risk_output["failure_probability"] = (
    xgb_prob
)

risk_output["failure_risk_pct"] = (
    xgb_prob * 100
)


risk_output["risk_level"] = pd.cut(

    risk_output["failure_probability"],

    bins=[
        -0.01,
        0.25,
        0.50,
        0.75,
        1.00
    ],

    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]
)


print("\n" + "=" * 75)
print("TOP PREDICTED FAILURE RISKS")
print("=" * 75)

display(
    risk_output
    .sort_values(
        "failure_probability",
        ascending=False
    )
    .head(20)
)


# =====================================================================
# FINAL
# =====================================================================

print("\n" + "=" * 75)
print("TASK 5 COMPLETE")
print("=" * 75)

print(f"""
Prediction target:
Failure occurring within NEXT 30 MINUTES

Best-model candidate:
XGBoost

XGBoost Precision : {xgb_precision:.4f}
XGBoost Recall    : {xgb_recall:.4f}
XGBoost F1        : {xgb_f1:.4f}
XGBoost ROC-AUC   : {xgb_auc:.4f}
XGBoost PR-AUC    : {xgb_pr_auc:.4f}

NEXT TASK:
Isolation Forest Anomaly Detection
""")

print("=" * 75)
