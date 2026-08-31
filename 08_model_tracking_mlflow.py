# Databricks notebook source
# DBTITLE 1,Install xgboost
# MAGIC %pip install xgboost

# COMMAND ----------

# ================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 8 — MLFLOW MODEL EXPERIMENT TRACKING
# Databricks + MLflow + XGBoost
# ================================================================

import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from mlflow.models import infer_signature


# ------------------------------------------------
# 1. LOAD EXISTING SILVER DATA
# ------------------------------------------------

df = spark.table(
    "workspace.default.network_telemetry_silver"
)

print("=" * 75)
print("TASK 8 — MLFLOW EXPERIMENT TRACKING")
print("=" * 75)

print(f"\nSource rows: {df.count():,}")


# ------------------------------------------------
# 2. CREATE SAME FUTURE-FAILURE TARGET
# ------------------------------------------------
# Predict whether a cell experiences a failure
# within the NEXT 30 MINUTES.

from pyspark.sql import functions as F
from pyspark.sql.window import Window

window_spec = (
    Window
    .partitionBy("cell_id")
    .orderBy(F.col("timestamp").cast("long"))
    .rangeBetween(1, 30 * 60)
)

df = df.withColumn(
    "failure_next_30m",
    F.max("failure_event").over(window_spec)
)

df = df.filter(
    F.col("failure_next_30m").isNotNull()
)


# ------------------------------------------------
# 3. MODEL FEATURES
# ------------------------------------------------

features = [
    "signal_strength_dbm",
    "sinr_db",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "traffic_load_pct",
    "connected_users",
    "cpu_utilization_pct",
    "alarm_count",
    "hour_of_day"
]

model_df = (
    df
    .select(
        "timestamp",
        "cell_id",
        "failure_next_30m",
        *features
    )
    .dropna()
    .orderBy("timestamp")
)


# ------------------------------------------------
# 4. CONVERT MODEL DATA TO PANDAS
# ------------------------------------------------

pdf = model_df.toPandas()

X = pdf[features]
y = pdf["failure_next_30m"].astype(int)

print(f"Model rows: {len(pdf):,}")
print(f"Future failures: {int(y.sum()):,}")
print(f"Future failure rate: {(y.mean() * 100):.2f}%")


# ------------------------------------------------
# 5. TIME-BASED TRAIN / TEST SPLIT
# ------------------------------------------------

split_index = int(len(pdf) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTIME-BASED SPLIT")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows : {len(X_test):,}")


# ------------------------------------------------
# 6. HANDLE CLASS IMBALANCE
# ------------------------------------------------

negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / max(positive, 1)

print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")


# ------------------------------------------------
# 7. DEFINE XGBOOST PARAMETERS
# ------------------------------------------------

model_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1
}


# ------------------------------------------------
# 8. START MLFLOW RUN
# ------------------------------------------------

mlflow.set_experiment(
    "/Shared/5G_Network_Failure_Intelligence"
)

with mlflow.start_run(
    run_name="XGBoost_30Min_Failure_Prediction"
) as run:

    print("\nTraining XGBoost model...")

    model = xgb.XGBClassifier(**model_params)

    model.fit(
        X_train,
        y_train
    )


    # ------------------------------------------------
    # 9. PREDICTIONS
    # ------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # ------------------------------------------------
    # 10. MODEL METRICS
    # ------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )


    # ------------------------------------------------
    # 11. LOG PARAMETERS
    # ------------------------------------------------

    mlflow.log_params({
        "algorithm": "XGBoost",
        "prediction_horizon": "30_minutes",
        "split_strategy": "time_based_80_20",
        "decision_threshold": threshold,
        "feature_count": len(features),
        **model_params
    })


    # ------------------------------------------------
    # 12. LOG METRICS
    # ------------------------------------------------

    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    })


    # ------------------------------------------------
    # 13. LOG FEATURE IMPORTANCE
    # ------------------------------------------------

    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance_file = "/tmp/xgboost_feature_importance.csv"

    feature_importance.to_csv(
        importance_file,
        index=False
    )

    mlflow.log_artifact(
        importance_file,
        artifact_path="analysis"
    )


    # ------------------------------------------------
    # 14. LOG MODEL
    # ------------------------------------------------

    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5)
    )


    run_id = run.info.run_id


# ------------------------------------------------
# 15. DISPLAY RESULTS
# ------------------------------------------------

print("\n" + "=" * 75)
print("MLFLOW TRACKING RESULTS")
print("=" * 75)

print(f"""
Run Name       : XGBoost_30Min_Failure_Prediction
Run ID         : {run_id}

Precision      : {precision:.4f}
Recall         : {recall:.4f}
F1 Score       : {f1:.4f}
ROC-AUC        : {roc_auc:.4f}
PR-AUC         : {pr_auc:.4f}

Threshold      : {threshold:.2f}
Features       : {len(features)}
""")


print("\nTOP FEATURE IMPORTANCE")
display(feature_importance)


print("\n" + "=" * 75)
print("TASK 8 COMPLETE")
print("=" * 75)

print("""
MLflow successfully tracked:

✓ XGBoost model
✓ Hyperparameters
✓ Precision
✓ Recall
✓ F1 Score
✓ ROC-AUC
✓ PR-AUC
✓ Decision threshold
✓ Feature importance artifact
✓ Model signature
✓ Input example
✓ Reproducible experiment run

Experiment:
5G_Network_Failure_Intelligence

NEXT:
Model operationalization / risk-scoring pipeline
""")

print("=" * 75)