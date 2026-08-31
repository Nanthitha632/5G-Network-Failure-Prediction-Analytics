# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

# DBTITLE 1,Install xgboost
# MAGIC %pip install xgboost

# COMMAND ----------

# =====================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 9 — PRODUCTION RISK-SCORING + ALERT PIPELINE
# Databricks + MLflow + XGBoost + Isolation Forest Outputs + Delta Lake
# =====================================================================

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from pyspark.sql import functions as F

print("=" * 78)
print("TASK 9 — PRODUCTION 5G RISK-SCORING & ALERT PIPELINE")
print("=" * 78)


# =====================================================================
# 1. FIND THE TRACKED XGBOOST MODEL FROM MLFLOW
# =====================================================================

experiment = mlflow.get_experiment_by_name(
    "/Shared/5G_Network_Failure_Intelligence"
)

if experiment is None:
    raise ValueError(
        "MLflow experiment not found. Task 8 must exist before Task 9."
    )

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'XGBoost_30Min_Failure_Prediction'",
    order_by=["start_time DESC"],
    max_results=1
)

if len(runs) == 0:
    raise ValueError(
        "Tracked XGBoost run not found in MLflow."
    )

run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{run_id}/model"

xgb_model = mlflow.xgboost.load_model(model_uri)

print("\nTracked model loaded successfully.")
print("MLflow Run ID:", run_id)


# =====================================================================
# 2. LOAD GOLD TELEMETRY
# =====================================================================

gold_table = "workspace.default.network_telemetry_gold"

gold_spark = spark.table(gold_table)

print("\nGold telemetry rows:", f"{gold_spark.count():,}")


# =====================================================================
# 3. MODEL FEATURES
# Must match the features used when training the XGBoost model.
# =====================================================================

feature_columns = [
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

id_columns = [
    "timestamp",
    "cell_id",
    "site_id",
    "region",
    "time_period",
    "failure_event"
]

score_spark = (
    gold_spark
    .select(
        *id_columns,
        *feature_columns
    )
    .dropna()
)


# =====================================================================
# 4. CONVERT SCORING DATA TO PANDAS
# Databricks/PySpark already handled heavy engineering.
# This step only performs model inference.
# =====================================================================

score_pdf = (
    score_spark
    .orderBy("timestamp", "cell_id")
    .toPandas()
)

print("Rows prepared for scoring:", f"{len(score_pdf):,}")


# =====================================================================
# 5. GENERATE XGBOOST FAILURE PROBABILITY
# =====================================================================

X_score = score_pdf[feature_columns]

failure_probability = (
    xgb_model
    .predict_proba(X_score)[:, 1]
)

score_pdf["failure_probability"] = failure_probability

score_pdf["failure_risk_pct"] = (
    score_pdf["failure_probability"] * 100
).round(2)


# =====================================================================
# 6. OPERATIONAL RISK LEVEL
#
# These are business-operational alert bands.
# They are NOT model-training class labels.
# =====================================================================

score_pdf["risk_level"] = pd.cut(
    score_pdf["failure_probability"],
    bins=[
        -0.001,
        0.25,
        0.50,
        0.75,
        1.001
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]
).astype(str)


# =====================================================================
# 7. CREATE ACTION PRIORITY
# =====================================================================

priority_map = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4
}

score_pdf["priority_code"] = (
    score_pdf["risk_level"]
    .map(priority_map)
    .astype(int)
)


# =====================================================================
# 8. ADD RECOMMENDED OPERATIONAL ACTION
# =====================================================================

def recommended_action(row):

    if row["risk_level"] == "CRITICAL":
        return (
            "Immediate investigation: review packet loss, latency, "
            "signal quality, alarms and cell capacity."
        )

    elif row["risk_level"] == "HIGH":
        return (
            "Prioritize network review and monitor KPI degradation."
        )

    elif row["risk_level"] == "MEDIUM":
        return (
            "Monitor cell closely and review recent KPI trends."
        )

    else:
        return (
            "Continue standard monitoring."
        )


score_pdf["recommended_action"] = score_pdf.apply(
    recommended_action,
    axis=1
)


# =====================================================================
# 9. LOAD ISOLATION FOREST RESULTS
#
# Task 7 scored a representative sample.
# We attach anomaly information only where it actually exists.
# =====================================================================

anomaly_table = "workspace.default.network_anomaly_results"

anomaly_spark = (
    spark.table(anomaly_table)
    .select(
        "timestamp",
        "cell_id",
        "anomaly_score",
        "is_anomaly"
    )
)

anomaly_pdf = anomaly_spark.toPandas()


# Remove accidental duplicates before join
anomaly_pdf = anomaly_pdf.drop_duplicates(
    subset=["timestamp", "cell_id"]
)


# =====================================================================
# 10. COMBINE PREDICTIVE RISK + ANOMALY INTELLIGENCE
# =====================================================================

risk_pdf = score_pdf.merge(
    anomaly_pdf,
    on=["timestamp", "cell_id"],
    how="left"
)


# Honest handling:
# NULL means Isolation Forest did not score this record,
# not that it was definitely normal.

risk_pdf["anomaly_status"] = np.where(
    risk_pdf["is_anomaly"].isna(),
    "NOT_SCORED",
    np.where(
        risk_pdf["is_anomaly"] == 1,
        "ANOMALY",
        "NORMAL"
    )
)


# =====================================================================
# 11. CREATE COMBINED INTELLIGENCE FLAG
# =====================================================================

risk_pdf["intelligence_flag"] = np.select(

    [
        (
            risk_pdf["risk_level"].eq("CRITICAL") &
            risk_pdf["is_anomaly"].eq(1)
        ),

        risk_pdf["risk_level"].eq("CRITICAL"),

        risk_pdf["risk_level"].eq("HIGH"),

        risk_pdf["is_anomaly"].eq(1)
    ],

    [
        "CRITICAL + ANOMALOUS",
        "CRITICAL ML RISK",
        "HIGH ML RISK",
        "ANOMALOUS BEHAVIOR"
    ],

    default="STANDARD MONITORING"
)


# =====================================================================
# 12. CREATE PRIORITIZED ALERT QUEUE
# =====================================================================

alert_queue = (
    risk_pdf[
        risk_pdf["risk_level"].isin(
            ["HIGH", "CRITICAL"]
        )
    ]
    .sort_values(
        [
            "priority_code",
            "failure_probability",
            "anomaly_score"
        ],
        ascending=[
            True,
            False,
            False
        ],
        na_position="last"
    )
    .copy()
)

alert_queue["priority_rank"] = (
    np.arange(1, len(alert_queue) + 1)
)


# =====================================================================
# 13. DISPLAY RISK DISTRIBUTION
# =====================================================================

risk_distribution = (
    risk_pdf["risk_level"]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        fill_value=0
    )
)

print("\n" + "=" * 78)
print("FAILURE RISK DISTRIBUTION")
print("=" * 78)

print(risk_distribution)

print(
    "\nHighest predicted failure risk:",
    f"{risk_pdf['failure_risk_pct'].max():.2f}%"
)


# =====================================================================
# 14. DISPLAY TOP PRIORITY ALERTS
# =====================================================================

print("\n" + "=" * 78)
print("TOP 20 PRIORITY NETWORK ALERTS")
print("=" * 78)

display(
    alert_queue[
        [
            "priority_rank",
            "timestamp",
            "cell_id",
            "site_id",
            "region",
            "failure_risk_pct",
            "risk_level",
            "anomaly_status",
            "anomaly_score",
            "latency_ms",
            "packet_loss_pct",
            "throughput_mbps",
            "alarm_count",
            "recommended_action"
        ]
    ].head(20)
)


# =====================================================================
# 15. SAVE FULL RISK-SCORING OUTPUT TO DELTA
# =====================================================================

# Spark may infer categorical columns poorly,
# so ensure normal string types first.

risk_pdf["risk_level"] = (
    risk_pdf["risk_level"].astype(str)
)

risk_pdf["anomaly_status"] = (
    risk_pdf["anomaly_status"].astype(str)
)

risk_pdf["intelligence_flag"] = (
    risk_pdf["intelligence_flag"].astype(str)
)

risk_pdf["recommended_action"] = (
    risk_pdf["recommended_action"].astype(str)
)

risk_spark = spark.createDataFrame(risk_pdf)

(
    risk_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.default.network_risk_scores"
    )
)


# =====================================================================
# 16. SAVE ALERT QUEUE TO DELTA
# =====================================================================

alert_spark = spark.createDataFrame(alert_queue)

(
    alert_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.default.network_alert_queue"
    )
)


# =====================================================================
# 17. FINAL VALIDATION
# =====================================================================

risk_rows_saved = spark.table(
    "workspace.default.network_risk_scores"
).count()

alert_rows_saved = spark.table(
    "workspace.default.network_alert_queue"
).count()

critical_count = int(
    (risk_pdf["risk_level"] == "CRITICAL").sum()
)

high_count = int(
    (risk_pdf["risk_level"] == "HIGH").sum()
)

medium_count = int(
    (risk_pdf["risk_level"] == "MEDIUM").sum()
)

low_count = int(
    (risk_pdf["risk_level"] == "LOW").sum()
)

anomalous_count = int(
    (risk_pdf["anomaly_status"] == "ANOMALY").sum()
)


print("\n" + "=" * 78)
print("TASK 9 COMPLETE — OPERATIONAL AI OUTPUT READY")
print("=" * 78)

print(f"""
MLflow Model Run
----------------
Run ID              : {run_id}

Risk Scoring
------------
Records scored      : {len(risk_pdf):,}

LOW                  : {low_count:,}
MEDIUM               : {medium_count:,}
HIGH                 : {high_count:,}
CRITICAL             : {critical_count:,}

Known anomalies      : {anomalous_count:,}

Alert queue
-----------
HIGH + CRITICAL      : {len(alert_queue):,}

Delta Outputs
-------------
workspace.default.network_risk_scores
Rows saved           : {risk_rows_saved:,}

workspace.default.network_alert_queue
Rows saved           : {alert_rows_saved:,}


Operational Architecture
------------------------

Gold Network Telemetry
        ↓
Tracked MLflow XGBoost Model
        ↓
30-Minute Failure Probability
        ↓
Failure Risk %
        ↓
LOW / MEDIUM / HIGH / CRITICAL
        +
Isolation Forest Anomaly Signal
        ↓
Prioritized Network Alert Queue
        ↓
Power BI / Incident Intelligence
""")

print("=" * 78)