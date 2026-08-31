# Databricks notebook source
# ================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 7 — AI ANOMALY DETECTION
# Databricks + Spark + Isolation Forest
# ================================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------
# 1. LOAD OUR EXISTING SILVER DATA
# ------------------------------------------------

spark_df = spark.table(
    "workspace.default.network_telemetry_silver"
)

print("=" * 75)
print("TASK 7 — NETWORK ANOMALY DETECTION")
print("=" * 75)

print(f"\nSilver rows: {spark_df.count():,}")


# ------------------------------------------------
# 2. SELECT NETWORK HEALTH FEATURES
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
    "alarm_count"
]

# Keep identifiers so we know WHERE anomalies occur
columns_needed = [
    "timestamp",
    "cell_id",
    "site_id",
    "region",
    "failure_event"
] + features

df_model = spark_df.select(*columns_needed).dropna()


# ------------------------------------------------
# 3. SAMPLE DATA FOR FREE-EDITION COMPUTE
# ------------------------------------------------

# Isolation Forest is being demonstrated on a representative
# sample to keep Databricks Free Edition resource usage reasonable.

sample_fraction = min(1.0, 100000 / df_model.count())

sample_spark = df_model.sample(
    withReplacement=False,
    fraction=sample_fraction,
    seed=42
)

pdf = sample_spark.toPandas()

# Cap at exactly 100K if sampling returns slightly more
if len(pdf) > 100000:
    pdf = pdf.sample(100000, random_state=42)

print(f"Rows used for anomaly detection: {len(pdf):,}")


# ------------------------------------------------
# 4. PREPARE FEATURES
# ------------------------------------------------

X = pdf[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------
# 5. TRAIN ISOLATION FOREST
# ------------------------------------------------

isolation_model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

pdf["anomaly_prediction"] = isolation_model.fit_predict(X_scaled)

# sklearn returns:
#  1 = normal
# -1 = anomaly

pdf["is_anomaly"] = (
    pdf["anomaly_prediction"] == -1
).astype(int)


# ------------------------------------------------
# 6. CREATE ANOMALY SCORE
# ------------------------------------------------

raw_scores = -isolation_model.score_samples(X_scaled)

score_min = raw_scores.min()
score_max = raw_scores.max()

pdf["anomaly_score"] = (
    (raw_scores - score_min) /
    (score_max - score_min)
)

pdf["anomaly_score"] = (
    pdf["anomaly_score"] * 100
).round(2)


# ------------------------------------------------
# 7. RESULTS
# ------------------------------------------------

total_records = len(pdf)
total_anomalies = int(pdf["is_anomaly"].sum())

anomaly_rate = (
    total_anomalies / total_records * 100
)

print("\n" + "=" * 75)
print("ANOMALY DETECTION RESULTS")
print("=" * 75)

print(f"""
Records analyzed : {total_records:,}
Anomalies found  : {total_anomalies:,}
Anomaly rate     : {anomaly_rate:.2f}%
""")


# ------------------------------------------------
# 8. ANOMALIES BY REGION
# ------------------------------------------------

region_summary = (
    pdf.groupby("region")
       .agg(
           records=("cell_id", "count"),
           anomalies=("is_anomaly", "sum")
       )
       .reset_index()
)

region_summary["anomaly_rate_pct"] = (
    region_summary["anomalies"] /
    region_summary["records"] * 100
).round(2)

region_summary = region_summary.sort_values(
    "anomaly_rate_pct",
    ascending=False
)

print("\nANOMALIES BY REGION")
display(region_summary)


# ------------------------------------------------
# 9. CHECK RELATIONSHIP WITH REAL FAILURES
# ------------------------------------------------

failure_validation = pd.crosstab(
    pdf["is_anomaly"],
    pdf["failure_event"],
    margins=True
)

print("\nANOMALY vs ACTUAL FAILURE")
display(failure_validation)


# ------------------------------------------------
# 10. TOP MOST ABNORMAL NETWORK RECORDS
# ------------------------------------------------

top_anomalies = (
    pdf[pdf["is_anomaly"] == 1]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    [
        [
            "timestamp",
            "cell_id",
            "site_id",
            "region",
            "anomaly_score",
            "latency_ms",
            "packet_loss_pct",
            "throughput_mbps",
            "traffic_load_pct",
            "signal_strength_dbm",
            "alarm_count",
            "failure_event"
        ]
    ]
    .head(20)
)

print("\nTOP 20 NETWORK ANOMALIES")
display(top_anomalies)


# ------------------------------------------------
# 11. COMPARE NORMAL vs ANOMALOUS NETWORK BEHAVIOR
# ------------------------------------------------

behavior_comparison = (
    pdf.groupby("is_anomaly")
       [
           [
               "latency_ms",
               "packet_loss_pct",
               "throughput_mbps",
               "traffic_load_pct",
               "signal_strength_dbm",
               "alarm_count"
           ]
       ]
       .mean()
       .round(2)
       .reset_index()
)

behavior_comparison["network_behavior"] = (
    behavior_comparison["is_anomaly"]
    .map({
        0: "NORMAL",
        1: "ANOMALOUS"
    })
)

print("\nNORMAL vs ANOMALOUS NETWORK BEHAVIOR")
display(behavior_comparison)


# ------------------------------------------------
# 12. SAVE RESULTS AS DELTA TABLE
# ------------------------------------------------

output_columns = [
    "timestamp",
    "cell_id",
    "site_id",
    "region",
    "failure_event",
    "anomaly_score",
    "is_anomaly"
] + features

anomaly_output = pdf[output_columns].copy()

anomaly_spark = spark.createDataFrame(anomaly_output)

(
    anomaly_spark.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.default.network_anomaly_results"
    )
)


# ------------------------------------------------
# 13. FINAL VALIDATION
# ------------------------------------------------

saved_rows = spark.table(
    "workspace.default.network_anomaly_results"
).count()

print("\n" + "=" * 75)
print("TASK 7 COMPLETE")
print("=" * 75)

print(f"""
Technology      : Databricks + Spark + Scikit-learn
Algorithm       : Isolation Forest
Input           : network_telemetry_silver
Records analyzed: {total_records:,}
Anomalies       : {total_anomalies:,}
Anomaly rate    : {anomaly_rate:.2f}%
Saved rows      : {saved_rows:,}

Output Delta Table:
workspace.default.network_anomaly_results

PURPOSE:
Identify abnormal 5G network behavior independently of the
supervised failure-prediction model.
""")

print("=" * 75)