# Databricks notebook source
# ================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 3 — DATABRICKS + PYSPARK DATA ENGINEERING
# ================================================================

from pyspark.sql import functions as F
from pyspark.sql.types import *

print("=" * 70)
print("5G NETWORK FAILURE INTELLIGENCE")
print("DATABRICKS + PYSPARK DATA ENGINEERING")
print("=" * 70)


# ================================================================
# 1. RAW DATA LOCATION — UNITY CATALOG VOLUME
# ================================================================

raw_path = (
    "/Volumes/workspace/default/5g_network_data/"
    "5g_network_telemetry_raw.csv"
)

print("\nReading raw telemetry from:")
print(raw_path)


# ================================================================
# 2. LOAD RAW CSV USING PYSPARK
# ================================================================

df_raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("mode", "PERMISSIVE")
    .csv(raw_path)
)


# ================================================================
# 3. BASIC DATASET INFORMATION
# ================================================================

row_count = df_raw.count()
column_count = len(df_raw.columns)

print("\n" + "=" * 70)
print("RAW DATASET INFORMATION")
print("=" * 70)

print(f"Rows:    {row_count:,}")
print(f"Columns: {column_count}")

print("\nColumns:")
for col_name in df_raw.columns:
    print(" -", col_name)


# ================================================================
# 4. PYSPARK SCHEMA
# ================================================================

print("\n" + "=" * 70)
print("PYSPARK INFERRED SCHEMA")
print("=" * 70)

df_raw.printSchema()


# ================================================================
# 5. MISSING-VALUE PROFILING
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUE PROFILE")
print("=" * 70)

missing_expr = [
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df_raw.columns
]

missing_profile = df_raw.select(missing_expr)

display(missing_profile)


# ================================================================
# 6. EXACT DUPLICATE PROFILING
# ================================================================

distinct_count = df_raw.dropDuplicates().count()
duplicate_count = row_count - distinct_count

print("\n" + "=" * 70)
print("DUPLICATE PROFILE")
print("=" * 70)

print(f"Total rows:       {row_count:,}")
print(f"Distinct rows:    {distinct_count:,}")
print(f"Exact duplicates: {duplicate_count:,}")


# ================================================================
# 7. REGION QUALITY CHECK
# ================================================================

print("\n" + "=" * 70)
print("REGION DISTRIBUTION")
print("=" * 70)

region_profile = (
    df_raw
    .groupBy("region")
    .count()
    .orderBy(F.desc("count"))
)

display(region_profile)


# ================================================================
# 8. FAILURE TARGET DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("FAILURE EVENT DISTRIBUTION")
print("=" * 70)

failure_distribution = (
    df_raw
    .groupBy("failure_event")
    .count()
    .orderBy("failure_event")
)

display(failure_distribution)


# ================================================================
# 9. FAILURE RATE
# ================================================================

failure_count = (
    df_raw
    .filter(F.col("failure_event") == 1)
    .count()
)

failure_rate = (failure_count / row_count) * 100

print(f"\nFailure Events: {failure_count:,}")
print(f"Failure Rate:   {failure_rate:.3f}%")


# ================================================================
# 10. NETWORK ENTITY PROFILE
# ================================================================

unique_cells = df_raw.select("cell_id").distinct().count()
unique_sites = df_raw.select("site_id").distinct().count()

print("\n" + "=" * 70)
print("NETWORK ENTITY PROFILE")
print("=" * 70)

print(f"Unique Cells: {unique_cells:,}")
print(f"Unique Sites: {unique_sites:,}")


# ================================================================
# 11. NUMERICAL NETWORK KPI SUMMARY
# ================================================================

kpi_columns = [
    "signal_strength_dbm",
    "sinr_db",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps",
    "connected_users",
    "traffic_load_pct",
    "cpu_utilization_pct",
    "alarm_count"
]

existing_kpis = [c for c in kpi_columns if c in df_raw.columns]

print("\n" + "=" * 70)
print("NETWORK KPI STATISTICAL PROFILE")
print("=" * 70)

df_raw.select(existing_kpis).summary().show(
    truncate=False
)


# ================================================================
# 12. SAMPLE RECORDS
# ================================================================

print("\n" + "=" * 70)
print("SAMPLE RAW TELEMETRY RECORDS")
print("=" * 70)

display(df_raw.limit(10))


# ================================================================
# FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("TASK 3 SUCCESSFULLY COMPLETED")
print("=" * 70)

print(f"""
Platform      : Databricks Free Edition
Processing    : Apache Spark / PySpark
Storage       : Unity Catalog Managed Volume
Dataset       : 5G Network Telemetry
Rows          : {row_count:,}
Columns       : {column_count}
Failures      : {failure_count:,}
Failure Rate  : {failure_rate:.3f}%

RAW DATA SUCCESSFULLY LOADED AND PROFILED USING PYSPARK.
READY FOR DISTRIBUTED DATA QUALITY + TRANSFORMATION.
""")

print("=" * 70)

# COMMAND ----------

# =====================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 3 — PYSPARK CLEANING + TRANSFORMATION + DELTA LAKE
# =====================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=" * 75)
print("TASK 3 — PYSPARK DATA CLEANING & TRANSFORMATION")
print("=" * 75)

# ---------------------------------------------------------------------
# 1. START FROM THE RAW DATAFRAME ALREADY LOADED
# ---------------------------------------------------------------------

df = df_raw

rows_before = df.count()

print(f"\nRaw rows received: {rows_before:,}")


# ---------------------------------------------------------------------
# 2. REMOVE EXACT DUPLICATES
# ---------------------------------------------------------------------

df = df.dropDuplicates()

rows_after_duplicates = df.count()
duplicates_removed = rows_before - rows_after_duplicates

print(f"Exact duplicates removed: {duplicates_removed:,}")


# ---------------------------------------------------------------------
# 3. STANDARDIZE REGION NAMES
# ---------------------------------------------------------------------

df = df.withColumn(
    "region",
    F.when(
        F.lower(F.trim(F.col("region"))) == "houston",
        "Houston"
    )
    .when(
        F.upper(F.trim(F.col("region"))).isin("FT WORTH", "FORT WORTH"),
        "Fort Worth"
    )
    .otherwise(F.trim(F.col("region")))
)


# ---------------------------------------------------------------------
# 4. PARSE AND VALIDATE TIMESTAMP
# ---------------------------------------------------------------------

df = df.withColumn(
    "timestamp",
    F.to_timestamp("timestamp")
)

invalid_timestamp_count = (
    df.filter(F.col("timestamp").isNull()).count()
)

df = df.filter(F.col("timestamp").isNotNull())


# ---------------------------------------------------------------------
# 5. VALIDATE TELECOM CELL AND SITE IDENTIFIERS
# Expected examples:
# CELL_0001
# SITE_0001
# ---------------------------------------------------------------------

invalid_cell_condition = (
    F.col("cell_id").isNull() |
    (~F.col("cell_id").rlike(r"^CELL_[0-9]{4}$"))
)

invalid_site_condition = (
    F.col("site_id").isNull() |
    (~F.col("site_id").rlike(r"^SITE_[0-9]{4}$"))
)

invalid_cell_count = df.filter(invalid_cell_condition).count()
invalid_site_count = df.filter(invalid_site_condition).count()

df = df.filter(
    (~invalid_cell_condition) &
    (~invalid_site_condition)
)


# ---------------------------------------------------------------------
# 6. REMOVE IMPOSSIBLE KPI VALUES
#
# We don't blindly clip impossible measurements because that would
# fabricate network measurements. Invalid records are excluded.
# ---------------------------------------------------------------------

valid_kpi_condition = (
    F.col("traffic_load_pct").between(0, 100) &
    F.col("cpu_utilization_pct").between(0, 100) &
    (F.col("connected_users") >= 0) &
    (F.col("throughput_mbps") >= 0) &
    (F.col("latency_ms") >= 0) &
    (F.col("latency_ms") <= 500) &
    (F.col("packet_loss_pct") >= 0) &
    (F.col("packet_loss_pct") <= 100) &
    F.col("failure_event").isin(0, 1)
)

invalid_kpi_count = df.filter(~valid_kpi_condition).count()

df = df.filter(valid_kpi_condition)


# ---------------------------------------------------------------------
# 7. HANDLE REMAINING NUMERIC NULL VALUES
#
# Median is more robust than mean for network KPI data containing
# spikes/outliers.
# ---------------------------------------------------------------------

numeric_columns = [
    "signal_strength_dbm",
    "sinr_db",
    "latency_ms",
    "packet_loss_pct",
    "throughput_mbps"
]

for column_name in numeric_columns:

    if column_name in df.columns:

        median_value = df.approxQuantile(
            column_name,
            [0.5],
            0.01
        )[0]

        df = df.fillna({
            column_name: median_value
        })


# ---------------------------------------------------------------------
# 8. CREATE ANALYTICAL TIME FEATURES
# ---------------------------------------------------------------------

df = (
    df
    .withColumn("hour_of_day", F.hour("timestamp"))
    .withColumn("day_of_week", F.date_format("timestamp", "EEEE"))
    .withColumn("event_date", F.to_date("timestamp"))

    .withColumn(
        "time_period",
        F.when(
            (F.col("hour_of_day") >= 6) &
            (F.col("hour_of_day") < 12),
            "Morning"
        )
        .when(
            (F.col("hour_of_day") >= 12) &
            (F.col("hour_of_day") < 17),
            "Afternoon"
        )
        .when(
            (F.col("hour_of_day") >= 17) &
            (F.col("hour_of_day") < 22),
            "Evening"
        )
        .otherwise("Night")
    )
)


# ---------------------------------------------------------------------
# 9. CREATE NETWORK OPERATIONS FEATURES
# ---------------------------------------------------------------------

df = df.withColumn(
    "high_load_flag",
    F.when(F.col("traffic_load_pct") >= 80, 1).otherwise(0)
)

df = df.withColumn(
    "high_latency_flag",
    F.when(F.col("latency_ms") >= 100, 1).otherwise(0)
)

df = df.withColumn(
    "packet_loss_flag",
    F.when(F.col("packet_loss_pct") >= 2, 1).otherwise(0)
)

df = df.withColumn(
    "network_health_flag",
    F.when(
        (F.col("latency_ms") >= 100) |
        (F.col("packet_loss_pct") >= 2) |
        (F.col("traffic_load_pct") >= 90) |
        (F.col("cpu_utilization_pct") >= 90),
        "DEGRADED"
    ).otherwise("NORMAL")
)


# ---------------------------------------------------------------------
# 10. FINAL DATA QUALITY VALIDATION
# ---------------------------------------------------------------------

final_rows = df.count()

remaining_duplicates = (
    final_rows - df.dropDuplicates().count()
)

missing_expressions = [
    F.sum(
        F.when(F.col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in df.columns
]

missing_result = (
    df.select(missing_expressions)
      .first()
      .asDict()
)

total_missing = sum(
    value or 0
    for value in missing_result.values()
)

failure_count_clean = (
    df.filter(F.col("failure_event") == 1).count()
)

failure_rate_clean = (
    failure_count_clean / final_rows * 100
)


# ---------------------------------------------------------------------
# 11. SAVE AS DELTA LAKE TABLE
# ---------------------------------------------------------------------

delta_table = "workspace.default.network_telemetry_silver"

(
    df.write
      .format("delta")
      .mode("overwrite")
      .option("overwriteSchema", "true")
      .saveAsTable(delta_table)
)


# ---------------------------------------------------------------------
# 12. VERIFY DELTA TABLE
# ---------------------------------------------------------------------

df_delta = spark.table(delta_table)

delta_rows = df_delta.count()


# ---------------------------------------------------------------------
# 13. FINAL ENGINEERING REPORT
# ---------------------------------------------------------------------

print("\n" + "=" * 75)
print("PYSPARK CLEANING RESULTS")
print("=" * 75)

print(f"""
Raw rows                     : {rows_before:,}
Exact duplicates removed     : {duplicates_removed:,}

Invalid timestamps detected  : {invalid_timestamp_count:,}
Invalid Cell IDs detected    : {invalid_cell_count:,}
Invalid Site IDs detected    : {invalid_site_count:,}
Invalid KPI records detected : {invalid_kpi_count:,}

Final rows                    : {final_rows:,}
Final columns                 : {len(df.columns)}
Remaining missing values      : {total_missing:,}
Remaining exact duplicates    : {remaining_duplicates:,}

Failure events                : {failure_count_clean:,}
Failure rate                  : {failure_rate_clean:.3f}%
""")

print("=" * 75)
print("REGION VALIDATION")
print("=" * 75)

df.groupBy("region") \
  .count() \
  .orderBy(F.desc("count")) \
  .show(truncate=False)


print("=" * 75)
print("DELTA LAKE VALIDATION")
print("=" * 75)

print(f"Delta table : {delta_table}")
print(f"Delta rows  : {delta_rows:,}")

if (
    total_missing == 0
    and remaining_duplicates == 0
    and delta_rows == final_rows
):

    print("\n✓ DATA QUALITY VALIDATION PASSED")
    print("✓ PYSPARK TRANSFORMATION PASSED")
    print("✓ DELTA LAKE WRITE PASSED")

else:

    print("\n⚠ REVIEW DATA QUALITY RESULTS")


print("\n" + "=" * 75)
print("TASK 3 COMPLETE")
print("=" * 75)

print("""
RAW TELEMETRY
      ↓
UNITY CATALOG VOLUME
      ↓
DATABRICKS
      ↓
PYSPARK DISTRIBUTED TRANSFORMATION
      ↓
DATA QUALITY VALIDATION
      ↓
FEATURE ENGINEERING
      ↓
DELTA LAKE SILVER TABLE

Official clean engineering dataset:
workspace.default.network_telemetry_silver
""")

print("=" * 75)

display(df.limit(10))
