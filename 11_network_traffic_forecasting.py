# Databricks notebook source
# ================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 11 — NETWORK TRAFFIC / DEMAND FORECASTING
# Technology: Databricks + Spark SQL + Pandas + Prophet
# ================================================================

# ------------------------------------------------
# 1. INSTALL PROPHET IF REQUIRED
# ------------------------------------------------

import sys
import subprocess
import importlib.util

if importlib.util.find_spec("prophet") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "prophet", "-q"]
    )

from prophet import Prophet

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import TimestampType, DoubleType


print("=" * 75)
print("TASK 11 — 5G NETWORK TRAFFIC FORECASTING")
print("=" * 75)


# ================================================================
# 2. LOAD CLEAN SILVER TELEMETRY
# ================================================================

df = spark.table(
    "workspace.default.network_telemetry_silver"
)

print(f"\nTelemetry rows loaded: {df.count():,}")


# ================================================================
# 3. CREATE HOURLY NETWORK DEMAND SERIES
# ================================================================

hourly_spark = (
    df
    .withColumn(
        "hour_timestamp",
        F.date_trunc("hour", F.col("timestamp"))
    )
    .groupBy("hour_timestamp")
    .agg(

        F.avg("traffic_load_pct")
        .alias("avg_traffic_load_pct"),

        F.avg("connected_users")
        .alias("avg_connected_users"),

        F.avg("throughput_mbps")
        .alias("avg_throughput_mbps"),

        F.avg("latency_ms")
        .alias("avg_latency_ms"),

        F.avg("packet_loss_pct")
        .alias("avg_packet_loss_pct"),

        F.sum("failure_event")
        .alias("failure_events"),

        F.countDistinct("cell_id")
        .alias("active_cells")
    )
    .orderBy("hour_timestamp")
)


print("\nHourly demand data created.")

display(hourly_spark.limit(20))


# ================================================================
# 4. CONVERT SMALL AGGREGATED DATA TO PANDAS
# ================================================================
# Important:
# We use Spark for the ~500K telemetry rows.
# Only the much smaller hourly aggregate is brought into Pandas.

hourly_pdf = hourly_spark.toPandas()

hourly_pdf["hour_timestamp"] = pd.to_datetime(
    hourly_pdf["hour_timestamp"]
).dt.tz_localize(None)

hourly_pdf = hourly_pdf.sort_values(
    "hour_timestamp"
).dropna(
    subset=["hour_timestamp", "avg_traffic_load_pct"]
)


print(
    f"\nHourly observations available for forecasting: "
    f"{len(hourly_pdf):,}"
)


# ================================================================
# 5. PREPARE PROPHET DATA
# ================================================================

prophet_df = (
    hourly_pdf[
        ["hour_timestamp", "avg_traffic_load_pct"]
    ]
    .rename(
        columns={
            "hour_timestamp": "ds",
            "avg_traffic_load_pct": "y"
        }
    )
)


print("\nForecast training sample:")
print(prophet_df.head())


# ================================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ================================================================

if len(prophet_df) < 48:
    raise ValueError(
        "Not enough hourly observations for a meaningful forecast."
    )

test_hours = min(
    24,
    max(6, int(len(prophet_df) * 0.20))
)

train_df = prophet_df.iloc[:-test_hours].copy()
test_df = prophet_df.iloc[-test_hours:].copy()


print("\n" + "=" * 75)
print("TIME-BASED FORECAST VALIDATION")
print("=" * 75)

print(f"Training hours : {len(train_df):,}")
print(f"Testing hours  : {len(test_df):,}")


# ================================================================
# 7. TRAIN PROPHET FORECASTING MODEL
# ================================================================

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    interval_width=0.95
)

model.fit(train_df)


# ================================================================
# 8. PREDICT TEST PERIOD
# ================================================================

test_future = test_df[["ds"]].copy()

test_forecast = model.predict(test_future)

evaluation = test_df.merge(
    test_forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ],
    on="ds",
    how="left"
)


# ================================================================
# 9. FORECAST PERFORMANCE
# ================================================================

actual = evaluation["y"].values
predicted = evaluation["yhat"].values

mae = np.mean(
    np.abs(actual - predicted)
)

rmse = np.sqrt(
    np.mean((actual - predicted) ** 2)
)

mape_mask = actual != 0

mape = np.mean(
    np.abs(
        (actual[mape_mask] - predicted[mape_mask])
        / actual[mape_mask]
    )
) * 100


print("\n" + "=" * 75)
print("PROPHET VALIDATION RESULTS")
print("=" * 75)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"MAPE : {mape:.2f}%")


# ================================================================
# 10. RETRAIN USING ALL AVAILABLE HISTORY
# ================================================================

final_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    interval_width=0.95
)

final_model.fit(prophet_df)


# ================================================================
# 11. FORECAST NEXT 24 HOURS
# ================================================================

future = final_model.make_future_dataframe(
    periods=24,
    freq="h"
)

full_forecast = final_model.predict(future)

last_observed_time = prophet_df["ds"].max()

future_forecast = (
    full_forecast[
        full_forecast["ds"] > last_observed_time
    ][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]
    .copy()
)


# Network load is a percentage.
for column in ["yhat", "yhat_lower", "yhat_upper"]:
    future_forecast[column] = (
        future_forecast[column]
        .clip(lower=0, upper=100)
    )


future_forecast = future_forecast.rename(
    columns={
        "ds": "forecast_timestamp",
        "yhat": "predicted_traffic_load_pct",
        "yhat_lower": "forecast_lower_pct",
        "yhat_upper": "forecast_upper_pct"
    }
)


# ================================================================
# 12. CAPACITY-RISK CLASSIFICATION
# ================================================================

future_forecast["capacity_risk"] = np.select(

    [
        future_forecast[
            "predicted_traffic_load_pct"
        ] >= 85,

        future_forecast[
            "predicted_traffic_load_pct"
        ] >= 70
    ],

    [
        "CRITICAL",
        "HIGH"
    ],

    default="NORMAL"
)


print("\nNEXT 24-HOUR NETWORK LOAD FORECAST")

display(future_forecast)


# ================================================================
# 13. SAVE FORECAST AS DELTA TABLE
# ================================================================

forecast_schema = StructType([
    StructField(
        "forecast_timestamp",
        TimestampType(),
        False
    ),
    StructField(
        "predicted_traffic_load_pct",
        DoubleType(),
        True
    ),
    StructField(
        "forecast_lower_pct",
        DoubleType(),
        True
    ),
    StructField(
        "forecast_upper_pct",
        DoubleType(),
        True
    ),
    StructField(
        "capacity_risk",
        __import__(
            "pyspark.sql.types",
            fromlist=["StringType"]
        ).StringType(),
        True
    )
])


forecast_save = future_forecast.copy()

forecast_save[
    "forecast_timestamp"
] = pd.to_datetime(
    forecast_save["forecast_timestamp"]
)

forecast_spark = spark.createDataFrame(
    forecast_save,
    schema=forecast_schema
)


forecast_spark.write.mode(
    "overwrite"
).format(
    "delta"
).saveAsTable(
    "workspace.default.network_traffic_forecast"
)


# ================================================================
# 14. FINAL OPERATIONAL SUMMARY
# ================================================================

peak_row = future_forecast.loc[
    future_forecast[
        "predicted_traffic_load_pct"
    ].idxmax()
]

high_capacity_hours = (
    future_forecast["capacity_risk"]
    .isin(["HIGH", "CRITICAL"])
    .sum()
)


print("\n" + "=" * 75)
print("TASK 11 COMPLETE")
print("=" * 75)

print(
    f"""
Technology:
Databricks + Apache Spark + Pandas + Prophet + Delta Lake

Forecast Target:
Average 5G Network Traffic Load

Forecast Horizon:
Next 24 Hours

Validation:
MAE  = {mae:.3f}
RMSE = {rmse:.3f}
MAPE = {mape:.2f}%

Peak Forecast:
{peak_row['predicted_traffic_load_pct']:.2f}% network load

Peak Time:
{peak_row['forecast_timestamp']}

High/Critical Capacity Hours:
{high_capacity_hours}

Delta Output:
workspace.default.network_traffic_forecast

Operational Purpose:
Forecast future network demand and identify upcoming
capacity-pressure periods before service degradation occurs.

TASK 11 COMPLETE
"""
)

print("=" * 75)

# COMMAND ----------

# ================================================================
# 5G NETWORK FAILURE INTELLIGENCE
# TASK 11 — NETWORK TRAFFIC / DEMAND FORECASTING
#
# Technology:
# Databricks + Apache Spark + Pandas + Prophet + Delta Lake
#
# Purpose:
# Forecast the next 24 hours of average 5G network traffic load,
# validate forecast quality, classify capacity risk, save the
# forecast to Delta, and prepare the serving output for
# PostgreSQL / Power BI.
# ================================================================


# ================================================================
# 1. INSTALL / IMPORT LIBRARIES
# ================================================================

import sys
import subprocess
import importlib.util

if importlib.util.find_spec("prophet") is None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "prophet", "-q"]
    )

from prophet import Prophet

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    TimestampType,
    DoubleType,
    StringType
)

print("=" * 75)
print("TASK 11 — 5G NETWORK TRAFFIC FORECASTING")
print("=" * 75)


# ================================================================
# 2. LOAD CLEAN SILVER TELEMETRY
# ================================================================

df = spark.table(
    "workspace.default.network_telemetry_silver"
)

source_count = df.count()

print(f"\nTelemetry rows loaded: {source_count:,}")


# ================================================================
# 3. CREATE HOURLY NETWORK DEMAND SERIES
# ================================================================

hourly_spark = (
    df
    .withColumn(
        "hour_timestamp",
        F.date_trunc("hour", F.col("timestamp"))
    )
    .groupBy("hour_timestamp")
    .agg(
        F.avg("traffic_load_pct")
        .alias("avg_traffic_load_pct"),

        F.avg("connected_users")
        .alias("avg_connected_users"),

        F.avg("throughput_mbps")
        .alias("avg_throughput_mbps"),

        F.avg("latency_ms")
        .alias("avg_latency_ms"),

        F.avg("packet_loss_pct")
        .alias("avg_packet_loss_pct"),

        F.sum("failure_event")
        .alias("failure_events"),

        F.countDistinct("cell_id")
        .alias("active_cells")
    )
    .orderBy("hour_timestamp")
)

print("\nHourly demand dataset created.")

display(hourly_spark.limit(20))


# ================================================================
# 4. CONVERT AGGREGATED DATA TO PANDAS
# ================================================================
#
# Spark processes the large telemetry dataset.
# Only the small hourly aggregate is moved into Pandas because
# Prophet expects Pandas-compatible input.
# ================================================================

hourly_pdf = hourly_spark.toPandas()

hourly_pdf["hour_timestamp"] = pd.to_datetime(
    hourly_pdf["hour_timestamp"]
).dt.tz_localize(None)

hourly_pdf = (
    hourly_pdf
    .sort_values("hour_timestamp")
    .dropna(
        subset=[
            "hour_timestamp",
            "avg_traffic_load_pct"
        ]
    )
)

print(
    f"\nHourly observations available for forecasting: "
    f"{len(hourly_pdf):,}"
)


# ================================================================
# 5. PREPARE PROPHET DATA
# ================================================================
#
# Prophet requires:
#
# ds = timestamp
# y  = value being forecast
# ================================================================

prophet_df = (
    hourly_pdf[
        [
            "hour_timestamp",
            "avg_traffic_load_pct"
        ]
    ]
    .rename(
        columns={
            "hour_timestamp": "ds",
            "avg_traffic_load_pct": "y"
        }
    )
)

print("\nForecast training sample:")

display(prophet_df.head(10))


# ================================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ================================================================

if len(prophet_df) < 48:
    raise ValueError(
        "Not enough hourly observations for a meaningful forecast."
    )

test_hours = min(
    24,
    max(
        6,
        int(len(prophet_df) * 0.20)
    )
)

train_df = prophet_df.iloc[:-test_hours].copy()
test_df = prophet_df.iloc[-test_hours:].copy()

print("\n" + "=" * 75)
print("TIME-BASED FORECAST VALIDATION")
print("=" * 75)

print(f"Training hours : {len(train_df):,}")
print(f"Testing hours  : {len(test_df):,}")


# ================================================================
# 7. TRAIN PROPHET VALIDATION MODEL
# ================================================================

validation_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    interval_width=0.95
)

validation_model.fit(train_df)


# ================================================================
# 8. PREDICT TEST PERIOD
# ================================================================

test_future = test_df[
    ["ds"]
].copy()

test_forecast = validation_model.predict(
    test_future
)

evaluation = test_df.merge(
    test_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ],
    on="ds",
    how="left"
)


# ================================================================
# 9. CALCULATE FORECAST PERFORMANCE
# ================================================================

actual = evaluation["y"].values
predicted = evaluation["yhat"].values

mae = np.mean(
    np.abs(actual - predicted)
)

rmse = np.sqrt(
    np.mean(
        (actual - predicted) ** 2
    )
)

mape_mask = actual != 0

mape = (
    np.mean(
        np.abs(
            (
                actual[mape_mask]
                - predicted[mape_mask]
            )
            / actual[mape_mask]
        )
    )
    * 100
)

print("\n" + "=" * 75)
print("PROPHET VALIDATION RESULTS")
print("=" * 75)

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"MAPE : {mape:.2f}%")

print("\nValidation comparison:")

display(
    evaluation[
        [
            "ds",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ]
)


# ================================================================
# 10. RETRAIN MODEL USING ALL AVAILABLE HISTORY
# ================================================================
#
# Validation is complete.
# We now retrain Prophet using the complete historical dataset
# before producing the operational future forecast.
# ================================================================

final_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    interval_width=0.95
)

final_model.fit(prophet_df)


# ================================================================
# 11. FORECAST NEXT 24 HOURS
# ================================================================

future = final_model.make_future_dataframe(
    periods=24,
    freq="h"
)

full_forecast = final_model.predict(
    future
)

last_observed_time = prophet_df["ds"].max()

future_forecast = (
    full_forecast[
        full_forecast["ds"] > last_observed_time
    ][
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ]
    .copy()
)


# ================================================================
# 12. APPLY BUSINESS CONSTRAINTS
# ================================================================
#
# Network traffic load is a percentage.
# Predictions therefore remain between 0 and 100.
# ================================================================

for column in [
    "yhat",
    "yhat_lower",
    "yhat_upper"
]:
    future_forecast[column] = (
        future_forecast[column]
        .clip(
            lower=0,
            upper=100
        )
    )


# Rename ML fields into business-friendly fields.

future_forecast = future_forecast.rename(
    columns={
        "ds":
            "forecast_timestamp",

        "yhat":
            "predicted_traffic_load_pct",

        "yhat_lower":
            "forecast_lower_pct",

        "yhat_upper":
            "forecast_upper_pct"
    }
)


# ================================================================
# 13. CAPACITY-RISK CLASSIFICATION
# ================================================================
#
# Business interpretation:
#
# NORMAL   < 70%
# HIGH     >= 70%
# CRITICAL >= 85%
# ================================================================

future_forecast["capacity_risk"] = np.select(
    [
        future_forecast[
            "predicted_traffic_load_pct"
        ] >= 85,

        future_forecast[
            "predicted_traffic_load_pct"
        ] >= 70
    ],
    [
        "CRITICAL",
        "HIGH"
    ],
    default="NORMAL"
)


print("\n" + "=" * 75)
print("NEXT 24-HOUR NETWORK LOAD FORECAST")
print("=" * 75)

display(future_forecast)


# ================================================================
# 14. CREATE SPARK FORECAST DATAFRAME
# ================================================================

forecast_schema = StructType(
    [
        StructField(
            "forecast_timestamp",
            TimestampType(),
            False
        ),

        StructField(
            "predicted_traffic_load_pct",
            DoubleType(),
            True
        ),

        StructField(
            "forecast_lower_pct",
            DoubleType(),
            True
        ),

        StructField(
            "forecast_upper_pct",
            DoubleType(),
            True
        ),

        StructField(
            "capacity_risk",
            StringType(),
            True
        )
    ]
)

forecast_save = future_forecast.copy()

forecast_save["forecast_timestamp"] = pd.to_datetime(
    forecast_save["forecast_timestamp"]
)

forecast_spark = spark.createDataFrame(
    forecast_save,
    schema=forecast_schema
)


# ================================================================
# 15. SAVE FORECAST TO DELTA SERVING TABLE
# ================================================================

delta_table = (
    "workspace.default.network_traffic_forecast"
)

forecast_spark.write \
    .mode("overwrite") \
    .format("delta") \
    .saveAsTable(delta_table)

print(
    "\nDelta forecast table saved successfully:"
)

print(delta_table)


# ================================================================
# 16. VALIDATE DELTA OUTPUT
# ================================================================

saved_forecast = spark.table(
    delta_table
)

delta_count = saved_forecast.count()

print(
    f"\nDelta forecast rows: {delta_count:,}"
)

display(saved_forecast)


# ================================================================
# 17. PREPARE POSTGRESQL / POWER BI SERVING EXPORT
# ================================================================
#
# PostgreSQL expects:
#
# forecast_timestamp
# predicted_traffic_load_pct
# forecast_lower_bound
# forecast_upper_bound
#
# capacity_risk remains available in Databricks but is not
# currently part of the PostgreSQL forecast-table schema.
# ================================================================

forecast_export = future_forecast[
    [
        "forecast_timestamp",
        "predicted_traffic_load_pct",
        "forecast_lower_pct",
        "forecast_upper_pct"
    ]
].copy()

forecast_export = forecast_export.rename(
    columns={
        "forecast_lower_pct":
            "forecast_lower_bound",

        "forecast_upper_pct":
            "forecast_upper_bound"
    }
)

print(
    "\nPOSTGRESQL / POWER BI FORECAST EXPORT"
)

display(forecast_export)


# ================================================================
# 18. CONVERT EXPORT TO SPARK
# ================================================================

export_schema = StructType(
    [
        StructField(
            "forecast_timestamp",
            TimestampType(),
            False
        ),

        StructField(
            "predicted_traffic_load_pct",
            DoubleType(),
            True
        ),

        StructField(
            "forecast_lower_bound",
            DoubleType(),
            True
        ),

        StructField(
            "forecast_upper_bound",
            DoubleType(),
            True
        )
    ]
)

forecast_export["forecast_timestamp"] = pd.to_datetime(
    forecast_export["forecast_timestamp"]
)

forecast_export_spark = spark.createDataFrame(
    forecast_export,
    schema=export_schema
)


# ================================================================
# 19. EXPORT FORECAST AS CSV
# ================================================================

export_path = (
    "/FileStore/5g_network_forecast_export"
)

forecast_export_spark \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(export_path)

export_count = forecast_export_spark.count()

print("\nForecast export completed.")

print(
    f"Export rows: {export_count:,}"
)

print(
    f"Export path: {export_path}"
)


# ================================================================
# 20. OPERATIONAL FORECAST SUMMARY
# ================================================================

peak_row = future_forecast.loc[
    future_forecast[
        "predicted_traffic_load_pct"
    ].idxmax()
]

high_capacity_hours = (
    future_forecast[
        "capacity_risk"
    ]
    .isin(
        [
            "HIGH",
            "CRITICAL"
        ]
    )
    .sum()
)

average_forecast_load = (
    future_forecast[
        "predicted_traffic_load_pct"
    ].mean()
)

critical_hours = (
    future_forecast[
        "capacity_risk"
    ]
    .eq("CRITICAL")
    .sum()
)

high_hours = (
    future_forecast[
        "capacity_risk"
    ]
    .eq("HIGH")
    .sum()
)


# ================================================================
# 21. FINAL VALIDATION
# ================================================================

if len(future_forecast) != 24:
    raise ValueError(
        "Expected exactly 24 future forecast rows."
    )

if delta_count != 24:
    raise ValueError(
        "Delta forecast table should contain 24 rows."
    )

if export_count != 24:
    raise ValueError(
        "PostgreSQL export should contain 24 rows."
    )


# ================================================================
# 22. FINAL PROJECT OUTPUT
# ================================================================

print("\n" + "=" * 75)
print("TASK 11 COMPLETE")
print("=" * 75)

print(
    f"""
TECHNOLOGY
Databricks
Apache Spark
Spark SQL
Pandas
Prophet
Delta Lake

FORECAST TARGET
Average 5G Network Traffic Load

FORECAST HORIZON
Next 24 Hours

MODEL VALIDATION
MAE  : {mae:.3f}
RMSE : {rmse:.3f}
MAPE : {mape:.2f}%

FORECAST SUMMARY
Average Forecast Load :
{average_forecast_load:.2f}%

Peak Forecast :
{peak_row['predicted_traffic_load_pct']:.2f}%

Peak Time :
{peak_row['forecast_timestamp']}

High Capacity Hours :
{high_hours}

Critical Capacity Hours :
{critical_hours}

Total High/Critical Hours :
{high_capacity_hours}

DELTA SERVING TABLE
{delta_table}

DELTA ROWS
{delta_count}

POSTGRESQL / POWER BI EXPORT
{export_path}

EXPORT ROWS
{export_count}

OPERATIONAL PURPOSE
Forecast future 5G network demand and identify upcoming
capacity-pressure periods before network performance
degradation occurs.

STATUS
TASK 11 SUCCESSFULLY COMPLETED
"""
)

print("=" * 75)

# COMMAND ----------

# Load the already-saved forecast table
forecast_export = spark.table(
    "workspace.default.network_traffic_forecast"
).select(
    "forecast_timestamp",
    "predicted_traffic_load_pct",
    "forecast_lower_pct",
    "forecast_upper_pct",
    "capacity_risk"
)

print("Forecast rows:", forecast_export.count())

display(forecast_export)