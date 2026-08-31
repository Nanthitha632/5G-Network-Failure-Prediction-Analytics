-- Databricks notebook source

-- ================================================================
-- 5G NETWORK FAILURE INTELLIGENCE
-- TASK 4 — SQL ANALYTICS + GOLD LAYER
-- Source: Silver Delta Table created in Task 3
-- ================================================================


-- ================================================================
-- 1. CREATE GOLD ANALYTICAL TABLE
-- ================================================================

CREATE OR REPLACE TABLE workspace.default.network_telemetry_gold
USING DELTA
AS

SELECT
    timestamp,
    event_date,
    hour_of_day,
    day_of_week,
    time_period,

    cell_id,
    site_id,
    region,

    signal_strength_dbm,
    sinr_db,
    latency_ms,
    packet_loss_pct,
    throughput_mbps,
    connected_users,
    traffic_load_pct,
    cpu_utilization_pct,
    alarm_count,

    high_load_flag,
    high_latency_flag,
    packet_loss_flag,
    network_health_flag,

    failure_event,

    -- ------------------------------------------------------------
    -- Operational severity score
    -- ------------------------------------------------------------

    CASE
        WHEN latency_ms >= 150
          OR packet_loss_pct >= 5
          OR alarm_count >= 8
        THEN 3

        WHEN latency_ms >= 100
          OR packet_loss_pct >= 2
          OR traffic_load_pct >= 90
        THEN 2

        WHEN latency_ms >= 70
          OR traffic_load_pct >= 80
        THEN 1

        ELSE 0
    END AS operational_severity_score,


    -- ------------------------------------------------------------
    -- Human-readable operational status
    -- ------------------------------------------------------------

    CASE
        WHEN failure_event = 1 THEN 'FAILURE'

        WHEN latency_ms >= 150
          OR packet_loss_pct >= 5
          OR alarm_count >= 8
        THEN 'CRITICAL'

        WHEN latency_ms >= 100
          OR packet_loss_pct >= 2
          OR traffic_load_pct >= 90
        THEN 'HIGH RISK'

        WHEN latency_ms >= 70
          OR traffic_load_pct >= 80
        THEN 'WATCH'

        ELSE 'HEALTHY'
    END AS operational_status


FROM workspace.default.network_telemetry_silver;


-- ================================================================
-- 2. VALIDATE GOLD TABLE
-- ================================================================

SELECT
    COUNT(*) AS total_records,
    COUNT(DISTINCT cell_id) AS total_cells,
    COUNT(DISTINCT site_id) AS total_sites,
    COUNT(DISTINCT region) AS total_regions,

    SUM(failure_event) AS total_failures,

    ROUND(
        100.0 * SUM(failure_event) / COUNT(*),
        3
    ) AS failure_rate_pct,

    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,

    ROUND(AVG(packet_loss_pct), 3) AS avg_packet_loss_pct,

    ROUND(AVG(throughput_mbps), 2) AS avg_throughput_mbps

FROM workspace.default.network_telemetry_gold;


-- ================================================================
-- 3. REGIONAL FAILURE INTELLIGENCE
-- ================================================================

SELECT
    region,

    COUNT(*) AS telemetry_records,

    COUNT(DISTINCT cell_id) AS cells,

    SUM(failure_event) AS failures,

    ROUND(
        100.0 * SUM(failure_event) / COUNT(*),
        3
    ) AS failure_rate_pct,

    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,

    ROUND(AVG(packet_loss_pct), 3) AS avg_packet_loss_pct,

    ROUND(AVG(throughput_mbps), 2) AS avg_throughput_mbps,

    ROUND(AVG(traffic_load_pct), 2) AS avg_traffic_load_pct

FROM workspace.default.network_telemetry_gold

GROUP BY region

ORDER BY failure_rate_pct DESC;


-- ================================================================
-- 4. TIME-PERIOD FAILURE INTELLIGENCE
-- ================================================================

SELECT
    time_period,

    COUNT(*) AS telemetry_records,

    SUM(failure_event) AS failures,

    ROUND(
        100.0 * SUM(failure_event) / COUNT(*),
        3
    ) AS failure_rate_pct,

    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,

    ROUND(AVG(traffic_load_pct), 2) AS avg_traffic_load_pct

FROM workspace.default.network_telemetry_gold

GROUP BY time_period

ORDER BY failure_rate_pct DESC;


-- ================================================================
-- 5. OPERATIONAL STATUS DISTRIBUTION
-- ================================================================

SELECT
    operational_status,

    COUNT(*) AS records,

    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_network

FROM workspace.default.network_telemetry_gold

GROUP BY operational_status

ORDER BY records DESC;


-- ================================================================
-- 6. TOP HIGH-RISK CELLS
-- ================================================================

SELECT
    cell_id,
    site_id,
    region,

    COUNT(*) AS observations,

    SUM(failure_event) AS failures,

    ROUND(
        100.0 * SUM(failure_event) / COUNT(*),
        3
    ) AS historical_failure_rate_pct,

    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,

    ROUND(AVG(packet_loss_pct), 3) AS avg_packet_loss_pct,

    ROUND(AVG(throughput_mbps), 2) AS avg_throughput_mbps,

    ROUND(AVG(traffic_load_pct), 2) AS avg_traffic_load_pct,

    ROUND(AVG(alarm_count), 2) AS avg_alarm_count

FROM workspace.default.network_telemetry_gold

GROUP BY
    cell_id,
    site_id,
    region

HAVING COUNT(*) >= 100

ORDER BY
    historical_failure_rate_pct DESC,
    failures DESC

LIMIT 20;


-- ================================================================
-- TASK 4 COMPLETE
-- ================================================================
