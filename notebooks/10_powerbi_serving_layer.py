# Databricks notebook source
# MAGIC %sql
# MAGIC -- ================================================================
# MAGIC -- 5G NETWORK FAILURE INTELLIGENCE
# MAGIC -- TASK 10 — POWER BI SERVING LAYER
# MAGIC -- Databricks SQL + Delta Lake
# MAGIC -- ================================================================
# MAGIC
# MAGIC
# MAGIC -- ================================================================
# MAGIC -- 1. CREATE POWER BI REPORTING VIEW
# MAGIC -- ================================================================
# MAGIC
# MAGIC CREATE OR REPLACE VIEW workspace.default.vw_5g_network_command_center AS
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- TIME
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.timestamp,
# MAGIC     DATE(r.timestamp) AS event_date,
# MAGIC     HOUR(r.timestamp) AS hour_of_day,
# MAGIC     r.time_period,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- NETWORK IDENTIFIERS
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.cell_id,
# MAGIC     r.site_id,
# MAGIC     r.region,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- CORE NETWORK KPIs
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.signal_strength_dbm,
# MAGIC     r.sinr_db,
# MAGIC     r.latency_ms,
# MAGIC     r.packet_loss_pct,
# MAGIC     r.throughput_mbps,
# MAGIC     r.traffic_load_pct,
# MAGIC     r.connected_users,
# MAGIC     r.cpu_utilization_pct,
# MAGIC     r.alarm_count,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- AI FAILURE INTELLIGENCE
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.failure_probability,
# MAGIC     r.failure_risk_pct,
# MAGIC     r.risk_level,
# MAGIC     r.priority_code,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- ANOMALY INTELLIGENCE
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.anomaly_status,
# MAGIC     r.anomaly_score,
# MAGIC     r.intelligence_flag,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- OPERATIONAL CONTEXT
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     r.failure_event,
# MAGIC     r.recommended_action,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- DASHBOARD-FRIENDLY FLAGS
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN r.risk_level = 'CRITICAL'
# MAGIC         THEN 1
# MAGIC         ELSE 0
# MAGIC     END AS critical_risk_flag,
# MAGIC
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN r.risk_level IN ('HIGH', 'CRITICAL')
# MAGIC         THEN 1
# MAGIC         ELSE 0
# MAGIC     END AS high_critical_flag,
# MAGIC
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN r.anomaly_status = 'ANOMALY'
# MAGIC         THEN 1
# MAGIC         ELSE 0
# MAGIC     END AS anomaly_flag,
# MAGIC
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN r.failure_risk_pct >= 75
# MAGIC         THEN 'Immediate Attention'
# MAGIC
# MAGIC         WHEN r.failure_risk_pct >= 50
# MAGIC         THEN 'Priority Review'
# MAGIC
# MAGIC         WHEN r.failure_risk_pct >= 25
# MAGIC         THEN 'Monitor'
# MAGIC
# MAGIC         ELSE 'Healthy'
# MAGIC     END AS operational_attention,
# MAGIC
# MAGIC
# MAGIC     -- ------------------------------------------------------------
# MAGIC     -- NETWORK HEALTH CATEGORY
# MAGIC     -- ------------------------------------------------------------
# MAGIC
# MAGIC     CASE
# MAGIC         WHEN r.packet_loss_pct >= 5
# MAGIC           OR r.latency_ms >= 150
# MAGIC         THEN 'SEVERE'
# MAGIC
# MAGIC         WHEN r.packet_loss_pct >= 2
# MAGIC           OR r.latency_ms >= 100
# MAGIC         THEN 'DEGRADED'
# MAGIC
# MAGIC         WHEN r.packet_loss_pct >= 1
# MAGIC           OR r.latency_ms >= 70
# MAGIC         THEN 'WATCH'
# MAGIC
# MAGIC         ELSE 'HEALTHY'
# MAGIC     END AS network_health_status
# MAGIC
# MAGIC
# MAGIC FROM workspace.default.network_risk_scores r;
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ================================================================
# MAGIC -- 2. VERIFY POWER BI VIEW
# MAGIC -- ================================================================
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     COUNT(*) AS total_records,
# MAGIC
# MAGIC     COUNT(DISTINCT cell_id) AS total_cells,
# MAGIC
# MAGIC     COUNT(DISTINCT site_id) AS total_sites,
# MAGIC
# MAGIC     COUNT(DISTINCT region) AS total_regions,
# MAGIC
# MAGIC     SUM(critical_risk_flag) AS critical_records,
# MAGIC
# MAGIC     SUM(high_critical_flag) AS high_critical_records,
# MAGIC
# MAGIC     SUM(anomaly_flag) AS anomaly_records,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(failure_risk_pct),
# MAGIC         2
# MAGIC     ) AS avg_failure_risk_pct,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(latency_ms),
# MAGIC         2
# MAGIC     ) AS avg_latency_ms,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(packet_loss_pct),
# MAGIC         2
# MAGIC     ) AS avg_packet_loss_pct,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(throughput_mbps),
# MAGIC         2
# MAGIC     ) AS avg_throughput_mbps
# MAGIC
# MAGIC FROM workspace.default.vw_5g_network_command_center;
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ================================================================
# MAGIC -- 3. REGIONAL RISK SUMMARY
# MAGIC -- ================================================================
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     region,
# MAGIC
# MAGIC     COUNT(DISTINCT cell_id) AS cells,
# MAGIC
# MAGIC     COUNT(*) AS telemetry_records,
# MAGIC
# MAGIC     SUM(critical_risk_flag) AS critical_records,
# MAGIC
# MAGIC     SUM(high_critical_flag) AS high_critical_records,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(failure_risk_pct),
# MAGIC         2
# MAGIC     ) AS avg_failure_risk_pct,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(latency_ms),
# MAGIC         2
# MAGIC     ) AS avg_latency_ms,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(packet_loss_pct),
# MAGIC         2
# MAGIC     ) AS avg_packet_loss_pct,
# MAGIC
# MAGIC     ROUND(
# MAGIC         AVG(throughput_mbps),
# MAGIC         2
# MAGIC     ) AS avg_throughput_mbps
# MAGIC
# MAGIC FROM workspace.default.vw_5g_network_command_center
# MAGIC
# MAGIC GROUP BY region
# MAGIC
# MAGIC ORDER BY avg_failure_risk_pct DESC;
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ================================================================
# MAGIC -- 4. TOP AI FAILURE WATCHLIST
# MAGIC -- ================================================================
# MAGIC
# MAGIC SELECT
# MAGIC
# MAGIC     timestamp,
# MAGIC     cell_id,
# MAGIC     site_id,
# MAGIC     region,
# MAGIC
# MAGIC     failure_risk_pct,
# MAGIC     risk_level,
# MAGIC
# MAGIC     anomaly_status,
# MAGIC     anomaly_score,
# MAGIC
# MAGIC     latency_ms,
# MAGIC     packet_loss_pct,
# MAGIC     throughput_mbps,
# MAGIC     traffic_load_pct,
# MAGIC
# MAGIC     alarm_count,
# MAGIC
# MAGIC     intelligence_flag,
# MAGIC     recommended_action
# MAGIC
# MAGIC FROM workspace.default.vw_5g_network_command_center
# MAGIC
# MAGIC WHERE risk_level IN ('HIGH', 'CRITICAL')
# MAGIC
# MAGIC ORDER BY
# MAGIC     failure_risk_pct DESC,
# MAGIC     anomaly_score DESC
# MAGIC
# MAGIC LIMIT 20;