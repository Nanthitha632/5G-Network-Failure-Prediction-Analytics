# 📡 5G Network Failure Prediction & Analytics

### End-to-End Telecom Data Engineering, Predictive Analytics & Explainable AI Project

An end-to-end telecommunications analytics project designed to process large-scale 5G network telemetry, identify network-performance patterns, and predict potential cell failures **30 minutes in advance**.

The project combines **PySpark, SQL, Databricks, Python, Machine Learning, XGBoost, Logistic Regression, SHAP, and Power BI** to demonstrate a production-style analytics workflow from raw telemetry to operational intelligence.

---

## 🎯 Business Problem

Modern telecom networks generate millions of telemetry observations across cell towers and network infrastructure.

Reactive monitoring identifies failures only after service degradation has already occurred.

This project addresses a more valuable question:

> **Can network telemetry be used to identify high-risk conditions and predict a cell failure before it happens?**

The solution analyzes network KPIs including traffic load, connected users, signal strength, latency, packet loss, throughput, CPU utilization, alarms, SINR, and time-based behavior.

The final prediction target identifies whether a network cell will experience a failure within the **next 30 minutes**.

---

## 🏗️ Solution Architecture

```text
Raw 5G Telemetry
        ↓
Data Profiling & Quality Assessment
        ↓
Python / Pandas Data Cleaning
        ↓
Curated Telemetry Dataset
        ↓
Databricks + PySpark Data Engineering
        ↓
Bronze → Silver → Gold Data Layers
        ↓
SQL KPI & Network Analytics
        ↓
Feature Engineering
        ↓
30-Minute Future Failure Target
        ↓
Logistic Regression Baseline
        ↓
XGBoost Failure Prediction
        ↓
Model Evaluation & Comparison
        ↓
Anomaly Detection
        ↓
SHAP Explainable AI
        ↓
Power BI Operational Dashboard
