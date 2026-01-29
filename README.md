# End-to-End Marketing Machine Learning Project
## Overview

This project is an end-to-end data & machine learning pipeline built in a retail marketing context, covering the full lifecycle from raw data ingestion to model monitoring.

It aims to predict customer churn and product purchase propensity on a monthly, customer-level basis, using scalable cloud data processing and Machine Learning best practices.

## Business Context

Domain: Retail / Marketing analytics

Granularity: Customer × Month

Targets:

Churn

Sport product purchase

Cinema product purchase

Model type: Binary classification (one model per target)

Main objective (Churn): Maximize recall of churners while maintaining reasonable false positives

## Architecture & Workflow
### 1. Data Ingestion

Synthetic but realistic customer interaction data

6 daily refreshed sources:

Internet usage

TV usage

Mobile usage

Socio-demographics

Customer journeys

Requests (used for targets)

Stored in Azure Blob Storage

Orchestrated via Databricks Jobs

### 2. Feature Engineering (PySpark)

Monthly aggregation of granular data

Rolling 3-month (L3M) behavioral features

Scheduled on the 1st of each month

Outputs written back to Blob Storage

### 3. Labelled Dataset Creation

Monthly customer-level dataset

Targets created using a 1-month prediction latency

Split into training and evaluation sets (most recent data used for evaluation)

### 4. Machine Learning

Algorithm: XGBoost

One binary classifier per target

Simple hyperparameter tuning

Model evaluation based on:

ROC AUC

Lift & quantile analysis

Precision / recall trade-offs

### 5. Model Tracking & Scoring

Models registered and versioned with MLflow

Monthly automated scoring using Databricks Workflows

Latest model selected via aliases

### 6. Reporting & Monitoring

Power BI dashboard for model performance monitoring

Stored in PBIP format

Versioned and maintained via Git & VS Code

Enables KPI historization and collaboration

## Tech Stack

Python, PySpark

XGBoost

Azure Blob Storage

Databricks (Jobs & Workflows)

MLflow

Power BI (PBIP)

Git / GitHub

## Key Takeaways

End-to-end, production-oriented data & ML project

Scalable cloud architecture

Business-driven model evaluation

Full traceability from raw data to dashboard