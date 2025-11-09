"""
Configuration File - Loan Default Prediction ML Pipeline
Centralizes all configuration parameters

Project: CS611 Machine Learning Engineering - Assignment 2
Version: 2.0.0
Key Features:
- Two-model comparison (LogReg + XGBoost)
- Weighted scoring for model selection
- Custom hyperparameter tuning
- Stringent monitoring thresholds
"""

import os
from pathlib import Path

# Project metadata
PROJECT_NAME = "Loan Default Prediction ML Pipeline"
PROJECT_VERSION = "2.0.0"
PROJECT_ID = "CS611_MLE_Assignment2"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
BRONZE_PATH = PROJECT_ROOT / "datamart" / "bronze"
SILVER_PATH = PROJECT_ROOT / "datamart" / "silver"
GOLD_PATH = PROJECT_ROOT / "datamart" / "gold"
MODEL_STORE_PATH = PROJECT_ROOT / "model_store"
RESULTS_PATH = PROJECT_ROOT / "results"

# Create directories if they don't exist
for path in [DATA_PATH, BRONZE_PATH, SILVER_PATH, GOLD_PATH, MODEL_STORE_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Gold layer parameters
FEATURE_MOB = 0  # Features at application time (MOB=0)
LABEL_MOB = 6    # Labels at 6 months (MOB=6)
DPD_THRESHOLD = 30  # Days past due threshold for default definition

# Temporal Window Configuration
TEMPORAL_WINDOW_MODE = "absolute"  # "absolute" or "relative"
TEMPORAL_SPLITS = {
    "train": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-01"
    },
    "validation": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-01"
    },
    "test": {
        "start_date": "2024-04-01",
        "end_date": "2024-05-01"
    },
    "oot": {
        "start_date": "2024-06-01",
        "end_date": "2024-06-01"
    }
}

# Data processing date range (for backfill)
DATA_PROCESSING_START_DATE = "2023-01-01"
DATA_PROCESSING_END_DATE = "2024-12-01"

# Model training trigger date (only train when all data is available)
MODEL_TRAINING_TRIGGER_DATE = "2024-12-01"

INFER_START_DATE = "2024-04-01"
INFER_END_DATE = "2024-06-01"
MONITOR_START_DATE = "2024-04-01"
MONITOR_END_DATE = "2024-06-01"

# Model training parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.
RANDOM_STATE = 42
CV_FOLDS = 5

# Model configurations
# Model hyperparameters - Custom tuned for loan default prediction
# Rationale: Tuned based on financial domain expertise and class imbalance handling
MODELS = {
    'LogisticRegression': {
        'C': 0.5,  # CUSTOM: Stronger regularization for better generalization
        'max_iter': 2000,  # CUSTOM: Increased for convergence
        'random_state': RANDOM_STATE,
        'solver': 'saga',  # CUSTOM: Better for large datasets and L1 penalty
        'penalty': 'l1',  # CUSTOM: L1 for feature selection
        'class_weight': 'balanced'
    },
    'XGBoost': {
        'n_estimators': 150,  # CUSTOM: More trees for better learning
        'max_depth': 5,  # CUSTOM: Reduced to prevent overfitting
        'learning_rate': 0.05,  # CUSTOM: Lower rate with more estimators
        'subsample': 0.7,  # CUSTOM: More aggressive subsampling
        'colsample_bytree': 0.7,  # CUSTOM: Feature subsampling per tree
        'min_child_weight': 3,  # CUSTOM: Added for regularization
        'gamma': 0.1,  # CUSTOM: Minimum loss reduction for split
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'scale_pos_weight': 2  # CUSTOM: Handle class imbalance (2:1 ratio)
    }
}

# Feature configuration - Features available at application time (MOB=0)
# These features do NOT cause temporal leakage
FEATURE_COLUMNS = [
    # Loan characteristics at application
    'tenure',
    'loan_amt',
    
    # Customer demographics
    'customer_age',
    
    # Financial features
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Num_of_Delayed_Payment',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance',
    'debt_to_income_ratio',
    
    # Clickstream features
    'fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5',
    'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10',
    'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15',
    'fe_16', 'fe_17', 'fe_18', 'fe_19', 'fe_20'
]

# Monitoring thresholds - Custom thresholds based on business requirements
# CUSTOM: More stringent thresholds for financial risk management
MONITORING_THRESHOLDS = {
    'auc_roc_min': 0.72,  # CUSTOM: Raised from 0.70 for better discrimination
    'precision_min': 0.65,  # CUSTOM: Higher precision to reduce false alarms
    'recall_min': 0.55,  # CUSTOM: Increased to catch more defaults
    'f1_score_min': 0.60,  # CUSTOM: Higher balanced metric requirement
    'psi_warning': 0.08,  # CUSTOM: More sensitive drift detection
    'psi_critical': 0.15,  # CUSTOM: Earlier critical alert
    'performance_degradation_threshold': 0.03  # CUSTOM: 3% drop triggers retraining (more aggressive)
}

# Monitoring metrics to track
MONITORING_METRICS = [
    'auc_roc',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'log_loss',
    'psi'
]

# Model selection criteria
MODEL_SELECTION_METRIC = 'weighted_score'  # Custom weighted scoring
MODEL_SELECTION_MODE = 'max'  # 'max' or 'min'
# Weighted scoring formula: 0.5*AUC + 0.3*F1 + 0.2*Precision
MODEL_SELECTION_WEIGHTS = {
    'auc_roc': 0.5,
    'f1_score': 0.3,
    'precision': 0.2
}

# Airflow parameters
AIRFLOW_SCHEDULE = '0 0 1 * *'  # Monthly at midnight on the 1st
AIRFLOW_EMAIL = ['ml-team@company.com']
AIRFLOW_RETRIES = 2
AIRFLOW_RETRY_DELAY_MINUTES = 5
AIRFLOW_EXECUTION_TIMEOUT_HOURS = 2

# Spark configuration
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_APP_NAME = "MLPipeline"

# File formats
PARQUET_COMPRESSION = 'snappy'
PARQUET_ENGINE = 'pyarrow'

# Training data windows (in months)
TRAINING_WINDOW_MONTHS = 12
VALIDATION_WINDOW_MONTHS = 2
TEST_WINDOW_MONTHS = 2

# Model versioning
MODEL_CONFIG_FILE = "model_config.json"
MODEL_EVALUATION_FILE = "model_evaluation.json"
MODEL_MONITORING_FILE = "model_monitoring.json"