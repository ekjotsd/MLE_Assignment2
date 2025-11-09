# Loan Default Prediction ML Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting loan defaults in a financial institution. The pipeline uses Apache Airflow for orchestration and is fully containerized with Docker for easy deployment and reproducibility.

**Key Innovation:** Instead of training 3 models and selecting based solely on AUC-ROC, this implementation trains 2 models (LogisticRegression and XGBoost) and uses a **weighted scoring system** (0.5×AUC + 0.3×F1 + 0.2×Precision) to select the best model, providing a more balanced evaluation that considers both discrimination and classification performance.

---

## Architecture

### Three-Pipeline Design

The ML pipeline is structured as **three independent Airflow DAGs**:

#### 1. **Data Pipeline** (`data_pipeline.py`)
- **Schedule**: Monthly (1st of each month), 2023-01-01 to 2024-12-01
- **Purpose**: Process raw data through Bronze → Silver → Gold layers
- **Output**: 24 monthly feature stores (MOB=0) and label stores (MOB=6)
- **Key Feature**: Temporal alignment ensures no data leakage

#### 2. **Model Trainer** (`model_trainer.py`)
- **Schedule**: Manual trigger only
- **Purpose**: Train and evaluate 2 ML models
- **Models**: 
  - LogisticRegression (baseline, interpretable)
  - XGBoost (high-performance gradient boosting)
- **Selection**: Weighted scoring (0.5×AUC + 0.3×F1 + 0.2×Precision)
- **Output**: Best model artifacts saved to `model_store/`

#### 3. **Prediction Monitor** (`prediction_monitor.py`)
- **Schedule**: Manual trigger after training
- **Purpose**: Run inference and monitor model performance
- **Range**: OOT data (2024-04-01 to 2024-06-01)
- **Output**: Predictions + monitoring results + visualizations

---

## Design Decisions

### Why Only 2 Models?
- **Efficiency**: RandomForest was excluded as XGBoost consistently outperforms it on tabular data
- **Focused Comparison**: LogisticRegression provides an interpretable baseline, XGBoost provides maximum predictive power
- **Faster Training**: Reduces training time is 30% faster without sacrificing model quality

### Why Weighted Scoring?
Traditional model selection uses only AUC-ROC, but in loan default prediction:
- **Precision matters**: False positives (wrongly predicting default) impact business operations
- **F1-Score balances**: Ensures we don't sacrifice recall for precision
- **Formula**: `Score = 0.5×AUC + 0.3×F1 + 0.2×Precision`

This provides a more holistic view of model performance aligned with business objectives.

### Temporal Leakage Prevention
- **Features extracted at MOB=0**: Only information available at loan application time
- **Labels extracted at MOB=6**: Default status observed 6 months after application
- **Fixed temporal windows**: No random splitting - strict chronological ordering
- **OOT validation**: True out-of-time testing on 2024-06-01 data

---

## Project Structure

```
.
├── dags/
│   ├── data_pipeline.py          # Bronze → Silver → Gold processing
│   ├── model_trainer.py          # Train & select best model
│   └── prediction_monitor.py     # Inference & monitoring
├── src/
│   ├── data_processing_*.py      # Data transformation scripts
│   ├── model_training.py         # Model training with weighted scoring
│   ├── model_inference.py        # Prediction generation
│   ├── model_monitoring.py       # Performance tracking
│   └── visualization.py          # Custom purple/teal themed charts
├── pipeline_config/
│   └── config.py                 # Centralized configuration
├── data/                         # Source data
├── datamart/
│   ├── bronze/                   # Raw snapshots
│   ├── silver/                   # Cleaned data
│   └── gold/                     # Feature/label stores, predictions
├── model_store/                  # Trained model artifacts
├── results/
│   └── monitoring_visualizations/ # Performance charts
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Docker Desktop installed
- At least 8GB RAM available
- Port 8080 available

### 1. Build & Start

```bash
docker-compose build
docker-compose up -d
```

### 2. Access Airflow UI
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 3. Run the Pipeline

**Step 1: Process Historical Data**
```
1. Enable "data_pipeline" DAG
2. It automatically backfills 24 months (2023-01 to 2024-12)
3. Wait ~30-40 minutes for completion
```

**Step 2: Train Models**
```
1. Enable "model_trainer" DAG
2. Click ▶ (Play) button to manually trigger
3. Training completes in ~5-10 seconds
4. Check model_store/model_config.json for best model
```

**Step 3: Monitor Performance**
```
1. Enable "prediction_monitor" DAG
2. Click ▶ (Play) button to manually trigger
3. Generates predictions and monitoring charts
4. View results in results/monitoring_visualizations/
```

---

## Key Features

### 1. Model Training
- ✅ Trains 2 models: LogisticRegression + XGBoost
- ✅ Weighted scoring for balanced model selection
- ✅ Handles class imbalance with balanced weights
- ✅ Saves all model artifacts (model.pkl, scaler.pkl, metadata.json)

### 2. Model Selection
Automatically selects best model using:
```python
weighted_score = (0.5 × AUC-ROC) + (0.3 × F1-Score) + (0.2 × Precision)
```

### 3. Monitoring Metrics
- AUC-ROC (discrimination)
- Accuracy (overall correctness)
- Precision (positive predictive value)
- Recall (true positive rate)
- F1-Score (balanced metric)
- PSI (Population Stability Index - drift detection)

### 4. Custom Visualizations
**Purple/Teal themed charts:**
- Performance metrics over time
- PSI trend with warning zones
- Confusion matrix components
- Threshold compliance tracking
- Prediction distribution evolution

---

## Model Governance

### Retraining Triggers
1. **Performance Degradation**: Any metric below threshold for 2+ periods
2. **Distribution Drift**: PSI > 0.2 (CRITICAL)
3. **Scheduled**: Monthly with 12-month rolling window

### Monitoring Thresholds
- AUC-ROC: ≥ 0.70
- Precision: ≥ 0.60
- Recall: ≥ 0.50
- F1-Score: ≥ 0.55
- PSI Warning: 0.1
- PSI Critical: 0.2

### Deployment Strategy
1. **Shadow Mode**: Run new model in parallel for 1 month
2. **Canary**: 10% → 50% → 100% traffic shift
3. **Rollback**: Keep previous model artifacts for quick reversion

---

## Results

Latest model performance (validation set):
- **Best Model**: XGBoost
- **Weighted Score**: Calculated from validation metrics
- **AUC-ROC**: ~0.80
- **Precision**: ~0.61
- **Recall**: ~0.67
- **F1-Score**: ~0.64

OOT Performance (2024-06-01):
- **PSI**: < 0.1 (Stable - no drift detected)
- **Threshold Compliance**: See `results/monitoring_visualizations/`

---

## Configuration

All parameters centralized in `pipeline_config/config.py`:

```python
# Temporal windows
TEMPORAL_SPLITS = {
    "train": {"start_date": "2023-01-01", "end_date": "2023-12-01"},
    "validation": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    "test": {"start_date": "2024-04-01", "end_date": "2024-05-01"},
    "oot": {"start_date": "2024-06-01", "end_date": "2024-06-01"}
}

# Model selection
MODEL_SELECTION_METRIC = 'weighted_score'
MODEL_SELECTION_WEIGHTS = {
    'auc_roc': 0.5,
    'f1_score': 0.3,
    'precision': 0.2
}
```

---

## Troubleshooting

**DAG not appearing?**
- Check `docker logs` for errors
- Verify all DAG files in `dags/` folder
- Refresh Airflow UI

**Model training fails?**
- Ensure data_pipeline completed for all 24 months
- Check `model_store/` has write permissions
- Verify feature/label stores exist in `datamart/gold/`

**Visualizations missing?**
- Run prediction_monitor DAG after model_trainer
- Check `results/monitoring_visualizations/` folder
- Verify monitoring history exists in `model_store/model_monitoring.json`

---

## Dependencies

See `requirements.txt`:
- `apache-airflow==2.10.4` - Workflow orchestration
- `pandas==2.2.3` - Data manipulation
- `scikit-learn==1.6.1` - ML models and metrics
- `xgboost==2.1.3` - Gradient boosting
- `pyspark==3.5.5` - Distributed data processing
- `matplotlib==3.10.0` + `seaborn==0.13.2` - Visualizations

---

## Future Enhancements

1. **Additional Models**: LightGBM, CatBoost
2. **Hyperparameter Tuning**: GridSearch or Optuna
3. **Feature Importance**: SHAP values for explainability
4. **Advanced Monitoring**: KS statistic, Gini coefficient
5. **Automated Retraining**: Trigger-based retraining pipeline
6. **A/B Testing**: Champion/Challenger comparison framework

---

## License

This project is developed for educational purposes as part of CS611 - Machine Learning Engineering.

---

**Author**: ML Engineering Team  
**Last Updated**: November 2024  
**Version**: 2.0
