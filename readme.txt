https://github.com/YOUR_USERNAME/MLE_Assignment2

Project: Loan Default Prediction ML Pipeline
Assignment: CS611 - Machine Learning Engineering - Assignment 2

IMPLEMENTATION NOTES:
- Custom 2-model approach (LogisticRegression + XGBoost)
- Weighted scoring for model selection (0.5*AUC + 0.3*F1 + 0.2*Precision)
- Purple/teal visualization theme
- Renamed DAGs for clarity

=== QUICK START ===
1. docker-compose build
2. docker-compose up -d
3. Access http://localhost:8080 (airflow/airflow)
4. Run DAGs: data_pipeline → model_trainer → prediction_monitor

See README.md for detailed documentation.
