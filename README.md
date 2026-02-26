Diacare NLP Triage System

This project builds a machine learning triage engine that converts patient support messages into:

• priority level (1–5)
• category (medical, billing, tech, etc.)

The system uses:

• TF-IDF features
• Linear SVM classifiers
• Probability calibration
• 5-fold cross validation
• Fixed holdout test evaluation

It is deployed as a FastAPI service and live on Railway.

This project demonstrates:

• interpretable decision support modeling
• class-imbalanced healthcare data handling
• deployable ML pipelines
