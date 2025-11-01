
import os
from datetime import datetime

# General Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_NAME = "credit_risk_modeling"

# Data Configuration
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "data.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRANSACTIONS_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "transactions_processed.csv")
CUSTOMER_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "customer_features.csv")

# Model Configuration
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PREPROCESSING_PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.joblib")
REGISTERED_MODEL_NAME = "credit_risk_model"

# Data Processing Configuration
SNAPSHOT_DATE = datetime.now().strftime('%Y-%m-%d')
N_CLUSTERS = 3
CLUSTER_FEATURES = ['recency', 'frequency', 'monetary_sum']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Training Configuration
MODEL_CONFIGS = {
    'logistic_regression': {
        'model': 'LogisticRegression',
        'params': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    'decision_tree': {
        'model': 'DecisionTreeClassifier',
        'params': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    },
    'random_forest': {
        'model': 'RandomForestClassifier',
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'gradient_boosting': {
        'model': 'GradientBoostingClassifier',
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}
HYPERPARAMETER_TUNING_CONFIG = {
    'use_grid_search': {
        'logistic_regression': True,
        'decision_tree': True
    },
    'use_random_search': {
        'random_forest': True,
        'gradient_boosting': True
    },
    'random_search_n_iter': 20
}
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'

# Logging Configuration
LOGGING_LEVEL = "INFO"
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "credit_risk_model.log")

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
