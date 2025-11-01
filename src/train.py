"""
Model Training Module for Credit Risk Model

This module handles model training, hyperparameter tuning,
and experiment tracking using MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import logging
from typing import Dict, Any, Tuple
import warnings
import config

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=config.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Comprehensive model training and evaluation pipeline.
    
    This class handles:
    - Model training with multiple algorithms
    - Hyperparameter tuning
    - Model evaluation and comparison
    - MLflow experiment tracking
    - Model registration
    """
    
    def __init__(self):
        """
        Initialize the ModelTrainer.
        """
        self.experiment_name = config.EXPERIMENT_NAME
        self.models = self._get_model_instances()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(self.experiment_name)

    def _get_model_instances(self) -> Dict[str, Any]:
        """
        Get model instances from configuration.
        """
        model_instances = {}
        for name, conf in config.MODEL_CONFIGS.items():
            model_class = globals()[conf['model']]
            model_instances[name] = model_class(random_state=config.RANDOM_STATE)
        return model_instances

    def prepare_data(self, customer_data: pd.DataFrame, 
                    preprocessing_pipeline: Pipeline) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            customer_data: Customer features with target variable
            preprocessing_pipeline: Preprocessing pipeline
            
        Returns:
            Tuple of train/test splits
        """
        logger.info("Preparing data for model training...")
        
        try:
            # Separate features and target
            exclude_cols = ['CustomerId', 'is_high_risk', 'cluster', 'last_transaction_date']
            feature_cols = [col for col in customer_data.columns if col not in exclude_cols]

            X = customer_data[feature_cols]
            y = customer_data['is_high_risk']

            # Fit and apply preprocessing
            X_processed = preprocessing_pipeline.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE, stratify=y
            )

            logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")

            return X_train, X_test, y_train, y_test
        
        except KeyError as e:
            logger.error(f"Missing expected column in data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            raise
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            return {}
    
    def train_model(self, model_name: str, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train, X_test, y_train, y_test: Train/test data splits
            
        Returns:
            Dictionary containing trained model and results
        """
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name, nested=True) as run:
            try:
                # Get model configuration
                model = self.models[model_name]
                params = config.MODEL_CONFIGS[model_name]['params']

                # Log model parameters
                mlflow.log_param("model_type", model_name)

                # Hyperparameter tuning
                logger.info(f"Performing hyperparameter tuning for {model_name}...")
                
                if model_name in config.HYPERPARAMETER_TUNING_CONFIG['use_random_search']:
                    search = RandomizedSearchCV(
                        model, params,
                        n_iter=config.HYPERPARAMETER_TUNING_CONFIG['random_search_n_iter'],
                        cv=config.CV_FOLDS, scoring=config.SCORING_METRIC,
                        random_state=config.RANDOM_STATE, n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        model, params, cv=config.CV_FOLDS,
                        scoring=config.SCORING_METRIC, n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Log best parameters
                mlflow.log_params(search.best_params_)
                    
                # Evaluate model
                train_metrics = self.evaluate_model(best_model, X_train, y_train)
                test_metrics = self.evaluate_model(best_model, X_test, y_test)

                # Log metrics
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                # Log model
                signature = infer_signature(X_train, best_model.predict(X_train))
                mlflow.sklearn.log_model(
                    best_model, model_name, signature=signature
                )

                # Store results
                result = {
                    'model': best_model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'run_id': run.info.run_id
                }

                logger.info(f"{model_name} training completed. Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")

                return result

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(e))
                return None
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data splits
            
        Returns:
            Dictionary of all model results
        """
        logger.info("Training all models...")
        
        with mlflow.start_run(run_name="Model_Comparison") as parent_run:
            for model_name in self.models.keys():
                result = self.train_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                if result:
                    self.results[model_name] = result
                    
                    # Track best model
                    test_score = result['test_metrics'].get(config.SCORING_METRIC, 0)
                    if test_score > self.best_score:
                        self.best_score = test_score
                        self.best_model = result['model']
                        self.best_model_name = model_name
        
        logger.info(f"All models trained. Best model: {self.best_model_name} (Score: {self.best_score:.4f})")
        
        return self.results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models have been trained yet.")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {'model': model_name}
            row.update({f"train_{k}": v for k, v in result['train_metrics'].items()})
            row.update({f"test_{k}": v for k, v in result['test_metrics'].items()})
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by test ROC-AUC or F1 score
        sort_col = f'test_{config.SCORING_METRIC}'
        if sort_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_col, ascending=False)
        
        return comparison_df
    
    def register_best_model(self) -> str:
        """
        Register the best model in MLflow Model Registry.
        
        Returns:
            Model version URI
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")
        
        logger.info(f"Registering best model ({self.best_model_name}) to MLflow Model Registry...")
        
        try:
            # Get the run ID of the best model
            run_id = self.results[self.best_model_name]['run_id']

            # Register model
            model_uri = f"runs:/{run_id}/{self.best_model_name}"

            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=config.REGISTERED_MODEL_NAME
            )
            
            logger.info(f"Model registered successfully. Version: {model_version.version}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save a model to disk.
        
        Args:
            model: The model to save.
            filepath: Path to save the model.
        """
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")
            raise


def main():
    """
    Main function to run the training pipeline.
    """
    import os
    import sys
    
    # Add the src directory to Python path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_processing import DataProcessor
    
    # Create necessary directories
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    try:
        # Check if raw data exists
        if not os.path.exists(config.RAW_DATA_PATH):
            logger.error(f"Error: Raw data file not found at {config.RAW_DATA_PATH}")
            sys.exit(1)
        
        # Initialize data processor and process raw data
        logger.info("Processing raw data...")
        processor = DataProcessor()
        
        transaction_data, customer_data = processor.process_full_pipeline()
        
        # Save processed data
        processor.save_processed_data(transaction_data, customer_data)
        logger.info("Processed data saved successfully")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare data for training
        logger.info("Preparing data for training...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            customer_data, processor.preprocessing_pipeline
        )
        
        # Train all models
        logger.info("Training models...")
        trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Compare models
        comparison = trainer.compare_models()
        logger.info("\nModel Comparison:")
        logger.info(comparison)
        
        # Register best model
        logger.info("Registering best model...")
        model_uri = trainer.register_best_model()
        
        # Save best model
        trainer.save_model(trainer.best_model, config.BEST_MODEL_PATH)
        
        # Save preprocessing pipeline
        trainer.save_model(processor.preprocessing_pipeline, config.PREPROCESSING_PIPELINE_PATH)
        
        logger.info("\nModel training completed successfully!")
        logger.info(f"Best model URI: {model_uri}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == "__main__":
    main()