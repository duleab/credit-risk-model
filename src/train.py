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
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import logging
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    
    def __init__(self, experiment_name: str = "credit_risk_modeling"):
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
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
        
        # Separate features and target
        exclude_cols = ['CustomerId', 'is_high_risk', 'cluster', 'last_transaction_date']
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]
        
        X = customer_data[feature_cols]
        y = customer_data['is_high_risk']
        
        # Fit and apply preprocessing
        X_processed = preprocessing_pipeline.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get model configurations for training.
        
        Returns:
            Dictionary of model configurations
        """
        return {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
    
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
    
    def train_model(self, model_name: str, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray, 
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train, X_test, y_train, y_test: Train/test data splits
            use_grid_search: Whether to use grid search for hyperparameter tuning
            
        Returns:
            Dictionary containing trained model and results
        """
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            # Get model configuration
            model_configs = self.get_model_configs()
            config = model_configs[model_name]
            
            # Log model parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("use_grid_search", use_grid_search)
            
            if use_grid_search and config['params']:
                # Hyperparameter tuning
                logger.info(f"Performing hyperparameter tuning for {model_name}...")
                
                if model_name in ['random_forest', 'gradient_boosting']:
                    # Use RandomizedSearchCV for complex models
                    search = RandomizedSearchCV(
                        config['model'], 
                        config['params'],
                        n_iter=20,
                        cv=5,
                        scoring='roc_auc',
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    # Use GridSearchCV for simpler models
                    search = GridSearchCV(
                        config['model'],
                        config['params'],
                        cv=5,
                        scoring='roc_auc',
                        n_jobs=-1
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Log best parameters
                for param, value in search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)
                    
            else:
                # Train with default parameters
                best_model = config['model']
                best_model.fit(X_train, y_train)
            
            # Evaluate model
            train_metrics = self.evaluate_model(best_model, X_train, y_train)
            test_metrics = self.evaluate_model(best_model, X_test, y_test)
            
            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric}", value)
            
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            
            # Log model
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model, 
                model_name,
                signature=signature
            )
            
            # Store results
            result = {
                'model': best_model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'run_id': mlflow.active_run().info.run_id
            }
            
            logger.info(f"{model_name} training completed. Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A'):.4f}")
            
            return result
    
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
        
        model_configs = self.get_model_configs()
        results = {}
        
        for model_name in model_configs.keys():
            try:
                result = self.train_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                results[model_name] = result
                
                # Track best model
                test_score = result['test_metrics'].get('roc_auc', result['test_metrics']['f1_score'])
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_model = result['model']
                    self.best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = results
        logger.info(f"All models trained. Best model: {self.best_model_name} (Score: {self.best_score:.4f})")
        
        return results
    
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
        sort_col = 'test_roc_auc' if 'test_roc_auc' in comparison_df.columns else 'test_f1_score'
        comparison_df = comparison_df.sort_values(sort_col, ascending=False)
        
        return comparison_df
    
    def register_best_model(self, model_name: str = "credit_risk_model") -> str:
        """
        Register the best model in MLflow Model Registry.
        
        Args:
            model_name: Name for the registered model
            
        Returns:
            Model version URI
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")
        
        logger.info(f"Registering best model ({self.best_model_name}) to MLflow Model Registry...")
        
        # Get the run ID of the best model
        best_result = self.results[self.best_model_name]
        run_id = best_result['run_id']
        
        # Register model
        model_uri = f"runs:/{run_id}/{self.best_model_name}"
        
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            logger.info(f"Model registered successfully. Version: {model_version.version}")
            return model_uri
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def save_model(self, filepath: str = "models/best_model.joblib") -> None:
        """
        Save the best model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Best model saved to {filepath}")
    
    def save_preprocessing_pipeline(self, pipeline, filepath: str = "models/preprocessing_pipeline.joblib") -> None:
        """
        Save the preprocessing pipeline to disk.
        
        Args:
            pipeline: The preprocessing pipeline to save
            filepath: Path to save the pipeline
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(pipeline, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")


if __name__ == "__main__":
    import os
    import sys
    
    # Add the src directory to Python path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from data_processing import DataProcessor
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    
    try:
        # Check if raw data exists
        raw_data_path = "data/raw/data.csv"
        if not os.path.exists(raw_data_path):
            print(f"Error: Raw data file not found at {raw_data_path}")
            print("Please copy the data files to data/raw/ directory first:")
            print('copy "d:\\10-Academy\\Week5\\Technical Content\\Data\\data.csv" "data\\raw\\"')
            print('copy "d:\\10-Academy\\Week5\\Technical Content\\Data\\Xente_Variable_Definitions.csv" "data\\raw\\"')
            sys.exit(1)
        
        # Initialize data processor and process raw data
        print("Processing raw data...")
        processor = DataProcessor()
        
        # Use the complete pipeline method instead of individual steps
        transaction_data, customer_data = processor.process_full_pipeline(raw_data_path)
        
        # Save processed data
        processor.save_processed_data(transaction_data, customer_data)
        print("Processed data saved successfully")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare data for training
        print("Preparing data for training...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            customer_data, processor.preprocessing_pipeline
        )
        
        # Train all models
        print("Training models...")
        results = trainer.train_all_models(X_train, X_test, y_train, y_test)
        
        # Compare models
        comparison = trainer.compare_models()
        print("\nModel Comparison:")
        print(comparison)
        
        # Register best model
        print("Registering best model...")
        model_uri = trainer.register_best_model()
        
        # Save best model
        trainer.save_model()
        
        # Save preprocessing pipeline
        trainer.save_preprocessing_pipeline(processor.preprocessing_pipeline)
        
        print("\nModel training completed successfully!")
        print(f"Best model URI: {model_uri}")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please ensure all required data files are in the correct locations.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()