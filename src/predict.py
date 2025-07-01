"""
Prediction Module for Credit Risk Model

This module handles model inference and risk probability prediction.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from typing import Union, Dict, List, Any
import logging
from datetime import datetime
import shap
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Credit risk prediction pipeline.
    
    This class handles:
    - Model loading from MLflow or disk
    - Feature preprocessing for new data
    - Risk probability prediction
    - Credit score calculation
    - Model explainability with SHAP
    """
    
    def __init__(self, model_path: str = None, model_name: str = None, model_version: str = None):
        """
        Initialize the RiskPredictor.
        
        Args:
            model_path: Path to saved model file
            model_name: Name of registered model in MLflow
            model_version: Version of the model to load
        """
        self.model = None
        self.preprocessing_pipeline = None
        self.feature_names = None
        self.explainer = None
        
        if model_path:
            self.load_model_from_file(model_path)
        elif model_name:
            self.load_model_from_mlflow(model_name, model_version)
    
    def load_model_from_file(self, model_path: str) -> None:
        """
        Load model from a saved file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info("Model loaded successfully from file")
        except Exception as e:
            logger.error(f"Error loading model from file: {str(e)}")
            raise
    
    def load_model_from_mlflow(self, model_name: str, model_version: str = "latest") -> None:
        """
        Load model from MLflow Model Registry.
        
        Args:
            model_name: Name of the registered model
            model_version: Version to load (default: "latest")
        """
        try:
            logger.info(f"Loading model {model_name} version {model_version} from MLflow")
            
            if model_version == "latest":
                model_uri = f"models:/{model_name}/Latest"
            else:
                model_uri = f"models:/{model_name}/{model_version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully from MLflow")
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {str(e)}")
            raise
    
    def load_preprocessing_pipeline(self, pipeline_path: str) -> None:
        """
        Load preprocessing pipeline.
        
        Args:
            pipeline_path: Path to the preprocessing pipeline
        """
        try:
            logger.info(f"Loading preprocessing pipeline from {pipeline_path}")
            self.preprocessing_pipeline = joblib.load(pipeline_path)
            logger.info("Preprocessing pipeline loaded successfully")
            
            # Try to extract feature names from pipeline
            if hasattr(self.preprocessing_pipeline, 'get_feature_names_out'):
                try:
                    self.feature_names = self.preprocessing_pipeline.get_feature_names_out()
                except:
                    logger.warning("Could not extract feature names from pipeline")
            
        except Exception as e:
            logger.error(f"Error loading preprocessing pipeline: {str(e)}")
            raise
    
    def load_model(self, model_path: str = "models/best_model.joblib", 
                   pipeline_path: str = "models/preprocessing_pipeline.joblib") -> None:
        """
        Convenience method to load both model and preprocessing pipeline.
        
        Args:
            model_path: Path to the model file
            pipeline_path: Path to the preprocessing pipeline
        """
        try:
            self.load_model_from_file(model_path)
            self.load_preprocessing_pipeline(pipeline_path)
            logger.info("Model and preprocessing pipeline loaded successfully")
            
            # Initialize SHAP explainer
            self._init_explainer()
            
        except Exception as e:
            logger.error(f"Error loading model and pipeline: {str(e)}")
            raise
    
    def _init_explainer(self) -> None:
        """
        Initialize SHAP explainer for the model.
        """
        if self.model is None:
            logger.warning("Cannot initialize explainer: No model loaded")
            return
            
        try:
            # Create a background dataset for the explainer
            # This is a simplified approach - in production, you'd want to use a representative sample
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("TreeExplainer initialized for model")
            else:
                # For other models
                logger.info("Model is not tree-based, explainer will be initialized on first use")
                self.explainer = None
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {str(e)}")
            self.explainer = None
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features for prediction.
        
        Args:
            data: Raw feature data
            
        Returns:
            Preprocessed features
        """
        if self.preprocessing_pipeline is None:
            logger.warning("No preprocessing pipeline loaded. Using data as-is.")
            return data.values
        
        try:
            # Apply preprocessing
            processed_data = self.preprocessing_pipeline.transform(data)
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def predict_risk_probability(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict risk probabilities for given data.
        
        Args:
            data: Feature data for prediction
            
        Returns:
            Array of risk probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        try:
            # Preprocess if DataFrame
            if isinstance(data, pd.DataFrame):
                processed_data = self.preprocess_features(data)
            else:
                processed_data = data
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[:, 1]
            else:
                # For models without predict_proba, use decision function or predict
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(processed_data)
                    # Convert to probabilities using sigmoid
                    probabilities = 1 / (1 + np.exp(-scores))
                else:
                    # Binary predictions converted to probabilities
                    predictions = self.model.predict(processed_data)
                    probabilities = predictions.astype(float)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting risk probabilities: {str(e)}")
            raise
    
    def predict_risk_class(self, data: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Predict risk classes (binary) for given data.
        
        Args:
            data: Feature data for prediction
            threshold: Probability threshold for classification
            
        Returns:
            Array of risk classes (0: low risk, 1: high risk)
        """
        probabilities = self.predict_risk_probability(data)
        return (probabilities >= threshold).astype(int)
    
    def calculate_credit_score(self, risk_probabilities: np.ndarray, 
                             min_score: int = 300, max_score: int = 850) -> np.ndarray:
        """
        Convert risk probabilities to credit scores.
        
        Args:
            risk_probabilities: Array of risk probabilities
            min_score: Minimum credit score
            max_score: Maximum credit score
            
        Returns:
            Array of credit scores
        """
        # Invert probabilities (higher risk = lower score)
        inverted_probs = 1 - risk_probabilities
        
        # Scale to credit score range
        scores = min_score + (inverted_probs * (max_score - min_score))
        
        # Round to integers
        return np.round(scores).astype(int)
    
    def predict_comprehensive(self, data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make comprehensive predictions including risk probability, class, and credit score.
        
        Args:
            data: Feature data for prediction
            threshold: Probability threshold for classification
            
        Returns:
            DataFrame with predictions
        """
        # Get risk probabilities
        risk_probs = self.predict_risk_probability(data)
        
        # Get risk classes
        risk_classes = (risk_probs >= threshold).astype(int)
        
        # Calculate credit scores
        credit_scores = self.calculate_credit_score(risk_probs)
        
        # Create risk level labels
        risk_levels = np.where(risk_classes == 1, 'High', 'Low')
        
        # Create result DataFrame
        result = pd.DataFrame({
            'customer_id': data['CustomerId'] if 'CustomerId' in data.columns else range(len(risk_probs)),
            'risk_probability': risk_probs,
            'risk_class': risk_classes,
            'risk_level': risk_levels,
            'credit_score': credit_scores
        })
        
        return result
    
    def batch_predict(self, data_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions on a batch of data from a file.
        
        Args:
            data_path: Path to the input data file
            output_path: Path to save the predictions
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Load data
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path)
            
            # Make predictions
            logger.info("Making predictions")
            predictions = self.predict_comprehensive(data)
            
            # Save predictions if output path is provided
            if output_path:
                logger.info(f"Saving predictions to {output_path}")
                predictions.to_csv(output_path, index=False)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def explain_prediction(self, data: pd.DataFrame, customer_index: int = 0) -> Dict[str, Any]:
        """
        Explain a prediction using SHAP values.
        
        Args:
            data: Feature data
            customer_index: Index of the customer to explain
            
        Returns:
            Dictionary with explanation data
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        try:
            # Preprocess data
            processed_data = self.preprocess_features(data)
            
            # Get feature names
            feature_names = self.feature_names
            if feature_names is None:
                feature_names = data.columns.tolist()
            
            # Initialize explainer if needed
            if self.explainer is None:
                if hasattr(self.model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # For non-tree models, use KernelExplainer with a background dataset
                    # This is a simplified approach - in production, you'd want to use a representative sample
                    background_data = processed_data[:min(50, len(processed_data))]
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                        else self.model.predict,
                        background_data
                    )
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(processed_data[customer_index:customer_index+1])
            
            # For classification models, shap_values might be a list of arrays
            if isinstance(shap_values, list):
                # Take the values for the positive class (high risk)
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create a dictionary of feature contributions
            contributions = {}
            for i, name in enumerate(feature_names):
                if i < shap_values.shape[1]:
                    contributions[name] = float(shap_values[0, i])
            
            # Sort by absolute value
            sorted_contributions = {
                k: v for k, v in sorted(
                    contributions.items(), 
                    key=lambda item: abs(item[1]), 
                    reverse=True
                )
            }
            
            # Generate a SHAP force plot
            plt.figure(figsize=(10, 3))
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                
            shap.force_plot(
                base_value, 
                shap_values[0], 
                features=processed_data[customer_index], 
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            
            # Save plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Create explanation object
            explanation = {
                'customer_index': customer_index,
                'base_value': float(base_value),
                'feature_contributions': sorted_contributions,
                'top_positive_features': {k: v for k, v in sorted_contributions.items() if v > 0}[:5],
                'top_negative_features': {k: v for k, v in sorted_contributions.items() if v < 0}[:5],
                'force_plot': plot_base64
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            raise


def predict_optimal_loan_terms(risk_probability: float, 
                              base_amount: float = 10000,
                              base_duration: int = 12) -> Dict[str, Union[float, int]]:
    """
    Calculate optimal loan terms based on risk probability.
    
    Args:
        risk_probability: Probability of default
        base_amount: Maximum loan amount for lowest risk
        base_duration: Minimum loan duration in months
        
    Returns:
        Dictionary with loan terms
    """
    # Risk adjustment factor (lower risk = higher amount, lower interest)
    risk_factor = 1 - risk_probability
    
    # Calculate maximum loan amount based on risk
    # Lower risk = higher loan amount
    max_amount = base_amount * risk_factor
    
    # Recommended amount (80% of max)
    recommended_amount = max_amount * 0.8
    
    # Calculate interest rate based on risk
    # Base rate of 5% + risk premium
    interest_rate = 0.05 + (risk_probability * 0.15)
    
    # Calculate loan duration (higher risk = longer duration)
    # Base duration + risk adjustment
    duration_months = base_duration + int(risk_probability * 24)
    
    # Calculate monthly payment
    # Simple loan calculation: P * r * (1+r)^n / ((1+r)^n - 1)
    monthly_rate = interest_rate / 12
    n_payments = duration_months
    
    if monthly_rate > 0:
        monthly_payment = recommended_amount * monthly_rate * (1 + monthly_rate) ** n_payments / ((1 + monthly_rate) ** n_payments - 1)
    else:
        monthly_payment = recommended_amount / n_payments
    
    return {
        'max_amount': round(max_amount, 2),
        'recommended_amount': round(recommended_amount, 2),
        'interest_rate': round(interest_rate * 100, 2),  # Convert to percentage
        'duration_months': duration_months,
        'monthly_payment': round(monthly_payment, 2)
    }


if __name__ == "__main__":
    # Example usage
    predictor = RiskPredictor(model_path="models/best_model.joblib")
    
    # Load preprocessing pipeline
    try:
        predictor.load_preprocessing_pipeline("models/preprocessing_pipeline.joblib")
    except FileNotFoundError:
        logger.warning("Preprocessing pipeline not found. You may need to retrain the model.")
    
    # Load test data
    test_data = pd.read_csv("data/processed/customer_features.csv")
    
    # Generate predictions
    predictions = predictor.predict_comprehensive(test_data.head(10))
    print("Sample Predictions:")
    print(predictions)
    
    # Example loan term prediction
    sample_risk = predictions['risk_probability'].iloc[0]
    loan_terms = predict_optimal_loan_terms(sample_risk)
    print(f"\nOptimal loan terms for risk probability {sample_risk:.3f}:")
    print(loan_terms)