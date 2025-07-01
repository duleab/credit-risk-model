import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import ModelTrainer
from data_processing import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'CustomerId': [f'CUST{i:04d}' for i in range(n_samples)],
        'TransactionId': [f'TXN{i:06d}' for i in range(n_samples)],
        'Amount': np.random.lognormal(5, 1, n_samples),
        'TransactionStartTime': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'PricingStrategy': np.random.randint(1, 5, n_samples),
        'FraudResult': np.random.randint(0, 2, n_samples),
        'ProviderId': [f'PROV{np.random.randint(1, 11):02d}' for _ in range(n_samples)],
        'ProductId': [f'PROD{np.random.randint(1, 21):03d}' for _ in range(n_samples)],
        'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples),
        'ChannelId': [f'CHAN{np.random.randint(1, 6):02d}' for _ in range(n_samples)],
        'Value': np.random.lognormal(5, 1, n_samples),
        'CurrencyCode': ['USD'] * n_samples,
        'CountryCode': ['US'] * n_samples
    }
    
    return pd.DataFrame(data)

def test_model_trainer_initialization():
    """Test ModelTrainer initialization"""
    trainer = ModelTrainer()
    assert trainer is not None
    assert hasattr(trainer, 'models')
    # The models dict is empty by default, populated by get_model_configs()
    model_configs = trainer.get_model_configs()
    assert len(model_configs) > 0

def test_data_preparation(sample_data):
    """Test data preparation pipeline"""
    processor = DataProcessor()
    
    # Save sample data to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Process the data using the correct method
        transaction_data, customer_data = processor.process_full_pipeline(temp_file)
        
        # Check that RFM features are created
        rfm_columns = ['recency', 'frequency', 'monetary_sum']
        for col in rfm_columns:
            assert col in customer_data.columns
        
        # Check that risk labels are created
        assert 'is_high_risk' in customer_data.columns
        assert customer_data['is_high_risk'].nunique() > 1
    finally:
        # Clean up temp file
        os.unlink(temp_file)

@patch('mlflow.start_run')
@patch('mlflow.log_params')
@patch('mlflow.log_metrics')
def test_model_training_pipeline(mock_log_metrics, mock_log_params, mock_start_run, sample_data):
    """Test the complete model training pipeline"""
    # Mock MLflow
    mock_start_run.return_value.__enter__ = Mock()
    mock_start_run.return_value.__exit__ = Mock()
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Save sample data to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Process data using the correct method
        transaction_data, customer_data = processor.process_full_pipeline(temp_file)
        
        # Prepare features and target using the correct method signature
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            customer_data, processor.preprocessing_pipeline
        )
        
        assert X_train is not None
        assert y_train is not None
        assert len(X_train) == len(y_train)
        assert len(X_train) > 0
    finally:
        # Clean up temp file
        os.unlink(temp_file)

def test_model_evaluation_metrics():
    """Test model evaluation metrics calculation"""
    from sklearn.ensemble import RandomForestClassifier
    
    trainer = ModelTrainer()
    
    # Create dummy data and train a simple model
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test evaluation with the correct method signature
    metrics = trainer.evaluate_model(model, X_test, y_test)
    
    # Check that all expected metrics are present
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in expected_metrics:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1