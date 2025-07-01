import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data

def test_predict_endpoint_structure():
    """Test predict endpoint structure (without model)"""
    # The API expects CustomerTransactions format
    sample_customer_data = {
        "customer_id": "CUST001",
        "transactions": [
            {
                "CustomerId": "CUST001",
                "TransactionId": "TXN001",
                "TransactionStartTime": "2023-01-01 10:00:00",
                "Amount": 1000.0,
                "CurrencyCode": "USD",
                "CountryCode": "US",
                "ProviderId": "PROV001",
                "ProductId": "PROD001",
                "ProductCategory": "Electronics",
                "ChannelId": "CHAN001",
                "PricingStrategy": 2,
                "FraudResult": 0
            }
        ]
    }
    
    response = client.post("/predict", json=sample_customer_data)
    # Expect 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]

def test_invalid_transaction_data():
    """Test with invalid transaction data"""
    invalid_customer_data = {
        "customer_id": "CUST001",
        "transactions": [
            {
                "CustomerId": "CUST001",
                "TransactionId": "TXN001",
                "TransactionStartTime": "2023-01-01 10:00:00",
                "Amount": -100.0,  # Invalid negative amount
                "CurrencyCode": "USD",
                "CountryCode": "US",
                "ProviderId": "PROV001",
                "ProductId": "PROD001",
                "ProductCategory": "Electronics",
                "ChannelId": "CHAN001",
                "PricingStrategy": 5,  # Invalid strategy (should be 1-4)
                "FraudResult": 2  # Invalid fraud result (should be 0 or 1)
            }
        ]
    }
    
    response = client.post("/predict", json=invalid_customer_data)
    assert response.status_code == 422  # Validation error

def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    # Should return 404 if no model loaded, or 200 if model is loaded
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert "model_loaded" in data
        assert "model_type" in data
    elif response.status_code == 404:
        data = response.json()
        assert "detail" in data

def test_batch_predict_endpoint():
    """Test batch prediction endpoint structure"""
    sample_batch_data = [
        {
            "customer_id": "CUST001",
            "transactions": [
                {
                    "CustomerId": "CUST001",
                    "TransactionId": "TXN001",
                    "TransactionStartTime": "2023-01-01 10:00:00",
                    "Amount": 1000.0,
                    "CurrencyCode": "USD",
                    "CountryCode": "US",
                    "ProviderId": "PROV001",
                    "ProductId": "PROD001",
                    "ProductCategory": "Electronics",
                    "ChannelId": "CHAN001",
                    "PricingStrategy": 2,
                    "FraudResult": 0
                }
            ]
        }
    ]
    
    response = client.post("/predict/batch", json=sample_batch_data)
    # Expect 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]