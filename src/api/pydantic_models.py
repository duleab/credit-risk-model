from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class TransactionData(BaseModel):
    """Single transaction data model"""
    CustomerId: str = Field(..., description="Customer ID")
    TransactionId: str = Field(..., description="Transaction ID")
    TransactionStartTime: str = Field(..., description="Transaction start time (YYYY-MM-DD HH:MM:SS)")
    Amount: float = Field(..., gt=0, description="Transaction amount")
    CurrencyCode: str = Field(..., description="Currency code")
    CountryCode: str = Field(..., description="Country code")
    ProviderId: str = Field(..., description="Provider ID")
    ProductId: str = Field(..., description="Product ID")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Channel ID")
    PricingStrategy: int = Field(..., description="Pricing strategy")
    FraudResult: int = Field(..., ge=0, le=1, description="Fraud result (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "CustomerId": "C123456",
                "TransactionId": "T987654",
                "TransactionStartTime": "2023-01-15 14:30:00",
                "Amount": 1500.0,
                "CurrencyCode": "USD",
                "CountryCode": "840",
                "ProviderId": "P001",
                "ProductId": "PROD123",
                "ProductCategory": "Electronics",
                "ChannelId": "WEB",
                "PricingStrategy": 1,
                "FraudResult": 0
            }
        }


class CustomerTransactions(BaseModel):
    """Customer transactions for batch prediction"""
    customer_id: str = Field(..., description="Customer ID")
    transactions: List[TransactionData] = Field(..., min_items=1, description="List of transactions")


class LoanTerms(BaseModel):
    """Loan terms recommendation model"""
    max_amount: float = Field(..., description="Maximum loan amount")
    recommended_amount: float = Field(..., description="Recommended loan amount")
    duration_months: int = Field(..., description="Recommended loan duration in months")
    interest_rate: float = Field(..., description="Recommended interest rate")
    monthly_payment: float = Field(..., description="Estimated monthly payment")


class PredictionResponse(BaseModel):
    """Prediction response model"""
    customer_id: str
    risk_probability: float
    risk_class: str
    credit_score: int
    recommended_loan_terms: Optional[LoanTerms] = None
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    model_version: str = Field(default="1.0.0")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    model_version: str
    creation_date: datetime
    features: List[str]
    metrics: Dict[str, float]
    description: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: int
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
