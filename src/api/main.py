from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import time
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import RiskPredictor, predict_optimal_loan_terms
from data_processing import DataProcessor
from api.pydantic_models import (
    TransactionData, CustomerTransactions, PredictionResponse,
    BatchPredictionResponse, ModelInfo, ErrorResponse, LoanTerms
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log")
    ]
)
logger = logging.getLogger("credit-risk-api")

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using alternative data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code=500,
            message="Internal server error",
            details={"error": str(exc)}
        ).dict()
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup"""
    global predictor
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize predictor with model path
        predictor = RiskPredictor(model_path="models/best_model.joblib")
        
        # Load preprocessing pipeline
        predictor.load_preprocessing_pipeline("models/preprocessing_pipeline.joblib")
        
        logger.info("Model and preprocessing pipeline loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.info("API will still work but you need to train a model first")
        # Initialize empty predictor for API to work
        predictor = RiskPredictor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None and predictor.model is not None
    }

def check_model_loaded():
    """Check if model is loaded and raise exception if not"""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first."
        )
    return predictor

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    customer_data: CustomerTransactions,
    predictor: RiskPredictor = Depends(check_model_loaded)
):
    """Predict credit risk for a single customer"""
    try:
        # Convert transactions to DataFrame
        transactions_df = pd.DataFrame([t.dict() for t in customer_data.transactions])
        
        # Process transactions to get customer features
        processor = DataProcessor()
        customer_features = processor.process_customer_transactions(
            customer_data.customer_id, 
            transactions_df
        )
        
        # Make prediction
        result = predictor.predict_comprehensive(customer_features)
        
        # Get recommended loan terms
        loan_terms_dict = predict_optimal_loan_terms(
            result['risk_probability'].iloc[0],
            base_amount=10000  # Default max loan amount
        )
        
        # Convert to LoanTerms model
        loan_terms = LoanTerms(
            max_amount=loan_terms_dict['max_amount'],
            recommended_amount=loan_terms_dict['recommended_amount'],
            duration_months=loan_terms_dict['duration_months'],
            interest_rate=loan_terms_dict['interest_rate'],
            monthly_payment=loan_terms_dict['monthly_payment']
        )
        
        return PredictionResponse(
            customer_id=customer_data.customer_id,
            risk_probability=float(result['risk_probability'].iloc[0]),
            risk_class=result['risk_level'].iloc[0],
            credit_score=int(result['credit_score'].iloc[0]),
            recommended_loan_terms=loan_terms,
            prediction_timestamp=datetime.now(),
            model_version="1.0.0"  # Replace with actual version from model metadata
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    customers_data: List[CustomerTransactions],
    predictor: RiskPredictor = Depends(check_model_loaded)
):
    """Predict credit risk for multiple customers"""
    try:
        start_time = time.time()
        predictions = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        
        processor = DataProcessor()
        
        for customer_data in customers_data:
            # Convert transactions to DataFrame
            transactions_df = pd.DataFrame([t.dict() for t in customer_data.transactions])
            
            # Process transactions to get customer features
            customer_features = processor.process_customer_transactions(
                customer_data.customer_id, 
                transactions_df
            )
            
            # Make prediction
            result = predictor.predict_comprehensive(customer_features)
            
            # Get recommended loan terms
            loan_terms_dict = predict_optimal_loan_terms(
                result['risk_probability'].iloc[0],
                base_amount=10000
            )
            
            # Convert to LoanTerms model
            loan_terms = LoanTerms(
                max_amount=loan_terms_dict['max_amount'],
                recommended_amount=loan_terms_dict['recommended_amount'],
                duration_months=loan_terms_dict['duration_months'],
                interest_rate=loan_terms_dict['interest_rate'],
                monthly_payment=loan_terms_dict['monthly_payment']
            )
            
            prediction = PredictionResponse(
                customer_id=customer_data.customer_id,
                risk_probability=float(result['risk_probability'].iloc[0]),
                risk_class=result['risk_level'].iloc[0],
                credit_score=int(result['credit_score'].iloc[0]),
                recommended_loan_terms=loan_terms,
                prediction_timestamp=datetime.now(),
                model_version="1.0.0"  # Replace with actual version
            )
            
            predictions.append(prediction)
            risk_counts[result['risk_level'].iloc[0]] += 1
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(customers_data),
            high_risk_count=risk_counts.get("High", 0),
            medium_risk_count=risk_counts.get("Medium", 0),
            low_risk_count=risk_counts.get("Low", 0),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.post("/explain/{customer_id}")
async def explain_prediction(
    customer_id: str,
    customer_data: CustomerTransactions,
    predictor: RiskPredictor = Depends(check_model_loaded)
):
    """Explain prediction for a customer using SHAP values"""
    try:
        # Verify customer ID matches
        if customer_id != customer_data.customer_id:
            raise HTTPException(
                status_code=400, 
                detail="Customer ID in path does not match customer ID in request body"
            )
        
        # Convert transactions to DataFrame
        transactions_df = pd.DataFrame([t.dict() for t in customer_data.transactions])
        
        # Process transactions to get customer features
        processor = DataProcessor()
        customer_features = processor.process_customer_transactions(
            customer_data.customer_id, 
            transactions_df
        )
        
        # Get explanation
        explanation = predictor.explain_prediction(customer_features)
        
        # Add prediction results
        result = predictor.predict_comprehensive(customer_features)
        explanation.update({
            'customer_id': customer_id,
            'risk_probability': float(result['risk_probability'].iloc[0]),
            'risk_class': result['risk_level'].iloc[0],
            'credit_score': int(result['credit_score'].iloc[0])
        })
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Explanation failed: {str(e)}")

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with new data"""
    try:
        from train import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer(experiment_name="credit_risk_model_retraining")
        
        # Load and process data
        logger.info("Loading and processing data for retraining")
        # Implement data loading and processing logic here
        
        # Train model
        logger.info("Training model")
        # Implement model training logic here
        
        # Update global predictor
        global predictor
        predictor = RiskPredictor(model_path="models/best_model.joblib")
        predictor.load_preprocessing_pipeline("models/preprocessing_pipeline.joblib")
        
        return {"status": "success", "message": "Model retrained successfully"}
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def model_info(predictor: RiskPredictor = Depends(check_model_loaded)):
    """Get information about the currently loaded model"""
    try:
        # Get model metadata
        model_metadata = {
            "model_name": "credit_risk_model",
            "model_version": "1.0.0",  # Replace with actual version
            "creation_date": datetime.now(),  # Replace with actual creation date
            "features": predictor.feature_names if predictor.feature_names is not None else [],
            "metrics": {
                "accuracy": 0.85,  # Replace with actual metrics
                "precision": 0.82,
                "recall": 0.79,
                "f1_score": 0.80,
                "roc_auc": 0.88
            },
            "description": "Credit risk model using alternative data from eCommerce transactions"
        }
        
        return ModelInfo(**model_metadata)
        
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)