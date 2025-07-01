# Credit Risk Model

A comprehensive credit risk modeling solution for predicting credit risk using alternative data from eCommerce transactions.

## Project Overview

This project implements a credit risk model for a buy-now-pay-later service, enabling a financial institution to evaluate creditworthiness based on eCommerce transaction data. The model uses behavioral patterns to create a proxy for traditional credit risk indicators, making it suitable for scenarios where direct default data is unavailable.

## Credit Scoring Business Understanding

### Basel II Accord Influence on Model Requirements

The Basel II Accord's emphasis on risk measurement directly influences our need for an interpretable and well-documented model because:

- **Regulatory Transparency**: Models must be explainable to regulators and auditors
- **Risk-Weighted Asset Calculation**: Clear methodology for quantifying credit risk exposure
- **Model Validation Requirements**: Comprehensive documentation for regulatory approval
- **Interpretability Standards**: Ability to explain individual predictions and overall model behavior

### Proxy Variable Necessity and Business Risks

Since we lack a direct "default" label, creating a proxy variable is necessary because:

**Why Proxy is Needed:**
- No historical default data available from eCommerce transactions
- Need to identify high-risk customer segments using behavioral patterns
- RFM analysis provides risk signals through engagement patterns

**Potential Business Risks:**
- **False Positives**: Rejecting creditworthy customers (lost revenue)
- **False Negatives**: Approving high-risk customers (potential losses)
- **Proxy Validity**: Behavioral disengagement may not correlate with credit risk
- **Regulatory Scrutiny**: Using non-traditional risk indicators

### Model Trade-offs: Simple vs Complex

**Simple Interpretable Models (Logistic Regression + WoE):**
- ✅ Regulatory compliance and explainability
- ✅ Clear audit trail and feature importance
- ✅ Easy to validate and maintain
- ❌ Potentially lower predictive accuracy
- ❌ Limited handling of complex interactions

**Complex High-Performance Models (Gradient Boosting):**
- ✅ Higher predictive accuracy
- ✅ Better handling of non-linear relationships
- ✅ Automatic feature interaction detection
- ❌ Reduced interpretability ("black box")
- ❌ Regulatory approval challenges
- ❌ Difficult to explain individual predictions

**Recommendation for Regulated Environment:**
Use Logistic Regression with WoE transformations as primary model for regulatory compliance, with Gradient Boosting as a benchmark for performance comparison.

## Features

- **Data Processing Pipeline**: Comprehensive data cleaning and preprocessing
- **Feature Engineering**: Temporal features, RFM analysis, WoE transformations
- **Proxy Target Creation**: Risk labeling using clustering techniques
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for model versioning and metrics
- **Model Explainability**: SHAP values for transparent predictions
- **API Service**: FastAPI endpoints for real-time predictions
- **Containerization**: Docker and docker-compose for easy deployment
- **CI/CD**: GitHub Actions workflow for automated testing and deployment

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data for training
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 1.0-eda.ipynb         # Exploratory data analysis
│   ├── 2.0-Advanced-analysis.ipynb  # Advanced feature analysis
│   ├── 3.0-model-performance.ipynb  # Model evaluation
│   └── 4.0-business-impact.ipynb    # Business impact analysis
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_processing.py     # Data preprocessing and feature engineering
│   ├── train.py              # Model training and evaluation
│   ├── predict.py            # Prediction and inference
│   └── api/                  # API service
│       ├── main.py           # FastAPI application
│       └── pydantic_models.py # Data validation models
├── tests/                     # Unit tests
│   ├── test_data_processing.py
│   ├── test_train.py
│   └── test_api.py
├── models/                    # Saved model artifacts
├── logs/                      # Application logs
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Multi-container setup
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker and docker-compose (for containerized deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-risk-model.git
   cd credit-risk-model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place raw data in the `data/raw/` directory

### Docker Setup

1. Build and run the containers:
   ```bash
   docker-compose up --build
   ```

## Usage

### Data Processing

Run the data processing pipeline to prepare the data for modeling:

```bash
python -m src.data_processing
```

### Model Training

Train the credit risk model:

```bash
python -m src.train
```

### API Service

Start the API service locally:

```bash
uvicorn src.api.main:app --reload
```

The API documentation will be available at http://localhost:8000/docs

### Running Tests

Run the test suite:

```bash
pytest tests/
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API status |
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Predict credit risk for a single customer |
| `/predict/batch` | POST | Predict credit risk for multiple customers |
| `/explain/{customer_id}` | POST | Explain prediction using SHAP values |
| `/model/info` | GET | Get information about the current model |
| `/retrain` | POST | Retrain the model with new data |

## Model Explainability

The model provides explainability features using SHAP (SHapley Additive exPlanations) values, which help understand the contribution of each feature to the prediction. This is crucial for regulatory compliance and business understanding.

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. Runs linting checks with flake8
2. Executes unit tests with pytest
3. Builds and pushes Docker images
4. Deploys to staging and production environments

## Future Work

- Implement model monitoring for drift detection
- Add A/B testing framework for model comparison
- Enhance API security with authentication and rate limiting
- Develop a dashboard for business metrics visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Basel II Accord for regulatory framework guidance
- MLflow for experiment tracking capabilities
- FastAPI for the high-performance API framework