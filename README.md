# Credit Risk Model - Task 1: Business Understanding

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

## Project Status
- [x] Business context analysis
- [x] Basel II requirements understanding
- [x] Risk assessment framework
- [ ] Data exploration (Task 2)
- [ ] Feature engineering (Task 3)
