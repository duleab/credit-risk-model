"""
Data Processing Module for Credit Risk Model

This module handles all data preprocessing, feature engineering,
and target variable creation for the credit risk model.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from typing import Tuple, Dict, Any
import warnings
import config

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=config.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Comprehensive data processing pipeline for credit risk modeling.
    
    This class handles:
    - Data loading and validation
    - Feature engineering
    - RFM analysis
    - Target variable creation using clustering
    - Data preprocessing for ML models
    """
    
    def __init__(self):
        """
        Initialize the DataProcessor.
        """
        self.snapshot_date = config.SNAPSHOT_DATE
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans_model = None
        self.preprocessing_pipeline = None
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the transaction data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation
            required_columns = [
                'TransactionId', 'CustomerId', 'Amount', 'Value', 
                'TransactionStartTime', 'ProductCategory', 'ChannelId'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Args:
            df: Raw transaction data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Convert TransactionStartTime to datetime
        df_clean['TransactionStartTime'] = pd.to_datetime(
            df_clean['TransactionStartTime'], errors='coerce'
        )
        
        # Remove rows with invalid dates
        df_clean = df_clean.dropna(subset=['TransactionStartTime'])
        
        # Handle missing values in Amount and Value
        df_clean['Amount'] = pd.to_numeric(df_clean['Amount'], errors='coerce')
        df_clean['Value'] = pd.to_numeric(df_clean['Value'], errors='coerce')
        
        # Remove transactions with zero or negative amounts
        df_clean = df_clean[df_clean['Value'] > 0]
        
        # Remove duplicate transactions
        df_clean = df_clean.drop_duplicates(subset=['TransactionId'])
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from transaction timestamps.
        
        Args:
            df: DataFrame with TransactionStartTime column
            
        Returns:
            pd.DataFrame: DataFrame with additional temporal features
        """
        logger.info("Extracting temporal features...")
        
        df_temp = df.copy()
        
        # Extract temporal components
        df_temp['transaction_hour'] = df_temp['TransactionStartTime'].dt.hour
        df_temp['transaction_day'] = df_temp['TransactionStartTime'].dt.day
        df_temp['transaction_month'] = df_temp['TransactionStartTime'].dt.month
        df_temp['transaction_year'] = df_temp['TransactionStartTime'].dt.year
        df_temp['transaction_dayofweek'] = df_temp['TransactionStartTime'].dt.dayofweek
        df_temp['transaction_quarter'] = df_temp['TransactionStartTime'].dt.quarter
        
        # Create time-based categories
        df_temp['is_weekend'] = df_temp['transaction_dayofweek'].isin([5, 6]).astype(int)
        df_temp['is_business_hours'] = (
            (df_temp['transaction_hour'] >= 9) & 
            (df_temp['transaction_hour'] <= 17)
        ).astype(int)
        
        logger.info("Temporal features extracted successfully")
        return df_temp
        
    def calculate_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) features for each customer.
        
        Args:
            df: Transaction data
            
        Returns:
            pd.DataFrame: Customer-level RFM features
        """
        logger.info("Calculating RFM features...")
        
        # Ensure snapshot_dt is timezone-naive to match TransactionStartTime
        snapshot_dt = pd.to_datetime(self.snapshot_date).tz_localize(None)
        
        # Calculate RFM metrics
        rfm_data = df.groupby('CustomerId').agg({
            'TransactionStartTime': ['max', 'count'],
            'Value': ['sum', 'mean', 'std', 'min', 'max'],
            'TransactionId': 'count'
        }).reset_index()
        
        # Flatten column names
        rfm_data.columns = ['CustomerId', 'last_transaction_date', 'frequency',
                           'monetary_sum', 'monetary_mean', 'monetary_std', 
                           'monetary_min', 'monetary_max', 'transaction_count']
        
        # Ensure last_transaction_date is also timezone-naive
        rfm_data['last_transaction_date'] = pd.to_datetime(rfm_data['last_transaction_date']).dt.tz_localize(None)
        
        # Calculate recency (days since last transaction)
        rfm_data['recency'] = (
            snapshot_dt - rfm_data['last_transaction_date']
        ).dt.days
        
        # Handle missing standard deviation (customers with single transaction)
        rfm_data['monetary_std'] = rfm_data['monetary_std'].fillna(0)
        
        # Add additional aggregate features
        customer_features = df.groupby('CustomerId').agg({
            'ProductCategory': 'nunique',
            'ChannelId': 'nunique',
            'transaction_hour': ['mean', 'std'],
            'is_weekend': 'mean',
            'is_business_hours': 'mean'
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'CustomerId', 'unique_categories', 'unique_channels',
            'avg_transaction_hour', 'std_transaction_hour',
            'weekend_transaction_ratio', 'business_hours_ratio'
        ]
        
        # Fill NaN values
        customer_features['std_transaction_hour'] = customer_features['std_transaction_hour'].fillna(0)
        
        # Merge RFM with additional features
        rfm_final = rfm_data.merge(customer_features, on='CustomerId', how='left')
        
        logger.info(f"RFM features calculated for {len(rfm_final)} customers")
        return rfm_final
    
    def create_risk_labels(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk labels using K-Means clustering on RFM features.
        
        Args:
            rfm_df: DataFrame with RFM features
            
        Returns:
            pd.DataFrame: DataFrame with risk labels
        """
        logger.info("Creating risk labels using clustering...")
        
        # Select features for clustering
        X_cluster = rfm_df[config.CLUSTER_FEATURES].copy()
        
        # Scale features for clustering
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Perform K-Means clustering
        self.kmeans_model = KMeans(
            n_clusters=config.N_CLUSTERS,
            random_state=config.RANDOM_STATE,
            n_init=10
        )
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels
        rfm_df['cluster'] = clusters
        
        # Analyze clusters to identify high-risk segment
        cluster_analysis = rfm_df.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary_sum': 'mean'
        })
        
        logger.info("Cluster Analysis:")
        logger.info(cluster_analysis)
        
        # Identify high-risk cluster (high recency, low frequency, low monetary)
        # Normalize metrics for comparison
        cluster_analysis_norm = cluster_analysis.copy()
        cluster_analysis_norm['recency_norm'] = cluster_analysis['recency'] / cluster_analysis['recency'].max()
        cluster_analysis_norm['frequency_norm'] = cluster_analysis['frequency'] / cluster_analysis['frequency'].max()
        cluster_analysis_norm['monetary_norm'] = cluster_analysis['monetary_sum'] / cluster_analysis['monetary_sum'].max()
        
        # Risk score: high recency (bad), low frequency (bad), low monetary (bad)
        cluster_analysis_norm['risk_score'] = (
            cluster_analysis_norm['recency_norm'] - 
            cluster_analysis_norm['frequency_norm'] - 
            cluster_analysis_norm['monetary_norm']
        )
        
        high_risk_cluster = cluster_analysis_norm['risk_score'].idxmax()
        
        # Create binary risk label
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        logger.info(f"High-risk cluster identified: {high_risk_cluster}")
        logger.info(f"High-risk customers: {rfm_df['is_high_risk'].sum()} ({rfm_df['is_high_risk'].mean():.2%})")
        
        return rfm_df
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Create a preprocessing pipeline for ML models.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Pipeline: Sklearn preprocessing pipeline
        """
        logger.info("Creating preprocessing pipeline...")
        
        # Identify feature types
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = ['CustomerId', 'is_high_risk', 'cluster', 'last_transaction_date']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        categorical_features = [col for col in categorical_features if col not in exclude_cols]
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.preprocessing_pipeline = preprocessor
        self.feature_names = numeric_features + categorical_features
        
        logger.info(f"Pipeline created with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        return preprocessor
    
    def process_full_pipeline(self, file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete data processing pipeline.
        
        Args:
            file_path: Path to the raw data file. Defaults to path from config.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (transaction_data, customer_features)
        """
        logger.info("Starting full data processing pipeline...")
        
        if file_path is None:
            file_path = config.RAW_DATA_PATH

        # Load and clean data
        df = self.load_data(file_path)
        df_clean = self.clean_data(df)
        
        # Extract temporal features
        df_features = self.extract_temporal_features(df_clean)
        
        # Calculate RFM features
        rfm_features = self.calculate_rfm_features(df_features)
        
        # Create risk labels
        customer_data = self.create_risk_labels(rfm_features)
        
        # Create preprocessing pipeline
        self.create_preprocessing_pipeline(customer_data)
        
        logger.info("Full pipeline completed successfully")
        return df_features, customer_data
    
    def save_processed_data(self, transaction_data: pd.DataFrame, 
                           customer_data: pd.DataFrame) -> None:
        """
        Save processed data to files.
        
        Args:
            transaction_data: Processed transaction data
            customer_data: Customer-level features with risk labels
        """
        import os
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        
        transaction_data.to_csv(config.TRANSACTIONS_PROCESSED_PATH, index=False)
        customer_data.to_csv(config.CUSTOMER_FEATURES_PATH, index=False)
        
        logger.info(f"Processed data saved to {config.PROCESSED_DATA_DIR}")


def calculate_woe_iv(df: pd.DataFrame, feature_col: str, target_col: str) -> Dict[str, Any]:
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a feature.
    
    Args:
        df: DataFrame containing the feature and target
        feature_col: Name of the feature column
        target_col: Name of the target column
        
    Returns:
        Dict containing WoE mapping and IV value
    """
    # Create crosstab
    crosstab = pd.crosstab(df[feature_col], df[target_col])
    
    # Calculate WoE and IV
    crosstab['Total'] = crosstab.sum(axis=1)
    crosstab['Good_Rate'] = crosstab[0] / crosstab[0].sum()
    crosstab['Bad_Rate'] = crosstab[1] / crosstab[1].sum()
    
    # Avoid division by zero
    crosstab['Good_Rate'] = crosstab['Good_Rate'].replace(0, 0.0001)
    crosstab['Bad_Rate'] = crosstab['Bad_Rate'].replace(0, 0.0001)
    
    crosstab['WoE'] = np.log(crosstab['Good_Rate'] / crosstab['Bad_Rate'])
    crosstab['IV'] = (crosstab['Good_Rate'] - crosstab['Bad_Rate']) * crosstab['WoE']
    
    iv_value = crosstab['IV'].sum()
    woe_mapping = crosstab['WoE'].to_dict()
    
    return {
        'woe_mapping': woe_mapping,
        'iv_value': iv_value,
        'crosstab': crosstab
    }


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    # Process data
    transaction_data, customer_data = processor.process_full_pipeline()
    
    # Save processed data
    processor.save_processed_data(transaction_data, customer_data)
    
    print("Data processing completed successfully!")