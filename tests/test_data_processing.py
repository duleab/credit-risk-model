import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import pytest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import DataProcessor, calculate_woe_iv


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create a sample DataFrame for testing
        self.test_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3'],
            'Amount': [100.0, 200.0, 150.0, 300.0, 50.0, 250.0],
            'Value': [100, 200, 150, 300, 50, 250],
            'TransactionStartTime': [
                '2023-01-01 10:00:00',
                '2023-01-15 14:30:00',
                '2023-01-05 09:15:00',
                '2023-01-20 16:45:00',
                '2023-01-10 11:30:00',
                '2023-01-25 13:00:00'
            ],
            'ProductCategory': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing', 'Food'],
            'ChannelId': ['WEB', 'APP', 'WEB', 'APP', 'WEB', 'APP']
        })
        
        # Convert TransactionStartTime to datetime
        self.test_data['TransactionStartTime'] = pd.to_datetime(self.test_data['TransactionStartTime'])
        
        # Initialize DataProcessor with a fixed snapshot date
        self.snapshot_date = '2023-02-01'
        self.processor = DataProcessor(snapshot_date=self.snapshot_date)
        
    def test_clean_data(self):
        """Test data cleaning functionality"""
        # Create test data with issues
        dirty_data = self.test_data.copy()
        dirty_data.loc[0, 'Amount'] = np.nan  # Add a missing value
        dirty_data.loc[1, 'Value'] = 0  # Add a zero value
        dirty_data.loc[2, 'TransactionStartTime'] = np.nan  # Add invalid date
        
        # Add duplicate transaction
        duplicate_row = dirty_data.iloc[3].copy()
        dirty_data = pd.concat([dirty_data, pd.DataFrame([duplicate_row])], ignore_index=True)
        
        # Clean the data
        cleaned_data = self.processor.clean_data(dirty_data)
        
        # Check that issues were resolved
        self.assertFalse(cleaned_data['Amount'].isna().any(), "Missing values should be handled")
        self.assertTrue((cleaned_data['Value'] > 0).all(), "Zero or negative values should be removed")
        self.assertEqual(len(cleaned_data), 5, "Rows with invalid dates and duplicates should be removed")
        
    def test_extract_temporal_features(self):
        """Test extraction of temporal features"""
        # Process the data
        processed_data = self.processor.extract_temporal_features(self.test_data)
        
        # Check that new columns were added
        expected_columns = [
            'transaction_hour', 'transaction_day', 'transaction_month',
            'transaction_year', 'transaction_dayofweek', 'transaction_quarter',
            'is_weekend', 'is_business_hours'
        ]
        
        for col in expected_columns:
            self.assertIn(col, processed_data.columns, f"Column {col} should be created")
        
        # Check specific values
        self.assertEqual(processed_data.loc[0, 'transaction_hour'], 10, "Hour should be extracted correctly")
        self.assertEqual(processed_data.loc[0, 'transaction_day'], 1, "Day should be extracted correctly")
        self.assertEqual(processed_data.loc[0, 'transaction_month'], 1, "Month should be extracted correctly")
        
    def test_calculate_rfm_features(self):
        """Test calculation of RFM features"""
        # Process the data
        rfm_data = self.processor.calculate_rfm_features(self.test_data)
        
        # Check that RFM metrics were calculated
        self.assertEqual(len(rfm_data), 3, "Should have one row per customer")
        
        # Check that recency is calculated correctly
        snapshot_dt = pd.to_datetime(self.snapshot_date)
        c1_last_date = pd.to_datetime('2023-01-15 14:30:00')
        expected_recency_c1 = (snapshot_dt - c1_last_date).days
        
        # Find the row for customer C1
        c1_row = rfm_data[rfm_data['CustomerId'] == 'C1']
        self.assertEqual(c1_row['recency'].values[0], expected_recency_c1, 
                         "Recency should be calculated correctly")
        
        # Check frequency
        self.assertEqual(c1_row['frequency'].values[0], 2, 
                         "Frequency should count transactions correctly")
        
        # Check monetary
        self.assertEqual(c1_row['monetary_sum'].values[0], 300, 
                         "Monetary sum should be calculated correctly")


def test_woe_iv_calculation():
    """Test Weight of Evidence and Information Value calculation"""
    # Create sample data
    data = pd.DataFrame({
        'feature': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
        'target': [0, 0, 0, 1, 1, 1, 1]
    })
    
    # Calculate WoE and IV
    result = calculate_woe_iv(data, 'feature', 'target')
    
    # Check that the result has the expected keys
    assert 'woe_dict' in result, "Result should contain WoE dictionary"
    assert 'iv' in result, "Result should contain IV value"
    
    # Check WoE values
    woe_dict = result['woe_dict']
    assert 'A' in woe_dict, "Category A should be in WoE dictionary"
    assert 'B' in woe_dict, "Category B should be in WoE dictionary"
    assert 'C' in woe_dict, "Category C should be in WoE dictionary"
    
    # A has only good cases (target=0), so WoE should be positive
    assert woe_dict['A'] > 0, "WoE for category A should be positive"
    
    # C has only bad cases (target=1), so WoE should be negative
    assert woe_dict['C'] < 0, "WoE for category C should be negative"
    
    # Check IV value is positive
    assert result['iv'] > 0, "IV should be positive"


if __name__ == '__main__':
    unittest.main()
