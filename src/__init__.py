"""
Credit Risk Model Package

A comprehensive credit risk modeling package for alternative data analysis.
"""

__version__ = "1.0.0"
__author__ = "Analytics Team"
__email__ = "analytics@batibank.com"

from .data_processing import DataProcessor
from .train import ModelTrainer
from .predict import RiskPredictor

__all__ = [
    "DataProcessor",
    "ModelTrainer", 
    "RiskPredictor"
]