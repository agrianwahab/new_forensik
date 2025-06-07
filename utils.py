"""
Utility functions for Forensic Image Analysis System
"""

import numpy as np
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

def detect_outliers_iqr(data, factor=1.5):
    """Detect outliers using IQR method"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator
