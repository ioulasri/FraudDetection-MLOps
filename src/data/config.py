"""
Configuration Template for Preprocessing Pipeline
Modify these settings based on your needs.
"""

# ============================================
# DATA LOADING CONFIGURATION
# ============================================
DATA_CONFIG = {
    'raw_data_path': 'data/raw/creditcard.csv',
    'required_columns': ['Time', 'Amount', 'Class'],
    'target_column': 'Class',
}

# ============================================
# DATA SPLITTING CONFIGURATION
# ============================================
SPLIT_CONFIG = {
    'test_size': 0.2,        # 20% for testing
    'val_size': 0.1,         # 10% of remaining for validation
    'random_state': 42,      # For reproducibility
    'stratify': True,        # Maintain class distribution
    'save_splits': True,     # Save splits to disk
    'splits_dir': 'data/splits',
}

# ============================================
# DATA CLEANING CONFIGURATION
# ============================================
CLEANING_CONFIG = {
    'remove_duplicates': True,
    'missing_strategy': 'drop',  # Options: 'drop', 'mean', 'median', 'mode'
    'outlier_columns': None,     # Set to ['Amount'] to remove outliers, or None to skip
    'outlier_method': 'iqr',     # Options: 'iqr', 'zscore'
    'outlier_threshold': 3.0,    # IQR multiplier or z-score threshold
}

# ============================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================
FEATURE_CONFIG = {
    'include_time': True,         # Time-based features (hour, is_night, cyclical)
    'include_amount': True,       # Amount transformations (log, flags, etc.)
    'include_stats': True,        # Statistical features (mean, std, etc.)
    'include_interactions': True, # Interaction features
}

# ============================================
# SCALING CONFIGURATION
# ============================================
SCALING_CONFIG = {
    'scaler_type': 'robust',  # Options: 'standard', 'robust'
    # 'standard': Good for normally distributed features
    # 'robust': Better for features with outliers (recommended for fraud detection)
}

# ============================================
# OUTPUT CONFIGURATION
# ============================================
OUTPUT_CONFIG = {
    'processed_data_dir': 'data/processed',
    'save_processed': True,      # Save processed data to disk
    'log_level': 'INFO',         # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
}

# ============================================
# EXPERIMENT TRACKING
# ============================================
EXPERIMENT_CONFIG = {
    'experiment_name': 'fraud_detection_baseline',
    'description': 'Initial preprocessing pipeline with all features',
    'version': '1.0.0',
}

# ============================================
# HELPER FUNCTION TO GET COMPLETE CONFIG
# ============================================
def get_complete_config():
    """Get all configuration as a single dictionary."""
    return {
        'data': DATA_CONFIG,
        'split': SPLIT_CONFIG,
        'cleaning': CLEANING_CONFIG,
        'features': FEATURE_CONFIG,
        'scaling': SCALING_CONFIG,
        'output': OUTPUT_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
    }


def print_config():
    """Print current configuration."""
    import json
    config = get_complete_config()
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    print_config()
