"""
Data Loading Module
Handles loading raw data from various sources.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw data for fraud detection."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the data file or directory
        """
        self.data_path = Path(data_path)

    def load_csv(self, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")

        try:
            df = pd.read_csv(self.data_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
        """
        Validate loaded data.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info("Data validation passed")
        return True

    def load_and_validate(self, required_columns: Optional[list] = None, **kwargs) -> pd.DataFrame:
        """
        Load and validate data in one step.

        Args:
            required_columns: List of required column names
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Validated DataFrame
        """
        df = self.load_csv(**kwargs)
        self.validate_data(df, required_columns)
        return df
