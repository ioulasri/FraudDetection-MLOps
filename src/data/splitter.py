"""
Data Splitting Module.

Handles train/validation/test splits with stratification.
"""

import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train, validation, and test sets."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """
        Initialize DataSplitter.

        Args
        ----
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits by target variable
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self, df: pd.DataFrame, target_col: str = "Class"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args
        ----
            df: Input DataFrame
            target_col: Name of target column

        Returns
        -------
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data: test_size={self.test_size}, val_size={self.val_size}")

        # First split: train+val vs test
        stratify_col = df[target_col] if self.stratify else None

        train_val, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state, stratify=stratify_col
        )

        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        stratify_col_train = train_val[target_col] if self.stratify else None

        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_col_train,
        )

        logger.info(f"Split complete: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        logger.info(f"Train fraud rate: {train[target_col].mean():.4f}")
        logger.info(f"Val fraud rate: {val[target_col].mean():.4f}")
        logger.info(f"Test fraud rate: {test[target_col].mean():.4f}")

        return train, val, test

    def split_X_y(self, df: pd.DataFrame, target_col: str = "Class") -> Tuple:
        """
        Split data and separate features from target.

        Args
        ----
            df: Input DataFrame
            target_col: Name of target column

        Returns
        -------
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train, val, test = self.split(df, target_col)

        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]

        X_val = val.drop(columns=[target_col])
        y_val = val[target_col]

        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str = "data/splits",
    ) -> None:
        """
        Save splits to CSV files.

        Args
        ----
            train: Training DataFrame
            val: Validation DataFrame
            test: Test DataFrame
            output_dir: Directory to save splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train.to_csv(output_path / "train.csv", index=False)
        val.to_csv(output_path / "val.csv", index=False)
        test.to_csv(output_path / "test.csv", index=False)

        logger.info(f"Splits saved to {output_dir}")

    def load_splits(
        self, input_dir: str = "data/splits"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load splits from CSV files.

        Args
        ----
            input_dir: Directory containing split files

        Returns
        -------
            Tuple of (train_df, val_df, test_df)
        """
        input_path = Path(input_dir)

        train = pd.read_csv(input_path / "train.csv")
        val = pd.read_csv(input_path / "val.csv")
        test = pd.read_csv(input_path / "test.csv")

        logger.info(f"Splits loaded from {input_dir}")
        return train, val, test
