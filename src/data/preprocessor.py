"""
Preprocessing and Feature Engineering Module.

Handles data cleaning, transformation, and feature creation.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handle data cleaning operations."""

    def __init__(self):
        self.cleaning_stats = {}

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)

        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
            self.cleaning_stats["duplicates_removed"] = removed

        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        """
        Handle missing values.

        Args
        ----
            df: Input DataFrame
            strategy: 'drop', 'mean', 'median', or 'mode'
        """
        missing_before = df.isnull().sum().sum()

        if missing_before == 0:
            logger.info("No missing values found")
            return df

        if strategy == "drop":
            df = df.dropna()
            logger.info(f"Dropped {missing_before} rows with missing values")
        elif strategy in ["mean", "median", "mode"]:
            numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
            numeric_cols: List[str] = numeric_df.columns.tolist()
            for col in numeric_cols:
                if df[col].isnull().any():
                    col_series = df[col]
                    if strategy == "mean":
                        fill_value = col_series.mean()
                        df[col] = col_series.fillna(fill_value)
                    elif strategy == "median":
                        fill_value = col_series.median()
                        df[col] = col_series.fillna(fill_value)
            logger.info(f"Filled {missing_before} missing values using {strategy}")

        self.cleaning_stats["missing_values_handled"] = missing_before
        return df

    def remove_outliers(
        self, df: pd.DataFrame, columns: List[str], method: str = "iqr", threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns.

        Args
        ----
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        """
        initial_len = len(df)

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask: Series = (df[col] >= lower_bound) & (df[col] <= upper_bound)
                df = df.loc[mask]

            elif method == "zscore":
                col_mean = df[col].mean()
                col_std = df[col].std()
                z_scores = np.abs((df[col] - col_mean) / col_std)
                mask = z_scores < threshold
                df = df[mask]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outlier rows")
            self.cleaning_stats["outliers_removed"] = removed

        return df

    def clean(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        missing_strategy: str = "drop",
        outlier_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply all cleaning operations.

        Args
        ----
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicates
            missing_strategy: Strategy for handling missing values
            outlier_columns: Columns to check for outliers
        """
        df = df.copy()
        logger.info("Starting data cleaning")

        if remove_duplicates:
            df = self.remove_duplicates(df)

        df = self.handle_missing_values(df, strategy=missing_strategy)

        if outlier_columns:
            df = self.remove_outliers(df, columns=outlier_columns)

        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df


class FeatureEngineer:
    """Create and transform features for fraud detection."""

    def __init__(self):
        self.feature_names = []

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hour, is_night, and cyclical time features."""
        df = df.copy()
        df["hour"] = (df["Time"] / 3600) % 24
        df["is_night"] = ((df["hour"] > 22) | (df["hour"] < 6)).astype(int)
        df["day_sin"] = np.sin(2 * np.pi * df["Time"] / 86400)
        df["day_cos"] = np.cos(2 * np.pi * df["Time"] / 86400)

        logger.info("Added time-based features")
        return df

    def add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log, small, and large amount features."""
        df = df.copy()
        df["log_amount"] = np.log1p(df["Amount"])
        df["is_small_amount"] = (df["Amount"] < 5).astype(int)
        df["is_large_amount"] = (df["Amount"] > 1000).astype(int)
        df["amount_squared"] = df["Amount"] ** 2
        df["amount_sqrt"] = np.sqrt(df["Amount"])

        logger.info("Added amount-based features")
        return df

    def add_statistical_features(
        self, df: pd.DataFrame, pca_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add statistical aggregations of PCA features.

        Args
        ----
            df: Input DataFrame
            pca_columns: List of PCA column names (e.g., ['V1', 'V2', ...])
        """
        df = df.copy()

        if pca_columns is None:
            pca_columns = [col for col in df.columns if col.startswith("V")]

        if pca_columns:
            df["pca_mean"] = df[pca_columns].mean(axis=1)
            df["pca_std"] = df[pca_columns].std(axis=1)
            df["pca_min"] = df[pca_columns].min(axis=1)
            df["pca_max"] = df[pca_columns].max(axis=1)
            df["pca_range"] = df["pca_max"] - df["pca_min"]

            logger.info("Added statistical features")

        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables."""
        df = df.copy()

        if "hour" in df.columns and "Amount" in df.columns:
            df["hour_amount_interaction"] = df["hour"] * df["Amount"]

        if "is_night" in df.columns and "log_amount" in df.columns:
            df["night_log_amount"] = df["is_night"] * df["log_amount"]

        logger.info("Added interaction features")
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        include_time: bool = True,
        include_amount: bool = True,
        include_stats: bool = True,
        include_interactions: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args
        ----
            df: Input DataFrame
            include_time: Add time-based features
            include_amount: Add amount-based features
            include_stats: Add statistical features
            include_interactions: Add interaction features
        """
        df = df.copy()
        logger.info("Starting feature engineering")

        if include_time:
            df = self.add_time_features(df)

        if include_amount:
            df = self.add_amount_features(df)

        if include_stats:
            df = self.add_statistical_features(df)

        if include_interactions:
            df = self.add_interaction_features(df)

        self.feature_names = df.columns.tolist()
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")

        return df


class FeatureScaler:
    """Scale features for modeling."""

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize scaler.

        Args
        ----
            scaler_type: 'standard' or 'robust'
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        self.feature_columns: Optional[List[str]] = None
        self.target_column: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_col: str = "Class") -> "FeatureScaler":
        """
        Fit scaler on training data.

        Args
        ----
            df: Training DataFrame
            target_col: Name of target column to exclude from scaling
        """
        self.target_column = target_col
        self.feature_columns = [col for col in df.columns if col != target_col]

        self.scaler.fit(df[self.feature_columns])
        logger.info(f"Fitted scaler on {len(self.feature_columns)} features")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        df = df.copy()

        if self.feature_columns is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        scaled_features = self.scaler.transform(df[self.feature_columns])
        df[self.feature_columns] = scaled_features

        logger.info("Data scaled successfully")
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df)


class PreprocessingPipeline:
    """Complete preprocessing pipeline orchestrator."""

    def __init__(
        self,
        cleaning_config: Optional[dict] = None,
        feature_config: Optional[dict] = None,
        scaler_type: str = "standard",
    ):
        """
        Initialize preprocessing pipeline.

        Args
        ----
            cleaning_config: Configuration for data cleaning
            feature_config: Configuration for feature engineering
            scaler_type: Type of scaler to use
        """
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.scaler = FeatureScaler(scaler_type=scaler_type)

        self.cleaning_config = cleaning_config or {}
        self.feature_config = feature_config or {}

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = "Class") -> "PreprocessingPipeline":
        """
        Fit pipeline on training data.

        Args
        ----
            df: Training DataFrame
            target_col: Target column name
        """
        logger.info("Fitting preprocessing pipeline")

        # Clean data
        df_clean = self.cleaner.clean(df, **self.cleaning_config)

        # Engineer features
        df_features = self.engineer.engineer_features(df_clean, **self.feature_config)

        # Fit scaler
        self.scaler.fit(df_features, target_col=target_col)

        self.is_fitted = True
        logger.info("Pipeline fitted successfully")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args
        ----
            df: DataFrame to transform
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        logger.info("Transforming data with fitted pipeline")

        # Clean data
        df_clean = self.cleaner.clean(df, **self.cleaning_config)

        # Engineer features
        df_features = self.engineer.engineer_features(df_clean, **self.feature_config)

        # Scale features
        df_scaled = self.scaler.transform(df_features)

        logger.info("Data transformation complete")
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df)
