"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from data.preprocessor import DataCleaner, FeatureEngineer, FeatureScaler, PreprocessingPipeline


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud detection data."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "Time": np.random.randint(0, 172800, n_samples),
        "Amount": np.random.exponential(50, n_samples),
        "Class": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }

    # Add PCA features V1-V5
    for i in range(1, 6):
        data[f"V{i}"] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def data_with_duplicates():
    """Create data with duplicates."""
    df = pd.DataFrame({"Time": [100, 200, 100], "Amount": [10.0, 20.0, 10.0], "Class": [0, 1, 0]})
    return df


@pytest.fixture
def data_with_missing():
    """Create data with missing values."""
    df = pd.DataFrame(
        {
            "Time": [100, 200, 300],
            "Amount": [10.0, np.nan, 30.0],
            "V1": [1.0, 2.0, np.nan],
            "Class": [0, 1, 0],
        }
    )
    return df


class TestDataCleaner:
    """Test cases for DataCleaner class."""

    def test_init(self):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner()
        assert cleaner.cleaning_stats == {}

    def test_remove_duplicates(self, data_with_duplicates):
        """Test duplicate removal."""
        cleaner = DataCleaner()
        df_clean = cleaner.remove_duplicates(data_with_duplicates)

        assert len(df_clean) == 2  # One duplicate removed
        assert "duplicates_removed" in cleaner.cleaning_stats

    def test_remove_duplicates_no_duplicates(self, sample_fraud_data):
        """Test duplicate removal when no duplicates exist."""
        cleaner = DataCleaner()
        df_clean = cleaner.remove_duplicates(sample_fraud_data)

        assert len(df_clean) == len(sample_fraud_data)

    def test_handle_missing_values_drop(self, data_with_missing):
        """Test missing value handling with drop strategy."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(data_with_missing, strategy="drop")

        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) < len(data_with_missing)

    def test_handle_missing_values_mean(self, data_with_missing):
        """Test missing value handling with mean strategy."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(data_with_missing, strategy="mean")

        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) == len(data_with_missing)

    def test_handle_missing_values_median(self, data_with_missing):
        """Test missing value handling with median strategy."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(data_with_missing, strategy="median")

        assert df_clean.isnull().sum().sum() == 0

    def test_handle_missing_values_no_missing(self, sample_fraud_data):
        """Test handling when no missing values exist."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(sample_fraud_data)

        assert len(df_clean) == len(sample_fraud_data)

    def test_remove_outliers_iqr(self, sample_fraud_data):
        """Test outlier removal using IQR method."""
        cleaner = DataCleaner()
        df_clean = cleaner.remove_outliers(
            sample_fraud_data, columns=["Amount"], method="iqr", threshold=1.5
        )

        assert len(df_clean) <= len(sample_fraud_data)

    def test_remove_outliers_zscore(self, sample_fraud_data):
        """Test outlier removal using z-score method."""
        cleaner = DataCleaner()
        df_clean = cleaner.remove_outliers(
            sample_fraud_data, columns=["Amount"], method="zscore", threshold=3.0
        )

        assert len(df_clean) <= len(sample_fraud_data)

    def test_remove_outliers_missing_column(self, sample_fraud_data):
        """Test outlier removal with non-existent column."""
        cleaner = DataCleaner()
        df_clean = cleaner.remove_outliers(sample_fraud_data, columns=["NonExistent"], method="iqr")

        # Should return unchanged data
        assert len(df_clean) == len(sample_fraud_data)

    def test_clean_full_pipeline(self, sample_fraud_data):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner()
        df_clean = cleaner.clean(
            sample_fraud_data,
            remove_duplicates=True,
            missing_strategy="drop",
            outlier_columns=["Amount"],
        )

        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) <= len(sample_fraud_data)


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer.feature_names == []

    def test_add_time_features(self, sample_fraud_data):
        """Test time-based feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer.add_time_features(sample_fraud_data)

        assert "hour" in df_features.columns
        assert "is_night" in df_features.columns
        assert "day_sin" in df_features.columns
        assert "day_cos" in df_features.columns

        # Check hour is in valid range
        assert df_features["hour"].between(0, 24).all()

    def test_add_amount_features(self, sample_fraud_data):
        """Test amount-based feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer.add_amount_features(sample_fraud_data)

        assert "log_amount" in df_features.columns
        assert "is_small_amount" in df_features.columns
        assert "is_large_amount" in df_features.columns
        assert "amount_squared" in df_features.columns
        assert "amount_sqrt" in df_features.columns

    def test_add_statistical_features(self, sample_fraud_data):
        """Test statistical feature creation."""
        engineer = FeatureEngineer()
        df_features = engineer.add_statistical_features(sample_fraud_data)

        assert "pca_mean" in df_features.columns
        assert "pca_std" in df_features.columns
        assert "pca_min" in df_features.columns
        assert "pca_max" in df_features.columns
        assert "pca_range" in df_features.columns

    def test_add_interaction_features(self, sample_fraud_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        # First add time and amount features for interactions
        df = engineer.add_time_features(sample_fraud_data)
        df = engineer.add_amount_features(df)
        df_features = engineer.add_interaction_features(df)

        assert "hour_amount_interaction" in df_features.columns
        assert "night_log_amount" in df_features.columns

    def test_engineer_features_full(self, sample_fraud_data):
        """Test full feature engineering pipeline."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(sample_fraud_data)

        # Should have more features than original
        assert len(df_features.columns) > len(sample_fraud_data.columns)
        assert engineer.feature_names == df_features.columns.tolist()

    def test_engineer_features_selective(self, sample_fraud_data):
        """Test selective feature engineering."""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(
            sample_fraud_data,
            include_time=True,
            include_amount=False,
            include_stats=False,
            include_interactions=False,
        )

        assert "hour" in df_features.columns
        assert "log_amount" not in df_features.columns


class TestFeatureScaler:
    """Test cases for FeatureScaler class."""

    def test_init_standard(self):
        """Test FeatureScaler initialization with standard scaler."""
        scaler = FeatureScaler(scaler_type="standard")
        assert scaler.feature_columns is None
        assert scaler.target_column is None

    def test_init_robust(self):
        """Test FeatureScaler initialization with robust scaler."""
        scaler = FeatureScaler(scaler_type="robust")
        assert scaler.feature_columns is None

    def test_init_invalid_scaler(self):
        """Test initialization with invalid scaler type."""
        with pytest.raises(ValueError, match="Unknown scaler type"):
            FeatureScaler(scaler_type="invalid")

    def test_fit(self, sample_fraud_data):
        """Test fitting the scaler."""
        scaler = FeatureScaler(scaler_type="standard")
        scaler.fit(sample_fraud_data, target_col="Class")

        assert scaler.target_column == "Class"
        assert "Class" not in scaler.feature_columns
        assert len(scaler.feature_columns) == len(sample_fraud_data.columns) - 1

    def test_transform(self, sample_fraud_data):
        """Test transforming data."""
        scaler = FeatureScaler(scaler_type="standard")
        scaler.fit(sample_fraud_data, target_col="Class")
        df_scaled = scaler.transform(sample_fraud_data)

        assert df_scaled.shape == sample_fraud_data.shape
        assert "Class" in df_scaled.columns

    def test_transform_not_fitted(self, sample_fraud_data):
        """Test transform without fitting first."""
        scaler = FeatureScaler(scaler_type="standard")

        with pytest.raises(ValueError, match="Scaler not fitted"):
            scaler.transform(sample_fraud_data)

    def test_fit_transform(self, sample_fraud_data):
        """Test fit and transform in one step."""
        scaler = FeatureScaler(scaler_type="standard")
        df_scaled = scaler.fit_transform(sample_fraud_data, target_col="Class")

        assert df_scaled.shape == sample_fraud_data.shape
        # Check that features are scaled (mean ~0, std ~1 for standard scaler)
        feature_cols = [col for col in df_scaled.columns if col != "Class"]
        assert abs(df_scaled[feature_cols].mean().mean()) < 1.0


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline class."""

    def test_init(self):
        """Test PreprocessingPipeline initialization."""
        pipeline = PreprocessingPipeline()

        assert pipeline.is_fitted is False
        assert pipeline.cleaning_config == {}
        assert pipeline.feature_config == {}

    def test_init_with_config(self):
        """Test initialization with custom config."""
        cleaning_config = {"remove_duplicates": True}
        feature_config = {"include_time": True}

        pipeline = PreprocessingPipeline(
            cleaning_config=cleaning_config, feature_config=feature_config, scaler_type="robust"
        )

        assert pipeline.cleaning_config == cleaning_config
        assert pipeline.feature_config == feature_config

    def test_fit(self, sample_fraud_data):
        """Test fitting the pipeline."""
        pipeline = PreprocessingPipeline()
        pipeline.fit(sample_fraud_data, target_col="Class")

        assert pipeline.is_fitted is True

    def test_transform_not_fitted(self, sample_fraud_data):
        """Test transform without fitting first."""
        pipeline = PreprocessingPipeline()

        with pytest.raises(ValueError, match="Pipeline not fitted"):
            pipeline.transform(sample_fraud_data)

    def test_transform(self, sample_fraud_data):
        """Test transforming data."""
        pipeline = PreprocessingPipeline()
        pipeline.fit(sample_fraud_data, target_col="Class")
        df_processed = pipeline.transform(sample_fraud_data)

        assert isinstance(df_processed, pd.DataFrame)
        assert "Class" in df_processed.columns

    def test_fit_transform(self, sample_fraud_data):
        """Test fit and transform in one step."""
        pipeline = PreprocessingPipeline(
            cleaning_config={"remove_duplicates": True, "missing_strategy": "drop"},
            feature_config={"include_time": True, "include_amount": True},
            scaler_type="standard",
        )

        df_processed = pipeline.fit_transform(sample_fraud_data, target_col="Class")

        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed.columns) > len(sample_fraud_data.columns)
        assert pipeline.is_fitted is True

    def test_pipeline_consistency(self, sample_fraud_data):
        """Test that pipeline produces consistent results."""
        pipeline = PreprocessingPipeline()
        pipeline.fit(sample_fraud_data, target_col="Class")

        df1 = pipeline.transform(sample_fraud_data)
        df2 = pipeline.transform(sample_fraud_data)

        pd.testing.assert_frame_equal(df1, df2)
