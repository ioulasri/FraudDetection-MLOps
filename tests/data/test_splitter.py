"""Unit tests for DataSplitter module."""

import pandas as pd
import pytest
from pathlib import Path

from data.splitter import DataSplitter


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Feature1': range(100),
        'Feature2': range(100, 200),
        'Class': [0] * 90 + [1] * 10  # 10% fraud rate
    })


class TestDataSplitter:
    """Test cases for DataSplitter class."""

    def test_init_default_params(self):
        """Test DataSplitter initialization with defaults."""
        splitter = DataSplitter()
        
        assert splitter.test_size == 0.2
        assert splitter.val_size == 0.1
        assert splitter.random_state == 42
        assert splitter.stratify is True

    def test_init_custom_params(self):
        """Test DataSplitter initialization with custom parameters."""
        splitter = DataSplitter(test_size=0.3, val_size=0.15, random_state=123, stratify=False)
        
        assert splitter.test_size == 0.3
        assert splitter.val_size == 0.15
        assert splitter.random_state == 123
        assert splitter.stratify is False

    def test_split_returns_three_sets(self, sample_dataframe):
        """Test that split returns train, val, and test sets."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_sizes(self, sample_dataframe):
        """Test that split sizes are approximately correct."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataframe)
        assert len(test) == pytest.approx(0.2 * len(sample_dataframe), abs=2)
        assert len(val) == pytest.approx(0.1 * len(sample_dataframe), abs=2)

    def test_split_no_data_leakage(self, sample_dataframe):
        """Test that splits don't contain overlapping rows."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        
        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_split_stratified(self, sample_dataframe):
        """Test that stratification maintains class distribution."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42, stratify=True)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        original_ratio = sample_dataframe['Class'].mean()
        train_ratio = train['Class'].mean()
        val_ratio = val['Class'].mean()
        test_ratio = test['Class'].mean()
        
        # All splits should have similar fraud rates
        assert train_ratio == pytest.approx(original_ratio, abs=0.05)
        assert val_ratio == pytest.approx(original_ratio, abs=0.05)
        assert test_ratio == pytest.approx(original_ratio, abs=0.05)

    def test_split_reproducible(self, sample_dataframe):
        """Test that splits are reproducible with same random state."""
        splitter1 = DataSplitter(random_state=42)
        train1, val1, test1 = splitter1.split(sample_dataframe, target_col='Class')
        
        splitter2 = DataSplitter(random_state=42)
        train2, val2, test2 = splitter2.split(sample_dataframe, target_col='Class')
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_split_X_y(self, sample_dataframe):
        """Test split_X_y returns features and targets separately."""
        splitter = DataSplitter(random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_X_y(
            sample_dataframe, target_col='Class'
        )
        
        # Check that Class column is removed from features
        assert 'Class' not in X_train.columns
        assert 'Class' not in X_val.columns
        assert 'Class' not in X_test.columns
        
        # Check that targets are Series
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Check sizes match
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)

    def test_save_splits(self, sample_dataframe, tmp_path):
        """Test saving splits to CSV files."""
        splitter = DataSplitter(random_state=42)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        output_dir = tmp_path / "splits"
        splitter.save_splits(train, val, test, output_dir=str(output_dir))
        
        # Check files exist
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()

    def test_load_splits(self, sample_dataframe, tmp_path):
        """Test loading splits from CSV files."""
        splitter = DataSplitter(random_state=42)
        train_original, val_original, test_original = splitter.split(sample_dataframe, target_col='Class')
        
        output_dir = tmp_path / "splits"
        splitter.save_splits(train_original, val_original, test_original, output_dir=str(output_dir))
        
        # Load splits back
        train_loaded, val_loaded, test_loaded = splitter.load_splits(input_dir=str(output_dir))
        
        # Compare loaded data with original
        pd.testing.assert_frame_equal(train_original.reset_index(drop=True), train_loaded)
        pd.testing.assert_frame_equal(val_original.reset_index(drop=True), val_loaded)
        pd.testing.assert_frame_equal(test_original.reset_index(drop=True), test_loaded)

    def test_split_without_stratify(self, sample_dataframe):
        """Test splitting without stratification."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42, stratify=False)
        train, val, test = splitter.split(sample_dataframe, target_col='Class')
        
        # Should still return valid splits
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(sample_dataframe)
