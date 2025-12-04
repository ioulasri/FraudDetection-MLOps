"""Unit tests for DataLoader module."""

from pathlib import Path

import pandas as pd
import pytest

from data.loader import DataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "test_data.csv"
    df = pd.DataFrame({"Time": [0, 100, 200], "Amount": [10.0, 20.0, 30.0], "Class": [0, 1, 0]})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def empty_csv(tmp_path):
    """Create an empty CSV file for testing."""
    csv_file = tmp_path / "empty.csv"
    df = pd.DataFrame()
    df.to_csv(csv_file, index=False)
    return csv_file


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self, sample_csv):
        """Test DataLoader initialization."""
        loader = DataLoader(sample_csv)
        assert loader.data_path == Path(sample_csv)

    def test_load_csv_success(self, sample_csv):
        """Test successful CSV loading."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["Time", "Amount", "Class"]

    def test_load_csv_with_kwargs(self, sample_csv):
        """Test CSV loading with additional kwargs."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv(usecols=["Time", "Amount"])

        assert len(df.columns) == 2
        assert "Class" not in df.columns

    def test_load_csv_file_not_found(self, tmp_path):
        """Test loading non-existent file."""
        loader = DataLoader(tmp_path / "nonexistent.csv")

        with pytest.raises(Exception):
            loader.load_csv()

    def test_validate_data_success(self, sample_csv):
        """Test successful data validation."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv()

        result = loader.validate_data(df, required_columns=["Time", "Amount", "Class"])
        assert result is True

    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        # Create an empty DataFrame directly instead of loading from CSV
        df = pd.DataFrame()
        loader = DataLoader("dummy_path.csv")

        with pytest.raises(ValueError, match="DataFrame is empty"):
            loader.validate_data(df)

    def test_validate_data_missing_columns(self, sample_csv):
        """Test validation with missing required columns."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv()

        with pytest.raises(ValueError, match="Missing required columns"):
            loader.validate_data(df, required_columns=["Time", "NonExistent"])

    def test_validate_data_no_required_columns(self, sample_csv):
        """Test validation without required columns specified."""
        loader = DataLoader(sample_csv)
        df = loader.load_csv()

        result = loader.validate_data(df)
        assert result is True

    def test_load_and_validate_success(self, sample_csv):
        """Test combined load and validate."""
        loader = DataLoader(sample_csv)
        df = loader.load_and_validate(required_columns=["Time", "Amount", "Class"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_load_and_validate_failure(self, sample_csv):
        """Test combined load and validate with missing columns."""
        loader = DataLoader(sample_csv)

        with pytest.raises(ValueError):
            loader.load_and_validate(required_columns=["Time", "NonExistent"])
