"""Unit tests for data configuration module."""

import json

from data.config import (
    CLEANING_CONFIG,
    DATA_CONFIG,
    EXPERIMENT_CONFIG,
    FEATURE_CONFIG,
    OUTPUT_CONFIG,
    SCALING_CONFIG,
    SPLIT_CONFIG,
    get_complete_config,
    print_config,
)


class TestConfigConstants:
    """Test cases for configuration constants."""

    def test_data_config_structure(self):
        """Test DATA_CONFIG has required keys."""
        assert "raw_data_path" in DATA_CONFIG
        assert "required_columns" in DATA_CONFIG
        assert "target_column" in DATA_CONFIG
        assert DATA_CONFIG["target_column"] == "Class"

    def test_split_config_structure(self):
        """Test SPLIT_CONFIG has required keys."""
        assert "test_size" in SPLIT_CONFIG
        assert "val_size" in SPLIT_CONFIG
        assert "random_state" in SPLIT_CONFIG
        assert "stratify" in SPLIT_CONFIG
        assert SPLIT_CONFIG["random_state"] == 42

    def test_cleaning_config_structure(self):
        """Test CLEANING_CONFIG has required keys."""
        assert "remove_duplicates" in CLEANING_CONFIG
        assert "missing_strategy" in CLEANING_CONFIG
        assert "outlier_method" in CLEANING_CONFIG
        assert CLEANING_CONFIG["outlier_method"] in ["iqr", "zscore"]

    def test_feature_config_structure(self):
        """Test FEATURE_CONFIG has required keys."""
        assert "include_time" in FEATURE_CONFIG
        assert "include_amount" in FEATURE_CONFIG
        assert "include_stats" in FEATURE_CONFIG
        assert "include_interactions" in FEATURE_CONFIG

    def test_scaling_config_structure(self):
        """Test SCALING_CONFIG has required keys."""
        assert "scaler_type" in SCALING_CONFIG
        assert SCALING_CONFIG["scaler_type"] in ["standard", "robust"]

    def test_output_config_structure(self):
        """Test OUTPUT_CONFIG has required keys."""
        assert "processed_data_dir" in OUTPUT_CONFIG
        assert "save_processed" in OUTPUT_CONFIG
        assert "log_level" in OUTPUT_CONFIG

    def test_experiment_config_structure(self):
        """Test EXPERIMENT_CONFIG has required keys."""
        assert "experiment_name" in EXPERIMENT_CONFIG
        assert "description" in EXPERIMENT_CONFIG
        assert "version" in EXPERIMENT_CONFIG


class TestConfigFunctions:
    """Test cases for configuration functions."""

    def test_get_complete_config_returns_dict(self):
        """Test get_complete_config returns a dictionary."""
        config = get_complete_config()
        assert isinstance(config, dict)

    def test_get_complete_config_has_all_sections(self):
        """Test get_complete_config contains all configuration sections."""
        config = get_complete_config()
        assert "data" in config
        assert "split" in config
        assert "cleaning" in config
        assert "features" in config
        assert "scaling" in config
        assert "output" in config
        assert "experiment" in config

    def test_get_complete_config_sections_match(self):
        """Test get_complete_config sections match individual configs."""
        config = get_complete_config()
        assert config["data"] == DATA_CONFIG
        assert config["split"] == SPLIT_CONFIG
        assert config["cleaning"] == CLEANING_CONFIG
        assert config["features"] == FEATURE_CONFIG
        assert config["scaling"] == SCALING_CONFIG
        assert config["output"] == OUTPUT_CONFIG
        assert config["experiment"] == EXPERIMENT_CONFIG

    def test_get_complete_config_is_json_serializable(self):
        """Test that complete config can be serialized to JSON."""
        config = get_complete_config()
        # Should not raise an exception
        json_str = json.dumps(config)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_print_config(self, capsys):
        """Test print_config outputs configuration."""
        print_config()
        captured = capsys.readouterr()

        # Verify output contains key configuration sections
        assert "data" in captured.out
        assert "split" in captured.out
        assert "cleaning" in captured.out
        assert "features" in captured.out

    def test_print_config_valid_json(self, capsys):
        """Test print_config outputs valid JSON."""
        print_config()
        captured = capsys.readouterr()

        # Should be able to parse the output as JSON
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)
        assert len(parsed) == 7  # Should have 7 sections
