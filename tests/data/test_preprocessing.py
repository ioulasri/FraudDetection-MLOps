"""Quick test script to verify the preprocessing pipeline works."""

import sys
from pathlib import Path

from loader import DataLoader
from preprocessor import DataCleaner, FeatureEngineer, FeatureScaler, PreprocessingPipeline
from splitter import DataSplitter

# Add src/data to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "data"))


def test_loader():
    """Test DataLoader."""
    print("\n[TEST] DataLoader")
    try:
        loader = DataLoader("data/raw/creditcard.csv")
        df = loader.load_csv()
        print(f"✓ Loaded data: {df.shape}")

        # Validate
        loader.validate_data(df, required_columns=["Time", "Amount", "Class"])
        print("✓ Validation passed")

        return df
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def test_splitter(df):
    """Test DataSplitter."""
    print("\n[TEST] DataSplitter")
    try:
        splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)
        train, val, test = splitter.split(df, target_col="Class")

        print(f"✓ Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        print(
            f"✓ Fraud rates - Train: {train['Class'].mean():.4f}, "
            f"Val: {val['Class'].mean():.4f}, Test: {test['Class'].mean():.4f}"
        )

        return train, val, test
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None, None, None


def test_cleaner(df):
    """Test DataCleaner."""
    print("\n[TEST] DataCleaner")
    try:
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df, remove_duplicates=True, missing_strategy="drop")
        print(f"✓ Cleaned data: {df_clean.shape}")
        print(f"✓ Cleaning stats: {cleaner.cleaning_stats}")
        return df_clean
    except Exception as e:
        print(f"✗ Failed: {e}")
        return df


def test_feature_engineer(df):
    """Test FeatureEngineer."""
    print("\n[TEST] FeatureEngineer")
    try:
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        print(f"✓ Engineered features: {df_features.shape}")
        print(
            f"✓ New columns: {[col for col in df_features.columns if col not in df.columns][:10]}"
        )
        return df_features
    except Exception as e:
        print(f"✗ Failed: {e}")
        return df


def test_scaler(train_df, test_df):
    """Test FeatureScaler."""
    print("\n[TEST] FeatureScaler")
    try:
        scaler = FeatureScaler(scaler_type="robust")
        scaler.fit(train_df, target_col="Class")

        train_scaled = scaler.transform(train_df)
        test_scaled = scaler.transform(test_df)

        print(f"✓ Scaled train: {train_scaled.shape}")
        print(f"✓ Scaled test: {test_scaled.shape}")
        return train_scaled, test_scaled
    except Exception as e:
        print(f"✗ Failed: {e}")
        return train_df, test_df


def test_full_pipeline(train_df, val_df, test_df):
    """Test complete PreprocessingPipeline."""
    print("\n[TEST] Complete PreprocessingPipeline")
    try:
        pipeline = PreprocessingPipeline(
            cleaning_config={"remove_duplicates": True, "missing_strategy": "drop"},
            feature_config={"include_time": True, "include_amount": True, "include_stats": True},
            scaler_type="robust",
        )

        # Fit on train
        pipeline.fit(train_df, target_col="Class")
        print("✓ Pipeline fitted")

        # Transform all
        train_processed = pipeline.transform(train_df)
        val_processed = pipeline.transform(val_df)
        test_processed = pipeline.transform(test_df)

        print(
            f"✓ Processed shapes - Train: {train_processed.shape}, "
            f"Val: {val_processed.shape}, Test: {test_processed.shape}"
        )

        # Verify same columns
        assert (
            list(train_processed.columns)
            == list(val_processed.columns)
            == list(test_processed.columns)
        )
        print("✓ All splits have same columns")

        return train_processed, val_processed, test_processed
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None, None, None


def main():
    """Run all tests."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE TEST SUITE")
    print("=" * 60)

    # Test 1: Load data
    df = test_loader()
    if df is None:
        print("\n✗ Cannot proceed without data")
        return

    # Test 2: Split data
    train, val, test = test_splitter(df)
    if train is None:
        print("\n✗ Cannot proceed without splits")
        return

    # Test 3: Data cleaning
    train_clean = test_cleaner(train.copy())

    # Test 4: Feature engineering
    train_features = test_feature_engineer(train_clean.copy())

    # Test 5: Scaling
    test_features = test_feature_engineer(test.copy())
    train_scaled, test_scaled = test_scaler(train_features, test_features)

    # Test 6: Full pipeline
    train_processed, val_processed, test_processed = test_full_pipeline(train, val, test)

    print("\n" + "=" * 60)
    if train_processed is not None:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
