"""
Main Preprocessing Script
Uses configuration file to run the complete pipeline.
"""

from config import (
    DATA_CONFIG,
    SPLIT_CONFIG,
    CLEANING_CONFIG,
    FEATURE_CONFIG,
    SCALING_CONFIG,
    OUTPUT_CONFIG,
    EXPERIMENT_CONFIG,
)
from splitter import DataSplitter
from preprocessor import PreprocessingPipeline
from loader import DataLoader
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging():
    """Set up logging configuration."""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, OUTPUT_CONFIG["log_level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "preprocessing.log")],
    )


def main():
    """Run complete preprocessing pipeline."""

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Starting Preprocessing Pipeline")
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['experiment_name']}")
    logger.info(f"Version: {EXPERIMENT_CONFIG['version']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info("=" * 60)

    try:
        # ============================================
        # STEP 1: LOAD DATA
        # ============================================
        logger.info("\n[STEP 1] Loading data...")

        loader = DataLoader(DATA_CONFIG["raw_data_path"])
        df = loader.load_and_validate(required_columns=DATA_CONFIG["required_columns"])

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        logger.info(f"Fraud rate: {df[DATA_CONFIG['target_column']].mean():.4f}")

        # ============================================
        # STEP 2: SPLIT DATA
        # ============================================
        logger.info("\n[STEP 2] Splitting data...")

        splitter = DataSplitter(
            test_size=SPLIT_CONFIG["test_size"],
            val_size=SPLIT_CONFIG["val_size"],
            random_state=SPLIT_CONFIG["random_state"],
            stratify=SPLIT_CONFIG["stratify"],
        )

        train_df, val_df, test_df = splitter.split(df, target_col=DATA_CONFIG["target_column"])

        # Save splits if configured
        if SPLIT_CONFIG["save_splits"]:
            splitter.save_splits(train_df, val_df, test_df, output_dir=SPLIT_CONFIG["splits_dir"])

        # ============================================
        # STEP 3: PREPROCESS DATA
        # ============================================
        logger.info("\n[STEP 3] Preprocessing data...")

        pipeline = PreprocessingPipeline(
            cleaning_config=CLEANING_CONFIG,
            feature_config=FEATURE_CONFIG,
            scaler_type=SCALING_CONFIG["scaler_type"],
        )

        # Fit on training data only
        logger.info("Fitting pipeline on training data...")
        pipeline.fit(train_df, target_col=DATA_CONFIG["target_column"])

        # Transform all splits
        logger.info("Transforming all splits...")
        train_processed = pipeline.transform(train_df)
        val_processed = pipeline.transform(val_df)
        test_processed = pipeline.transform(test_df)

        logger.info(f"Train processed: {train_processed.shape}")
        logger.info(f"Val processed: {val_processed.shape}")
        logger.info(f"Test processed: {test_processed.shape}")

        # ============================================
        # STEP 4: SAVE PROCESSED DATA
        # ============================================
        if OUTPUT_CONFIG["save_processed"]:
            logger.info("\n[STEP 4] Saving processed data...")

            output_dir = Path(OUTPUT_CONFIG["processed_data_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            train_processed.to_csv(output_dir / "train_processed.csv", index=False)
            val_processed.to_csv(output_dir / "val_processed.csv", index=False)
            test_processed.to_csv(output_dir / "test_processed.csv", index=False)

            logger.info(f"Processed data saved to {output_dir}")

        # ============================================
        # STEP 5: SUMMARY
        # ============================================
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total features created: {len(train_processed.columns)}")
        logger.info(f"Training samples: {len(train_processed)}")
        logger.info(f"Validation samples: {len(val_processed)}")
        logger.info(f"Test samples: {len(test_processed)}")
        logger.info("\nNext steps:")
        logger.info("1. Handle class imbalance (SMOTE, undersampling, class weights)")
        logger.info("2. Train baseline models")
        logger.info("3. Evaluate and iterate")
        logger.info("=" * 60)

        return train_processed, val_processed, test_processed

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    train, val, test = main()
