"""
Example Usage of the Preprocessing Pipeline
Demonstrates how to use the complete data processing workflow.
"""
import logging
from loader import DataLoader
from preprocessor import PreprocessingPipeline
from splitter import DataSplitter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run complete preprocessing pipeline example."""
    
    # 1. LOAD DATA
    print("\n" + "="*50)
    print("STEP 1: LOADING DATA")
    print("="*50)
    
    loader = DataLoader('data/raw/creditcard.csv')
    df = loader.load_and_validate(
        required_columns=['Time', 'Amount', 'Class']
    )
    print(f"Loaded shape: {df.shape}")
    
    # 2. SPLIT DATA (before preprocessing to avoid data leakage)
    print("\n" + "="*50)
    print("STEP 2: SPLITTING DATA")
    print("="*50)
    
    splitter = DataSplitter(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify=True
    )
    
    train_df, val_df, test_df = splitter.split(df, target_col='Class')
    
    # Optional: Save splits for later use
    splitter.save_splits(train_df, val_df, test_df, output_dir='data/splits')
    
    # 3. PREPROCESS DATA
    print("\n" + "="*50)
    print("STEP 3: PREPROCESSING DATA")
    print("="*50)
    
    # Configure preprocessing
    cleaning_config = {
        'remove_duplicates': True,
        'missing_strategy': 'drop',
        'outlier_columns': None  # Or specify columns like ['Amount']
    }
    
    feature_config = {
        'include_time': True,
        'include_amount': True,
        'include_stats': True,
        'include_interactions': True
    }
    
    # Create pipeline
    pipeline = PreprocessingPipeline(
        cleaning_config=cleaning_config,
        feature_config=feature_config,
        scaler_type='robust'  # Use 'robust' for outlier-resistant scaling
    )
    
    # Fit on training data only
    pipeline.fit(train_df, target_col='Class')
    
    # Transform all splits
    train_processed = pipeline.transform(train_df)
    val_processed = pipeline.transform(val_df)
    test_processed = pipeline.transform(test_df)
    
    print(f"Processed train shape: {train_processed.shape}")
    print(f"Processed val shape: {val_processed.shape}")
    print(f"Processed test shape: {test_processed.shape}")
    
    # 4. PREPARE FOR MODELING
    print("\n" + "="*50)
    print("STEP 4: PREPARING FOR MODELING")
    print("="*50)
    
    # Separate features and targets
    X_train = train_processed.drop(columns=['Class'])
    y_train = train_processed['Class']
    
    X_val = val_processed.drop(columns=['Class'])
    y_val = val_processed['Class']
    
    X_test = test_processed.drop(columns=['Class'])
    y_test = test_processed['Class']
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    
    # 5. SAVE PROCESSED DATA (OPTIONAL)
    print("\n" + "="*50)
    print("STEP 5: SAVING PROCESSED DATA")
    print("="*50)
    
    train_processed.to_csv('data/processed/train_processed.csv', index=False)
    val_processed.to_csv('data/processed/val_processed.csv', index=False)
    test_processed.to_csv('data/processed/test_processed.csv', index=False)
    
    print("Processed data saved to data/processed/")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = main()
