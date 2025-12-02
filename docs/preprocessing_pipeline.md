# Preprocessing Pipeline Structure

## Overview
This preprocessing pipeline provides a modular, production-ready structure for fraud detection data processing. It follows best practices including separation of concerns, avoiding data leakage, and maintaining reproducibility.

## Architecture

```
src/data/
├── loader.py           # Data loading and validation
├── preprocessor.py     # Cleaning, feature engineering, scaling
├── splitter.py         # Train/val/test splitting
└── pipeline_example.py # Complete usage example
```

## Components

### 1. DataLoader (`loader.py`)
**Purpose**: Load and validate raw data

**Key Features**:
- Load data from CSV files
- Validate required columns
- Log loading statistics

**Usage**:
```python
from loader import DataLoader

loader = DataLoader('data/raw/creditcard.csv')
df = loader.load_and_validate(required_columns=['Time', 'Amount', 'Class'])
```

### 2. PreprocessingPipeline (`preprocessor.py`)
**Purpose**: Complete preprocessing orchestration

**Components**:

#### a. DataCleaner
- Remove duplicates
- Handle missing values (drop, mean, median)
- Remove outliers (IQR, Z-score methods)

#### b. FeatureEngineer
- **Time features**: hour, is_night, cyclical features (sin/cos)
- **Amount features**: log_amount, small/large flags, squared/sqrt
- **Statistical features**: mean, std, min, max, range of PCA features
- **Interaction features**: hour × amount, night × log_amount

#### c. FeatureScaler
- StandardScaler or RobustScaler
- Fit on training data only
- Transform train/val/test consistently

**Usage**:
```python
from preprocessor import PreprocessingPipeline

# Configure
pipeline = PreprocessingPipeline(
    cleaning_config={
        'remove_duplicates': True,
        'missing_strategy': 'drop',
        'outlier_columns': ['Amount']
    },
    feature_config={
        'include_time': True,
        'include_amount': True,
        'include_stats': True,
        'include_interactions': True
    },
    scaler_type='robust'
)

# Fit on training data
pipeline.fit(train_df, target_col='Class')

# Transform all splits
train_processed = pipeline.transform(train_df)
val_processed = pipeline.transform(val_df)
test_processed = pipeline.transform(test_df)
```

### 3. DataSplitter (`splitter.py`)
**Purpose**: Split data with stratification

**Key Features**:
- Stratified train/val/test splits
- Maintains fraud rate distribution
- Save/load splits functionality
- Separate X and y automatically

**Usage**:
```python
from splitter import DataSplitter

splitter = DataSplitter(
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    stratify=True
)

# Get splits
train_df, val_df, test_df = splitter.split(df, target_col='Class')

# Or get X, y directly
X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_X_y(df)

# Save splits
splitter.save_splits(train_df, val_df, test_df, output_dir='data/splits')
```

## Complete Workflow

### Correct Order (Prevents Data Leakage):

1. **Load Data** → `DataLoader`
2. **Split Data** → `DataSplitter` (BEFORE any preprocessing)
3. **Fit Pipeline** → `PreprocessingPipeline.fit()` on training data only
4. **Transform Data** → Apply to train/val/test
5. **Model Training** → Use processed data

### Why This Order Matters:
- **Splitting first** prevents information from test set leaking into training
- **Fitting on train only** ensures validation/test remain unseen during preprocessing
- **Consistent transforms** maintain data distribution across all sets

## Example Usage

See `pipeline_example.py` for a complete end-to-end example.

```bash
cd src/data
python pipeline_example.py
```

## Configuration Options

### Cleaning Configuration
```python
cleaning_config = {
    'remove_duplicates': True,          # Remove duplicate rows
    'missing_strategy': 'drop',         # 'drop', 'mean', 'median'
    'outlier_columns': ['Amount']       # Columns to check for outliers
}
```

### Feature Engineering Configuration
```python
feature_config = {
    'include_time': True,          # Time-based features
    'include_amount': True,        # Amount transformations
    'include_stats': True,         # Statistical aggregations
    'include_interactions': True   # Feature interactions
}
```

### Scaler Options
- `'standard'`: StandardScaler (assumes normal distribution)
- `'robust'`: RobustScaler (resistant to outliers)

## Best Practices

1. **Always split before preprocessing** to avoid data leakage
2. **Fit only on training data**, transform on all sets
3. **Use stratification** for imbalanced datasets
4. **Log all operations** for debugging and monitoring
5. **Save processed data** for reproducibility
6. **Version your splits** by saving them to disk

## Extensibility

The modular design allows easy extensions:

### Add New Feature Engineering
```python
class FeatureEngineer:
    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Your custom feature logic here
        df['custom_feature'] = ...
        return df
```

### Add New Cleaning Method
```python
class DataCleaner:
    def custom_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your custom cleaning logic
        return df
```

### Use Different Scalers
```python
from sklearn.preprocessing import MinMaxScaler

class FeatureScaler:
    def __init__(self, scaler_type='minmax'):
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        # ...
```

## Testing Your Pipeline

```python
# Test the pipeline
from pipeline_example import main

X_train, X_val, X_test, y_train, y_val, y_test = main()

# Verify shapes
assert X_train.shape[0] == y_train.shape[0]
assert X_val.shape[0] == y_val.shape[0]
assert X_test.shape[0] == y_test.shape[0]

# Verify no data leakage (same columns)
assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns)

# Verify fraud rates are similar
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Val fraud rate: {y_val.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")
```

## Next Steps

After preprocessing:
1. Handle class imbalance (SMOTE, undersampling, class weights)
2. Train models using processed data
3. Evaluate on validation set
4. Fine-tune based on results
5. Final evaluation on test set

## Questions to Consider

As you implement this pipeline, think about:
- What features are most important for fraud detection?
- Should outliers be removed or are they informative?
- What's the right train/val/test split ratio?
- How does scaling affect different models?
- Are there any domain-specific features to add?
