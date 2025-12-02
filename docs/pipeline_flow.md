# Preprocessing Pipeline Visual Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE STRUCTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────┐
│  RAW DATA     │
│  creditcard   │
│  .csv         │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          1. DATA LOADER                                     │
│  • Load CSV file                                                            │
│  • Validate required columns                                                │
│  • Log statistics                                                           │
└───────────────────────────────────────┬───────────────────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────┐
                            │   FULL DATASET    │
                            │   (284,807 rows)  │
                            └─────────┬─────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          2. DATA SPLITTER                                   │
│  • Stratified split (maintains fraud rate)                                  │
│  • Train: 70% | Val: 10% | Test: 20%                                       │
│  • Random state for reproducibility                                         │
└───────────────────────────────────────┬───────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
            │ TRAIN DATA   │    │  VAL DATA    │   │  TEST DATA   │
            │  (~199k)     │    │  (~28k)      │   │  (~57k)      │
            └──────┬───────┘    └──────┬───────┘   └──────┬───────┘
                   │                   │                   │
                   │                   │                   │
                   ▼                   │                   │
┌─────────────────────────────────┐    │                   │
│  3. FIT PREPROCESSING PIPELINE  │    │                   │
│  (TRAINING DATA ONLY!)          │    │                   │
│                                 │    │                   │
│  ┌───────────────────────────┐ │    │                   │
│  │  a. DATA CLEANING         │ │    │                   │
│  │  • Remove duplicates      │ │    │                   │
│  │  • Handle missing values  │ │    │                   │
│  │  • Remove outliers (opt)  │ │    │                   │
│  └───────────────────────────┘ │    │                   │
│                                 │    │                   │
│  ┌───────────────────────────┐ │    │                   │
│  │  b. FEATURE ENGINEERING   │ │    │                   │
│  │  • Time features          │ │    │                   │
│  │  │  - hour                │ │    │                   │
│  │  │  - is_night            │ │    │                   │
│  │  │  - cyclical (sin/cos)  │ │    │                   │
│  │  • Amount features        │ │    │                   │
│  │  │  - log_amount          │ │    │                   │
│  │  │  - small/large flags   │ │    │                   │
│  │  │  - squared/sqrt        │ │    │                   │
│  │  • Statistical features   │ │    │                   │
│  │  │  - mean, std, min, max │ │    │                   │
│  │  • Interaction features   │ │    │                   │
│  │  │  - hour × amount       │ │    │                   │
│  │  │  - night × log_amount  │ │    │                   │
│  └───────────────────────────┘ │    │                   │
│                                 │    │                   │
│  ┌───────────────────────────┐ │    │                   │
│  │  c. FEATURE SCALING       │ │    │                   │
│  │  • Fit StandardScaler or  │ │    │                   │
│  │    RobustScaler           │ │    │                   │
│  │  • Learn μ and σ          │ │    │                   │
│  └───────────────────────────┘ │    │                   │
└─────────────────┬───────────────┘    │                   │
                  │                    │                   │
                  │  Apply same        │   Apply same      │
                  │  transforms ───────┼───────────────────┤
                  │                    │                   │
                  ▼                    ▼                   ▼
          ┌──────────────┐     ┌──────────────┐   ┌──────────────┐
          │ TRAIN        │     │ VAL          │   │ TEST         │
          │ PROCESSED    │     │ PROCESSED    │   │ PROCESSED    │
          │              │     │              │   │              │
          │ • Cleaned    │     │ • Cleaned    │   │ • Cleaned    │
          │ • Engineered │     │ • Engineered │   │ • Engineered │
          │ • Scaled     │     │ • Scaled     │   │ • Scaled     │
          └──────┬───────┘     └──────┬───────┘   └──────┬───────┘
                 │                    │                   │
                 └────────────────────┼───────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   SAVE TO DISK        │
                          │   data/processed/     │
                          └───────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │   READY FOR           │
                          │   MODEL TRAINING      │
                          └───────────────────────┘
```

## Key Principles

### 1. **No Data Leakage**
```
✓ CORRECT:   Split → Fit on Train → Transform All
✗ WRONG:     Fit on All → Split → Transform
```

### 2. **Fit vs Transform**
```
FIT:        Learn parameters (mean, std, etc.) from TRAINING data only
TRANSFORM:  Apply learned parameters to ANY dataset (train/val/test)
```

### 3. **Order Matters**
```
1. Load Raw Data
2. Split into Train/Val/Test  ← Do this FIRST!
3. Fit pipeline on Train only
4. Transform all splits using same fitted pipeline
5. Train models
```

## File Structure After Running Pipeline

```
fraud-detection-system/
├── data/
│   ├── raw/
│   │   └── creditcard.csv                    # Original data
│   ├── splits/
│   │   ├── train.csv                         # Raw train split
│   │   ├── val.csv                           # Raw val split
│   │   └── test.csv                          # Raw test split
│   └── processed/
│       ├── train_processed.csv               # Processed train
│       ├── val_processed.csv                 # Processed val
│       └── test_processed.csv                # Processed test
├── src/
│   └── data/
│       ├── loader.py                         # Loading module
│       ├── preprocessor.py                   # Main preprocessing
│       ├── splitter.py                       # Splitting module
│       ├── config.py                         # Configuration
│       ├── run_pipeline.py                   # Main execution script
│       ├── pipeline_example.py               # Usage example
│       ├── test_pipeline.py                  # Test suite
│       └── PIPELINE_README.md                # Documentation
└── preprocessing.log                         # Execution logs
```

## Usage Commands

```bash
# 1. Configure pipeline
nano src/data/config.py

# 2. Run complete pipeline
cd src/data
python run_pipeline.py

# 3. Test pipeline (optional)
python test_pipeline.py

# 4. View configuration
python config.py
```

## Common Configurations

### For Initial Exploration
```python
CLEANING_CONFIG = {
    'remove_duplicates': True,
    'missing_strategy': 'drop',
    'outlier_columns': None,  # Keep outliers initially
}
```

### For Production
```python
CLEANING_CONFIG = {
    'remove_duplicates': True,
    'missing_strategy': 'median',  # More robust
    'outlier_columns': ['Amount'],  # Remove outliers
}
```

### For Different Models
```python
# Tree-based models (XGBoost, Random Forest)
SCALING_CONFIG = {'scaler_type': None}  # Don't need scaling

# Linear models (Logistic Regression)
SCALING_CONFIG = {'scaler_type': 'standard'}  # Need scaling

# For data with outliers
SCALING_CONFIG = {'scaler_type': 'robust'}  # Robust to outliers
```
