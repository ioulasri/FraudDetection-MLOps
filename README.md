# FraudDetection-MLOps

![CI Status](https://github.com/ioulasri/FraudDetection-MLOps/workflows/CI%2FCD%20Pipeline/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

A production-ready fraud detection system with MLOps best practices.

## Project Structure

```
fraud-detection-system/
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data ready for modeling
│   └── splits/           # Train/val/test splits
├── docs/
│   ├── preprocessing_pipeline.md    # Preprocessing documentation
│   └── pipeline_flow.md             # Visual pipeline flow
├── models/
│   └── saved/            # Trained model artifacts
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/             # Data processing modules
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   ├── splitter.py
│   │   ├── config.py
│   │   ├── run_pipeline.py
│   │   └── pipeline_example.py
│   ├── models/           # Model training modules
│   └── utils/            # Utility functions
├── tests/                # Test files
│   └── data/
│       └── test_preprocessing.py
├── requirements.txt      # Production dependencies
└── requirements-dev.txt  # Development dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv_fraud
source venv_fraud/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Setup pre-commit hooks (recommended for contributors)
pre-commit install
```

### 2. Run Preprocessing Pipeline

```bash
cd src/data
python run_pipeline.py
```

### 3. Run Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Or use make commands
make test          # Run tests
make test-cov      # Run tests with coverage report
make quality       # Run all quality checks
```

## Documentation

- **[Preprocessing Pipeline Guide](docs/preprocessing_pipeline.md)** - Complete guide to the preprocessing pipeline
- **[Pipeline Flow Diagram](docs/pipeline_flow.md)** - Visual representation of data flow
- **[Tests Documentation](tests/README.md)** - How to run and write tests

## Features

✅ Modular preprocessing pipeline
✅ Configurable data cleaning and feature engineering
✅ Stratified train/val/test splitting
✅ No data leakage (fit on train, transform all)
✅ Comprehensive logging
✅ Production-ready code structure
✅ Test suite with 70%+ coverage
✅ CI/CD pipeline with GitHub Actions
✅ Pre-commit hooks for code quality
✅ Automated security scanning
✅ Multi-Python version testing (3.9, 3.10, 3.11)

## Next Steps

1. Explore the data in notebooks
2. Configure preprocessing in `src/data/config.py`
3. Run the pipeline
4. Train baseline models
5. Evaluate and iterate
