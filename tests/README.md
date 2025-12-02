# Tests

This directory contains test files for the fraud detection system.

## Structure

```
tests/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── test_preprocessing.py    # Tests for data preprocessing pipeline
└── models/                       # (Future) Tests for model training
```

## Running Tests

### Run all tests
```bash
cd tests/data
python test_preprocessing.py
```

### Using pytest (recommended)
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/data/test_preprocessing.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=src tests/
```

## Test Guidelines

1. **Keep tests isolated** - Each test should be independent
2. **Use fixtures** - For common setup/teardown
3. **Mock external dependencies** - Don't rely on external services
4. **Test edge cases** - Not just happy paths
5. **Keep tests fast** - Use small datasets for testing

## Adding New Tests

Create new test files following the pattern:
```python
"""Test description."""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'module_name'))

from your_module import YourClass

def test_your_function():
    """Test your function."""
    # Arrange
    # Act
    # Assert
    pass
```
