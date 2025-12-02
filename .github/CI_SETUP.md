# CI/CD Setup Instructions

This document provides instructions for setting up and using the CI/CD pipeline.

## Overview

The CI/CD pipeline includes:
- **Code Quality Checks**: Black, Flake8, isort, MyPy
- **Security Scanning**: Bandit, Safety
- **Testing**: pytest with 70% coverage requirement
- **Multi-Python Version Testing**: 3.9, 3.10, 3.11
- **Pre-commit Hooks**: Local quality gates

## Quick Setup

### 1. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### 2. Setup Pre-commit Hooks

```bash
pre-commit install
```

This will run quality checks automatically before each commit.

### 3. Run Pre-commit on All Files (Optional)

```bash
pre-commit run --all-files
```

## Local Development Workflow

### Before Committing

Run these checks locally:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/ -ll

# Run tests
pytest tests/ -v --cov=src
```

### Or Use the Makefile (to be created)

```bash
make format    # Format code
make lint      # Run all linters
make test      # Run tests
make quality   # Run all quality checks
```

## CI/CD Pipeline Details

### GitHub Actions Workflows

Located in `.github/workflows/ci.yml`

#### Jobs:

1. **code-quality**: Runs formatters, linters, type checkers
2. **tests**: Runs pytest on multiple Python versions
3. **integration-tests**: Runs integration tests
4. **dependency-security**: Scans for vulnerable dependencies
5. **model-validation**: Validates ML pipeline configuration
6. **build-docs**: Validates documentation

### Branches

- **main**: Production branch (protected)
- **develop**: Development branch (protected)
- **feature/***: Feature branches (PR to develop)

### Pull Request Requirements

Before merging to `develop` or `main`:
- ✅ All CI checks must pass
- ✅ Code coverage ≥ 70%
- ✅ No security vulnerabilities
- ✅ Code review approved

## Configuration Files

### `.flake8`
Linting rules and exclusions

### `mypy.ini`
Type checking configuration

### `pyproject.toml`
Black, isort, pytest, coverage configuration

### `pytest.ini`
Test discovery and coverage settings

### `.pre-commit-config.yaml`
Pre-commit hooks configuration

## Coverage Requirements

- **Minimum**: 70% overall coverage
- **Target**: 80%+ for production code
- **Exclusions**: Tests, notebooks, generated files

View coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Badges (Add to README)

```markdown
![CI Status](https://github.com/ioulasri/FraudDetection-MLOps/workflows/CI%2FCD%20Pipeline/badge.svg)
![Coverage](https://codecov.io/gh/ioulasri/FraudDetection-MLOps/branch/main/graph/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
```

## Troubleshooting

### Pre-commit Hook Failures

Skip hooks temporarily (not recommended):
```bash
git commit --no-verify
```

Update hooks:
```bash
pre-commit autoupdate
```

### CI Failures

Check logs:
1. Go to GitHub Actions tab
2. Click on failed workflow
3. Review job logs
4. Fix issues locally and push again

## Best Practices

1. **Run pre-commit before pushing** to catch issues early
2. **Write tests** for new features (maintain coverage)
3. **Update documentation** when changing APIs
4. **Use type hints** for better code quality
5. **Keep dependencies updated** (check for security issues)
6. **Small, focused commits** for easier review

## Adding New Checks

To add a new check to the CI pipeline:

1. Update `.github/workflows/ci.yml`
2. Add configuration file if needed
3. Update `requirements-dev.txt` if new tool required
4. Add to `.pre-commit-config.yaml` for local checks
5. Document in this file

## Contact

For questions about the CI/CD setup, contact the team lead.
