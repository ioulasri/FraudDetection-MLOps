# Makefile for Fraud Detection System

.PHONY: help install install-dev format lint type-check security test test-cov quality clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make format        - Format code with black and isort"
	@echo "  make lint          - Run flake8 linting"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make security      - Run security checks (bandit, safety)"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make quality       - Run all quality checks"
	@echo "  make clean         - Clean generated files"
	@echo "  make pre-commit    - Install and run pre-commit hooks"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	@echo "Formatting code with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/

lint:
	@echo "Running flake8..."
	flake8 src/ tests/

type-check:
	@echo "Running mypy type checking..."
	mypy src/ --ignore-missing-imports --no-strict-optional

security:
	@echo "Running security scan with bandit..."
	bandit -r src/ -ll
	@echo "Checking dependencies for vulnerabilities..."
	safety check

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/"

quality: format lint type-check security test-cov
	@echo "All quality checks passed!"

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .tox htmlcov .coverage
	rm -rf build dist
	@echo "Clean complete!"

pre-commit:
	pre-commit install
	pre-commit run --all-files

# Data pipeline commands
run-pipeline:
	python src/data/run_pipeline.py

validate-config:
	python src/data/config.py

# Setup directories
setup-dirs:
	mkdir -p data/raw data/processed data/splits logs models/saved
