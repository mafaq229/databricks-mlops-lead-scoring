# Lead Scoring MLOps Project

A complete end-to-end MLOps project demonstrating production ML practices for a lead scoring model.

## Project Overview

This project predicts the probability of a sales lead converting into a paying customer. It demonstrates:

- **ML Pipeline**: Data processing, feature engineering, model training
- **Experiment Tracking**: MLflow for tracking experiments and model versioning
- **Model Serving**: FastAPI REST API for real-time predictions
- **Monitoring**: Drift detection using PSI and custom metrics
- **CI/CD**: Automated testing with GitHub Actions

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv (macOS)

# 2. Install dependencies
uv sync
uv sync --extra dev  # Include dev tools

# 3. Generate sample data
uv run python src/data_generator.py

# 4. Train a model (with MLflow tracking)
uv run python src/train_with_mlflow.py

# 5. Start MLflow UI (in separate terminal)
uv run mlflow ui

# 6. Start the API
uv run uvicorn src.api:app --reload

# 7. Run tests
uv run pytest tests/ -v
```

## Project Structure

```
lead_scoring_mlops/
├── data/                    # Data files (generated)
├── notebooks/               # Jupyter notebooks for exploration
├── src/
│   ├── data_generator.py    # Generate synthetic lead data
│   ├── features.py          # Feature engineering pipeline
│   ├── train_with_mlflow.py # Model training with tracking
│   ├── api.py               # FastAPI serving endpoint
│   └── monitoring.py        # Drift detection utilities
├── tests/                   # Unit tests
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
└── requirements.txt         # Python dependencies
```

## Learning Path

If you're new to MLOps, follow the [LEARNING_PLAN.md](../LEARNING_PLAN.md) for a step-by-step guide.

## Tech Stack

- **ML**: XGBoost, scikit-learn
- **Tracking**: MLflow
- **API**: FastAPI
- **Monitoring**: Custom PSI implementation
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## Key Concepts Demonstrated

### 1. Feature Engineering
- Consistent transformations between training and inference
- Feature versioning and reproducibility

### 2. Experiment Tracking
- Parameter and metric logging
- Model versioning with MLflow Registry

### 3. Model Serving
- REST API with FastAPI
- Input validation with Pydantic
- Batch and single prediction endpoints

### 4. Monitoring
- Population Stability Index (PSI) for drift detection
- Alert thresholds and severity levels

### 5. Testing
- Unit tests for features and monitoring
- Automated CI pipeline

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Score single lead |
| `/predict/batch` | POST | Score multiple leads |

## Metrics Tracked

- **Model Performance**: AUC-ROC, Precision, Recall, Brier Score
- **Drift**: PSI per feature, score distribution shift
- **Operational**: Latency, throughput, error rate
