# Lead Scoring MLOps Project

A production-ready, end-to-end MLOps system that predicts lead conversion probability. This project demonstrates real-world ML engineering practices: from synthetic data generation and experiment tracking to API serving, monitoring, and CI/CD automation.

## 🎯 Project Overview

**Business Problem**: Identify high-probability sales leads to optimize sales team efforts and improve ROI.

**Solution**: A machine learning pipeline that scores leads in real-time via a REST API, tracks experiments systematically, monitors data drift, and maintains code quality through automated testing.

**Key Features**:
- ✅ **End-to-End ML Pipeline**: Synthetic data generation → Feature engineering → XGBoost training → REST API serving
- ✅ **Experiment Tracking**: MLflow integration for reproducible model versioning and model registry
- ✅ **Production API**: FastAPI with async/lifespan pattern, Pydantic validation, batch prediction support
- ✅ **Data Drift Detection**: PSI-based monitoring with configurable alert thresholds
- ✅ **Automated Testing**: Pytest with coverage reporting, type checking with mypy
- ✅ **CI/CD Pipeline**: GitHub Actions for linting, testing, and type validation

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- `uv` package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

### Installation & Setup

```bash
# 1. Clone and navigate to project
cd lead_scoring_mlops

# 2. Install dependencies with uv
uv sync
uv sync --extra dev  # Includes pytest, mypy, ruff

# 3. Generate synthetic training data
uv run python src/data_generator.py
# Output: data/leads.csv with 10,000 synthetic leads

# 4. Train model and track with MLflow
uv run python src/train_with_mlflow.py
# Output: Model registered in MLflow, metrics logged
```

### Running the API

```bash
# Terminal 1: Start the FastAPI server
uv run uvicorn src.api:app --reload --port 8000

# Terminal 2 (optional): Start MLflow UI
uv run mlflow ui --port 5000
# View at http://localhost:5000
```

### Testing the API

```bash
# Single lead prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lead_id": "LEAD_001",
    "company_size": 150,
    "industry": "Technology",
    "lead_source": "Website",
    "engagement_score": 75.5,
    "page_views": 12,
    "email_opens": 5,
    "demo_requested": 1
  }'

# Health check
curl http://localhost:8000/health
```

### Running Tests & Quality Checks

```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src

# Type checking
uv run mypy src/ --ignore-missing-imports

# Linting
uv run ruff check src/

# Simulate data drift detection
uv run python src/simulate_drift.py
```

## 🔄 Workflow: Training to Serving

```
1. Data Generation (synthetic, realistic)
         ↓
2. Feature Engineering (fit transformers)
         ↓
3. Model Training (XGBoost, MLflow tracking)
         ↓
4. Model Evaluation (AUC, Precision, Recall)
         ↓
5. Register to MLflow (version & stage)
         ↓
6. API Server Startup (load model from registry/latest run)
         ↓
7. Serve Predictions (REST API)
         ↓
8. Monitor (PSI, drift detection)
         ↓
9. Retrain if needed (drift detected)
```

## 📁 Project Structure

```
lead_scoring_mlops/
├── data/                         # Generated data directory
│   ├── leads.csv                # Synthetic training data (10k samples)
│   ├── feature_engineer.joblib  # Fitted feature transformer
│   └── monitoring_log.csv       # Drift detection history
│
├── src/
│   ├── data_generator.py        # Synthetic data generation (realistic distributions)
│   ├── features.py              # Feature engineering (consistent train/inference)
│   ├── train_with_mlflow.py     # Model training with MLflow experiment tracking
│   ├── api.py                   # FastAPI REST API (async lifespan pattern)
│   ├── monitoring.py            # PSI-based drift detection
│   ├── simulate_drift.py        # Generate drifted data for testing
│   └── register_model.py        # Register trained model to MLflow registry
│
├── tests/
│   ├── test_features.py         # Feature engineering unit tests
│   ├── test_monitoring.py       # Drift detection tests
│   └── test_api.py              # API endpoint tests
│
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI/CD pipeline
│
├── pyproject.toml               # uv project config (Python 3.10, dependencies)
├── README.md                    # This file
```

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML** | XGBoost, scikit-learn | Classification model |
| **Feature Engineering** | pandas, scikit-learn | Data transformation & scaling |
| **Experiment Tracking** | MLflow | Model versioning & registry |
| **API** | FastAPI, Pydantic | REST API with async/validation |
| **Monitoring** | NumPy, pandas | PSI-based drift detection |
| **Testing** | pytest, coverage | Unit & integration tests |
| **Type Safety** | mypy | Static type checking |
| **Linting** | ruff | Code quality |
| **CI/CD** | GitHub Actions | Automated testing & checks |
| **Package Manager** | uv | Fast Python dependency management |
