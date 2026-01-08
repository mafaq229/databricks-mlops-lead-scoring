"""FastAPI application for lead scoring."""

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import LeadFeatureEngineer

sys.path.append('..')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model
    global feature_engineer

    # load the latest registered model from MLflow Model Registry
    model_name = "lead_scoring_model"
    try:
        # try to load from model registry (if registered)
        # Use sklearn.load_model to get actual model with predict_proba()
        # Option 2: mlflow.pyfunc.load_model() has predict() only which gives probabilities directly
        # Options 3: load using pyfunc and then access the underlying model using actual_model = pyfunc_model._model_impl.python_model
        # score = actual_model.predict_proba(X)[:, 1][0]
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/latest")
        print(f"Loaded model: {model_name} from MLflow Model Registry.")
    except Exception as e:
        # Fallback: load from latest run
        print(f"Registry load failed ({e}), loading from latest run...")
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name("lead_scoring")
        if experiment is None:
            print("Experiment 'lead_scoring' not found")
            model = None
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["metrics.roc_auc DESC"],
                max_results=1
            )
            if runs:
                model = mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/lead_scoring_model")
                print(f"Loaded model from run ID: {runs[0].info.run_id}")

    # initialize feature engineer
    feature_engineer = LeadFeatureEngineer().load('data/feature_engineer.joblib')
    print("Feature engineer loaded.")

    yield  # app runs here

    # shutdown
    print("Shutting down Lead Scoring API.")


# initialize fastapi app
app = FastAPI(
    title="Lead Scoring API",
    description="API for predicting lead conversion",
    version="1.0.0",
    lifespan=lifespan
)

# global variables for model and feature engineer
model = None
feature_engineer = None


class LeadInput(BaseModel):
    """
    Input schema for a single lead.
    """
    lead_id: str = Field(..., description="Unique identifier for the lead") # ... means required, no default
    company_size: int = Field(..., description="Number of employees")
    industry: str = Field(..., description="Industry sector")
    lead_source: str = Field(..., description="How lead was acquired")
    engagement_score: float = Field(..., description="Engagement score of the lead 0-100")
    page_views: int = Field(..., description="Website page views")
    email_opens: int = Field(..., description="Number of email opens")
    demo_requested: int = Field(..., description="Whether demo was requested (1 or 0)")


class LeadPrediction(BaseModel):
    """
    Output schema for lead conversion prediction.
    """
    lead_id: str = Field(..., description="Unique identifier for the lead")
    score: float = Field(..., description="Predicted conversion probability (0-1)")
    score_tier: str = Field(..., description="Score tier: Low, Medium, High")
    model_version: str = Field(..., description="Version of the model used for prediction")
    scored_at: str = Field(..., description="Timestamp when the lead was scored")


class BatchInput(BaseModel):
    """
    Input schema for batch lead predictions.
    """
    leads: List[LeadInput] = Field(..., description="List of leads to be scored")


class BatchOutput(BaseModel):
    """
    Output schema for batch lead predictions.
    """
    predictions: List[LeadPrediction] = Field(..., description="List of lead predictions")


def get_score_tier(score: float) -> str:
    """
    Determine score tier based on score.
    """
    if score < 0.4:
        return "Low"
    elif score < 0.7:
        return "Medium"
    else:
        return "High"


@app.get("/")
async def root():
    """health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/health")
async def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_engineer_loaded": feature_engineer is not None
    }

@app.post("/predict", response_model=LeadPrediction)
async def predict(lead: LeadInput):
    """Predict lead conversion probability for a single lead."""
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model or feature engineer not loaded.")

    # convert input to dataframe
    lead_df = pd.DataFrame([lead.model_dump()])

    # feature engineering
    X = feature_engineer.transform(lead_df)

    # predict
    score = model.predict_proba(X)[:, 1][0]  # probability of positive class

    # prepare output
    prediction = LeadPrediction(
        lead_id=lead.lead_id,
        score=round(score, 4),
        score_tier=get_score_tier(score),
        model_version="1.0.0", # hardcoded for simplicity
        scored_at=datetime.now().isoformat()
    )
    return prediction

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    """Predict lead conversion probabilities for a batch of leads."""
    if model is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Model or feature engineer not loaded.")

    # convert input to dataframe
    leads_df = pd.DataFrame([lead.model_dump() for lead in batch.leads])

    # feature engineering
    X = feature_engineer.transform(leads_df)

    # predict
    scores = model.predict_proba(X)[:, 1]  # probabilities of positive class

    predictions = [
        LeadPrediction(
            lead_id=lead.lead_id,
            score=round(float(score), 4),
            score_tier=get_score_tier(float(score)),
            model_version="1.0.0",
            scored_at=datetime.utcnow().isoformat()
        )
        for lead, score in zip(batch.leads, scores)
    ]

    return BatchOutput(predictions=predictions)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
