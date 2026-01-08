"""Train lead scoring model with MLflow tracking."""

import json
import sys
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import LeadFeatureEngineer

sys.path.append('..')


def train_model(
        data_path: str = "data/leads.csv",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        experiment_name: str = "lead_scoring",
):
    """
    Train model with full MLflow tracking.
    """
    # set up mlflow experiment
    mlflow.set_experiment(experiment_name)

    # start mlflow run
    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # log training parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "data_path": data_path
        })
        # load and prepare data
        df = pd.read_csv(data_path)
        mlflow.log_param("n_samples", len(df))

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['converted']
        )

        # feature engineering
        feature_engineer = LeadFeatureEngineer()
        X_train = feature_engineer.fit_transform(train_df)
        X_test = feature_engineer.transform(test_df)
        y_train = train_df['converted']
        y_test = test_df['converted']

        mlflow.log_param("n_features", X_train.shape[1])

        # train model
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='auc'
        )
        model.fit(X_train, y_train)

        # evaluate model (model.predict_proba returns probabilities for both classes, we take the probability of the positive class)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "brier_score": brier_score_loss(y_test, y_pred_proba)
        }

        # log metrics
        mlflow.log_metrics(metrics)

        # log model with signature
        signature = infer_signature(X_test, y_pred_proba) # returns both classes probabilities
        mlflow.sklearn.log_model(
            sk_model=model,
            name="lead_scoring_model",
            signature=signature,
            input_example=X_test.iloc[:5]
        )

        # save and log feature engineer
        feature_engineer.save("data/feature_engineer.joblib")
        mlflow.log_artifact("data/feature_engineer.joblib")

        importance = dict(zip(
            feature_engineer.get_feature_names(),
            model.feature_importances_.tolist()
        ))
        with open("data/feature_importance.json", "w") as f:
            json.dump(importance, f)
        mlflow.log_artifact("data/feature_importance.json")

        # print metrics
        print("-------TRAINING COMPLETE-------")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run found")
        return active_run.info.run_id


if __name__ == "__main__":
    run_id = train_model()
    print(f"MLflow run ID: {run_id}")
    print("View results in MLflow UI by running 'mlflow ui' command and at http://localhost:5000")
