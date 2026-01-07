"""Register the best model to MLflow Model Registry."""

import mlflow
from mlflow import MlflowClient


def register_best_model(
    experiment_name: str = "lead_scoring",
    model_name: str = "lead_scoring_model"
):
    """
    Find the best model from the given experiment and register it to the MLflow Model Registry.
    """
    client = MlflowClient()

    # get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # find the best run based on validation AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'", # attributes.status is older mlflow syntax
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No finished runs found in experiment '{experiment_name}'.")

    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id} with roc_auc: {best_run.data.metrics['roc_auc']:.4f}")

    # register the model
    model_uri = f"runs:/{best_run.info.run_id}/lead_scoring_model"
    result = mlflow.register_model(model_uri, model_name)

    print(f"Model registered: {result.name}, version: {result.version}")
    return result

if __name__ == "__main__":
    register_best_model()
