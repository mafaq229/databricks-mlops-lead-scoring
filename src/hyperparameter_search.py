import itertools
from typing import Dict, List, Tuple

from src.train_with_mlflow import train_model

# define hyperparameter grid
param_grid: Dict[str, List] = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
}

# run all combinations
for n_estimators, max_depth, learning_rate in itertools.product(
        param_grid['n_estimators'],  # type: ignore
        param_grid['max_depth'],  # type: ignore
        param_grid['learning_rate'],  # type: ignore
):
    print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    train_model(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        experiment_name="lead_scoring_hyperparameter_search",
    )

print("\nHyperparameter search completed. Check MLflow UI for results.")
