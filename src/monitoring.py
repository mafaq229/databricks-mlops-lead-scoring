"""Monitoring utilities for lead scoring model."""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10
) -> float:
    """
    Calculate Population Stability Index.

    PSI measures how much a distribution has shifted.

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, monitor
    - PSI >= 0.2: Significant change, investigate

    WHY PSI?
    - Industry standard for drift detection
    - Works for any feature type
    - Easy to interpret and explain
    """
    # create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # calculate proportions in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # convert to proportions (add small value to avoid log(0))
    expected_pct = (expected_counts + 0.0001) / len(expected)
    actual_pct = (actual_counts + 0.0001) / len(actual)

    # calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi) # to standard float type from numpy float64


def check_prediction_drift(
    baseline_scores: pd.Series,
    current_scores: pd.Series
) -> Dict:
    """
    Check if model predictions have drifted.

    Returns metrics and alerts.
    """
    metrics = {
        'baseline_mean': baseline_scores.mean(),
        'current_mean': current_scores.mean(),
        'mean_shift': abs(current_scores.mean() - baseline_scores.mean()),
        'baseline_std': baseline_scores.std(),
        'current_std': current_scores.std(),
        'psi': calculate_psi(baseline_scores, current_scores),
    }

    # Calculate relative shift
    metrics['mean_shift_pct'] = (
        metrics['mean_shift'] / metrics['baseline_mean'] * 100
        if metrics['baseline_mean'] > 0 else 0
    )

    # Generate alerts
    alerts = []
    if metrics['psi'] > 0.2:
        alerts.append(f"HIGH: PSI={metrics['psi']:.3f} exceeds 0.2 threshold")
    elif metrics['psi'] > 0.1:
        alerts.append(f"MEDIUM: PSI={metrics['psi']:.3f} exceeds 0.1 threshold")

    if metrics['mean_shift_pct'] > 10:
        alerts.append(f"HIGH: Mean shifted by {metrics['mean_shift_pct']:.1f}%")

    metrics['alerts'] = alerts
    metrics['status'] = 'ALERT' if any('HIGH' in a for a in alerts) else (
        'WATCH' if alerts else 'OK'
    )

    return metrics


def check_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: List[str]
) -> Dict[str, Dict]:
    """
    Check drift for multiple features.
    """
    results = {}

    for feature in numerical_features:
        psi = calculate_psi(baseline_df[feature], current_df[feature])
        status = 'ALERT' if psi > 0.2 else ('WATCH' if psi > 0.1 else 'OK')

        results[feature] = {
            'psi': psi,
            'baseline_mean': baseline_df[feature].mean(),
            'current_mean': current_df[feature].mean(),
            'status': status
        }

    return results


class MonitoringDashboard:
    """
    Simple monitoring dashboard that logs metrics.

    In production, this would write to a database
    and feed a visualization tool.
    """

    def __init__(self, log_path: str = 'data/monitoring_log.csv'):
        self.log_path = log_path
        self.metrics_history = []

    def log_metrics(self, metrics: Dict):
        """Log metrics with timestamp."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)

        # Save to CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.log_path, index=False)

    def get_history(self) -> pd.DataFrame:
        """Get metrics history."""
        try:
            return pd.read_csv(self.log_path)
        except FileNotFoundError:
            return pd.DataFrame()

    def print_summary(self):
        """Print current status summary."""
        history = self.get_history()
        if len(history) == 0:
            print("No monitoring data available")
            return

        latest = history.iloc[-1]
        print("\n" + "="*50)
        print("MONITORING SUMMARY")
        print("="*50)
        print(f"Last check: {latest['timestamp']}")
        print(f"Status: {latest.get('status', 'N/A')}")
        print(f"PSI: {latest.get('psi', 'N/A'):.4f}")
        print(f"Mean shift: {latest.get('mean_shift_pct', 'N/A'):.2f}%")

        if 'alerts' in latest and latest['alerts']:
            print(f"\nAlerts: {latest['alerts']}")


# Demo usage
if __name__ == "__main__":
    # Generate baseline and "drifted" data
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0.4, 0.15, 1000))
    current = pd.Series(np.random.normal(0.45, 0.18, 1000))  # Slight drift

    # Check drift
    result = check_prediction_drift(baseline, current)
    print(f"PSI: {result['psi']:.4f}")
    print(f"Status: {result['status']}")
    print(f"Alerts: {result['alerts']}")
