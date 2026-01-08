"""
Simulate data drift for testing monitoring.

This helps you understand what drift looks like
and test your monitoring alerts.
"""

import numpy as np
import pandas as pd

from src.data_generator import generate_drifted_data, generate_lead_data
from src.monitoring import check_feature_drift


def simulate_drift(drift_type: str = "gradual"):
    """
    Generate data with different types of drift.

    drift_type:
    - 'none': No drift (control)
    - 'gradual': Slow shift over time
    - 'sudden': Abrupt distribution change
    - 'seasonal': Cyclical changes
    """
    np.random.seed(42)

    # Baseline data
    baseline = generate_lead_data(5000, seed=42)

    if drift_type == 'none':
        current = generate_lead_data(1000, seed=123)
    else:
        # Use centralized drift generation (DRY principle)
        current = generate_drifted_data(1000, drift_type=drift_type)

    return baseline, current


def run_drift_analysis():
    """Run drift analysis for all drift types."""
    print("\n" + "="*60)
    print("DRIFT ANALYSIS SIMULATION")
    print("="*60)

    numerical_features = ['company_size', 'engagement_score', 'page_views', 'email_opens']

    for drift_type in ['none', 'gradual', 'sudden', 'seasonal']:
        print(f"\n{'='*40}")
        print(f"Drift Type: {drift_type.upper()}")
        print('='*40)

        baseline, current = simulate_drift(drift_type)

        # Check feature drift
        feature_drift = check_feature_drift(baseline, current, numerical_features)

        print("\nFeature Drift (PSI):")
        for feature, result in feature_drift.items():
            status_emoji = '   ' if result['status'] == 'OK' else ' ! ' if result['status'] == 'WATCH' else '!!!'
            print(f"  {status_emoji} {feature}: {result['psi']:.4f} ({result['status']})")


if __name__ == "__main__":
    run_drift_analysis()
