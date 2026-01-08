"""Tests for monitoring functions."""

import sys

import numpy as np
import pandas as pd

from src.monitoring import calculate_psi, check_prediction_drift

sys.path.append('.')

def test_psi_no_drift():
    """PSI should be low when distributions are identical."""
    np.random.seed(42)
    data = pd.Series(np.random.normal(0, 1, 1000))

    psi = calculate_psi(data, data)

    # PSI should be very close to 0
    assert psi < 0.01


def test_psi_high_drift():
    """PSI should be high when distributions differ significantly."""
    np.random.seed(42)
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(2, 1, 1000))  # Shifted mean

    psi = calculate_psi(expected, actual)

    # PSI should exceed alert threshold
    assert psi > 0.2


def test_check_prediction_drift_status():
    """Test that status is correctly assigned."""
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0.5, 0.1, 1000))

    # No drift
    current_ok = pd.Series(np.random.normal(0.5, 0.1, 1000))
    result = check_prediction_drift(baseline, current_ok)
    assert result['status'] == 'OK'

    # High drift
    current_drift = pd.Series(np.random.normal(0.8, 0.1, 1000))
    result = check_prediction_drift(baseline, current_drift)
    assert result['status'] == 'ALERT'
