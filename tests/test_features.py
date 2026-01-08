"""Tests for feature engineering."""

import sys

import numpy as np
import pandas as pd
import pytest

from src.features import LeadFeatureEngineer

sys.path.append('.')

# inserts at the top of the path
# project_root = Path().absolute().parent.parent
# sys.path.insert(0, str(project_root))

# using pytest fixtures for reusable test data.
# Pytest sees the fixture and automatically calls it, then injects its return value into any test function that has a parameter named sample_data.
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'company_size': [100, 500, 1000],
        'industry': ['Technology', 'Finance', 'Healthcare'],
        'lead_source': ['Website', 'Referral', 'Event'],
        'engagement_score': [50.0, 75.0, 25.0],
        'page_views': [5, 10, 2],
        'email_opens': [3, 8, 1],
        'demo_requested': [0, 1, 0],
    })

# functions start with 'test_' to be auto-discovered by pytest
def test_feature_engineer_fit_transforms(sample_data):
    """Test that fit_transform produces expected output."""
    fe = LeadFeatureEngineer()
    result = fe.fit_transform(sample_data)

    # Check output shape
    assert len(result) == len(sample_data)
    assert len(result.columns) == len(fe.get_feature_names())

    # Check no null values
    assert result.isnull().sum().sum() == 0

def test_feature_engineer_save_load(sample_data, tmp_path):
    """Test save and load functionality."""
    fe = LeadFeatureEngineer()
    fe.fit(sample_data)
    original_result = fe.transform(sample_data)

    # Save
    save_path = tmp_path / "feature_engineer.joblib"
    fe.save(str(save_path))

    # Load
    fe_loaded = LeadFeatureEngineer.load(str(save_path))
    loaded_result = fe_loaded.transform(sample_data)

    # Should be equal
    pd.testing.assert_frame_equal(original_result, loaded_result)

def test_derived_features(sample_data):
    """Test that derived features are calculated correctly."""
    fe = LeadFeatureEngineer()
    result = fe.fit_transform(sample_data)

    # Check engagement_per_pageview
    expected = sample_data['engagement_score'] / (sample_data['page_views'] + 1)
    np.testing.assert_array_almost_equal(
        result['engagement_per_pageview'].values,
        expected.values
    )

    # Check total_interactions
    expected = sample_data['page_views'] + sample_data['email_opens']
    np.testing.assert_array_almost_equal(
        result['total_interactions'].values,
        expected.values
    )

