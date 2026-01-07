"""
Feature engineering for lead scoring.

WHY A FEATURE CLASS?
- Ensures consistency between training and inference
- Documents feature transformations
- Enables easy testing
- Prevents train-serve skew (a common ML bug)

USAGE:
    from src.features import LeadFeatureEngineer

    # Training
    fe = LeadFeatureEngineer()
    X_train = fe.fit_transform(train_df)
    fe.save('feature_engineer.joblib')

    # Inference
    fe = LeadFeatureEngineer.load('feature_engineer.joblib')
    X_new = fe.transform(new_data)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List
import joblib


class LeadFeatureEngineer:
    """
    Transforms raw lead data into model-ready features.

    This class ensures the same transformations are applied
    during training and inference (critical for production!).

    Features created:
    - Scaled numerical features (z-score normalization)
    - Encoded categorical features (label encoding)
    - Derived features (engagement_per_pageview, total_interactions)
    """

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.fitted = False

        # Define feature groups
        self.numerical_features = [
            'company_size',
            'engagement_score',
            'page_views',
            'email_opens'
        ]
        self.categorical_features = ['industry', 'lead_source']
        self.binary_features = ['demo_requested']

    def fit(self, df: pd.DataFrame) -> 'LeadFeatureEngineer':
        """
        Fit transformers on training data.

        WHY FIT SEPARATELY?
        - Prevents data leakage from test set
        - Stores parameters for inference
        - Makes transformations reproducible

        Args:
            df: Training DataFrame with raw features

        Returns:
            self (for method chaining)
        """
        # Fit scalers for numerical features
        for col in self.numerical_features:
            self.scalers[col] = StandardScaler()
            self.scalers[col].fit(df[[col]])

        # Fit encoders for categorical features
        for col in self.categorical_features:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with transformed features

        Raises:
            ValueError: If fit() hasn't been called
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        result = pd.DataFrame(index=df.index)

        # Scale numerical features
        for col in self.numerical_features:
            result[f'{col}_scaled'] = self.scalers[col].transform(df[[col]]).flatten()

        # Encode categorical features
        for col in self.categorical_features:
            # Handle unseen categories gracefully
            known_categories = set(self.encoders[col].classes_)
            col_values = df[col].apply(
                lambda x: x if x in known_categories else self.encoders[col].classes_[0]
            )
            result[f'{col}_encoded'] = self.encoders[col].transform(col_values)

        # Copy binary features
        for col in self.binary_features:
            result[col] = df[col].values

        # Create derived features
        # WHY DERIVED FEATURES?
        # - Capture relationships between features
        # - Often more predictive than raw features
        result['engagement_per_pageview'] = (
            df['engagement_score'] / (df['page_views'] + 1)  # +1 to avoid division by zero
        )
        result['total_interactions'] = df['page_views'] + df['email_opens']

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """
        Return list of output feature names.

        Useful for:
        - Feature importance analysis
        - Model interpretation
        - Documentation
        """
        names = []
        names.extend([f'{col}_scaled' for col in self.numerical_features])
        names.extend([f'{col}_encoded' for col in self.categorical_features])
        names.extend(self.binary_features)
        names.extend(['engagement_per_pageview', 'total_interactions'])
        return names

    def save(self, path: str):
        """
        Save fitted transformers to disk.

        WHY SAVE?
        - Use same transformers during inference
        - Reproducible predictions
        - Version control for features
        """
        joblib.dump({
            'scalers': self.scalers,
            'encoders': self.encoders,
            'fitted': self.fitted,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'binary_features': self.binary_features,
        }, path)
        print(f"Feature engineer saved to: {path}")

    @classmethod
    def load(cls, path: str) -> 'LeadFeatureEngineer':
        """
        Load fitted transformers from disk.

        Args:
            path: Path to saved feature engineer

        Returns:
            LeadFeatureEngineer instance with fitted transformers
        """
        data = joblib.load(path)
        engineer = cls()
        engineer.scalers = data['scalers']
        engineer.encoders = data['encoders']
        engineer.fitted = data['fitted']
        engineer.numerical_features = data.get('numerical_features', engineer.numerical_features)
        engineer.categorical_features = data.get('categorical_features', engineer.categorical_features)
        engineer.binary_features = data.get('binary_features', engineer.binary_features)
        return engineer


if __name__ == "__main__":
    # Demo usage
    import pandas as pd
    from pathlib import Path

    # Load sample data
    data_path = Path(__file__).parent.parent / 'data' / 'leads.csv'

    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} leads")

        # Create and fit feature engineer
        fe = LeadFeatureEngineer()
        X = fe.fit_transform(df)

        print(f"\nOriginal features: {df.columns.tolist()}")
        print(f"Transformed features: {fe.get_feature_names()}")
        print(f"\nShape: {df.shape} -> {X.shape}")

        print(f"\nSample transformed data:")
        print(X.head())
    else:
        print(f"Data not found at {data_path}")
        print("Run 'python src/data_generator.py' first")
