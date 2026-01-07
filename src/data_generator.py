"""
Generate synthetic lead data for the scoring model.

WHY SYNTHETIC DATA?
- No need for real customer data (privacy safe)
- Control over data characteristics
- Can simulate drift later for testing

USAGE:
    python src/data_generator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_lead_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic lead data with realistic characteristics.

    Features:
    - company_size: Number of employees (log-normal distribution)
    - industry: Business sector (categorical)
    - lead_source: How the lead was acquired (categorical)
    - engagement_score: Based on interactions (0-100, beta distribution)
    - page_views: Website visits (Poisson)
    - email_opens: Email engagement (Poisson)
    - demo_requested: Binary flag

    Target:
    - converted: Whether lead became customer (0/1)

    The conversion probability is realistically modeled based on features:
    - Higher engagement = higher conversion
    - Demo requests = strong signal
    - Referrals convert better
    - Technology industry slightly higher
    """
    np.random.seed(seed)

    # Generate features
    data = {
        'lead_id': [f'LEAD_{i:06d}' for i in range(n_samples)],

        # Company size follows log-normal (many small, few large)
        'company_size': np.random.lognormal(mean=4, sigma=1.5, size=n_samples).astype(int).clip(1, 50000),

        # Industry distribution based on typical B2B patterns
        'industry': np.random.choice(
            ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Other'],
            size=n_samples,
            p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        ),

        # Lead source distribution
        'lead_source': np.random.choice(
            ['Website', 'Referral', 'Event', 'Cold_Outreach', 'Partner'],
            size=n_samples,
            p=[0.35, 0.25, 0.15, 0.15, 0.10]
        ),

        # Engagement follows beta distribution (most are low-medium)
        'engagement_score': np.random.beta(2, 5, size=n_samples) * 100,

        # Page views and email opens follow Poisson
        'page_views': np.random.poisson(lam=5, size=n_samples),
        'email_opens': np.random.poisson(lam=3, size=n_samples),

        # Demo requests are rare but important
        'demo_requested': np.random.binomial(1, 0.3, size=n_samples),
    }

    df = pd.DataFrame(data)

    # Generate conversion based on features (realistic relationship)
    conversion_prob = (
        0.05 +  # Base rate (5%)
        0.15 * (df['engagement_score'] / 100) +  # Engagement is key
        0.10 * df['demo_requested'] +  # Demo requests are strong signal
        0.10 * (df['lead_source'] == 'Referral').astype(int) +  # Referrals convert well
        0.05 * np.log1p(df['page_views']) / 5 +  # More visits = more interest
        0.05 * (df['industry'] == 'Technology').astype(int)  # Tech slightly higher
    ).clip(0, 0.8)  # Cap at 80%

    df['converted'] = np.random.binomial(1, conversion_prob)

    # Add timestamps (leads over past year)
    base_date = datetime(2024, 1, 1)
    df['created_at'] = [
        base_date + timedelta(days=np.random.randint(0, 365))
        for _ in range(n_samples)
    ]

    return df


def generate_drifted_data(n_samples: int = 1000, drift_type: str = 'gradual') -> pd.DataFrame:
    """
    Generate data with drift for testing monitoring.

    drift_type:
    - 'gradual': Slow shift in distributions
    - 'sudden': Abrupt change in patterns
    - 'seasonal': Cyclical changes (e.g., holiday patterns)
    """
    np.random.seed(99)  # Different seed for drift data
    df = generate_lead_data(n_samples, seed=99)

    if drift_type == 'gradual':
        # Engagement scores trending up 15%
        df['engagement_score'] = df['engagement_score'] * 1.15
        df['engagement_score'] = df['engagement_score'].clip(0, 100)

    elif drift_type == 'sudden':
        # Company sizes dramatically increased (new market segment)
        df['company_size'] = df['company_size'] * 3

        # New partner channel dominating
        partner_mask = np.random.random(len(df)) > 0.3
        df.loc[partner_mask, 'lead_source'] = 'Partner'

    elif drift_type == 'seasonal':
        # Holiday season: lower engagement
        df['engagement_score'] = df['engagement_score'] * 0.7
        df['page_views'] = (df['page_views'] * 0.5).astype(int)

    return df


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # Generate main dataset
    df = generate_lead_data(10000)
    df.to_csv(data_dir / 'leads.csv', index=False)

    print(f"Generated {len(df)} leads")
    print(f"Conversion rate: {df['converted'].mean():.2%}")
    print(f"\nSaved to: {data_dir / 'leads.csv'}")

    print(f"\nFeature distributions:")
    print(f"  - Company size: mean={df['company_size'].mean():.0f}, median={df['company_size'].median():.0f}")
    print(f"  - Engagement score: mean={df['engagement_score'].mean():.1f}")
    print(f"  - Demo requested: {df['demo_requested'].mean():.1%}")

    print(f"\nConversion by lead source:")
    print(df.groupby('lead_source')['converted'].mean().sort_values(ascending=False).to_string())
