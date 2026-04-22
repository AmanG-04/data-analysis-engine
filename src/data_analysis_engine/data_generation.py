from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customer_data(n_customers: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data for analytics experiments."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 66, n_customers)
    tenure_months = rng.integers(1, 60, n_customers)
    sessions = rng.poisson(lam=8, size=n_customers) + 1
    support_tickets = rng.poisson(lam=1.4, size=n_customers)
    last_30d_activity = np.clip(rng.normal(loc=14, scale=8, size=n_customers), 0, 30)

    channels = rng.choice(["Organic", "Paid", "Referral", "Affiliate"], p=[0.36, 0.32, 0.2, 0.12], size=n_customers)
    regions = rng.choice(["North", "South", "East", "West"], p=[0.27, 0.24, 0.22, 0.27], size=n_customers)
    plan_type = rng.choice(["Basic", "Standard", "Premium"], p=[0.46, 0.38, 0.16], size=n_customers)

    plan_multiplier = pd.Series(plan_type).map({"Basic": 0.9, "Standard": 1.2, "Premium": 1.8}).to_numpy()
    channel_multiplier = pd.Series(channels).map({"Organic": 1.05, "Paid": 0.95, "Referral": 1.25, "Affiliate": 1.0}).to_numpy()

    monthly_spend = np.clip(
        (25 + tenure_months * 0.6 + sessions * 3.4 + last_30d_activity * 1.2) * plan_multiplier * channel_multiplier
        + rng.normal(0, 20, n_customers),
        8,
        None,
    )

    subscribed = ((sessions >= 4) & (last_30d_activity >= 8)) | (monthly_spend > np.percentile(monthly_spend, 60))
    converted = ((sessions >= 6) & (monthly_spend > np.percentile(monthly_spend, 45))).astype(int)

    churn_logit = (
        -2.4
        + 0.06 * support_tickets
        - 0.02 * tenure_months
        - 0.035 * last_30d_activity
        - 0.0025 * monthly_spend
        + rng.normal(0, 0.55, n_customers)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churned = (rng.random(n_customers) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_customers + 1),
            "age": age,
            "tenure_months": tenure_months,
            "sessions": sessions,
            "support_tickets": support_tickets,
            "last_30d_activity": np.round(last_30d_activity, 2),
            "monthly_spend": np.round(monthly_spend, 2),
            "channel": channels,
            "region": regions,
            "plan_type": plan_type,
            "subscribed": subscribed.astype(int),
            "converted": converted,
            "churned": churned,
        }
    )
