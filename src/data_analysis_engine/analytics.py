from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_SEGMENT_FEATURES = [
    "age",
    "tenure_months",
    "sessions",
    "support_tickets",
    "last_30d_activity",
    "monthly_spend",
]


@dataclass
class ChurnModelResult:
    accuracy: float
    f1: float
    roc_auc: float
    feature_weights: pd.DataFrame


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster customers into behavior-based segments."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")),
        ]
    )

    segments = model.fit_predict(df[NUMERIC_SEGMENT_FEATURES])
    segmented_df = df.copy()
    segmented_df["segment"] = segments

    profile = (
        segmented_df.groupby("segment", as_index=False)[NUMERIC_SEGMENT_FEATURES + ["churned"]]
        .mean(numeric_only=True)
        .round(2)
        .sort_values("monthly_spend", ascending=False)
    )
    return segmented_df, profile


def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a simplified conversion funnel."""
    visited = len(df)
    engaged = int((df["sessions"] >= 3).sum())
    activated = int((df["last_30d_activity"] >= 7).sum())
    subscribed = int(df["subscribed"].sum())
    converted = int(df["converted"].sum())

    funnel = pd.DataFrame(
        {
            "stage": ["Visited", "Engaged", "Activated", "Subscribed", "Converted"],
            "users": [visited, engaged, activated, subscribed, converted],
        }
    )
    funnel["conversion_rate"] = (funnel["users"] / visited).round(4)
    return funnel


def train_churn_model(df: pd.DataFrame) -> ChurnModelResult:
    """Train a churn classifier and return core quality metrics."""
    features = [
        "age",
        "tenure_months",
        "sessions",
        "support_tickets",
        "last_30d_activity",
        "monthly_spend",
        "channel",
        "region",
        "plan_type",
    ]

    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    categorical = ["channel", "region", "plan_type"]
    numeric = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=400, random_state=42)),
        ]
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
    coefficients = clf.named_steps["classifier"].coef_[0]
    weights = (
        pd.DataFrame({"feature": feature_names, "weight": coefficients})
        .assign(abs_weight=lambda frame: frame["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns="abs_weight")
    )

    return ChurnModelResult(
        accuracy=round(float(accuracy), 4),
        f1=round(float(f1), 4),
        roc_auc=round(float(roc_auc), 4),
        feature_weights=weights,
    )


def market_research_snapshot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate channel and regional performance views."""
    channel_view = (
        df.groupby("channel", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            churn_rate=("churned", "mean"),
        )
        .round({"avg_monthly_spend": 2, "churn_rate": 4})
        .sort_values("customers", ascending=False)
    )

    region_view = (
        df.groupby("region", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            avg_monthly_spend=("monthly_spend", "mean"),
            conversion_rate=("converted", "mean"),
        )
        .round({"avg_monthly_spend": 2, "conversion_rate": 4})
        .sort_values("customers", ascending=False)
    )

    return channel_view, region_view
