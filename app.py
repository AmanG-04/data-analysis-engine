from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_analysis_engine.analytics import (  # noqa: E402
    build_funnel,
    market_research_snapshot,
    run_segmentation,
    train_churn_model,
)
from data_analysis_engine.data_generation import generate_customer_data  # noqa: E402


st.set_page_config(page_title="Data Analytics Engine", layout="wide")

st.title("Data Analytics & Insights Engine")
st.caption("Customer segmentation, funnel analytics, churn prediction, and market research in one app.")

with st.sidebar:
    st.header("Configuration")
    n_customers = st.slider("Customers", min_value=500, max_value=15000, value=5000, step=500)
    random_seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    n_clusters = st.slider("Segments (KMeans)", min_value=3, max_value=8, value=4, step=1)


@st.cache_data(show_spinner=False)
def load_data(size: int, seed: int) -> pd.DataFrame:
    return generate_customer_data(n_customers=size, random_state=seed)


df = load_data(n_customers, int(random_seed))
segmented_df, segment_profile = run_segmentation(df, n_clusters=n_clusters)
funnel_df = build_funnel(df)
churn_result = train_churn_model(df)
channel_view, region_view = market_research_snapshot(df)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Customers", f"{len(df):,}")
kpi_2.metric("Avg Spend", f"INR {df['monthly_spend'].mean():.2f}")
kpi_3.metric("Conversion Rate", f"{df['converted'].mean() * 100:.2f}%")
kpi_4.metric("Churn Rate", f"{df['churned'].mean() * 100:.2f}%")

tab_overview, tab_segmentation, tab_funnel, tab_churn, tab_market = st.tabs(
    ["Overview", "Segmentation", "Funnel", "Churn Model", "Market Research"]
)

with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        spend_hist = px.histogram(df, x="monthly_spend", nbins=40, title="Monthly Spend Distribution")
        st.plotly_chart(spend_hist, use_container_width=True)
    with col_right:
        churn_by_plan = (
            df.groupby("plan_type", as_index=False)["churned"].mean().rename(columns={"churned": "churn_rate"})
        )
        churn_chart = px.bar(churn_by_plan, x="plan_type", y="churn_rate", title="Churn Rate by Plan")
        st.plotly_chart(churn_chart, use_container_width=True)

with tab_segmentation:
    st.subheader("Customer Segments")
    scatter = px.scatter(
        segmented_df,
        x="tenure_months",
        y="monthly_spend",
        color=segmented_df["segment"].astype(str),
        hover_data=["age", "sessions", "support_tickets", "channel", "region"],
        title="Segments by Tenure and Spend",
    )
    st.plotly_chart(scatter, use_container_width=True)
    st.dataframe(segment_profile, use_container_width=True)

with tab_funnel:
    st.subheader("Conversion Funnel")
    funnel_chart = px.funnel(funnel_df, x="users", y="stage", title="User Journey Funnel")
    st.plotly_chart(funnel_chart, use_container_width=True)
    st.dataframe(funnel_df, use_container_width=True)

with tab_churn:
    st.subheader("Churn Prediction Quality")
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Accuracy", churn_result.accuracy)
    metric_col_2.metric("F1", churn_result.f1)
    metric_col_3.metric("ROC-AUC", churn_result.roc_auc)

    st.markdown("Top churn drivers by model weight")
    st.dataframe(churn_result.feature_weights.head(15), use_container_width=True)

with tab_market:
    st.subheader("Channel View")
    channel_chart = px.bar(
        channel_view,
        x="channel",
        y="avg_monthly_spend",
        color="customers",
        title="Average Monthly Spend by Channel",
    )
    st.plotly_chart(channel_chart, use_container_width=True)
    st.dataframe(channel_view, use_container_width=True)

    st.subheader("Regional View")
    region_chart = px.bar(
        region_view,
        x="region",
        y="conversion_rate",
        color="customers",
        title="Conversion Rate by Region",
    )
    st.plotly_chart(region_chart, use_container_width=True)
    st.dataframe(region_view, use_container_width=True)
