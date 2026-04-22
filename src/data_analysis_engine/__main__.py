from __future__ import annotations

from data_analysis_engine.analytics import build_funnel, market_research_snapshot, run_segmentation, train_churn_model
from data_analysis_engine.data_generation import generate_customer_data


def main() -> None:
    df = generate_customer_data(n_customers=3000, random_state=42)
    segmented_df, profile = run_segmentation(df, n_clusters=4)
    funnel = build_funnel(df)
    churn = train_churn_model(df)
    channel_view, region_view = market_research_snapshot(df)

    print("Data Analysis Engine - CLI Snapshot")
    print(f"Rows: {len(df)}")
    print(f"Segments: {segmented_df['segment'].nunique()}")
    print(f"Funnel converted users: {int(funnel.loc[funnel['stage'] == 'Converted', 'users'].iloc[0])}")
    print(f"Churn metrics -> accuracy: {churn.accuracy}, f1: {churn.f1}, roc_auc: {churn.roc_auc}")
    print("Top channels by customer count:")
    print(channel_view[['channel', 'customers']].to_string(index=False))
    print("Top regions by customer count:")
    print(region_view[['region', 'customers']].to_string(index=False))
    print("Segment profile preview:")
    print(profile.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
