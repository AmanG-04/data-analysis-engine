# Data Analytics & Insights Engine

A working project built from the challenge brief in `problemstatement.txt`.

## What it includes

- Customer segmentation with KMeans clustering
- Funnel analytics across key journey stages
- Churn prediction with a logistic regression model
- Market research views by channel and region
- Interactive Streamlit dashboard

## Project structure

```text
.
|- app.py
|- problemstatement.txt
|- pyproject.toml
|- requirements.txt
|- src/
|  \- data_analysis_engine/
|     |- __init__.py
|     |- analytics.py
|     \- data_generation.py
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- The app uses synthetic data generation so it can run without external datasets.
- Tune customer count, random seed, and number of clusters from the sidebar.
