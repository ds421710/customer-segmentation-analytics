# Customer Segmentation & Buying Behaviour Analysis

An end-to-end Marketing Analytics, Machine Learning, and Business Intelligence system built to help retail and e-commerce teams identify high-value customer segments, analyze purchasing behaviour, estimate Historical vs. Predictive Customer Lifetime Value (CLV), and execute targeted CRM retention strategies.

## Core Business Question

> *"How can businesses identify high-value customers, understand their purchasing behaviour, and develop targeted marketing strategies to improve customer retention and profitability?"*

## Dataset

This project is designed for the [**Online Retail Dataset (UCI Machine Learning Repository)**](https://archive.ics.uci.edu/dataset/352/online+retail) (541,909 transactions across 8 attributes from December 1, 2010 to December 9, 2011 for a UK-based online gift retailer) and also supports custom CSV/Excel uploads directly in the dashboard.

## Architecture & Modules

* `sql/schema.sql` & `sql/business_queries.sql`: Normalized MySQL schema and analytical SQL queries (KPIs, monthly trends, top SKUs, repeat vs. one-time buyer share).

* `src/data_preprocessing.py`: Data quality auditing, duplicate removal, `'C'`-prefixed cancellation separation, service-fee flagging, and revenue feature engineering.

* `src/eda_analysis.py`: Executive KPIs, seasonality trends, product rankings, and Seaborn/Matplotlib visualizations.

* `src/rfm_segmentation.py`: Deterministic RFM quintile scoring (8 CRM lifecycle segments) + log-scaled K-Means clustering with Elbow/Silhouette evaluation and programmatic cluster labeling.

* `src/clv_behaviour.py`: Segment buying behaviour diagnostics, cadence-based inactivity risk detection, and Historical vs. Predictive 12-Month CLV modeling.

* `src/recommendation.py`: Data-grounded CRM playbook mapping segment metrics to prioritized retention and win-back actions.

* `app/streamlit_app.py`: 7-tab interactive Streamlit BI dashboard with filters and downloadable CSV action reports.

## Quickstart

```
git clone <your-repo-url>
cd customer-segmentation-analytics
pip install -r requirements.txt

# Place UCI Online Retail.xlsx or online_retail.csv in data/raw/ (or upload via UI)
python -m unittest discover -s tests
streamlit run app/streamlit_app.py

```