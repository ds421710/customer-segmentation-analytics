"""
Phase 7: Interactive Business Intelligence Dashboard (Streamlit)
7-Tab Executive Dashboard integrating Data Upload/Cleaning, EDA,
RFM + K-Means Segmentation, Buying Behaviour, CLV, Retention, and CRM Actions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project root is on sys.path for modular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.clv_behaviour import (
    analyze_segment_buying_behaviour,
    compare_clv_by_segment,
    compute_clv_metrics,
    identify_inactivity_risk,
)
from src.data_preprocessing import clean_transactions, load_dataset
from src.eda_analysis import (
    compute_eda_kpis,
    get_customer_spending_summary,
    get_monthly_trends,
    get_top_products,
)
from src.recommendation import assign_customer_actions, generate_segment_recommendations
from src.rfm_segmentation import (
    compute_rfm_table,
    evaluate_kmeans_k,
    run_kmeans_segmentation,
)

st.set_page_config(
    page_title="Customer Segmentation & Buying Behaviour BI",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Segmentation & Buying Behaviour Analytics")
st.caption(
    "MBA Marketing Analytics & Machine Learning Decision Support System | "
    "RFM · K-Means Clustering · Historical & Predictive CLV · CRM Strategy"
)


@st.cache_data(show_spinner="Cleaning and preparing transaction data...")
def prepare_data(raw_df: pd.DataFrame):
    valid_df, cancelled_df, audit = clean_transactions(raw_df)
    return valid_df, cancelled_df, audit


# ---------------------------------------------------------------------------
# Sidebar: Data Source & Global Filters
# ---------------------------------------------------------------------------
st.sidebar.header("1. Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Retail Transaction CSV/Excel", type=["csv", "xlsx", "xls"]
)

default_paths = [
    ROOT_DIR / "data" / "raw" / "Online Retail.xlsx",
    ROOT_DIR / "data" / "raw" / "online_retail.csv",
]
raw_df = None
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", dtype={"InvoiceNo": str, "StockCode": str})
    else:
        raw_df = pd.read_excel(uploaded_file, dtype={"InvoiceNo": str, "StockCode": str})
else:
    for p in default_paths:
        if p.exists():
            raw_df = load_dataset(p)
            st.sidebar.success(f"Loaded local dataset: {p.name}")
            break

if raw_df is None:
    st.info(
        "Upload a transaction CSV/Excel file in the sidebar (or place `Online Retail.xlsx` "
        "in `data/raw/`) to launch the interactive dashboard."
    )
    st.stop()

valid_df, cancelled_df, audit = prepare_data(raw_df)

st.sidebar.header("2. Filters & Model Settings")
countries = ["All Countries"] + sorted(valid_df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

min_date = valid_df["InvoiceDate"].min().date()
max_date = valid_df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input("Invoice Date Range", value=(min_date, max_date))

n_clusters = st.sidebar.slider("K-Means Clusters (k)", min_value=3, max_value=6, value=4)
inactivity_days = st.sidebar.slider("Inactivity Threshold (Days)", min_value=30, max_value=180, value=90, step=15)

# Apply filters
filtered_df = valid_df.copy()
if selected_country != "All Countries":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered_df = filtered_df[(filtered_df["InvoiceDate"] >= start_d) & (filtered_df["InvoiceDate"] < end_d)]

if filtered_df.empty:
    st.warning("No transactions match the selected filters. Please widen the date range or country.")
    st.stop()

# Run downstream analytics on filtered slice
kpis = compute_eda_kpis(filtered_df)
rfm_df = compute_rfm_table(filtered_df)
k_eval_df = evaluate_kmeans_k(rfm_df, k_range=range(2, 7))
rfm_clustered, cluster_profile = run_kmeans_segmentation(rfm_df, n_clusters=n_clusters)
risk_df = identify_inactivity_risk(filtered_df, rfm_clustered, inactivity_threshold_days=inactivity_days)
clv_df = compute_clv_metrics(risk_df)
clv_seg_df = compare_clv_by_segment(clv_df)
rec_table = generate_segment_recommendations(clv_df)
customer_actions_df = assign_customer_actions(clv_df)

# ---------------------------------------------------------------------------
# 7 Executive Dashboard Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "1. Overall Performance",
    "2. Customer Segmentation",
    "3. Buying Behaviour",
    "4. RFM Analysis",
    "5. Customer Lifetime Value",
    "6. Retention Insights",
    "7. Marketing Recommendations",
])

# Tab 1: Overall Business Performance
with tabs[0]:
    st.subheader("Executive Business Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"£{kpis['total_revenue']:,.2f}")
    c2.metric("Total Customers", f"{kpis['total_customers']:,}")
    c3.metric("Total Orders", f"{kpis['total_orders']:,}")
    c4.metric("Avg Order Value", f"£{kpis['avg_order_value']:,.2f}")
    c5.metric("Repeat Customer Rate", f"{kpis['repeat_rate_pct']}%")

    col_a, col_b = st.columns(2)
    with col_a:
        monthly = get_monthly_trends(filtered_df)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.lineplot(data=monthly, x="YearMonth", y="Revenue", marker="o", ax=ax, color="#1f77b4")
        ax.set_title("Monthly Revenue Trend")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        top_prod = get_top_products(filtered_df, top_n=10)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=top_prod, x="Revenue", y="Description", ax=ax, color="#2ca02c")
        ax.set_title("Top 10 Merchandise SKUs by Revenue")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Data Quality Audit Summary"):
        st.json(audit)

# Tab 2: Customer Segmentation (K-Means)
with tabs[1]:
    st.subheader(f"K-Means Customer Segmentation (k = {n_clusters})")
    st.dataframe(cluster_profile, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.lineplot(data=k_eval_df, x="k", y="SilhouetteScore", marker="s", ax=ax, color="#2ca02c")
        ax.set_title("Silhouette Score by k")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.scatterplot(
            data=rfm_clustered, x="Recency", y="Monetary",
            hue="ClusterLabel", size="Frequency", alpha=0.75, ax=ax,
        )
        ax.set_title("Clusters: Recency vs. Monetary Spend")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# Tab 3: Buying Behaviour Analysis
with tabs[2]:
    st.subheader("Segment Buying Behaviour & Repeat Patterns")
    seg_stats, top_skus = analyze_segment_buying_behaviour(filtered_df, rfm_clustered)
    st.dataframe(seg_stats.merge(top_skus, on="ClusterLabel", how="left"), use_container_width=True)

    cust_summary = get_customer_spending_summary(filtered_df)
    buyer_split = cust_summary.groupby("BuyerType", as_index=False).agg(
        Customers=("CustomerID", "count"), Revenue=("TotalSpend", "sum")
    )
    st.bar_chart(buyer_split.set_index("BuyerType")[["Customers", "Revenue"]])

# Tab 4: RFM Analysis
with tabs[3]:
    st.subheader("RFM Quintile & Lifecycle Segment Breakdown")
    crm_counts = (
        rfm_clustered.groupby("CRM_Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .round(2)
        .sort_values("AvgMonetary", ascending=False)
    )
    st.dataframe(crm_counts, use_container_width=True)
    st.dataframe(
        rfm_clustered[["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "CRM_Segment", "ClusterLabel"]].head(100),
        use_container_width=True,
    )

# Tab 5: Customer Lifetime Value
with tabs[4]:
    st.subheader("Historical vs. Predictive 12-Month CLV (Gross Revenue Basis)")
    st.caption(
        "Historical CLV reflects cumulative past spend; Predictive CLV estimates 12-month "
        "expected revenue weighted by recency-based retention probability."
    )
    st.dataframe(clv_seg_df, use_container_width=True)

    melted = clv_seg_df.melt(
        id_vars="ClusterLabel",
        value_vars=["AvgHistoricalCLV", "AvgPredictiveCLV"],
        var_name="CLV_Type",
        value_name="Amount_GBP",
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=melted, x="ClusterLabel", y="Amount_GBP", hue="CLV_Type", ax=ax)
    ax.set_title("Average Historical vs. Predictive CLV by Segment")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 6: Retention Insights
with tabs[5]:
    st.subheader("Cadence-Based Inactivity & Churn Risk")
    flagged = clv_df[clv_df["BecomingInactive"]].sort_values("Predictive_CLV", ascending=False)
    r1, r2, r3 = st.columns(3)
    r1.metric("Repeat Customers Flagged Inactive", f"{len(flagged):,}")
    r2.metric("At-Risk Historical Spend", f"£{flagged['Historical_CLV'].sum():,.2f}")
    r3.metric("At-Risk Predictive CLV", f"£{flagged['Predictive_CLV'].sum():,.2f}")

    st.dataframe(
        flagged[["CustomerID", "CRM_Segment", "ClusterLabel", "Recency", "AvgInterPurchaseDays", "Frequency", "Historical_CLV", "Predictive_CLV", "RetentionProb"]],
        use_container_width=True,
    )

# Tab 7: Marketing Recommendations & Downloads
with tabs[6]:
    st.subheader("Actionable CRM & Retention Playbook")
    st.dataframe(rec_table, use_container_width=True)

    csv_rec = rec_table.to_csv(index=False).encode("utf-8")
    csv_cust = customer_actions_df.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Segment Strategy Report (CSV)",
        data=csv_rec,
        file_name="segment_marketing_recommendations.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download Customer-Level Action List (CSV)",
        data=csv_cust,
        file_name="customer_action_list.csv",
        mime="text/csv",
    )
