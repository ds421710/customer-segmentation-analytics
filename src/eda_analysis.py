"""
Phase 3: Exploratory Data Analysis (EDA) & Visualization Module
Computes executive KPIs, monthly/seasonal trends, top products,
spending distributions, and repeat vs. one-time buyer metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def compute_eda_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate core business and customer KPIs from cleaned transactions."""
    total_customers = int(df["CustomerID"].nunique())
    total_orders = int(df["InvoiceNo"].nunique())
    total_revenue = float(df["TotalAmount"].sum())
    total_units = int(df["Quantity"].sum())

    order_totals = df.groupby("InvoiceNo")["TotalAmount"].sum()
    avg_order_value = float(order_totals.mean()) if len(order_totals) > 0 else 0.0

    cust_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
    avg_purchase_freq = float(cust_orders.mean()) if len(cust_orders) > 0 else 0.0
    repeat_customers = int((cust_orders > 1).sum())
    repeat_rate_pct = (repeat_customers / total_customers * 100) if total_customers > 0 else 0.0

    return {
        "total_customers": total_customers,
        "total_orders": total_orders,
        "total_revenue": round(total_revenue, 2),
        "total_units_sold": total_units,
        "avg_order_value": round(avg_order_value, 2),
        "avg_purchase_frequency": round(avg_purchase_freq, 2),
        "repeat_customers": repeat_customers,
        "one_time_customers": total_customers - repeat_customers,
        "repeat_rate_pct": round(repeat_rate_pct, 2),
    }


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly revenue, order count, and active customers."""
    monthly = (
        df.groupby("YearMonth", as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
            ActiveCustomers=("CustomerID", "nunique"),
        )
        .sort_values("YearMonth")
    )
    monthly["Revenue"] = monthly["Revenue"].round(2)
    return monthly


def get_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top N products by revenue, excluding operational service fees."""
    merch = df[~df["IsServiceFee"]] if "IsServiceFee" in df.columns else df
    top_prod = (
        merch.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            UnitsSold=("Quantity", "sum"),
            Revenue=("TotalAmount", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )
    top_prod["Revenue"] = top_prod["Revenue"].round(2)
    return top_prod


def get_customer_spending_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer spend, order count, and buyer type."""
    cust = (
        df.groupby("CustomerID", as_index=False)
        .agg(
            TotalSpend=("TotalAmount", "sum"),
            OrderCount=("InvoiceNo", "nunique"),
            TotalUnits=("Quantity", "sum"),
        )
    )
    cust["BuyerType"] = cust["OrderCount"].apply(
        lambda x: "Repeat Buyer" if x > 1 else "One-Time Buyer"
    )
    cust["TotalSpend"] = cust["TotalSpend"].round(2)
    return cust


def generate_eda_charts(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Path]:
    """Generate and save 4 publication-grade EDA charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Monthly Revenue Trend
    monthly = get_monthly_trends(df)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.lineplot(data=monthly, x="YearMonth", y="Revenue", marker="o", linewidth=2.5, ax=ax, color="#1f77b4")
    ax.set_title("Monthly Revenue Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p1 = out / "eda_monthly_revenue.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["monthly_revenue"] = p1

    # 2. Top 10 Products by Revenue
    top_prod = get_top_products(df, top_n=10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_prod, x="Revenue", y="Description", ax=ax, color="#2ca02c")
    ax.set_title("Top 10 Products by Revenue", fontsize=13, fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("")
    plt.tight_layout()
    p2 = out / "eda_top_products.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["top_products"] = p2

    # 3. Customer Spending Distribution (capped at 95th percentile for readability)
    cust_df = get_customer_spending_summary(df)
    cap = cust_df["TotalSpend"].quantile(0.95)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(cust_df[cust_df["TotalSpend"] <= cap]["TotalSpend"], bins=30, kde=True, ax=ax, color="#ff7f0e")
    ax.set_title("Customer Spending Distribution (Up to 95th Percentile)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Customer Spend (£)")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    p3 = out / "eda_spending_distribution.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths["spending_distribution"] = p3

    # 4. Repeat vs One-Time Customers (Count & Revenue Share)
    buyer_summary = (
        cust_df.groupby("BuyerType", as_index=False)
        .agg(Customers=("CustomerID", "count"), Revenue=("TotalSpend", "sum"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(
        data=buyer_summary, x="BuyerType", y="Customers", hue="BuyerType",
        ax=axes[0], palette=["#4c72b0", "#dd8452"], legend=False
    )
    axes[0].set_title("Customer Count: Repeat vs. One-Time")
    sns.barplot(
        data=buyer_summary, x="BuyerType", y="Revenue", hue="BuyerType",
        ax=axes[1], palette=["#4c72b0", "#dd8452"], legend=False
    )
    axes[1].set_title("Revenue Share (£): Repeat vs. One-Time")
    plt.tight_layout()
    p4 = out / "eda_repeat_vs_onetime.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths["repeat_vs_onetime"] = p4

    return paths
