"""
Phase 5: Buying Behaviour & Customer Lifetime Value (CLV) Analysis
Combines segment-level purchasing behaviour diagnostics, inactivity-risk
detection, and Historical vs. Predictive CLV estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def analyze_segment_buying_behaviour(
    tx_df: pd.DataFrame, rfm_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare spend, order frequency, basket size, and top products across
    derived K-Means customer segments.
    """
    merged = tx_df.merge(
        rfm_df[["CustomerID", "ClusterLabel", "CRM_Segment"]],
        on="CustomerID",
        how="inner",
    )

    segment_stats = (
        merged.groupby("ClusterLabel", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique"),
            TotalRevenue=("TotalAmount", "sum"),
            TotalUnits=("Quantity", "sum"),
        )
    )
    segment_stats["AvgSpendPerCustomer"] = (
        segment_stats["TotalRevenue"] / segment_stats["Customers"]
    ).round(2)
    segment_stats["AvgOrderValue"] = (
        segment_stats["TotalRevenue"] / segment_stats["Orders"]
    ).round(2)
    segment_stats["AvgOrdersPerCustomer"] = (
        segment_stats["Orders"] / segment_stats["Customers"]
    ).round(2)
    segment_stats["TotalRevenue"] = segment_stats["TotalRevenue"].round(2)
    segment_stats = segment_stats.sort_values("TotalRevenue", ascending=False)

    # Top product per segment (excluding service fees if flagged)
    merch = merged[~merged["IsServiceFee"]] if "IsServiceFee" in merged.columns else merged
    top_sku_by_seg = (
        merch.groupby(["ClusterLabel", "Description"], as_index=False)["TotalAmount"]
        .sum()
        .sort_values(["ClusterLabel", "TotalAmount"], ascending=[True, False])
        .groupby("ClusterLabel", as_index=False)
        .first()
        .rename(columns={"Description": "TopProduct", "TotalAmount": "TopProductRevenue"})
    )
    top_sku_by_seg["TopProductRevenue"] = top_sku_by_seg["TopProductRevenue"].round(2)

    return segment_stats, top_sku_by_seg


def identify_inactivity_risk(
    tx_df: pd.DataFrame, rfm_df: pd.DataFrame, inactivity_threshold_days: int = 90
) -> pd.DataFrame:
    """
    Flag previously repeat customers who have exceeded both their normal
    inter-purchase cadence and the inactivity threshold.
    """
    # Compute average inter-purchase interval per repeat customer
    order_dates = (
        tx_df.groupby(["CustomerID", "InvoiceNo"], as_index=False)["InvoiceDate"]
        .min()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    order_dates["DaysBetween"] = order_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    cadence = (
        order_dates.groupby("CustomerID", as_index=False)["DaysBetween"]
        .mean()
        .rename(columns={"DaysBetween": "AvgInterPurchaseDays"})
    )

    risk_df = rfm_df.merge(cadence, on="CustomerID", how="left")
    risk_df["AvgInterPurchaseDays"] = risk_df["AvgInterPurchaseDays"].fillna(0).round(1)
    risk_df["BecomingInactive"] = (
        (risk_df["Frequency"] >= 2)
        & (
            (risk_df["Recency"] > inactivity_threshold_days)
            | (
                (risk_df["AvgInterPurchaseDays"] > 0)
                & (risk_df["Recency"] > 2.0 * risk_df["AvgInterPurchaseDays"])
            )
        )
    )
    return risk_df


def compute_clv_metrics(
    rfm_df: pd.DataFrame, horizon_months: int = 12, decay_days: float = 120.0
) -> pd.DataFrame:
    """
    Compute Historical CLV vs. Predictive CLV (gross-revenue basis).

    - Historical_CLV: Actual cumulative gross revenue generated to date.
    - Predictive_CLV: Forward-looking 12-month expected gross revenue:
        AOV * ExpectedAnnualOrders * RetentionProbability
      where RetentionProbability = exp(-Recency / decay_days).
    """
    clv = rfm_df.copy()
    clv["Historical_CLV"] = clv["Monetary"].round(2)
    clv["AOV"] = (clv["Monetary"] / clv["Frequency"].clip(lower=1)).round(2)

    # Active observation window in months (minimum 1 month to avoid division explosion)
    active_months = np.maximum(clv["TenureDays"] / 30.0, 1.0)
    monthly_order_rate = clv["Frequency"] / active_months
    expected_horizon_orders = monthly_order_rate * horizon_months

    # Smooth recency-based retention probability
    clv["RetentionProb"] = np.exp(-clv["Recency"] / decay_days).round(3)
    clv["Predictive_CLV"] = np.round(
        clv["AOV"] * expected_horizon_orders * clv["RetentionProb"], 2
    )
    clv["CLV_Tier"] = pd.qcut(
        clv["Predictive_CLV"].rank(method="first"),
        q=3,
        labels=["Low Potential", "Medium Potential", "High Potential"],
    )
    return clv


def compare_clv_by_segment(clv_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Historical vs. Predictive CLV across K-Means segments."""
    summary = (
        clv_df.groupby("ClusterLabel", as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            AvgHistoricalCLV=("Historical_CLV", "mean"),
            AvgPredictiveCLV=("Predictive_CLV", "mean"),
            TotalHistoricalCLV=("Historical_CLV", "sum"),
            TotalPredictiveCLV=("Predictive_CLV", "sum"),
            AvgRetentionProb=("RetentionProb", "mean"),
        )
        .sort_values("AvgPredictiveCLV", ascending=False)
    )
    for col in [
        "AvgHistoricalCLV",
        "AvgPredictiveCLV",
        "TotalHistoricalCLV",
        "TotalPredictiveCLV",
        "AvgRetentionProb",
    ]:
        summary[col] = summary[col].round(2)
    return summary


def generate_behaviour_clv_charts(
    clv_df: pd.DataFrame, clv_seg_df: pd.DataFrame, output_dir: str | Path
) -> Dict[str, Path]:
    """Generate Segment CLV Comparison and Historical vs. Predictive CLV charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Historical vs. Predictive Average CLV by Segment
    melted = clv_seg_df.melt(
        id_vars="ClusterLabel",
        value_vars=["AvgHistoricalCLV", "AvgPredictiveCLV"],
        var_name="CLV_Type",
        value_name="Amount_GBP",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=melted, x="ClusterLabel", y="Amount_GBP", hue="CLV_Type", ax=ax, palette=["#4c72b0", "#55a868"])
    ax.set_title("Historical vs. Predictive 12-Month CLV by Segment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Average CLV (£)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p1 = out / "clv_segment_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["clv_segment_comparison"] = p1

    # 2. Retention Probability vs. Predictive CLV
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(
        data=clv_df,
        x="RetentionProb",
        y="Predictive_CLV",
        hue="ClusterLabel",
        size="Historical_CLV",
        sizes=(40, 220),
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Retention Probability vs. Predictive 12-Month CLV", fontsize=13, fontweight="bold")
    ax.set_xlabel("Estimated Retention Probability")
    ax.set_ylabel("Predictive CLV (£)")
    plt.tight_layout()
    p2 = out / "clv_retention_vs_predictive.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["clv_retention_vs_predictive"] = p2

    return paths
