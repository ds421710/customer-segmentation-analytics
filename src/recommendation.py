"""
Phase 6: Customer Retention & Marketing Recommendation Engine
Generates data-grounded CRM strategies and customer-level campaign
assignments derived directly from RFM, K-Means, Inactivity, and CLV metrics.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


SEGMENT_PLAYBOOKS: Dict[str, Dict[str, str]] = {
    "Champions": {
        "Priority": "P1 - Protect & Expand",
        "PrimaryGoal": "Retention & Wallet-Share Expansion",
        "RecommendedAction": "VIP tier with early access to seasonal launches, priority fulfillment, and referral incentives (avoid margin-eroding discounts).",
        "Channel": "Dedicated Account Email / VIP Portal",
        "SuccessKPI": "Repeat Rate & Annualized CLV Retention",
    },
    "Loyal Customers": {
        "Priority": "P1 - Nurture to Champion",
        "PrimaryGoal": "Increase Basket Size (AOV) & Referral Advocacy",
        "RecommendedAction": "Cross-sell complementary top-performing SKUs via tiered bundle pricing and launch member-get-member referral rewards.",
        "Channel": "Personalized Email + In-Cart Bundles",
        "SuccessKPI": "AOV Lift & Champion Upgrade Rate",
    },
    "Potential Loyalists": {
        "Priority": "P2 - Build Habit",
        "PrimaryGoal": "Convert Recent Buyers into Repeat Habit",
        "RecommendedAction": "Send 2nd/3rd-order milestone incentives within 21 days of last purchase featuring personalized category recommendations.",
        "Channel": "Lifecycle Email Sequence",
        "SuccessKPI": "2nd-to-3rd Order Conversion Rate",
    },
    "New Customers": {
        "Priority": "P2 - Onboard",
        "PrimaryGoal": "Drive Second Purchase Within 30 Days",
        "RecommendedAction": "Automated post-purchase onboarding series with usage inspiration, top-rated gift pairings, and a time-bound welcome-back perk.",
        "Channel": "Automated Welcome Email Flow",
        "SuccessKPI": "30-Day Second-Purchase Conversion",
    },
    "At-Risk Customers": {
        "Priority": "P1 - Urgent Intervention",
        "PrimaryGoal": "Prevent High-Value Churn",
        "RecommendedAction": "Trigger personalized 'We Miss You' outreach referencing past favorite SKUs, paired with a limited-time free-shipping or tiered voucher.",
        "Channel": "Triggered Re-Engagement Email / Direct Outreach",
        "SuccessKPI": "Reactivation Rate within 30 Days",
    },
    "Customers Needing Attention": {
        "Priority": "P2 - Re-Activate",
        "PrimaryGoal": "Restore Purchase Cadence",
        "RecommendedAction": "Deploy seasonal reminder campaigns aligned to Q4/occasion peaks with curated best-seller bundles.",
        "Channel": "Segmented Promotional Email",
        "SuccessKPI": "Click-Through & Repeat Order Rate",
    },
    "Hibernating Customers": {
        "Priority": "P3 - Cost-Capped Win-Back",
        "PrimaryGoal": "Low-Cost Reactivation",
        "RecommendedAction": "Run automated win-back campaign with a strict offer cap (<=10% of Predictive CLV) and preference survey; suppress high-cost ad spend if unresponsive.",
        "Channel": "Low-Cost Automated Email",
        "SuccessKPI": "Win-Back ROI & Unsubscribe Hygiene",
    },
    "Lost Customers": {
        "Priority": "P4 - Suppress / Sunset",
        "PrimaryGoal": "Protect Marketing ROI",
        "RecommendedAction": "Exclude from paid retargeting and high-frequency promos; send a single low-cost annual clearance/catalog reactivation notice.",
        "Channel": "Low-Frequency Sunset List",
        "SuccessKPI": "Marketing Cost Savings & Deliverability",
    },
}


def generate_segment_recommendations(clv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate actual segment metrics (count, revenue share, recency, frequency,
    Historical/Predictive CLV, inactivity risk) and attach data-grounded actions.
    """
    total_customers = max(len(clv_df), 1)
    total_revenue = max(float(clv_df["Historical_CLV"].sum()), 1e-9)

    has_risk = "BecomingInactive" in clv_df.columns
    agg_dict = {
        "Customers": ("CustomerID", "count"),
        "AvgRecencyDays": ("Recency", "mean"),
        "AvgFrequency": ("Frequency", "mean"),
        "TotalHistoricalRevenue": ("Historical_CLV", "sum"),
        "AvgPredictiveCLV": ("Predictive_CLV", "mean"),
    }
    if has_risk:
        agg_dict["AtRiskFlagged"] = ("BecomingInactive", "sum")

    summary = clv_df.groupby("CRM_Segment", as_index=False).agg(**agg_dict)
    summary["CustomerSharePct"] = (summary["Customers"] / total_customers * 100).round(2)
    summary["RevenueSharePct"] = (summary["TotalHistoricalRevenue"] / total_revenue * 100).round(2)

    for col in ["AvgRecencyDays", "AvgFrequency", "TotalHistoricalRevenue", "AvgPredictiveCLV"]:
        summary[col] = summary[col].round(2)

    # Attach playbook metadata & data-grounded rationale
    rows = []
    for _, row in summary.iterrows():
        seg = row["CRM_Segment"]
        pb = SEGMENT_PLAYBOOKS.get(seg, SEGMENT_PLAYBOOKS["Customers Needing Attention"])
        risk_count = int(row["AtRiskFlagged"]) if has_risk else 0
        rationale = (
            f"{int(row['Customers'])} customers ({row['CustomerSharePct']}% of base) driving "
            f"{row['RevenueSharePct']}% of revenue (£{row['TotalHistoricalRevenue']:,.2f}); "
            f"avg recency {row['AvgRecencyDays']}d, avg {row['AvgFrequency']} orders, "
            f"avg Predictive CLV £{row['AvgPredictiveCLV']:,.2f}"
            + (f", {risk_count} cadence-inactive." if has_risk else ".")
        )
        rows.append({
            "CRM_Segment": seg,
            "Priority": pb["Priority"],
            "Customers": int(row["Customers"]),
            "RevenueSharePct": row["RevenueSharePct"],
            "AvgPredictiveCLV": row["AvgPredictiveCLV"],
            "DataRationale": rationale,
            "RecommendedAction": pb["RecommendedAction"],
            "Channel": pb["Channel"],
            "SuccessKPI": pb["SuccessKPI"],
        })

    rec_df = pd.DataFrame(rows).sort_values(
        ["RevenueSharePct", "AvgPredictiveCLV"], ascending=[False, False]
    ).reset_index(drop=True)
    return rec_df


def assign_customer_actions(clv_df: pd.DataFrame) -> pd.DataFrame:
    """Attach segment-level priority and next-best-action to every customer row."""
    out = clv_df.copy()
    out["Priority"] = out["CRM_Segment"].map(
        lambda s: SEGMENT_PLAYBOOKS.get(s, SEGMENT_PLAYBOOKS["Customers Needing Attention"])["Priority"]
    )
    out["RecommendedAction"] = out["CRM_Segment"].map(
        lambda s: SEGMENT_PLAYBOOKS.get(s, SEGMENT_PLAYBOOKS["Customers Needing Attention"])["RecommendedAction"]
    )
    return out
