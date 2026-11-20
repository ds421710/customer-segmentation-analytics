"""
Phase 8: End-to-End Automated Test Suite
Validates preprocessing, EDA, RFM + K-Means segmentation, CLV,
inactivity detection, and CRM recommendation generation.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.data_preprocessing import clean_transactions
from src.eda_analysis import compute_eda_kpis, get_monthly_trends, get_top_products
from src.rfm_segmentation import compute_rfm_table, evaluate_kmeans_k, run_kmeans_segmentation
from src.clv_behaviour import analyze_segment_buying_behaviour, identify_inactivity_risk, compute_clv_metrics
from src.recommendation import generate_segment_recommendations, assign_customer_actions


def _build_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    skus = [("22423", "REGENCY CAKESTAND 3 TIER"), ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER")]
    rows = []
    inv = 500000
    for c_idx in range(1, 25):
        cid = 10000 + c_idx
        n_orders = 8 if c_idx <= 6 else (4 if c_idx <= 14 else 1)
        days_max = 15 if c_idx <= 6 else (90 if c_idx <= 14 else 240)
        spend = 150.0 if c_idx <= 6 else (75.0 if c_idx <= 14 else 30.0)
        for _ in range(n_orders):
            inv += 1
            d = pd.Timestamp("2011-12-09") - pd.Timedelta(days=int(rng.integers(1, days_max)))
            code, desc = skus[c_idx % 2]
            rows.append({
                "InvoiceNo": str(inv),
                "StockCode": code,
                "Description": desc,
                "Quantity": 4,
                "InvoiceDate": d,
                "UnitPrice": round(spend / 4.0, 2),
                "CustomerID": cid,
                "Country": "United Kingdom",
            })
    # Add edge cases: cancellation & missing CustomerID
    rows.append({
        "InvoiceNo": "C500999", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER",
        "Quantity": -2, "InvoiceDate": pd.Timestamp("2011-11-01"), "UnitPrice": 12.75,
        "CustomerID": 10001, "Country": "United Kingdom",
    })
    rows.append({
        "InvoiceNo": "500998", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER",
        "Quantity": 2, "InvoiceDate": pd.Timestamp("2011-11-01"), "UnitPrice": 12.75,
        "CustomerID": None, "Country": "United Kingdom",
    })
    return pd.DataFrame(rows)


class TestCustomerAnalyticsPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        raw_df = _build_fixture()
        valid_df, cancelled_df, audit = clean_transactions(raw_df)
        self.assertEqual(audit["cancellations"], 1)
        self.assertEqual(audit["missing_customer_id"], 1)
        self.assertEqual(len(cancelled_df), 1)

        kpis = compute_eda_kpis(valid_df)
        self.assertEqual(kpis["total_customers"], 24)
        self.assertGreater(kpis["total_revenue"], 0)
        self.assertFalse(get_monthly_trends(valid_df).empty)
        self.assertFalse(get_top_products(valid_df).empty)

        rfm = compute_rfm_table(valid_df)
        k_eval = evaluate_kmeans_k(rfm, k_range=range(2, 6))
        self.assertEqual(len(k_eval), 4)
        rfm_clustered, profile = run_kmeans_segmentation(rfm, n_clusters=4)
        self.assertEqual(len(profile), 4)
        self.assertAlmostEqual(profile["RevenueSharePct"].sum(), 100.0, delta=0.2)

        seg_stats, _ = analyze_segment_buying_behaviour(valid_df, rfm_clustered)
        self.assertEqual(len(seg_stats), 4)
        risk_df = identify_inactivity_risk(valid_df, rfm_clustered)
        clv_df = compute_clv_metrics(risk_df)
        self.assertTrue((clv_df["Predictive_CLV"] >= 0).all())

        rec_table = generate_segment_recommendations(clv_df)
        cust_actions = assign_customer_actions(clv_df)
        self.assertFalse(rec_table.empty)
        self.assertIn("RecommendedAction", cust_actions.columns)


if __name__ == "__main__":
    unittest.main()
