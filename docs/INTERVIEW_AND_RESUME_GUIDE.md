# Resume Positioning & MBA Interview Preparation Guide

## 1. Resume-Ready Project Description
**Customer Segmentation, CLV Modeling & CRM Analytics Platform** *(Python, Scikit-learn, SQL, Streamlit)*
- Engineered an end-to-end marketing analytics system on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (541K+ transactions) using a modular Python ETL pipeline and normalized MySQL schema.
- Implemented RFM scoring and log-scaled K-Means clustering (validated via Elbow & Silhouette metrics) to segment customers into 8 actionable CRM lifecycle tiers.
- Built a gross-revenue Historical vs. Predictive 12-Month CLV model and cadence-based churn-risk detector (`Recency > 2 × AvgInterPurchaseDays`) to prioritize retention outreach.
- Deployed a 7-tab interactive Streamlit BI dashboard with dynamic country/date filters, KPI tracking, and downloadable campaign action lists.

---

## 2. MBA & Technical Interview Q&A

**Q1. Why did you use `log1p` transformation before K-Means clustering?**  
*Answer:* Retail datasets with wholesale buyers are heavily right-skewed—a small fraction of B2B buyers spend 50x more than retail shoppers. Without log-scaling, K-Means centroids get pulled toward extreme outliers, collapsing normal customers into a single undifferentiated cluster.

**Q2. How did you avoid assigning cluster labels arbitrarily?**  
*Answer:* Rather than hardcoding cluster IDs (since K-Means cluster numbers permute across runs), the pipeline computes a standardized Value Index (`z_Frequency + z_Monetary - z_Recency`) on each cluster centroid and derives labels programmatically from relative rank and recency.

**Q3. What is the difference between Historical CLV and Predictive CLV in your project, and why does it matter?**  
*Answer:* Historical CLV measures sunk revenue already earned, whereas Predictive CLV estimates forward 12-month expected spend (`AOV × ExpectedAnnualOrders × exp(-Recency / decay)`). A lapsed whale has high Historical CLV but low Predictive CLV; a recently acquired repeat buyer has modest Historical CLV but high Predictive CLV and deserves immediate nurturing.

**Q4. Why not give a 20% discount to Champions to keep them loyal?**  
*Answer:* Champions already purchase frequently at full margin. Blanket discounting erodes profitability without changing their buying intent. Instead, we retain Champions via non-discount value—VIP early access, priority fulfillment, and referral perks—and reserve targeted discounts for `At-Risk` or `Hibernating` buyers (capped below their Predictive CLV).

**Q5. How did you handle missing `CustomerID`s and cancelled invoices (`'C'` prefix)?**  
*Answer:* Guest checkouts without `CustomerID` cannot be linked across visits, so they are excluded from RFM/CLV modeling while logged in the data-quality audit. Cancelled invoices are separated into a `cancelled_transactions` table rather than deleted—preventing inflated order counts while preserving return-rate visibility.
