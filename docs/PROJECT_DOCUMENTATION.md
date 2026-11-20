# Customer Segmentation & Buying Behaviour Analysis: Project Documentation & Business Case

## Executive Business Case
**Business Problem:** E-commerce and wholesale retailers frequently deploy uniform promotional campaigns across their entire customer base. This erodes gross margins on loyal buyers who would purchase at full price, leaves high-value repeat accounts vulnerable to silent churn, and wastes marketing spend on one-time low-value buyers.  
**Proposed Solution:** A modular Python, SQL, Machine Learning, and Streamlit Business Intelligence system that transforms raw transaction logs into RFM behavioral segments, K-Means clusters, Historical vs. Predictive Customer Lifetime Value (CLV) estimates, cadence-based inactivity alerts, and prioritized CRM playbooks.  
**Expected Business Value:** Protects core Pareto revenue (concentrated in Champion and Loyal segments), catches repeat-buyer churn early via inter-purchase cadence monitoring, and improves marketing ROI by capping win-back spend on low-CLV accounts.  
**Managerial Implications:** Shifts marketing from intuition-led mass discounting to segment-level budget allocation, requiring cross-functional alignment between CRM, merchandising (Q4 hero SKU stocking), and checkout data capture.

---

## 1. Abstract
This project integrates MBA marketing and CRM frameworks with Python data engineering and unsupervised machine learning to solve customer retention and profitability challenges in online retail. Using the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) schema (541,909 transactions, Dec 2010–Dec 2011), the system cleans noisy transaction logs, computes Recency, Frequency, and Monetary (RFM) metrics, applies log-scaled K-Means clustering, distinguishes Historical CLV from Predictive 12-Month CLV, and surfaces actionable CRM recommendations in an interactive 7-tab Streamlit dashboard.

## 2. Introduction
Modern retail generates vast transactional data, yet many mid-sized businesses lack the analytical pipeline to translate invoice logs into customer strategy. Combining Information Science & Engineering (ISE) skills with MBA marketing concepts bridges this gap—turning raw `InvoiceNo` and `CustomerID` records into executive decisions.

## 3. Problem Statement
Businesses struggle to identify which customers drive long-term profitability, detect when repeat buyers begin to lapse, and tailor retention offers without sacrificing margin.

## 4. Project Objectives
1. Build a reproducible ETL and SQL storage pipeline for retail transaction data, including custom CSV uploads.
2. Quantify baseline business KPIs, seasonal trends, product concentration, and repeat vs. one-time buyer dynamics.
3. Segment customers using RFM quintiles and log-scaled K-Means clustering with programmatic label derivation.
4. Model Historical vs. Predictive CLV and cadence-based inactivity risk.
5. Deliver a 7-tab Streamlit BI dashboard with downloadable CRM action lists.

## 5. Scope of the Project
**In Scope:** Batch transaction ingestion (`.csv`/`.xlsx`), SQL schema and business queries, EDA, RFM + K-Means segmentation, gross-revenue CLV estimation, rule-based CRM recommendations, and interactive BI reporting.  
**Out of Scope:** Real-time event streaming, unit-cost/net-margin accounting (cost data is absent), and demographic profiling (not present in transactional logs).

## 6. Literature Review / Background Study
RFM modeling (Hughes, 1994) and data-mining extensions on the [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (Chen et al., 2012, *Journal of Database Marketing and Customer Strategy Management*) establish that behavioral recency, frequency, and monetary value outperform static demographics in predicting repeat purchase probability. Combining RFM with K-Means clustering and recency-weighted CLV provides a transparent, interpretable alternative to black-box deep learning for managerial decision-making.

## 7. MBA Concepts Applied
1. **Marketing Management:** Aligning promotional spend and retention offers to measurable segment profitability.
2. **Consumer Behaviour:** Analyzing basket size (AOV), order cadence, and Q4 seasonal gift-buying spikes.
3. **Market Segmentation:** Behavioral segmentation via RFM quintiles and K-Means centroids.
4. **Customer Relationship Management (CRM):** Lifecycle-stage playbooks (`Champions` through `Lost`).
5. **Marketing Research:** Hypothesis-driven EDA and data-quality auditing before modeling.
6. **Customer Retention Strategies:** Cadence-based early-warning flags (`Recency > 2 × AvgInterPurchaseDays`).
7. **Customer Lifetime Value (CLV):** Separating sunk Historical CLV from forward-looking Predictive 12-Month CLV.
8. **Marketing Strategy:** Avoiding margin-eroding discounts on high-loyalty tiers; capping win-back spend on low-CLV tiers.
9. **Data-Driven Decision Making:** Structuring every module around *Data → Business Meaning → Management Action*.
10. **Customer Acquisition & Retention:** Quantifying the revenue gap between one-time buyers and repeat cohorts.

## 8. Technical Requirements
Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SQL / MySQL (with SQLite compatibility), Streamlit, OpenPyXL, Pytest, Git/GitHub.

## 9. System Architecture
A 4-tier modular architecture:
1. **Storage Layer (`sql/schema.sql`, `sql/business_queries.sql`):** Relational tables (`customers`, `products`, `transactions`, `cancelled_transactions`).
2. **Processing Layer (`src/data_preprocessing.py`):** Quality audit, deduplication, cancellation separation, and feature engineering.
3. **Analytics & ML Layer (`src/eda_analysis.py`, `src/rfm_segmentation.py`, `src/clv_behaviour.py`, `src/recommendation.py`):** KPIs, K-Means clustering, CLV, and CRM playbooks.
4. **Presentation Layer (`app/streamlit_app.py`):** 7-tab interactive BI dashboard with CSV exports.

## 10. Dataset Description
The **[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)** contains 541,909 rows and 8 attributes (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`) spanning December 1, 2010 to December 9, 2011 for a UK-based non-store online gift retailer with many wholesale customers.  
**Limitations & Assumptions:** Guest checkouts lack `CustomerID` and are excluded from customer-level RFM/CLV; `'C'`-prefixed invoices represent cancellations and are separated into `cancelled_transactions`; lack of COGS/margin data means CLV is modeled on gross revenue.

## 11. Methodology
Iterative 8-phase engineering and business-analysis workflow: Planning → Preprocessing → EDA → RFM & K-Means → Behaviour & CLV → CRM Recommendation Engine → Streamlit BI Dashboard → Testing & Documentation.

## 12. Data Preprocessing
`src/data_preprocessing.py` audits raw data, removes exact duplicates, separates `'C'`-prefixed cancellations, drops missing `CustomerID` and non-positive `Quantity`/`UnitPrice` rows, derives `TotalAmount`, `YearMonth`, `DayOfWeek`, `Hour`, and flags operational service codes (`POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`).

## 13. Exploratory Data Analysis
`src/eda_analysis.py` and `sql/business_queries.sql` compute total revenue, unique customers, orders, AOV, purchase frequency, monthly seasonality, top merchandise SKUs, spending distributions (capped at the 95th percentile for readability), and repeat vs. one-time buyer revenue share.

## 14. Machine Learning Implementation
`src/rfm_segmentation.py` applies `log1p` transformation (to tame wholesale right-skew) and z-score standardization (`StandardScaler`), evaluates `k = 2..7` via Elbow Inertia and Silhouette Score, and fits K-Means++ clustering.

## 15. Customer Segmentation Results
Cluster labels are **never** assigned randomly; they are programmatically derived from each cluster's standardized Value Index (`z_Frequency + z_Monetary - z_Recency`) and mapped alongside deterministic RFM quintile segments (`Champions`, `Loyal Customers`, `Potential Loyalists`, `New Customers`, `At-Risk Customers`, `Customers Needing Attention`, `Hibernating Customers`, `Lost Customers`).

## 16. Business Findings
Across verified pipeline runs, revenue exhibits strong Pareto concentration: top-tier `Champions` generate the majority of revenue with high order frequency and recent activity, while `Hibernating / Lost` buyers exhibit long recency gaps and low AOV.

## 17. Marketing Recommendations
`src/recommendation.py` pairs live segment metrics with prioritized CRM actions: P1 VIP perks (no margin-cutting discounts) for `Champions`, cross-sell bundles for `Loyal Customers`, urgent SKU-personalized outreach for `At-Risk Customers`, cost-capped win-back offers for `Hibernating`, and paid-ad suppression for `Lost Customers`.

## 18. Results and Discussion
Combining unsupervised K-Means clusters with rule-based RFM quintiles and Predictive CLV bridges data science rigor with day-to-day marketing usability.

## 19. Limitations
1. Gross-revenue CLV (no unit-cost/COGS data).
2. Single-year observation window (Dec 2010–Dec 2011).
3. Guest checkouts without `CustomerID` cannot be tracked longitudinally.

## 20. Future Scope
Integrate BG/NBD + Gamma-Gamma probabilistic CLV models, market-basket association rules (Apriori/FP-Growth) for automated bundle recommendations, and live MySQL/API connectors.

## 21. Conclusion
This system demonstrates how MBA strategic frameworks and Python/ML engineering unite to turn raw transactional data into measurable retention and profitability decisions.

## 22. References
1. Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/352/online+retail
2. Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing and Customer Strategy Management*, 19(3), 197–208.
3. Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.
