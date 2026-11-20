-- ============================================================================
-- Phase 3: Business Intelligence & EDA SQL Queries
-- ============================================================================

-- 1. Executive KPI Summary
SELECT
    COUNT(DISTINCT customer_id) AS total_customers,
    COUNT(DISTINCT invoice_no) AS total_orders,
    ROUND(SUM(total_amount), 2) AS total_revenue,
    ROUND(SUM(total_amount) / COUNT(DISTINCT invoice_no), 2) AS avg_order_value
FROM transactions;

-- 2. Monthly Revenue & Active Customer Trend
SELECT
    DATE_FORMAT(invoice_date, '%Y-%m') AS year_month,
    ROUND(SUM(total_amount), 2) AS monthly_revenue,
    COUNT(DISTINCT invoice_no) AS total_orders,
    COUNT(DISTINCT customer_id) AS active_customers
FROM transactions
GROUP BY DATE_FORMAT(invoice_date, '%Y-%m')
ORDER BY year_month;

-- 3. Top 10 Products by Revenue
SELECT
    stock_code,
    description,
    SUM(quantity) AS total_units_sold,
    ROUND(SUM(total_amount), 2) AS total_revenue
FROM transactions
WHERE stock_code NOT IN ('POST', 'DOT', 'M', 'BANK CHARGES', 'AMAZONFEE', 'CRUK')
GROUP BY stock_code, description
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Repeat vs. One-Time Customer Revenue Contribution
WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS order_count,
        SUM(total_amount) AS total_spend
    FROM transactions
    GROUP BY customer_id
)
SELECT
    CASE WHEN order_count > 1 THEN 'Repeat Buyer' ELSE 'One-Time Buyer' END AS buyer_type,
    COUNT(*) AS customer_count,
    ROUND(SUM(total_spend), 2) AS total_revenue,
    ROUND(100.0 * SUM(total_spend) / SUM(SUM(total_spend)) OVER (), 2) AS revenue_share_pct
FROM customer_orders
GROUP BY CASE WHEN order_count > 1 THEN 'Repeat Buyer' ELSE 'One-Time Buyer' END;
