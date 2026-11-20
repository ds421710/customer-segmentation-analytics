-- ============================================================================
-- Customer Segmentation & Buying Behaviour Analysis
-- Phase 2: Relational Database Schema (MySQL 8.0+ / SQLite compatible)
-- ============================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    country VARCHAR(100) NOT NULL,
    first_purchase_date DATETIME,
    last_purchase_date DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    stock_code VARCHAR(30) PRIMARY KEY,
    description VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(30) NOT NULL,
    description VARCHAR(255),
    quantity INT NOT NULL,
    invoice_date DATETIME NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_amount DECIMAL(12, 2) NOT NULL,
    customer_id VARCHAR(20) NOT NULL,
    country VARCHAR(100) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_customer_date (customer_id, invoice_date),
    INDEX idx_invoice (invoice_no),
    INDEX idx_date (invoice_date)
);

CREATE TABLE IF NOT EXISTS cancelled_transactions (
    cancel_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(30) NOT NULL,
    quantity INT NOT NULL,
    invoice_date DATETIME NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_amount DECIMAL(12, 2) NOT NULL,
    customer_id VARCHAR(20),
    country VARCHAR(100)
);
