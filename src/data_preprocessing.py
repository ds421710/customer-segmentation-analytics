"""
Phase 2: Customer Data Management & Preprocessing Pipeline
Handles ingestion (CSV/Excel), data quality auditing, cleaning,
cancellation separation, feature derivation, and SQLite/MySQL export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load raw retail transaction data from CSV or Excel."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(
            path,
            encoding="ISO-8859-1",
            dtype={"InvoiceNo": str, "StockCode": str},
            low_memory=False,
        )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    logger.info("Loaded raw dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


def audit_data_quality(df: pd.DataFrame) -> Dict[str, int]:
    """Generate a pre-cleaning data quality audit report."""
    invoice_str = df["InvoiceNo"].astype(str).str.strip()
    audit = {
        "raw_rows": int(len(df)),
        "exact_duplicates": int(df.duplicated().sum()),
        "missing_customer_id": int(df["CustomerID"].isna().sum()),
        "missing_description": int(df["Description"].isna().sum()),
        "cancellations": int(invoice_str.str.startswith("C").sum()),
        "non_positive_quantity": int((df["Quantity"] <= 0).sum()),
        "non_positive_unit_price": int((df["UnitPrice"] <= 0).sum()),
    }
    return audit


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Clean raw transaction data and separate cancellations from valid sales.

    Returns:
        valid_df: Cleaned sales transactions ready for EDA, RFM, and CLV.
        cancelled_df: Cancelled/return records preserved for return-rate analysis.
        summary: Audit metrics before and after cleaning.
    """
    audit = audit_data_quality(df)
    work = df.copy()

    # 1. Remove exact duplicate line items
    work = work.drop_duplicates().reset_index(drop=True)

    # 2. Standardize strings & parse timestamps
    work["InvoiceNo"] = work["InvoiceNo"].astype(str).str.strip()
    work["StockCode"] = work["StockCode"].astype(str).str.strip()
    work["Description"] = (
        work["Description"].fillna("UNKNOWN PRODUCT").astype(str).str.strip()
    )
    work["Country"] = work["Country"].fillna("Unknown").astype(str).str.strip()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")
    work = work.dropna(subset=["InvoiceDate"])

    # 3. Separate cancellations (InvoiceNo starting with 'C')
    is_cancelled = work["InvoiceNo"].str.upper().str.startswith("C")
    cancelled_df = work[is_cancelled].copy()
    cancelled_df["TotalAmount"] = cancelled_df["Quantity"] * cancelled_df["UnitPrice"]

    # 4. Filter valid sales: non-cancelled, valid CustomerID, Quantity > 0, UnitPrice > 0
    valid_df = work[~is_cancelled].copy()
    valid_df = valid_df.dropna(subset=["CustomerID"])
    valid_df["CustomerID"] = (
        valid_df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
    valid_df = valid_df[(valid_df["Quantity"] > 0) & (valid_df["UnitPrice"] > 0)].copy()

    # 5. Derive revenue and calendar features
    valid_df["TotalAmount"] = np.round(valid_df["Quantity"] * valid_df["UnitPrice"], 2)
    valid_df["YearMonth"] = valid_df["InvoiceDate"].dt.to_period("M").astype(str)
    valid_df["DayOfWeek"] = valid_df["InvoiceDate"].dt.day_name()
    valid_df["Hour"] = valid_df["InvoiceDate"].dt.hour

    # 6. Flag non-product operational codes (e.g., POST, DOT, M, BANK CHARGES)
    non_product_codes = {"POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK"}
    valid_df["IsServiceFee"] = valid_df["StockCode"].str.upper().isin(non_product_codes)

    audit["clean_valid_rows"] = int(len(valid_df))
    audit["cancelled_rows_saved"] = int(len(cancelled_df))
    audit["unique_customers_clean"] = int(valid_df["CustomerID"].nunique())
    audit["unique_invoices_clean"] = int(valid_df["InvoiceNo"].nunique())
    audit["net_retention_pct"] = round(
        (len(valid_df) / max(audit["raw_rows"], 1)) * 100, 2
    )

    logger.info(
        "Cleaning complete: %d valid rows (%d unique customers), %d cancellations separated.",
        audit["clean_valid_rows"],
        audit["unique_customers_clean"],
        audit["cancelled_rows_saved"],
    )
    return valid_df, cancelled_df, audit


def save_processed_data(
    valid_df: pd.DataFrame, cancelled_df: pd.DataFrame, output_dir: str | Path
) -> None:
    """Persist cleaned datasets to CSV for downstream modules."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(out / "cleaned_transactions.csv", index=False)
    cancelled_df.to_csv(out / "cancelled_transactions.csv", index=False)
    logger.info("Saved processed CSVs to %s", out)
