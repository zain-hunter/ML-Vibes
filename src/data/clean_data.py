"""Clean and prepare Online Retail II transactions without changing raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.data_understanding import REQUIRED_COLUMNS, load_dataset


DEFAULT_INPUT_PATH = Path("data/raw/dataset.xlsx")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise an error when the expected Online Retail columns are missing."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")


def standardize_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent types for cleaning and calculations."""
    cleaned = dataframe.copy()
    cleaned["Invoice"] = cleaned["Invoice"].astype("string")
    cleaned["StockCode"] = cleaned["StockCode"].astype("string")
    cleaned["Description"] = cleaned["Description"].astype("string")
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")
    cleaned["Customer ID"] = pd.to_numeric(cleaned["Customer ID"], errors="coerce")
    cleaned["Country"] = cleaned["Country"].astype("string")
    return cleaned


def add_quality_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add transparent quality and transaction-type flags."""
    flagged = dataframe.copy()
    invoice_text = flagged["Invoice"].str.upper()
    flagged["CustomerIDMissing"] = flagged["Customer ID"].isna()
    flagged["DescriptionMissing"] = flagged["Description"].isna()
    flagged["IsExactDuplicate"] = flagged.duplicated()
    flagged["IsReturn"] = flagged["Quantity"].lt(0)
    flagged["IsCancelledInvoice"] = invoice_text.str.startswith("C", na=False)
    flagged["IsNegativePrice"] = flagged["Price"].lt(0)
    flagged["IsZeroPrice"] = flagged["Price"].eq(0)
    return flagged


def calculate_revenue(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add signed and absolute revenue while preserving quantity and price."""
    calculated = dataframe.copy()
    calculated["Revenue"] = calculated["Quantity"] * calculated["Price"]
    calculated["AbsoluteRevenue"] = calculated["Revenue"].abs()
    return calculated


def prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize types, add audit flags, and calculate revenue."""
    validate_required_columns(dataframe)
    cleaned = standardize_data_types(dataframe)
    cleaned = add_quality_flags(cleaned)
    return calculate_revenue(cleaned)


def split_transactions(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the master, normal-sales, and returns/cancellations datasets."""
    sales_mask = (
        ~dataframe["IsCancelledInvoice"]
        & dataframe["Quantity"].gt(0)
        & dataframe["Price"].ge(0)
    )
    exceptions_mask = dataframe["IsReturn"] | dataframe["IsCancelledInvoice"]
    sales = dataframe.loc[sales_mask].copy()
    returns_and_cancellations = dataframe.loc[exceptions_mask].copy()
    return dataframe.copy(), sales, returns_and_cancellations


def validate_prepared_data(
    original: pd.DataFrame,
    prepared: pd.DataFrame,
    sales: pd.DataFrame,
    returns_and_cancellations: pd.DataFrame,
) -> None:
    """Validate row preservation, derived fields, and output relationships."""
    if len(original) != len(prepared):
        raise ValueError("Cleaning changed the master dataset row count.")
    if prepared["InvoiceDate"].isna().any():
        raise ValueError("Cleaning produced invalid or missing InvoiceDate values.")

    expected_revenue = prepared["Quantity"] * prepared["Price"]
    if not prepared["Revenue"].equals(expected_revenue):
        raise ValueError("Revenue does not equal Quantity multiplied by Price.")
    if not sales.index.isin(prepared.index).all():
        raise ValueError("Sales output contains rows outside the master dataset.")
    if not returns_and_cancellations.index.isin(prepared.index).all():
        raise ValueError(
            "Returns/cancellations output contains rows outside the master dataset."
        )
    if not prepared["IsReturn"].equals(prepared["Quantity"].lt(0)):
        raise ValueError("IsReturn flag is inconsistent with Quantity.")
    if not prepared["IsCancelledInvoice"].equals(
        prepared["Invoice"].str.upper().str.startswith("C", na=False)
    ):
        raise ValueError("IsCancelledInvoice flag is inconsistent with Invoice.")


def build_summary(
    original: pd.DataFrame,
    prepared: pd.DataFrame,
    sales: pd.DataFrame,
    returns_and_cancellations: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact audit summary for the cleaning run."""
    metrics = {
        "input_rows": len(original),
        "master_rows": len(prepared),
        "sales_rows": len(sales),
        "returns_cancellations_rows": len(returns_and_cancellations),
        "exact_duplicate_rows": int(prepared["IsExactDuplicate"].sum()),
        "missing_customer_id_rows": int(prepared["CustomerIDMissing"].sum()),
        "missing_description_rows": int(prepared["DescriptionMissing"].sum()),
        "negative_quantity_rows": int(prepared["IsReturn"].sum()),
        "cancelled_invoice_rows": int(prepared["IsCancelledInvoice"].sum()),
        "negative_price_rows": int(prepared["IsNegativePrice"].sum()),
        "zero_price_rows": int(prepared["IsZeroPrice"].sum()),
        "input_revenue": float((original["Quantity"] * original["Price"]).sum()),
        "master_revenue": float(prepared["Revenue"].sum()),
        "sales_revenue": float(sales["Revenue"].sum()),
        "returns_cancellations_revenue": float(
            returns_and_cancellations["Revenue"].sum()
        ),
    }
    return pd.DataFrame([metrics])


def save_outputs(
    output_dir: str | Path,
    prepared: pd.DataFrame,
    sales: pd.DataFrame,
    returns_and_cancellations: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write derived datasets only to the processed-data directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(destination / "transactions_clean.csv", index=False)
    sales.to_csv(destination / "sales_transactions.csv", index=False)
    returns_and_cancellations.to_csv(
        destination / "returns_cancellations.csv", index=False
    )
    summary.to_csv(destination / "data_cleaning_summary.csv", index=False)


def run_cleaning(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the cleaning workflow and return its validation summary."""
    original = load_dataset(input_path)
    prepared = prepare_dataset(original)
    master, sales, returns_and_cancellations = split_transactions(prepared)
    validate_prepared_data(
        original, master, sales, returns_and_cancellations
    )
    summary = build_summary(
        original, master, sales, returns_and_cancellations
    )
    save_outputs(output_dir, master, sales, returns_and_cancellations, summary)
    return summary


def main() -> None:
    """Run the cleaning workflow from the command line."""
    parser = argparse.ArgumentParser(
        description="Prepare Online Retail II transaction datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw CSV or Excel dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived datasets and the cleaning summary.",
    )
    args = parser.parse_args()

    try:
        summary = run_cleaning(args.input, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
