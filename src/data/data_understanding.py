"""Profile the Online Retail II dataset without changing the source data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


@dataclass(frozen=True)
class DataUnderstandingReport:
    """Results from inspecting a retail transactions dataframe."""

    first_ten_rows: pd.DataFrame
    dimensions: tuple[int, int]
    columns_and_dtypes: dict[str, str]
    missing_values: pd.Series
    duplicate_records: int
    unique_counts: dict[str, int]
    negative_quantity_records: int
    negative_price_records: int
    cancelled_invoice_records: int
    cancelled_invoice_numbers: tuple[str, ...]
    earliest_transaction: Any
    latest_transaction: Any


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file and return its records unchanged.

    The returned dataframe is a new object created by pandas. This function
    never writes to, deletes, or modifies the input file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported types: {supported}"
        )

    try:
        if path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(path)
        else:
            dataframe = pd.read_excel(path)
    except (OSError, ValueError, ImportError) as error:
        raise RuntimeError(f"Could not read dataset '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return dataframe


def analyze_dataset(dataframe: pd.DataFrame) -> DataUnderstandingReport:
    """Calculate data understanding findings without changing ``dataframe``."""
    required_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if required_columns:
        missing = ", ".join(sorted(required_columns))
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    invoice_text = dataframe["Invoice"].astype("string")
    cancelled_invoice_mask = invoice_text.str.upper().str.startswith("C", na=False)
    invoice_dates = pd.to_datetime(dataframe["InvoiceDate"], errors="coerce")

    return DataUnderstandingReport(
        first_ten_rows=dataframe.head(10).copy(),
        dimensions=dataframe.shape,
        columns_and_dtypes={
            column: str(dtype) for column, dtype in dataframe.dtypes.items()
        },
        missing_values=dataframe.isna().sum(),
        duplicate_records=int(dataframe.duplicated().sum()),
        unique_counts={
            "invoices": int(dataframe["Invoice"].nunique(dropna=True)),
            "products": int(dataframe["StockCode"].nunique(dropna=True)),
            "customers": int(dataframe["Customer ID"].nunique(dropna=True)),
        },
        negative_quantity_records=int((dataframe["Quantity"] < 0).sum()),
        negative_price_records=int((dataframe["Price"] < 0).sum()),
        cancelled_invoice_records=int(cancelled_invoice_mask.sum()),
        cancelled_invoice_numbers=tuple(
            invoice_text.loc[cancelled_invoice_mask].dropna().unique().tolist()
        ),
        earliest_transaction=invoice_dates.min(),
        latest_transaction=invoice_dates.max(),
    )


def print_report(report: DataUnderstandingReport) -> None:
    """Print the requested findings in a readable format."""
    print("\nFirst 10 rows:")
    print(report.first_ten_rows.to_string(index=False))

    print(f"\nDataset dimensions: {report.dimensions[0]} rows x {report.dimensions[1]} columns")

    print("\nColumn names and data types:")
    for column, dtype in report.columns_and_dtypes.items():
        print(f"- {column}: {dtype}")

    print("\nMissing values by column:")
    print(report.missing_values.to_string())

    print(f"\nDuplicate records: {report.duplicate_records}")

    print("\nUnique values:")
    for name, count in report.unique_counts.items():
        print(f"- {name}: {count}")

    print(f"\nRecords with negative quantities: {report.negative_quantity_records}")
    print(f"Records with negative prices: {report.negative_price_records}")
    print(f"\nCancelled invoice records: {report.cancelled_invoice_records}")
    print(
        "Cancelled invoice numbers: "
        f"{len(report.cancelled_invoice_numbers)} unique"
    )
    if report.cancelled_invoice_numbers:
        print(", ".join(report.cancelled_invoice_numbers))

    print(f"\nEarliest transaction date: {report.earliest_transaction}")
    print(f"Latest transaction date: {report.latest_transaction}")


def main() -> None:
    """Run the data understanding report from the command line."""
    parser = argparse.ArgumentParser(
        description="Inspect a Smart Retail transaction dataset."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/raw/dataset.xlsx"),
        help="Path to a CSV or Excel dataset.",
    )
    args = parser.parse_args()

    try:
        dataframe = load_dataset(args.path)
        report = analyze_dataset(dataframe)
        print_report(report)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
