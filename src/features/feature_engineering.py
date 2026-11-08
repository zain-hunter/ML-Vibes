"""Build the first daily product-demand dataset for forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/processed/sales_transactions.csv")
DEFAULT_OUTPUT_PATH = Path("data/features/daily_demand.csv")
REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "Revenue",
    "IsReturn",
    "IsCancelledInvoice",
}


def load_sales_dataset(input_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load and validate the processed sales dataset."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sales dataset was not found: {path}")

    try:
        sales = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read sales dataset '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(sales.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Sales dataset is missing columns: {missing}")

    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"], errors="coerce")
    if sales["InvoiceDate"].isna().any():
        raise ValueError("Sales dataset contains invalid InvoiceDate values.")
    return sales


def select_valid_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Keep only positive, non-return, non-cancelled sales for the target."""
    valid_sales = sales.loc[
        sales["Quantity"].gt(0)
        & ~sales["IsReturn"].astype(bool)
        & ~sales["IsCancelledInvoice"].astype(bool)
    ].copy()
    if valid_sales.empty:
        raise ValueError("No valid positive sales are available for aggregation.")
    return valid_sales


def aggregate_daily_demand(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid sales to one row per product and calendar date."""
    valid_sales = select_valid_sales(sales)
    valid_sales["Date"] = valid_sales["InvoiceDate"].dt.normalize()
    daily_demand = (
        valid_sales.groupby(["StockCode", "Date"], as_index=False)
        .agg(
            DailyQuantityDemand=("Quantity", "sum"),
            DailyRevenue=("Revenue", "sum"),
            InvoiceCount=("Invoice", "nunique"),
        )
        .assign(HadSalesRecord=True)
        .sort_values(["StockCode", "Date"])
        .reset_index(drop=True)
    )
    return daily_demand


def validate_daily_demand(
    sales: pd.DataFrame, daily_demand: pd.DataFrame
) -> None:
    """Validate aggregation keys, target values, and totals."""
    if daily_demand.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Daily demand contains duplicate product-date rows.")
    if daily_demand["DailyQuantityDemand"].lt(0).any():
        raise ValueError("Daily demand contains negative quantities.")
    if daily_demand["HadSalesRecord"].ne(True).any():
        raise ValueError("Daily demand rows must represent observed sales records.")

    valid_sales = select_valid_sales(sales)
    expected_quantity = valid_sales["Quantity"].sum()
    expected_revenue = valid_sales["Revenue"].sum()
    if daily_demand["DailyQuantityDemand"].sum() != expected_quantity:
        raise ValueError("Daily quantity total does not match valid sales.")
    if not abs(daily_demand["DailyRevenue"].sum() - expected_revenue) < 1e-6:
        raise ValueError("Daily revenue total does not match valid sales.")


def save_daily_demand(
    daily_demand: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save daily demand to a new feature-data file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_demand.to_csv(path, index=False)


def run_daily_aggregation(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Run, validate, and save the daily aggregation step."""
    sales = load_sales_dataset(input_path)
    daily_demand = aggregate_daily_demand(sales)
    validate_daily_demand(sales, daily_demand)
    save_daily_demand(daily_demand, output_path)
    return daily_demand


def main() -> None:
    """Run daily product-demand aggregation from the command line."""
    parser = argparse.ArgumentParser(
        description="Aggregate valid retail sales by product and day."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        daily_demand = run_daily_aggregation(args.input, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Daily demand rows: {len(daily_demand)}")
    print(f"Products: {daily_demand['StockCode'].nunique()}")
    print(f"Date range: {daily_demand['Date'].min()} to {daily_demand['Date'].max()}")
    print(f"Quantity total: {daily_demand['DailyQuantityDemand'].sum()}")
    print(f"Revenue total: {daily_demand['DailyRevenue'].sum():.2f}")
    print("\nFirst 10 rows:")
    print(daily_demand.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
