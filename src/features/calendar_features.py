"""Create a consecutive product-date calendar for demand forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/features/daily_demand.csv")
DEFAULT_OUTPUT_PATH = Path("data/features/product_daily_calendar.csv")
REQUIRED_COLUMNS = {
    "StockCode",
    "Date",
    "DailyQuantityDemand",
    "DailyRevenue",
    "InvoiceCount",
    "HadSalesRecord",
}


def load_daily_demand(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the observed daily-demand dataset."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Daily demand file was not found: {path}")

    try:
        daily = pd.read_csv(path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read daily demand file '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(daily.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Daily demand file is missing columns: {missing}")

    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    if daily["Date"].isna().any():
        raise ValueError("Daily demand contains missing or invalid dates.")
    if daily.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Daily demand contains duplicate StockCode and Date rows.")
    if daily["DailyQuantityDemand"].lt(0).any():
        raise ValueError("Daily demand contains negative quantities.")
    return daily


def create_product_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    """Create all dates between each product's first and last sale date."""
    calendar_parts = [
        pd.DataFrame(
            {
                "StockCode": stock_code,
                "Date": pd.date_range(
                    dates["Date"].min(), dates["Date"].max(), freq="D"
                ),
            }
        )
        for stock_code, dates in daily.groupby("StockCode", sort=False)
    ]
    if not calendar_parts:
        raise ValueError("Daily demand is empty; no product calendar can be created.")

    calendar = pd.concat(calendar_parts, ignore_index=True)
    calendar = calendar.merge(
        daily,
        on=["StockCode", "Date"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    observed = calendar["HadSalesRecord"].notna()
    calendar["HadSalesRecord"] = observed
    for column in ["DailyQuantityDemand", "DailyRevenue", "InvoiceCount"]:
        calendar[column] = calendar[column].fillna(0)
    return calendar


def validate_product_calendar(
    daily: pd.DataFrame, calendar: pd.DataFrame
) -> None:
    """Validate consecutive dates, flags, totals, and preserved observations."""
    if calendar.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Calendar contains duplicate StockCode and Date rows.")
    if calendar["DailyQuantityDemand"].lt(0).any():
        raise ValueError("Calendar contains negative daily demand.")

    ordered = calendar.sort_values(["StockCode", "Date"])
    date_gaps = ordered.groupby("StockCode")["Date"].diff().dropna()
    if not date_gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Calendar dates are not consecutive for every product.")

    original_keys = pd.MultiIndex.from_frame(daily[["StockCode", "Date"]])
    observed_keys = pd.MultiIndex.from_frame(
        calendar.loc[calendar["HadSalesRecord"], ["StockCode", "Date"]]
    )
    if not original_keys.equals(observed_keys):
        raise ValueError("Observed product-date records were not preserved.")

    if not calendar.loc[~calendar["HadSalesRecord"], "DailyQuantityDemand"].eq(0).all():
        raise ValueError("Generated dates must have zero daily demand.")
    if not calendar.loc[~calendar["HadSalesRecord"], "DailyRevenue"].eq(0).all():
        raise ValueError("Generated dates must have zero daily revenue.")
    if not calendar.loc[~calendar["HadSalesRecord"], "InvoiceCount"].eq(0).all():
        raise ValueError("Generated dates must have zero invoice count.")

    if calendar["DailyQuantityDemand"].sum() != daily["DailyQuantityDemand"].sum():
        raise ValueError("Calendar changed the original quantity total.")
    if not abs(calendar["DailyRevenue"].sum() - daily["DailyRevenue"].sum()) < 1e-6:
        raise ValueError("Calendar changed the original revenue total.")


def save_product_calendar(
    calendar: pd.DataFrame, output_path: str | Path
) -> None:
    """Save the product calendar to a new feature file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(path, index=False)


def run_calendar_construction(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, int]:
    """Load, construct, validate, and save the product-date calendar."""
    daily = load_daily_demand(input_path)
    calendar = create_product_calendar(daily)
    validate_product_calendar(daily, calendar)
    save_product_calendar(calendar, output_path)
    return calendar, len(daily)


def main() -> None:
    """Run product-date calendar construction from the command line."""
    parser = argparse.ArgumentParser(
        description="Create consecutive active-window dates for each product."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        calendar, original_rows = run_calendar_construction(args.input, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    product_rows = calendar.loc[
        calendar["StockCode"].eq("10002")
    ].head(15)
    print(f"Rows before calendar construction: {original_rows}")
    print(f"Rows after calendar construction: {len(calendar)}")
    print(f"Products: {calendar['StockCode'].nunique()}")
    print(f"Date range: {calendar['Date'].min()} to {calendar['Date'].max()}")
    print(f"Generated zero-demand rows: {(~calendar['HadSalesRecord']).sum()}")
    print("\nFirst 15 rows for product 10002:")
    print(product_rows.to_string(index=False))
    print("\nValidation results: passed")


if __name__ == "__main__":
    main()
