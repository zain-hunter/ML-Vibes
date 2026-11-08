"""Create calendar-day lag features for product demand forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/features/product_daily_calendar.csv")
DEFAULT_OUTPUT_PATH = Path("data/features/product_daily_lag_features.csv")
REQUIRED_COLUMNS = {
    "StockCode",
    "Date",
    "DailyQuantityDemand",
    "DailyRevenue",
}
LAG_DAYS = (1, 7, 14, 28)


def load_product_calendar(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the complete product-date calendar."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Product calendar was not found: {path}")

    try:
        calendar = pd.read_csv(path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read product calendar '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(calendar.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Product calendar is missing columns: {missing}")

    calendar["Date"] = pd.to_datetime(calendar["Date"], errors="coerce")
    if calendar["Date"].isna().any():
        raise ValueError("Product calendar contains missing or invalid dates.")
    if calendar["StockCode"].isna().any():
        raise ValueError("Product calendar contains missing StockCode values.")
    if calendar.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Product calendar contains duplicate StockCode and Date rows.")
    return calendar


def validate_calendar_continuity(calendar: pd.DataFrame) -> None:
    """Ensure every product has one row for every date in its active window."""
    ordered = calendar.sort_values(["StockCode", "Date"])
    gaps = ordered.groupby("StockCode", sort=False)["Date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Product calendar dates are not consecutive.")


def add_lag_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """Add prior calendar-day demand values separately for each product."""
    validate_calendar_continuity(calendar)
    featured = calendar.sort_values(["StockCode", "Date"]).copy()
    demand_by_product = featured.groupby("StockCode", sort=False)[
        "DailyQuantityDemand"
    ]
    for lag_days in LAG_DAYS:
        featured[f"DemandLag{lag_days}"] = demand_by_product.shift(lag_days)
    return featured.reset_index(drop=True)


def validate_lag_features(
    calendar: pd.DataFrame, featured: pd.DataFrame
) -> None:
    """Validate lag alignment, boundaries, missing values, and totals."""
    if len(calendar) != len(featured):
        raise ValueError("Lag feature engineering changed the row count.")
    if featured.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Lag output contains duplicate StockCode and Date rows.")

    validate_calendar_continuity(featured)
    if featured["DailyQuantityDemand"].sum() != calendar["DailyQuantityDemand"].sum():
        raise ValueError("Lag feature engineering changed the quantity total.")
    if not abs(featured["DailyRevenue"].sum() - calendar["DailyRevenue"].sum()) < 1e-6:
        raise ValueError("Lag feature engineering changed the revenue total.")

    expected = featured.groupby("StockCode", sort=False)[
        "DailyQuantityDemand"
    ]
    for lag_days in LAG_DAYS:
        column = f"DemandLag{lag_days}"
        expected_values = expected.shift(lag_days)
        if not featured[column].equals(expected_values):
            raise ValueError(f"{column} does not match the product-level shift.")

        missing_by_product = featured.groupby("StockCode", sort=False)[column].head(
            lag_days
        )
        if not missing_by_product.isna().all():
            raise ValueError(f"Initial missing values for {column} are incorrect.")


def save_lag_features(
    featured: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save lag features to a new file without changing the calendar."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(path, index=False)


def run_lag_features(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, create, validate, and save product demand lag features."""
    calendar = load_product_calendar(input_path)
    featured = add_lag_features(calendar)
    validate_lag_features(calendar, featured)
    save_lag_features(featured, output_path)
    return featured


def main() -> None:
    """Run lag feature engineering from the command line."""
    parser = argparse.ArgumentParser(
        description="Create product-level calendar-day demand lags."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        featured = run_lag_features(args.input, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    product = featured.loc[featured["StockCode"].eq("10002")].head(35)
    print(f"Input filename: {args.input}")
    print(f"Output filename: {args.output}")
    print(f"Total rows: {len(featured)}")
    print(f"Unique products: {featured['StockCode'].nunique()}")
    print(f"Date range: {featured['Date'].min()} to {featured['Date'].max()}")
    for lag_days in LAG_DAYS:
        print(f"Missing DemandLag{lag_days}: {featured[f'DemandLag{lag_days}'].isna().sum()}")
    print(f"Quantity total: {featured['DailyQuantityDemand'].sum():.0f}")
    print(f"Revenue total: {featured['DailyRevenue'].sum():.2f}")
    print("\nProduct 10002, first 35 rows:")
    print(
        product[
            [
                "StockCode",
                "Date",
                "DailyQuantityDemand",
                "DemandLag1",
                "DemandLag7",
                "DemandLag14",
                "DemandLag28",
            ]
        ].to_string(index=False)
    )
    print("\nValidation results: passed")


if __name__ == "__main__":
    main()
