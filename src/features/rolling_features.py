"""Create leakage-safe rolling demand averages and a demand trend."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/features/product_daily_lag_features.csv")
DEFAULT_OUTPUT_PATH = Path("data/features/product_daily_rolling_features.csv")
REQUIRED_COLUMNS = {
    "StockCode",
    "Date",
    "DailyQuantityDemand",
    "DailyRevenue",
    "DemandLag1",
    "DemandLag7",
    "DemandLag14",
    "DemandLag28",
}
ROLLING_WINDOWS = (7, 14, 28)


def load_lag_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the lag-feature dataset."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Lag feature dataset was not found: {path}")

    try:
        dataframe = pd.read_csv(path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read lag dataset '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Lag dataset is missing columns: {missing}")

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dataframe["Date"].isna().any():
        raise ValueError("Lag dataset contains missing or invalid dates.")
    if dataframe["StockCode"].isna().any():
        raise ValueError("Lag dataset contains missing StockCode values.")
    if dataframe.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Lag dataset contains duplicate StockCode and Date rows.")
    return dataframe


def validate_calendar_continuity(dataframe: pd.DataFrame) -> None:
    """Ensure every product has consecutive dates."""
    ordered = dataframe.sort_values(["StockCode", "Date"])
    gaps = ordered.groupby("StockCode", sort=False)["Date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Dates are not consecutive for every product.")


def add_rolling_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add prior-day rolling means separately for each product."""
    validate_calendar_continuity(dataframe)
    featured = dataframe.sort_values(["StockCode", "Date"]).copy()
    grouped_demand = featured.groupby("StockCode", sort=False)[
        "DailyQuantityDemand"
    ]
    shifted_demand = grouped_demand.shift(1)
    for window in ROLLING_WINDOWS:
        featured[f"DemandRollingMean{window}"] = (
            shifted_demand.groupby(featured["StockCode"], sort=False)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return featured


def add_trend_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the difference between the seven-day and 28-day rolling means."""
    featured = dataframe.copy()
    featured["DemandTrend7_28"] = (
        featured["DemandRollingMean7"] - featured["DemandRollingMean28"]
    )
    return featured


def validate_rolling_features(
    original: pd.DataFrame, featured: pd.DataFrame
) -> None:
    """Validate leakage safety, rolling values, trend values, and totals."""
    if len(original) != len(featured):
        raise ValueError("Rolling features changed the row count.")
    if featured.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Rolling output contains duplicate product-date rows.")
    validate_calendar_continuity(featured)

    if featured["DailyQuantityDemand"].sum() != original["DailyQuantityDemand"].sum():
        raise ValueError("Rolling features changed the quantity total.")
    if not abs(featured["DailyRevenue"].sum() - original["DailyRevenue"].sum()) < 1e-6:
        raise ValueError("Rolling features changed the revenue total.")

    for window in ROLLING_WINDOWS:
        column = f"DemandRollingMean{window}"
        expected = (
            featured.groupby("StockCode", sort=False)["DailyQuantityDemand"]
            .shift(1)
            .groupby(featured["StockCode"], sort=False)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        if not featured[column].reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        ):
            raise ValueError(f"{column} does not match the leakage-safe calculation.")

        unavailable = featured.groupby("StockCode", sort=False)[column].head(window)
        if not unavailable.isna().all():
            raise ValueError(f"Initial missing values for {column} are incorrect.")

    expected_trend = (
        featured["DemandRollingMean7"] - featured["DemandRollingMean28"]
    )
    if not featured["DemandTrend7_28"].equals(expected_trend):
        raise ValueError("DemandTrend7_28 does not match the requested calculation.")


def save_rolling_features(
    featured: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save rolling features to a new dataset."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(path, index=False)


def run_rolling_features(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, create, validate, and save rolling and trend features."""
    original = load_lag_dataset(input_path)
    featured = add_rolling_features(original)
    featured = add_trend_feature(featured)
    validate_rolling_features(original, featured)
    save_rolling_features(featured, output_path)
    return featured


def main() -> None:
    """Run rolling and trend feature engineering from the command line."""
    parser = argparse.ArgumentParser(
        description="Create leakage-safe rolling demand features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        featured = run_rolling_features(args.input, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    product = featured.loc[featured["StockCode"].eq("10002")].head(35)
    print(f"Input filename: {args.input}")
    print(f"Output filename: {args.output}")
    print(f"Total rows: {len(featured)}")
    print(f"Unique products: {featured['StockCode'].nunique()}")
    print(f"Date range: {featured['Date'].min()} to {featured['Date'].max()}")
    for window in ROLLING_WINDOWS:
        print(
            f"Missing DemandRollingMean{window}: "
            f"{featured[f'DemandRollingMean{window}'].isna().sum()}"
        )
    print(f"Missing DemandTrend7_28: {featured['DemandTrend7_28'].isna().sum()}")
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
                "DemandRollingMean7",
                "DemandRollingMean14",
                "DemandRollingMean28",
                "DemandTrend7_28",
            ]
        ].to_string(index=False)
    )
    print("\nValidation results: passed")


if __name__ == "__main__":
    main()
