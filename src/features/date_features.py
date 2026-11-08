"""Add calendar features to the daily product-demand dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_DIR = Path("data/features")
DEFAULT_DAILY_PATH = DEFAULT_FEATURE_DIR / "daily_demand.csv"
DEFAULT_CALENDAR_PATH = DEFAULT_FEATURE_DIR / "product_daily_calendar.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_FEATURE_DIR / "product_daily_date_features.csv"
REQUIRED_COLUMNS = {
    "StockCode",
    "Date",
    "DailyQuantityDemand",
    "DailyRevenue",
    "HadSalesRecord",
}
DATE_FEATURE_COLUMNS = {
    "Year",
    "Month",
    "Day",
    "WeekdayNumber",
    "Weekday",
    "IsWeekend",
}


def load_feature_input(
    daily_path: str | Path = DEFAULT_DAILY_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
) -> tuple[pd.DataFrame, str, bool]:
    """Load a valid calendar when available, otherwise load daily demand.

    The boolean result indicates whether calendar construction is still pending.
    """
    selected_path = Path(daily_path)
    calendar_is_pending = True
    calendar_candidate = Path(calendar_path)

    if calendar_candidate.is_file():
        candidate = pd.read_csv(calendar_candidate)
        missing_columns = REQUIRED_COLUMNS.difference(candidate.columns)
        has_unique_keys = not candidate.duplicated(["StockCode", "Date"]).any()
        if not missing_columns and has_unique_keys:
            selected_path = calendar_candidate
            calendar_is_pending = False

    if not selected_path.is_file():
        raise FileNotFoundError(f"Feature input was not found: {selected_path}")

    try:
        dataframe = pd.read_csv(selected_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read feature input '{selected_path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Feature input is missing columns: {missing}")
    return dataframe, selected_path.name, calendar_is_pending


def add_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar features derived from the Date column."""
    featured = dataframe.copy()
    dates = pd.to_datetime(featured["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains missing or invalid values.")

    featured["Date"] = dates
    featured["Year"] = dates.dt.year
    featured["Month"] = dates.dt.month
    featured["Day"] = dates.dt.day
    featured["WeekdayNumber"] = dates.dt.day_of_week
    featured["Weekday"] = dates.dt.day_name()
    featured["IsWeekend"] = dates.dt.day_of_week >= 5
    return featured


def validate_date_features(
    original: pd.DataFrame, featured: pd.DataFrame
) -> None:
    """Validate dates, derived calendar values, row preservation, and totals."""
    if len(original) != len(featured):
        raise ValueError("Date feature engineering changed the row count.")
    if featured["Date"].isna().any():
        raise ValueError("Output contains missing Date values.")
    if featured["Year"].tolist() != featured["Date"].dt.year.tolist():
        raise ValueError("Year does not match Date.")
    if featured["Month"].tolist() != featured["Date"].dt.month.tolist():
        raise ValueError("Month does not match Date.")
    if featured["Day"].tolist() != featured["Date"].dt.day.tolist():
        raise ValueError("Day does not match Date.")
    if not featured["WeekdayNumber"].between(0, 6).all():
        raise ValueError("WeekdayNumber must be between 0 and 6.")
    if featured["Weekday"].tolist() != featured["Date"].dt.day_name().tolist():
        raise ValueError("Weekday does not match Date.")
    expected_weekend = featured["Date"].dt.day_of_week >= 5
    if featured["IsWeekend"].tolist() != expected_weekend.tolist():
        raise ValueError("IsWeekend does not match the calendar day.")
    if not featured["StockCode"].notna().all():
        raise ValueError("Output contains missing StockCode values.")
    if featured.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Output contains duplicate StockCode and Date combinations.")
    if featured["DailyQuantityDemand"].sum() != original["DailyQuantityDemand"].sum():
        raise ValueError("Daily quantity values changed.")
    if not abs(
        featured["DailyRevenue"].sum() - original["DailyRevenue"].sum()
    ) < 1e-6:
        raise ValueError("Daily revenue values changed.")


def save_date_features(
    featured: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save date features to a new CSV without modifying the input."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(path, index=False)


def run_date_features(
    daily_path: str | Path = DEFAULT_DAILY_PATH,
    calendar_path: str | Path = DEFAULT_CALENDAR_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> tuple[pd.DataFrame, str, bool]:
    """Load input, add date features, validate, and save the output."""
    original, input_name, calendar_is_pending = load_feature_input(
        daily_path, calendar_path
    )
    featured = add_date_features(original)
    validate_date_features(original, featured)
    save_date_features(featured, output_path)
    return featured, input_name, calendar_is_pending


def main() -> None:
    """Run date feature engineering from the command line."""
    parser = argparse.ArgumentParser(
        description="Add calendar features to daily product demand."
    )
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY_PATH)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        featured, input_name, calendar_is_pending = run_date_features(
            args.daily, args.calendar, args.output
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Input filename: {input_name}")
    print(f"Output filename: {args.output}")
    print(f"Total rows: {len(featured)}")
    print(f"Unique products: {featured['StockCode'].nunique()}")
    print(f"Date range: {featured['Date'].min()} to {featured['Date'].max()}")
    print(f"Quantity total: {featured['DailyQuantityDemand'].sum()}")
    print(f"Revenue total: {featured['DailyRevenue'].sum():.2f}")
    print(
        "Calendar status: "
        + ("pending; daily_demand.csv was used." if calendar_is_pending else "valid calendar used.")
    )
    print("\nValidation results: passed")
    print("\nFirst 10 output rows:")
    print(featured.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
