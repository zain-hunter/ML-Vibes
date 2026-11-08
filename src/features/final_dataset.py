"""Review and consolidate the final product-demand modeling dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURE_DIR = Path("data/features")
DEFAULT_OUTPUT_PATH = DEFAULT_FEATURE_DIR / "final_modeling_dataset.csv"
KEY_COLUMNS = ["StockCode", "Date"]
DATE_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "WeekdayNumber",
    "Weekday",
    "IsWeekend",
]
LAG_COLUMNS = ["DemandLag1", "DemandLag7", "DemandLag14", "DemandLag28"]
ROLLING_COLUMNS = [
    "DemandRollingMean7",
    "DemandRollingMean14",
    "DemandRollingMean28",
]
EXPECTED_COLUMNS = [
    "StockCode",
    "Date",
    "DailyQuantityDemand",
    "DailyRevenue",
    "DailyInvoiceCount",
    "HadSalesRecord",
    *DATE_COLUMNS,
    *LAG_COLUMNS,
    *ROLLING_COLUMNS,
    "DemandTrend7_28",
]


def load_feature_dataset(path: str | Path) -> pd.DataFrame:
    """Load one feature file with StockCode treated as an identifier."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature file was not found: {file_path}")
    try:
        dataframe = pd.read_csv(file_path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read feature file '{file_path}': {error}") from error

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    return dataframe


def review_dataset(name: str, dataframe: pd.DataFrame) -> dict[str, object]:
    """Return review information for one input dataset."""
    missing_keys = {
        column: int(dataframe[column].isna().sum())
        for column in KEY_COLUMNS
        if column in dataframe
    }
    return {
        "name": name,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "column_names": list(dataframe.columns),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "products": int(dataframe["StockCode"].nunique(dropna=True)),
        "date_min": dataframe["Date"].min(),
        "date_max": dataframe["Date"].max(),
        "missing_values": dataframe.isna().sum().to_dict(),
        "missing_keys": missing_keys,
        "duplicate_keys": int(dataframe.duplicated(KEY_COLUMNS).sum()),
    }


def validate_merge_keys(
    date_features: pd.DataFrame,
    lag_features: pd.DataFrame,
    rolling_features: pd.DataFrame,
) -> None:
    """Confirm all inputs have complete, matching one-to-one product-date keys."""
    datasets = {
        "date features": date_features,
        "lag features": lag_features,
        "rolling features": rolling_features,
    }
    for name, dataframe in datasets.items():
        if dataframe[KEY_COLUMNS].isna().any().any():
            raise ValueError(f"{name} contains missing StockCode or Date values.")
        if dataframe.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} contains duplicate StockCode and Date keys.")

    date_keys = pd.MultiIndex.from_frame(date_features[KEY_COLUMNS])
    for name, dataframe in datasets.items():
        if not date_keys.equals(pd.MultiIndex.from_frame(dataframe[KEY_COLUMNS])):
            raise ValueError(f"{name} does not contain the same product-date keys.")


def merge_feature_datasets(
    date_features: pd.DataFrame,
    lag_features: pd.DataFrame,
    rolling_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge missing calendar features into the rolling-feature base dataset."""
    validate_merge_keys(date_features, lag_features, rolling_features)
    calendar_columns = [
        column for column in DATE_COLUMNS if column not in rolling_features.columns
    ]
    merged = rolling_features.merge(
        date_features[KEY_COLUMNS + calendar_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Calendar merge contains unmatched product-date records.")
    merged = merged.drop(columns="_merge")
    merged["DailyInvoiceCount"] = merged.pop("InvoiceCount")
    return merged[EXPECTED_COLUMNS]


def validate_final_dataset(
    base: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    """Validate final schema, quality, totals, and preserved missing history."""
    missing_columns = set(EXPECTED_COLUMNS).difference(final.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Final dataset is missing columns: {missing}")
    if len(final) != len(base):
        raise ValueError("Final dataset row count changed.")
    if final["StockCode"].dtype.name != "string":
        raise ValueError("StockCode is not stored as a string identifier.")
    if final["Date"].isna().any() or final["StockCode"].isna().any():
        raise ValueError("Final dataset contains missing key values.")
    if final.duplicated(KEY_COLUMNS).any():
        raise ValueError("Final dataset contains duplicate product-date keys.")
    if not pd.api.types.is_numeric_dtype(final["DailyQuantityDemand"]):
        raise ValueError("DailyQuantityDemand is not numeric.")
    if final["DailyQuantityDemand"].lt(0).any():
        raise ValueError("DailyQuantityDemand contains negative values.")

    numeric_columns = final.select_dtypes(include=np.number).columns
    if np.isinf(final[numeric_columns].to_numpy()).any():
        raise ValueError("Final dataset contains infinite numeric values.")

    for column in [
        "DailyQuantityDemand",
        "DailyRevenue",
        "DailyInvoiceCount",
        "HadSalesRecord",
        *DATE_COLUMNS,
    ]:
        if final[column].isna().any():
            raise ValueError(f"Unexpected missing values in original calendar column: {column}")

    if final["DailyQuantityDemand"].sum() != base["DailyQuantityDemand"].sum():
        raise ValueError("Quantity total changed during consolidation.")
    if not abs(final["DailyRevenue"].sum() - base["DailyRevenue"].sum()) < 1e-6:
        raise ValueError("Revenue total changed during consolidation.")

    for column in LAG_COLUMNS + ROLLING_COLUMNS:
        if final[column].isna().sum() != base[column].isna().sum():
            raise ValueError(f"Initial missing values changed in {column}.")
    expected_trend = final["DemandRollingMean7"] - final["DemandRollingMean28"]
    if not final["DemandTrend7_28"].equals(expected_trend):
        raise ValueError("DemandTrend7_28 is inconsistent with rolling features.")


def save_final_dataset(
    dataframe: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save the consolidated modeling dataset."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def print_report(
    reviews: list[dict[str, object]],
    final: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Print input reviews and final dataset details."""
    for review in reviews:
        print(
            f"{review['name']}: {review['rows']} rows x "
            f"{review['columns']} columns; products={review['products']}; "
            f"date range={review['date_min']} to {review['date_max']}; "
            f"duplicate keys={review['duplicate_keys']}"
        )
    print(f"\nOutput filename: {output_path}")
    print(f"Final row count: {len(final)}")
    print(f"Final column count: {len(final.columns)}")
    print(f"Unique products: {final['StockCode'].nunique()}")
    print(f"Date range: {final['Date'].min()} to {final['Date'].max()}")
    print(f"Quantity total: {final['DailyQuantityDemand'].sum():.0f}")
    print(f"Revenue total: {final['DailyRevenue'].sum():.2f}")
    print(f"Duplicate count: {final.duplicated(KEY_COLUMNS).sum()}")
    print("\nAll columns:")
    print(", ".join(final.columns))
    print("\nData types:")
    print(final.dtypes.to_string())
    print("\nMissing-value counts:")
    print(final.isna().sum().to_string())
    print("\nMerge validation results: passed")
    print("\nFirst 20 rows for product 10002:")
    print(final.loc[final["StockCode"].eq("10002")].head(20).to_string(index=False))


def run_final_preparation(
    feature_dir: str | Path = DEFAULT_FEATURE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, review, merge, validate, save, and report the final dataset."""
    directory = Path(feature_dir)
    date_features = load_feature_dataset(directory / "product_daily_date_features.csv")
    lag_features = load_feature_dataset(directory / "product_daily_lag_features.csv")
    rolling_features = load_feature_dataset(
        directory / "product_daily_rolling_features.csv"
    )
    reviews = [
        review_dataset("Date features", date_features),
        review_dataset("Lag features", lag_features),
        review_dataset("Rolling features", rolling_features),
    ]
    final = merge_feature_datasets(date_features, lag_features, rolling_features)
    validate_final_dataset(rolling_features, final)
    save_final_dataset(final, output_path)
    print_report(reviews, final, output_path)
    return final


def main() -> None:
    """Run final feature review and dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Consolidate product-demand features for future modeling."
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        run_final_preparation(args.feature_dir, args.output)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
