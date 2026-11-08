"""Create chronological train and test datasets for demand forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/features/final_modeling_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("data/features")
TRAIN_FILENAME = "train_dataset.csv"
TEST_FILENAME = "test_dataset.csv"
REQUIRED_COLUMNS = {"StockCode", "Date", "DailyQuantityDemand"}


def load_final_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load and validate the final modeling dataset."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Final modeling dataset was not found: {path}")
    try:
        dataframe = pd.read_csv(path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read final dataset '{path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Final dataset is missing columns: {missing}")
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dataframe["Date"].isna().any():
        raise ValueError("Final dataset contains missing or invalid dates.")
    if dataframe["StockCode"].isna().any():
        raise ValueError("Final dataset contains missing StockCode values.")
    if dataframe.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Final dataset contains duplicate product-date records.")
    return dataframe


def split_by_date(
    dataframe: pd.DataFrame, train_proportion: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split all rows using the earliest proportion of unique calendar dates."""
    if not 0 < train_proportion < 1:
        raise ValueError("train_proportion must be between 0 and 1.")

    ordered = dataframe.sort_values(["Date", "StockCode"]).reset_index(drop=True)
    unique_dates = ordered["Date"].drop_duplicates().sort_values().tolist()
    if len(unique_dates) < 2:
        raise ValueError("At least two unique dates are required for a split.")

    train_date_count = int(len(unique_dates) * train_proportion)
    train_date_count = max(1, min(train_date_count, len(unique_dates) - 1))
    cutoff_date = unique_dates[train_date_count - 1]
    train = ordered.loc[ordered["Date"] <= cutoff_date].copy()
    test = ordered.loc[ordered["Date"] > cutoff_date].copy()
    return train, test, cutoff_date


def validate_split(
    original: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_proportion: float = 0.8,
) -> None:
    """Validate chronological separation, row preservation, and identifiers."""
    if len(train) + len(test) != len(original):
        raise ValueError("Train and test rows do not preserve the input row count.")
    if set(train["Date"]).intersection(set(test["Date"])):
        raise ValueError("Training and testing dates overlap.")
    if train["Date"].max() >= test["Date"].min():
        raise ValueError("Training dates are not strictly earlier than test dates.")
    if not (test["Date"] > train["Date"].max()).all():
        raise ValueError("A test date is not later than every training date.")
    if train["StockCode"].dtype.name != "string" or test["StockCode"].dtype.name != "string":
        raise ValueError("Product identifiers are not stored as strings.")
    for name, dataset in [("train", train), ("test", test)]:
        if dataset.duplicated(["StockCode", "Date"]).any():
            raise ValueError(f"{name} contains duplicate product-date records.")

    original_keys = set(zip(original["StockCode"], original["Date"]))
    split_keys = set(zip(train["StockCode"], train["Date"])) | set(
        zip(test["StockCode"], test["Date"])
    )
    if original_keys != split_keys:
        raise ValueError("Train and test keys do not preserve all original records.")

    expected_train_dates = int(original["Date"].nunique() * train_proportion)
    expected_train_dates = max(1, min(expected_train_dates, original["Date"].nunique() - 1))
    actual_train_dates = train["Date"].nunique()
    if actual_train_dates != expected_train_dates:
        raise ValueError("Training date proportion was not applied as expected.")


def save_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save train and test datasets without modifying the input."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train_path = directory / TRAIN_FILENAME
    test_path = directory / TEST_FILENAME
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def print_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    train_path: Path,
    test_path: Path,
) -> None:
    """Print split sizes, date ranges, proportions, and target statistics."""
    total_dates = train["Date"].nunique() + test["Date"].nunique()
    print(f"Training output: {train_path}")
    print(f"Testing output: {test_path}")
    print(f"Date cutoff: {cutoff_date.date()}")
    for name, dataset in [("Training", train), ("Testing", test)]:
        print(f"\n{name}:")
        print(f"- Rows: {len(dataset)}")
        print(f"- Products: {dataset['StockCode'].nunique()}")
        print(f"- Date range: {dataset['Date'].min()} to {dataset['Date'].max()}")
        print(f"- Unique dates: {dataset['Date'].nunique()}")
        print(f"- Date proportion: {dataset['Date'].nunique() / total_dates:.2%}")
        print(
            f"- Quantity total: {dataset['DailyQuantityDemand'].sum():.0f}; "
            f"mean daily product demand: {dataset['DailyQuantityDemand'].mean():.2f}"
        )
    print("\nValidation results: passed")


def run_time_split(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, split, validate, save, and report chronological datasets."""
    original = load_final_dataset(input_path)
    train, test, cutoff_date = split_by_date(original)
    validate_split(original, train, test)
    train_path, test_path = save_split(train, test, output_dir)
    print_report(train, test, cutoff_date, train_path, test_path)
    return train, test


def main() -> None:
    """Run the chronological train/test split."""
    parser = argparse.ArgumentParser(
        description="Split the final modeling dataset chronologically."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run_time_split(args.input, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
