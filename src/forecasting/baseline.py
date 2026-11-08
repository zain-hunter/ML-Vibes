"""Seasonal-naive baseline forecasting using seven-day demand lags."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_TRAIN_PATH = Path("data/features/train_dataset.csv")
DEFAULT_TEST_PATH = Path("data/features/test_dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/predictions/baseline_predictions.csv")
REQUIRED_COLUMNS = {"StockCode", "Date", "DailyQuantityDemand", "DemandLag7"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a train or test dataset and validate its required fields."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")
    try:
        dataframe = pd.read_csv(dataset_path, dtype={"StockCode": "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read dataset '{dataset_path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path.name} is missing columns: {missing}")
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    if dataframe["Date"].isna().any():
        raise ValueError(f"{dataset_path.name} contains invalid Date values.")
    if dataframe["StockCode"].isna().any():
        raise ValueError(f"{dataset_path.name} contains missing StockCode values.")
    if dataframe.duplicated(["StockCode", "Date"]).any():
        raise ValueError(f"{dataset_path.name} contains duplicate product-date rows.")
    return dataframe


def validate_time_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ensure every training date precedes every testing date."""
    if train["Date"].max() >= test["Date"].min():
        raise ValueError("Training dates must be earlier than testing dates.")
    if set(train["Date"]).intersection(set(test["Date"])):
        raise ValueError("Training and testing dates overlap.")


def generate_predictions(test: pd.DataFrame) -> pd.DataFrame:
    """Create seasonal-naive predictions from the existing seven-day lag."""
    predictions = test[["StockCode", "Date", "DailyQuantityDemand"]].copy()
    predictions = predictions.rename(columns={"DailyQuantityDemand": "ActualDemand"})
    predictions["PredictedDemand"] = test["DemandLag7"].to_numpy()
    predictions["ForecastError"] = (
        predictions["ActualDemand"] - predictions["PredictedDemand"]
    )
    predictions["AbsoluteError"] = predictions["ForecastError"].abs()
    return predictions


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate metrics using only rows with available predictions."""
    valid = predictions["PredictedDemand"].notna()
    scored = predictions.loc[valid]
    if scored.empty:
        raise ValueError("No test rows have an available DemandLag7 prediction.")

    errors = scored["ForecastError"]
    actual = scored["ActualDemand"]
    absolute_error = scored["AbsoluteError"]
    actual_total = actual.sum()
    wape = absolute_error.sum() / actual_total * 100 if actual_total else 0.0
    return {
        "forecastable_rows": int(valid.sum()),
        "rows_without_predictions": int((~valid).sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(math.sqrt((errors**2).mean())),
        "wape_percent": float(wape),
        "mean_actual_demand": float(actual.mean()),
        "mean_predicted_demand": float(scored["PredictedDemand"].mean()),
    }


def validate_predictions(
    test: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Validate prediction source, arithmetic, keys, and non-negative demand."""
    if len(predictions) != len(test):
        raise ValueError("Prediction row count does not match the test dataset.")
    if predictions.duplicated(["StockCode", "Date"]).any():
        raise ValueError("Predictions contain duplicate product-date rows.")
    if not predictions["StockCode"].dtype.name == "string":
        raise ValueError("Prediction StockCode values are not strings.")
    if not predictions["ActualDemand"].equals(test["DailyQuantityDemand"].reset_index(drop=True)):
        raise ValueError("ActualDemand does not match the test target.")

    expected_prediction = test["DemandLag7"].reset_index(drop=True)
    actual_prediction = predictions["PredictedDemand"].reset_index(drop=True)
    if not actual_prediction.equals(expected_prediction):
        raise ValueError("Predictions are not exactly equal to DemandLag7.")
    available = actual_prediction.notna()
    if actual_prediction.loc[available].lt(0).any():
        raise ValueError("Predicted demand contains negative values.")

    expected_error = (
        predictions["ActualDemand"] - predictions["PredictedDemand"]
    )
    if not predictions["ForecastError"].equals(expected_error):
        raise ValueError("ForecastError is calculated incorrectly.")
    if not predictions["AbsoluteError"].equals(predictions["ForecastError"].abs()):
        raise ValueError("AbsoluteError is calculated incorrectly.")


def save_predictions(
    predictions: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Save baseline predictions without modifying input datasets."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)


def print_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, float | int],
    output_path: str | Path,
) -> None:
    """Print the baseline evaluation report and a sample of predictions."""
    print(f"Training row count: {len(train)}")
    print(f"Testing row count: {len(test)}")
    print(f"Test date range: {test['Date'].min()} to {test['Date'].max()}")
    print(f"Test products: {test['StockCode'].nunique()}")
    print(f"Forecastable rows: {metrics['forecastable_rows']}")
    print(f"Rows without predictions: {metrics['rows_without_predictions']}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape_percent']:.4f}%")
    print(f"Mean actual demand: {metrics['mean_actual_demand']:.4f}")
    print(f"Mean predicted demand: {metrics['mean_predicted_demand']:.4f}")
    print(f"Output filename: {output_path}")
    print("\nSample prediction rows:")
    print(predictions.head(20).to_string(index=False))


def run_baseline(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, float | int]:
    """Run, validate, save, and report the seasonal-naive baseline."""
    train = load_dataset(train_path)
    test = load_dataset(test_path)
    validate_time_order(train, test)
    predictions = generate_predictions(test)
    validate_predictions(test, predictions)
    metrics = calculate_metrics(predictions)
    save_predictions(predictions, output_path)
    print_report(train, test, predictions, metrics, output_path)
    return metrics


def main() -> None:
    """Run the seasonal-naive baseline from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a seven-day seasonal-naive demand baseline."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        run_baseline(args.train, args.test, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
