"""Random Forest demand forecasting using leakage-safe engineered features."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TRAIN_PATH = Path("data/features/train_dataset.csv")
DEFAULT_TEST_PATH = Path("data/features/test_dataset.csv")
DEFAULT_PREDICTION_PATH = Path("data/predictions/random_forest_predictions.csv")
DEFAULT_MODEL_PATH = Path("models/random_forest_pipeline.joblib")
TARGET_COLUMN = "DailyQuantityDemand"
IDENTIFIER_COLUMN = "StockCode"
DATE_COLUMN = "Date"
CATEGORICAL_FEATURES = [IDENTIFIER_COLUMN]
NUMERICAL_FEATURES = [
    "Year",
    "Month",
    "Day",
    "WeekdayNumber",
    "IsWeekend",
    "DemandLag1",
    "DemandLag7",
    "DemandLag14",
    "DemandLag28",
    "DemandRollingMean7",
    "DemandRollingMean14",
    "DemandRollingMean28",
    "DemandTrend7_28",
]
REQUIRED_COLUMNS = {IDENTIFIER_COLUMN, DATE_COLUMN, TARGET_COLUMN, *NUMERICAL_FEATURES}
PREDICTION_COLUMNS = [
    IDENTIFIER_COLUMN,
    DATE_COLUMN,
    "ActualDemand",
    "PredictedDemand",
    "ForecastError",
    "AbsoluteError",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a train or test dataset with StockCode preserved as a string."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {dataset_path}")
    try:
        dataframe = pd.read_csv(dataset_path, dtype={IDENTIFIER_COLUMN: "string"})
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read dataset '{dataset_path}': {error}") from error

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path.name} is missing columns: {missing}")
    dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
    if dataframe[DATE_COLUMN].isna().any():
        raise ValueError(f"{dataset_path.name} contains invalid dates.")
    if dataframe[IDENTIFIER_COLUMN].isna().any():
        raise ValueError(f"{dataset_path.name} contains missing StockCode values.")
    if dataframe.duplicated([IDENTIFIER_COLUMN, DATE_COLUMN]).any():
        raise ValueError(f"{dataset_path.name} contains duplicate product-date rows.")
    return dataframe


def validate_time_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ensure the supplied train and test files are chronological."""
    if train[DATE_COLUMN].max() >= test[DATE_COLUMN].min():
        raise ValueError("Training dates must be earlier than testing dates.")
    if set(train[DATE_COLUMN]).intersection(set(test[DATE_COLUMN])):
        raise ValueError("Training and testing dates overlap.")


def select_model_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows with complete required model features."""
    required_columns = [IDENTIFIER_COLUMN, DATE_COLUMN, TARGET_COLUMN, *NUMERICAL_FEATURES]
    complete = dataframe[required_columns].notna().all(axis=1)
    selected = dataframe.loc[complete].copy()
    return selected, int((~complete).sum())


def build_pipeline() -> Pipeline:
    """Build the preprocessing and Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "stock_code",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        max_samples=0.5,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model(train: pd.DataFrame) -> Pipeline:
    """Train the Random Forest on complete historical feature rows."""
    model = build_pipeline()
    model.fit(train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES], train[TARGET_COLUMN])
    return model


def generate_predictions(
    model: Pipeline, test: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Predict forecastable test rows and retain unavailable rows with NaN values."""
    complete = test[CATEGORICAL_FEATURES + NUMERICAL_FEATURES].notna().all(axis=1)
    predictions = test[[IDENTIFIER_COLUMN, DATE_COLUMN, TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "ActualDemand"})
    predictions["PredictedDemand"] = float("nan")
    predictions.loc[complete, "PredictedDemand"] = model.predict(
        test.loc[complete, CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    )
    predictions["PredictedDemand"] = predictions["PredictedDemand"].clip(lower=0)
    predictions["ForecastError"] = (
        predictions["ActualDemand"] - predictions["PredictedDemand"]
    )
    predictions["AbsoluteError"] = predictions["ForecastError"].abs()
    return predictions, int((~complete).sum())


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate MAE, RMSE, and WAPE only for forecastable rows."""
    valid = predictions["PredictedDemand"].notna()
    scored = predictions.loc[valid]
    if scored.empty:
        raise ValueError("No test rows have complete model features.")
    errors = scored["ForecastError"]
    actual = scored["ActualDemand"]
    absolute_error = scored["AbsoluteError"]
    actual_total = actual.sum()
    return {
        "forecastable_rows": int(valid.sum()),
        "rows_without_predictions": int((~valid).sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(math.sqrt((errors**2).mean())),
        "wape_percent": float(
            absolute_error.sum() / actual_total * 100 if actual_total else 0.0
        ),
        "mean_actual_demand": float(actual.mean()),
        "mean_predicted_demand": float(scored["PredictedDemand"].mean()),
    }


def validate_predictions(
    test: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Validate prediction keys, target integrity, and error arithmetic."""
    if len(predictions) != len(test):
        raise ValueError("Prediction row count does not match the test dataset.")
    if predictions.duplicated([IDENTIFIER_COLUMN, DATE_COLUMN]).any():
        raise ValueError("Predictions contain duplicate product-date rows.")
    if predictions[IDENTIFIER_COLUMN].dtype.name != "string":
        raise ValueError("Prediction StockCode values are not strings.")
    if not predictions["ActualDemand"].equals(
        test[TARGET_COLUMN].reset_index(drop=True)
    ):
        raise ValueError("ActualDemand does not match the test target.")
    valid = predictions["PredictedDemand"].notna()
    if predictions.loc[valid, "PredictedDemand"].lt(0).any():
        raise ValueError("Predictions contain negative values.")
    expected_error = predictions["ActualDemand"] - predictions["PredictedDemand"]
    if not predictions["ForecastError"].equals(expected_error):
        raise ValueError("ForecastError is incorrect.")
    if not predictions["AbsoluteError"].equals(predictions["ForecastError"].abs()):
        raise ValueError("AbsoluteError is incorrect.")


def compare_with_baseline(
    predictions: pd.DataFrame, baseline_path: str | Path
) -> dict[str, float | int]:
    """Compare Random Forest and baseline metrics on common forecastable rows."""
    baseline_path = Path(baseline_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline predictions were not found: {baseline_path}")
    baseline = pd.read_csv(baseline_path, dtype={IDENTIFIER_COLUMN: "string"})
    baseline[DATE_COLUMN] = pd.to_datetime(baseline[DATE_COLUMN], errors="coerce")
    joined = predictions.merge(
        baseline[[IDENTIFIER_COLUMN, DATE_COLUMN, "PredictedDemand"]],
        on=[IDENTIFIER_COLUMN, DATE_COLUMN],
        how="inner",
        suffixes=("_rf", "_baseline"),
        validate="one_to_one",
    )
    common = joined.dropna(subset=["PredictedDemand_rf", "PredictedDemand_baseline"])
    actual = common["ActualDemand"]

    def metrics_for(predicted: pd.Series) -> tuple[float, float, float]:
        errors = actual - predicted
        absolute = errors.abs()
        wape = absolute.sum() / actual.sum() * 100 if actual.sum() else 0.0
        return float(absolute.mean()), float(math.sqrt((errors**2).mean())), float(wape)

    rf_mae, rf_rmse, rf_wape = metrics_for(common["PredictedDemand_rf"])
    base_mae, base_rmse, base_wape = metrics_for(common["PredictedDemand_baseline"])

    def improvement(baseline_metric: float, model_metric: float) -> float:
        return (
            (baseline_metric - model_metric) / baseline_metric * 100
            if baseline_metric
            else 0.0
        )

    return {
        "common_rows": len(common),
        "rf_mae_common": rf_mae,
        "rf_rmse_common": rf_rmse,
        "rf_wape_common": rf_wape,
        "baseline_mae_common": base_mae,
        "baseline_rmse_common": base_rmse,
        "baseline_wape_common": base_wape,
        "mae_improvement_percent": improvement(base_mae, rf_mae),
        "rmse_improvement_percent": improvement(base_rmse, rf_rmse),
        "wape_improvement_percent": improvement(base_wape, rf_wape),
    }


def save_outputs(
    predictions: pd.DataFrame,
    model: Pipeline,
    prediction_path: str | Path,
    model_path: str | Path,
) -> None:
    """Save predictions and the fitted pipeline."""
    prediction_file = Path(prediction_path)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_file, index=False)
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)


def print_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, float | int],
    comparison: dict[str, float | int],
    dropped_train: int,
    dropped_test: int,
    prediction_path: str | Path,
    model_path: str | Path,
) -> None:
    """Print model configuration, evaluation, and baseline comparison."""
    print(f"Training rows: {len(train)}; complete feature rows used: {len(train) - dropped_train}")
    print(f"Training rows excluded for missing features: {dropped_train}")
    print(f"Testing rows: {len(test)}; test rows without predictions: {dropped_test}")
    print(f"Test date range: {test[DATE_COLUMN].min()} to {test[DATE_COLUMN].max()}")
    print(f"Test products: {test[IDENTIFIER_COLUMN].nunique()}")
    print(f"Forecastable rows: {metrics['forecastable_rows']}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape_percent']:.4f}%")
    print(f"Mean actual demand: {metrics['mean_actual_demand']:.4f}")
    print(f"Mean predicted demand: {metrics['mean_predicted_demand']:.4f}")
    print(f"\nCommon baseline/model rows: {comparison['common_rows']}")
    print(
        f"Baseline common MAE/RMSE/WAPE: {comparison['baseline_mae_common']:.4f} / "
        f"{comparison['baseline_rmse_common']:.4f} / {comparison['baseline_wape_common']:.4f}%"
    )
    print(
        f"Random Forest common MAE/RMSE/WAPE: {comparison['rf_mae_common']:.4f} / "
        f"{comparison['rf_rmse_common']:.4f} / {comparison['rf_wape_common']:.4f}%"
    )
    print(
        f"Improvement MAE/RMSE/WAPE: {comparison['mae_improvement_percent']:.2f}% / "
        f"{comparison['rmse_improvement_percent']:.2f}% / "
        f"{comparison['wape_improvement_percent']:.2f}%"
    )
    print(f"\nPredictions: {prediction_path}")
    print(f"Pipeline: {model_path}")
    print("\nSample predictions:")
    print(predictions.head(20).to_string(index=False))


def run_random_forest(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    prediction_path: str | Path = DEFAULT_PREDICTION_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    baseline_path: str | Path = "data/predictions/baseline_predictions.csv",
) -> dict[str, float | int]:
    """Run, validate, save, and report the Random Forest benchmark."""
    train = load_dataset(train_path)
    test = load_dataset(test_path)
    validate_time_order(train, test)
    train_model_data, dropped_train = select_model_data(train)
    test_model_data, dropped_test = select_model_data(test)
    model = train_model(train_model_data)
    predictions, generated_dropped_test = generate_predictions(model, test)
    if generated_dropped_test != dropped_test:
        raise ValueError("Test feature availability changed during prediction.")
    validate_predictions(test, predictions)
    metrics = calculate_metrics(predictions)
    comparison = compare_with_baseline(predictions, baseline_path)
    save_outputs(predictions, model, prediction_path, model_path)
    print_report(
        train,
        test,
        predictions,
        metrics,
        comparison,
        dropped_train,
        dropped_test,
        prediction_path,
        model_path,
    )
    return metrics


def main() -> None:
    """Run the Random Forest benchmark from the command line."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest demand forecast."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/predictions/baseline_predictions.csv"),
    )
    args = parser.parse_args()
    try:
        run_random_forest(
            args.train, args.test, args.predictions, args.model, args.baseline
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
