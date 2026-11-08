"""Gradient Boosting demand forecasting with a safe scikit-learn fallback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.forecasting.random_forest import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    IDENTIFIER_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_dataset,
    select_model_data,
    validate_time_order,
)


DEFAULT_PREDICTION_PATH = Path(
    "data/predictions/gradient_boosting_predictions.csv"
)
DEFAULT_MODEL_PATH = Path("models/gradient_boosting_pipeline.joblib")
DEFAULT_BASELINE_PATH = Path("data/predictions/baseline_predictions.csv")
DEFAULT_RANDOM_FOREST_PATH = Path(
    "data/predictions/random_forest_predictions.csv"
)


def build_pipeline() -> Pipeline:
    """Build the dense preprocessing and Gradient Boosting pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "stock_code",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        loss="squared_error",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model(train: pd.DataFrame) -> Pipeline:
    """Train Gradient Boosting on rows with complete historical features."""
    model = build_pipeline()
    model.fit(
        train[CATEGORICAL_FEATURES + NUMERICAL_FEATURES],
        train[TARGET_COLUMN],
    )
    return model


def generate_predictions(
    model: Pipeline, test: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Predict complete test rows while retaining unavailable rows."""
    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    complete = test[feature_columns].notna().all(axis=1)
    predictions = test[[IDENTIFIER_COLUMN, DATE_COLUMN, TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "ActualDemand"})
    predictions["PredictedDemand"] = float("nan")
    predictions.loc[complete, "PredictedDemand"] = model.predict(
        test.loc[complete, feature_columns]
    )
    predictions["PredictedDemand"] = predictions["PredictedDemand"].clip(lower=0)
    predictions["ForecastError"] = (
        predictions["ActualDemand"] - predictions["PredictedDemand"]
    )
    predictions["AbsoluteError"] = predictions["ForecastError"].abs()
    return predictions, int((~complete).sum())


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate MAE, RMSE, and WAPE on available predictions."""
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
            absolute_error.sum() / actual_total * 100
            if actual_total
            else 0.0
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
    if not predictions["AbsoluteError"].equals(
        predictions["ForecastError"].abs()
    ):
        raise ValueError("AbsoluteError is incorrect.")


def load_prediction_file(path: str | Path) -> pd.DataFrame:
    """Load an existing prediction file for common-observation comparison."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file was not found: {prediction_path}")
    predictions = pd.read_csv(
        prediction_path,
        dtype={IDENTIFIER_COLUMN: "string"},
        parse_dates=[DATE_COLUMN],
    )
    required = {IDENTIFIER_COLUMN, DATE_COLUMN, "ActualDemand", "PredictedDemand"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"{prediction_path.name} is missing columns: {', '.join(sorted(missing))}"
        )
    return predictions


def metrics_for_predictions(
    actual: pd.Series, predicted: pd.Series
) -> tuple[float, float, float]:
    """Calculate the three requested metrics for aligned predictions."""
    errors = actual - predicted
    absolute_error = errors.abs()
    actual_total = actual.sum()
    wape = absolute_error.sum() / actual_total * 100 if actual_total else 0.0
    return (
        float(absolute_error.mean()),
        float(math.sqrt((errors**2).mean())),
        float(wape),
    )


def compare_models(
    predictions: pd.DataFrame,
    random_forest_path: str | Path,
    baseline_path: str | Path,
) -> dict[str, float | int]:
    """Compare all models on exactly the same valid product-day rows."""
    key_columns = [IDENTIFIER_COLUMN, DATE_COLUMN]
    random_forest = load_prediction_file(random_forest_path)
    baseline = load_prediction_file(baseline_path)
    gradient_boosting = predictions.rename(
        columns={"PredictedDemand": "PredictedDemand_gradient_boosting"}
    )
    random_forest = random_forest.rename(
        columns={"PredictedDemand": "PredictedDemand_random_forest"}
    )
    baseline = baseline.rename(
        columns={"PredictedDemand": "PredictedDemand_baseline"}
    )
    joined = gradient_boosting.merge(
        random_forest[key_columns + ["PredictedDemand_random_forest"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    ).merge(
        baseline[key_columns + ["PredictedDemand_baseline"]],
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    joined = joined.dropna(
        subset=[
            "PredictedDemand_gradient_boosting",
            "PredictedDemand_random_forest",
            "PredictedDemand_baseline",
        ]
    )
    if joined.empty:
        raise ValueError("No common valid observations are available for comparison.")
    actual = joined["ActualDemand"]
    gradient_metrics = metrics_for_predictions(
        actual, joined["PredictedDemand_gradient_boosting"]
    )
    random_forest_metrics = metrics_for_predictions(
        actual, joined["PredictedDemand_random_forest"]
    )
    baseline_metrics = metrics_for_predictions(
        actual, joined["PredictedDemand_baseline"]
    )
    return {
        "common_rows": len(joined),
        "gradient_boosting_mae": gradient_metrics[0],
        "gradient_boosting_rmse": gradient_metrics[1],
        "gradient_boosting_wape": gradient_metrics[2],
        "random_forest_mae": random_forest_metrics[0],
        "random_forest_rmse": random_forest_metrics[1],
        "random_forest_wape": random_forest_metrics[2],
        "baseline_mae": baseline_metrics[0],
        "baseline_rmse": baseline_metrics[1],
        "baseline_wape": baseline_metrics[2],
    }


def save_outputs(
    predictions: pd.DataFrame,
    model: Pipeline,
    prediction_path: str | Path,
    model_path: str | Path,
) -> None:
    """Save predictions and the fitted pipeline without touching inputs."""
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
    """Print the evaluation report and a prediction sample."""
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
    print(f"\nCommon rows across all models: {comparison['common_rows']}")
    print(
        "Gradient Boosting common MAE/RMSE/WAPE: "
        f"{comparison['gradient_boosting_mae']:.4f} / "
        f"{comparison['gradient_boosting_rmse']:.4f} / "
        f"{comparison['gradient_boosting_wape']:.4f}%"
    )
    print(
        "Random Forest common MAE/RMSE/WAPE: "
        f"{comparison['random_forest_mae']:.4f} / "
        f"{comparison['random_forest_rmse']:.4f} / "
        f"{comparison['random_forest_wape']:.4f}%"
    )
    print(
        "Seasonal Naive common MAE/RMSE/WAPE: "
        f"{comparison['baseline_mae']:.4f} / "
        f"{comparison['baseline_rmse']:.4f} / "
        f"{comparison['baseline_wape']:.4f}%"
    )
    print(f"\nPredictions: {prediction_path}")
    print(f"Pipeline: {model_path}")
    print("\nSample predictions:")
    print(predictions.head(20).to_string(index=False))


def run_gradient_boosting(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    prediction_path: str | Path = DEFAULT_PREDICTION_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    random_forest_path: str | Path = DEFAULT_RANDOM_FOREST_PATH,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
) -> dict[str, float | int]:
    """Run, validate, save, and report Gradient Boosting."""
    train = load_dataset(train_path)
    test = load_dataset(test_path)
    validate_time_order(train, test)
    train_model_data, dropped_train = select_model_data(train)
    _, dropped_test = select_model_data(test)
    model = train_model(train_model_data)
    predictions, generated_dropped_test = generate_predictions(model, test)
    if generated_dropped_test != dropped_test:
        raise ValueError("Test feature availability changed during prediction.")
    validate_predictions(test, predictions)
    metrics = calculate_metrics(predictions)
    comparison = compare_models(predictions, random_forest_path, baseline_path)
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
    """Run Gradient Boosting from the command line."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Gradient Boosting demand forecasts."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--random-forest",
        type=Path,
        default=DEFAULT_RANDOM_FOREST_PATH,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()
    try:
        run_gradient_boosting(
            args.train,
            args.test,
            args.predictions,
            args.model,
            args.random_forest,
            args.baseline,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
