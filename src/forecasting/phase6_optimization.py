"""Phase 6 Random Forest optimization, diagnostics, fallback, and selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

ROOT = Path(".")
KEY_COLUMNS = [IDENTIFIER_COLUMN, DATE_COLUMN]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
BASELINE_PATH = ROOT / "data/predictions/baseline_predictions.csv"
ORIGINAL_RF_PATH = ROOT / "data/predictions/random_forest_predictions.csv"
GB_PATH = ROOT / "data/predictions/gradient_boosting_predictions.csv"
OPTIMIZED_PREDICTIONS_PATH = ROOT / "data/predictions/optimized_random_forest_predictions.csv"
FALLBACK_PREDICTIONS_PATH = ROOT / "data/predictions/final_predictions_with_fallback.csv"
OPTIMIZED_MODEL_PATH = ROOT / "models/random_forest_optimized_pipeline.joblib"
FINAL_MODEL_PATH = ROOT / "models/final_demand_forecasting_pipeline.joblib"
FINAL_METADATA_PATH = ROOT / "models/final_model_metadata.json"
BEFORE_REPORT_PATH = ROOT / "reports/model_comparison_before_optimization.csv"
OPTIMIZATION_REPORT_PATH = ROOT / "reports/random_forest_optimization_results.csv"
FINAL_REPORT_PATH = ROOT / "reports/final_model_comparison.csv"
MISSING_REPORT_PATH = ROOT / "reports/missing_prediction_analysis.csv"
ERROR_REPORTS = {
    "product": ROOT / "reports/product_forecast_errors.csv",
    "month": ROOT / "reports/monthly_forecast_errors.csv",
    "weekday": ROOT / "reports/weekday_forecast_errors.csv",
    "range": ROOT / "reports/demand_range_errors.csv",
}


def metric_values(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    errors = actual - predicted
    absolute = errors.abs()
    denominator = actual.sum()
    return {
        "MAE": float(absolute.mean()),
        "RMSE": float(math.sqrt((errors**2).mean())),
        "WAPE": float(absolute.sum() / denominator * 100 if denominator else 0.0),
        "MeanActualDemand": float(actual.mean()),
        "MeanPredictedDemand": float(predicted.mean()),
    }


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file was not found: {path}")
    frame = pd.read_csv(path, dtype={IDENTIFIER_COLUMN: "string"})
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError(f"{path.name} contains invalid dates.")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path.name} contains duplicate product-date rows.")
    return frame


def common_metrics(
    prediction_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined: pd.DataFrame | None = None
    for name, frame in prediction_frames.items():
        selected = frame[KEY_COLUMNS + ["ActualDemand", "PredictedDemand"]].rename(
            columns={
                "ActualDemand": f"ActualDemand_{name}",
                "PredictedDemand": f"PredictedDemand_{name}",
            }
        )
        combined = (
            selected
            if combined is None
            else combined.merge(
                selected, on=KEY_COLUMNS, how="inner", validate="one_to_one"
            )
        )
    if combined is None:
        raise ValueError("No prediction frames were provided.")
    actual_columns = [c for c in combined if c.startswith("ActualDemand_")]
    combined["ActualDemand"] = combined[actual_columns[0]]
    for column in actual_columns[1:]:
        if not combined["ActualDemand"].equals(combined[column]):
            raise ValueError("Prediction files contain different actual targets.")
    prediction_columns = [f"PredictedDemand_{name}" for name in prediction_frames]
    common = combined.dropna(subset=prediction_columns).copy()
    if common.empty:
        raise ValueError("No common forecastable observations exist.")
    rows = []
    for name in prediction_frames:
        values = metric_values(
            common["ActualDemand"], common[f"PredictedDemand_{name}"]
        )
        rows.append({"Model": name, "EvaluatedObservations": len(common), **values})
    return common, pd.DataFrame(rows)


def audit_before_optimization(train: pd.DataFrame) -> pd.DataFrame:
    train_complete, _ = select_model_data(train)
    frames = {
        "Seasonal Naive": load_predictions(BASELINE_PATH),
        "Random Forest": load_predictions(ORIGINAL_RF_PATH),
        "Gradient Boosting": load_predictions(GB_PATH),
    }
    _, report = common_metrics(frames)
    parameters = {
        "Seasonal Naive": "DemandLag7",
        "Random Forest": "n_estimators=100; max_depth=15; min_samples_leaf=5; max_samples=0.5; random_state=42",
        "Gradient Boosting": "n_estimators=100; learning_rate=0.1; max_depth=3; min_samples_leaf=5; random_state=42",
    }
    locations = {
        "Seasonal Naive": "",
        "Random Forest": "models/random_forest_pipeline.joblib",
        "Gradient Boosting": "models/gradient_boosting_pipeline.joblib",
    }
    report["TrainingObservations"] = report["Model"].map(
        {"Seasonal Naive": 0, "Random Forest": len(train_complete), "Gradient Boosting": len(train_complete)}
    )
    report["ModelParameters"] = report["Model"].map(parameters)
    report["ModelFile"] = report["Model"].map(locations)
    BEFORE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(BEFORE_REPORT_PATH, index=False)
    return report


def build_random_forest(params: dict[str, Any]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("stock_code", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def chronological_validation_split(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(train[DATE_COLUMN].drop_duplicates())
    cutoff = max(1, int(len(dates) * 0.8))
    validation_dates = set(dates[cutoff:])
    fitting = train[~train[DATE_COLUMN].isin(validation_dates)].copy()
    validation = train[train[DATE_COLUMN].isin(validation_dates)].copy()
    if fitting.empty or validation.empty:
        raise ValueError("Chronological validation produced an empty partition.")
    if fitting[DATE_COLUMN].max() >= validation[DATE_COLUMN].min():
        raise ValueError("Chronological validation dates overlap.")
    return fitting, validation


def optimize_random_forest(
    train: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fitting, validation = chronological_validation_split(train)
    configurations = [
        {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5, "max_samples": 0.5, "max_features": 1.0},
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5, "max_samples": 0.5, "max_features": 0.7},
        {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 3, "max_samples": 0.5, "max_features": 0.7},
    ]
    records = []
    for index, parameters in enumerate(configurations, start=1):
        model = build_random_forest(parameters)
        model.fit(fitting[MODEL_FEATURES], fitting[TARGET_COLUMN])
        predicted = np.maximum(model.predict(validation[MODEL_FEATURES]), 0)
        scores = metric_values(
            validation[TARGET_COLUMN], pd.Series(predicted, index=validation.index)
        )
        records.append(
            {
                "Configuration": index,
                **parameters,
                "TrainingRows": len(fitting),
                "ValidationRows": len(validation),
                "ValidationStart": validation[DATE_COLUMN].min(),
                "ValidationEnd": validation[DATE_COLUMN].max(),
                **{f"Validation{key}": value for key, value in scores.items()},
            }
        )
        print(f"Completed optimization configuration {index}/{len(configurations)}")
    results = pd.DataFrame(records).sort_values("ValidationMAE").reset_index(drop=True)
    best = results.iloc[0]
    best_parameters = {key: best[key] for key in configurations[0]}
    for key in ("n_estimators", "max_depth", "min_samples_leaf"):
        best_parameters[key] = int(best_parameters[key])
    for key in ("max_samples", "max_features"):
        best_parameters[key] = float(best_parameters[key])
    OPTIMIZATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OPTIMIZATION_REPORT_PATH, index=False)
    return best_parameters, results


def predict_with_model(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    complete = test[MODEL_FEATURES].notna().all(axis=1)
    output = test[KEY_COLUMNS + [TARGET_COLUMN]].rename(
        columns={TARGET_COLUMN: "ActualDemand"}
    ).copy()
    output["PredictedDemand"] = np.nan
    output.loc[complete, "PredictedDemand"] = np.maximum(
        model.predict(test.loc[complete, MODEL_FEATURES]), 0
    )
    output["ForecastError"] = output["ActualDemand"] - output["PredictedDemand"]
    output["AbsoluteError"] = output["ForecastError"].abs()
    return output


def save_optimized(model: Pipeline, predictions: pd.DataFrame) -> None:
    OPTIMIZED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OPTIMIZED_MODEL_PATH)
    predictions.to_csv(OPTIMIZED_PREDICTIONS_PATH, index=False)


def create_final_comparison(optimized: pd.DataFrame, train_rows: int) -> pd.DataFrame:
    frames = {
        "Seasonal Naive": load_predictions(BASELINE_PATH),
        "Random Forest": load_predictions(ORIGINAL_RF_PATH),
        "Gradient Boosting": load_predictions(GB_PATH),
        "Optimized Random Forest": optimized,
    }
    common, report = common_metrics(frames)
    original_mae = report.loc[report["Model"].eq("Random Forest"), "MAE"].iloc[0]
    baseline_mae = report.loc[report["Model"].eq("Seasonal Naive"), "MAE"].iloc[0]
    report["ImprovementVsOriginalRFPercent"] = (
        (original_mae - report["MAE"]) / original_mae * 100
    )
    report["ImprovementVsSeasonalNaivePercent"] = (
        (baseline_mae - report["MAE"]) / baseline_mae * 100
    )
    report["TrainingObservations"] = train_rows
    report["CommonObservationSet"] = len(common)
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(FINAL_REPORT_PATH, index=False)
    return report


def create_error_reports(predictions: pd.DataFrame, test: pd.DataFrame) -> None:
    valid = predictions["PredictedDemand"].notna()
    frame = test.loc[
        valid, KEY_COLUMNS + [TARGET_COLUMN, "Month", "WeekdayNumber"]
    ].rename(columns={TARGET_COLUMN: "ActualDemand"})
    frame = frame.merge(
        predictions[KEY_COLUMNS + ["PredictedDemand", "ForecastError", "AbsoluteError"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    frame["DemandRange"] = pd.cut(
        frame["ActualDemand"],
        bins=[-np.inf, 0, 1, 10, 50, 100, np.inf],
        labels=["0", "1", "2-10", "11-50", "51-100", "101+"],
    )

    def grouped(columns: list[str]) -> pd.DataFrame:
        result = frame.groupby(columns, observed=True).agg(
            Observations=("ActualDemand", "size"),
            MAE=("AbsoluteError", "mean"),
            RMSE=("ForecastError", lambda values: float(np.sqrt(np.mean(values**2)))),
            MeanActualDemand=("ActualDemand", "mean"),
            MeanPredictedDemand=("PredictedDemand", "mean"),
            MeanForecastError=("ForecastError", "mean"),
            ZeroActualRate=("ActualDemand", lambda values: float((values == 0).mean())),
            ZeroPredictionRate=("PredictedDemand", lambda values: float((values == 0).mean())),
            MaxActualDemand=("ActualDemand", "max"),
        ).reset_index()
        result["UnderpredictionRate"] = (result["MeanForecastError"] < 0).astype(float)
        result["OverpredictionRate"] = (result["MeanForecastError"] > 0).astype(float)
        return result

    for name, columns in {
        "product": [IDENTIFIER_COLUMN],
        "month": ["Month"],
        "weekday": ["WeekdayNumber"],
        "range": ["DemandRange"],
    }.items():
        path = ERROR_REPORTS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        grouped(columns).to_csv(path, index=False)


def analyze_missing_predictions(
    test: pd.DataFrame, full_history: pd.DataFrame | None = None
) -> pd.DataFrame:
    missing = test[MODEL_FEATURES].isna()
    rows = []
    for feature in NUMERICAL_FEATURES:
        rows.append(
            {
                "Feature": feature,
                "MissingRows": int(missing[feature].sum()),
                "MissingPercentOfTest": float(missing[feature].mean() * 100),
                "MissingRowsWithOtherMissingFeatures": int(
                    (missing[feature] & (missing.sum(axis=1) > 1)).sum()
                ),
            }
        )
    any_missing = missing.any(axis=1)
    product_missing = test.loc[any_missing].groupby(IDENTIFIER_COLUMN).agg(
        MissingRows=("Date", "size"),
        FirstMissingDate=("Date", "min"),
        LastMissingDate=("Date", "max"),
    )
    product_first = (
        full_history.groupby(IDENTIFIER_COLUMN)["Date"].min()
        if full_history is not None
        else test.groupby(IDENTIFIER_COLUMN)["Date"].min()
    )
    if not product_missing.empty:
        product_missing["ProductFirstTestDate"] = product_missing.index.map(product_first)
        product_missing["MissingNearProductStart"] = (
            product_missing["FirstMissingDate"] - product_missing["ProductFirstTestDate"]
        ).dt.days.le(28)
        rows.append(
            {
                "Feature": "Product-level missingness summary",
                "MissingRows": int(len(product_missing)),
                "MissingPercentOfTest": float(any_missing.mean() * 100),
                "MissingRowsWithOtherMissingFeatures": int(
                    product_missing["MissingNearProductStart"].sum()
                ),
            }
        )
        test_start = test[DATE_COLUMN].min()
        products_starting_in_test = (
            product_first.loc[product_first >= test_start].index
        )
        missing_product_ids = product_missing.index
        rows.append(
            {
                "Feature": "New-product and insufficient-history summary",
                "MissingRows": int(
                    product_missing.loc[
                        product_missing.index.isin(products_starting_in_test),
                        "MissingRows",
                    ].sum()
                ),
                "MissingPercentOfTest": float(
                    product_missing.index.isin(products_starting_in_test).mean() * 100
                ),
                "MissingRowsWithOtherMissingFeatures": int(
                    product_missing["MissingRows"].sum()
                ),
                "ProductsWithMissingFeatures": int(len(missing_product_ids)),
                "ProductsStartingInTest": int(len(products_starting_in_test)),
                "RollingWindowCause": "Rows are missing until the required historical lag/rolling window exists.",
            }
        )
    report = pd.DataFrame(rows)
    report["TestRows"] = len(test)
    report["RowsWithAnyMissingModelFeature"] = int(any_missing.sum())
    report["ProductsWithMissingFeatures"] = int(test.loc[any_missing, IDENTIFIER_COLUMN].nunique())
    report["ProductsStartingInTest"] = int(
        (
            (
                (
                    full_history.groupby(IDENTIFIER_COLUMN)[DATE_COLUMN].min()
                    if full_history is not None
                    else test.groupby(IDENTIFIER_COLUMN)[DATE_COLUMN].min()
                )
                >= test[DATE_COLUMN].min()
            )
        ).sum()
    )
    report["Interpretation"] = (
        "Missing rolling and lag values are retained because historical windows "
        "are unavailable; no imputation is performed."
    )
    MISSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(MISSING_REPORT_PATH, index=False)
    return report


def create_fallback_predictions(
    optimized: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    output = test[KEY_COLUMNS + [TARGET_COLUMN, "DemandLag1", "DemandLag7"]].rename(
        columns={TARGET_COLUMN: "ActualDemand"}
    ).copy()
    output["PredictedDemand"] = optimized["PredictedDemand"].to_numpy()
    output["PredictionMethod"] = np.where(
        output["PredictedDemand"].notna(), "optimized_random_forest", "unavailable"
    )
    seasonal = output["PredictedDemand"].isna() & output["DemandLag7"].notna()
    output.loc[seasonal, "PredictedDemand"] = output.loc[seasonal, "DemandLag7"]
    output.loc[seasonal, "PredictionMethod"] = "seasonal_naive_fallback"
    lag_one = output["PredictedDemand"].isna() & output["DemandLag1"].notna()
    output.loc[lag_one, "PredictedDemand"] = output.loc[lag_one, "DemandLag1"]
    output.loc[lag_one, "PredictionMethod"] = "lag1_fallback"
    output["ForecastError"] = output["ActualDemand"] - output["PredictedDemand"]
    output["AbsoluteError"] = output["ForecastError"].abs()
    valid = output["PredictedDemand"].notna()
    scores = metric_values(
        output.loc[valid, "ActualDemand"], output.loc[valid, "PredictedDemand"]
    )
    summary: dict[str, float | int] = {
        **{
            f"Coverage_{key}": int(value)
            for key, value in output["PredictionMethod"].value_counts().items()
        },
        "CoveredRows": int(valid.sum()),
        "RemainingUnavailable": int((~valid).sum()),
        **{f"Combined{key}": value for key, value in scores.items()},
    }
    FALLBACK_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(FALLBACK_PREDICTIONS_PATH, index=False)
    return output, summary


def select_final_model(
    optimized_model: Pipeline, comparison: pd.DataFrame
) -> tuple[str, Pipeline]:
    original = comparison[comparison["Model"].eq("Random Forest")].iloc[0]
    optimized = comparison[comparison["Model"].eq("Optimized Random Forest")].iloc[0]
    if optimized["MAE"] < original["MAE"] and optimized["WAPE"] < original["WAPE"]:
        return "Optimized Random Forest", optimized_model
    return "Random Forest", joblib.load(ROOT / "models/random_forest_pipeline.joblib")


def save_final_metadata(
    name: str,
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    comparison: pd.DataFrame,
    fallback_summary: dict[str, float | int],
) -> None:
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FINAL_MODEL_PATH)
    parameters = model.named_steps["model"].get_params()
    selected = comparison[comparison["Model"].eq(name)].iloc[0].to_dict()
    metadata = {
        "final_model_name": name,
        "model_type": type(model.named_steps["model"]).__name__,
        "selected_hyperparameters": {
            key: parameters[key]
            for key in [
                "n_estimators", "max_depth", "min_samples_leaf",
                "max_features", "max_samples", "random_state",
            ]
            if key in parameters
        },
        "feature_columns": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "training_date_range": [
            str(train[DATE_COLUMN].min().date()),
            str(train[DATE_COLUMN].max().date()),
        ],
        "test_date_range": [
            str(test[DATE_COLUMN].min().date()),
            str(test[DATE_COLUMN].max().date()),
        ],
        "evaluation_metrics": selected,
        "training_row_count": len(train),
        "test_row_count": len(test),
        "model_version": "phase6-v1",
        "forecasting_assumptions": [
            "Chronological train/test evaluation is preserved.",
            "Only historical lag and rolling features are used.",
            "Unavailable historical features are not imputed.",
            "Fallbacks use DemandLag7, then DemandLag1, without future demand.",
        ],
        "fallback_summary": fallback_summary,
    }
    FINAL_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def validate_final_artifacts(
    predictions: pd.DataFrame, test: pd.DataFrame
) -> None:
    loaded = joblib.load(FINAL_MODEL_PATH)
    if list(loaded.feature_names_in_) != MODEL_FEATURES:
        raise ValueError("Saved final model feature columns do not match.")
    if len(predictions) != len(test):
        raise ValueError("Final prediction row count does not match test data.")
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("Final predictions contain duplicate product-date rows.")
    valid = predictions["PredictedDemand"].notna()
    if predictions.loc[valid, "PredictedDemand"].lt(0).any():
        raise ValueError("Final predictions contain negative values.")
    expected_error = predictions["ActualDemand"] - predictions["PredictedDemand"]
    if not predictions["ForecastError"].equals(expected_error):
        raise ValueError("Final forecast errors are incorrect.")
    sample = test.loc[test[MODEL_FEATURES].notna().all(axis=1)].head(3)
    if not sample.empty and len(loaded.predict(sample[MODEL_FEATURES])) != len(sample):
        raise ValueError("Saved final model cannot generate predictions.")


def run_phase6() -> None:
    train = load_dataset(DEFAULT_TRAIN_PATH)
    test = load_dataset(DEFAULT_TEST_PATH)
    validate_time_order(train, test)
    train_complete, _ = select_model_data(train)
    audit = audit_before_optimization(train)
    parameters, optimization_results = optimize_random_forest(train_complete)
    optimized_model = build_random_forest(parameters)
    optimized_model.fit(train_complete[MODEL_FEATURES], train_complete[TARGET_COLUMN])
    optimized_predictions = predict_with_model(optimized_model, test)
    save_optimized(optimized_model, optimized_predictions)
    final_comparison = create_final_comparison(
        optimized_predictions, len(train_complete)
    )
    create_error_reports(optimized_predictions, test)
    analyze_missing_predictions(
        test, pd.concat([train, test], ignore_index=True)
    )
    fallback_predictions, fallback_summary = create_fallback_predictions(
        optimized_predictions, test
    )
    final_name, final_model = select_final_model(
        optimized_model, final_comparison
    )
    save_final_metadata(
        final_name, final_model, train, test, final_comparison, fallback_summary
    )
    validate_final_artifacts(optimized_predictions, test)
    print("Best model before optimization:", audit.sort_values("MAE").iloc[0]["Model"])
    print("Selected hyperparameters:", parameters)
    print("\nOptimization validation results:")
    print(optimization_results.to_string(index=False))
    print("\nFinal model comparison:")
    print(final_comparison.to_string(index=False))
    print("\nFallback coverage:", fallback_summary)
    print("Final model selected:", final_name)
    print("Generated fallback rows:", len(fallback_predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 optimization.")
    parser.parse_args()
    try:
        run_phase6()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
