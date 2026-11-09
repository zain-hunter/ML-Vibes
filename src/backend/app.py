"""Flask application factory for the demand forecasting API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.backend.config import (
    API_VERSION,
    METADATA_PATH,
    MODEL_PATH,
    PRODUCT_DATA_PATH,
    SALES_DATA_PATH,
)
from src.backend.services.analytics_service import AnalyticsService
from src.backend.services.model_service import ModelService
from src.backend.services.prediction_service import PredictionService
from src.backend.utils.validators import validate_row


@lru_cache(maxsize=1)
def load_products() -> tuple[str, ...]:
    """Load unique product identifiers once from the prepared dataset."""
    if not PRODUCT_DATA_PATH.is_file():
        raise FileNotFoundError(f"Product dataset was not found: {PRODUCT_DATA_PATH}")
    frame = pd.read_csv(PRODUCT_DATA_PATH, usecols=["StockCode"], dtype="string")
    return tuple(sorted(frame["StockCode"].dropna().unique().tolist()))


def error_response(message: str, status: int):
    return jsonify({"error": message, "status": status}), status


def create_app(
    model_service: ModelService | None = None,
    product_codes: set[str] | None = None,
    analytics_service: AnalyticsService | None = None,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    service = model_service or ModelService(MODEL_PATH, METADATA_PATH)
    prediction_service = PredictionService(service)
    analytics = analytics_service or AnalyticsService(SALES_DATA_PATH, PRODUCT_DATA_PATH)

    def products() -> set[str]:
        return product_codes if product_codes is not None else set(load_products())

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/products")
    def products_page():
        return render_template("products.html")

    @app.get("/forecasting")
    def forecasting_page():
        return render_template("forecasting.html", feature_columns=service.feature_columns)

    @app.get("/api/health")
    def health():
        payload = {
            "api_status": "ok",
            "model_loaded": service.is_loaded,
            "model_name": service.model_name,
            "api_version": API_VERSION,
        }
        if not service.is_loaded:
            payload["error"] = service.error or "model unavailable"
            return jsonify(payload), 503
        return jsonify(payload), 200

    @app.get("/api/products")
    def product_list():
        try:
            values = sorted(products())
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)
        return jsonify({"products": values, "count": len(values)}), 200

    @app.post("/api/forecast")
    def forecast():
        if not service.is_loaded:
            return error_response("forecasting model is unavailable", 503)
        payload: Any = request.get_json(silent=True)
        errors = validate_row(payload, service.feature_columns, products())
        if errors:
            return jsonify({"error": "invalid forecast request", "details": errors}), 400
        try:
            prediction = prediction_service.predict([payload])[0]
        except (ValueError, TypeError, RuntimeError) as error:
            return error_response(f"prediction failed: {error}", 500)
        return jsonify(
            {"StockCode": payload["StockCode"], "predicted_demand": prediction}
        ), 200

    @app.post("/api/forecast/batch")
    def forecast_batch():
        if not service.is_loaded:
            return error_response("forecasting model is unavailable", 503)
        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, list):
            return error_response("batch request must be a JSON array", 400)
        known_products = products()
        results: list[dict[str, Any]] = []
        valid_rows: list[dict[str, object]] = []
        valid_indices: list[int] = []
        for index, row in enumerate(payload):
            errors = validate_row(row, service.feature_columns, known_products)
            if errors:
                results.append({"index": index, "success": False, "errors": errors})
            else:
                valid_rows.append(row)
                valid_indices.append(index)
                results.append({"index": index, "success": True})
        try:
            predictions = prediction_service.predict(valid_rows) if valid_rows else []
        except (ValueError, TypeError, RuntimeError) as error:
            return error_response(f"batch prediction failed: {error}", 500)
        for index, prediction in zip(valid_indices, predictions):
            results[index]["prediction"] = prediction
        return jsonify(
            {
                "total": len(payload),
                "successful": len(valid_rows),
                "failed": len(payload) - len(valid_rows),
                "results": results,
            }
        ), 200

    @app.get("/api/analytics/summary")
    def analytics_summary():
        try:
            return jsonify(analytics.summary()), 200
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)

    @app.get("/api/analytics/top-products")
    def analytics_top_products():
        try:
            return jsonify(analytics.top_products()), 200
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)

    @app.get("/api/analytics/countries")
    def analytics_countries():
        try:
            return jsonify({"countries": analytics.countries()}), 200
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)

    @app.get("/api/analytics/product/<stock_code>")
    def analytics_product(stock_code: str):
        if stock_code not in products():
            return error_response("product was not found", 404)
        try:
            result = analytics.product(stock_code)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)
        return jsonify(result), 200

    @app.get("/api/forecast-rows/<stock_code>")
    def forecast_rows(stock_code: str):
        if stock_code not in products():
            return error_response("product was not found", 404)
        try:
            return jsonify({"rows": analytics.forecast_rows(stock_code)}), 200
        except (OSError, ValueError, pd.errors.ParserError) as error:
            return error_response(str(error), 500)

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
