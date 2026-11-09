"""Focused Flask backend tests using the persisted final model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backend.app import create_app, load_products
from src.backend.config import METADATA_PATH, MODEL_PATH
from src.backend.services.model_service import ModelService


@pytest.fixture()
def client():
    app = create_app(
        model_service=ModelService(MODEL_PATH, METADATA_PATH),
        product_codes=set(load_products()),
    )
    app.config.update(TESTING=True)
    return app.test_client()


def valid_row() -> dict[str, object]:
    return {
        "StockCode": "10002",
        "Year": 2010,
        "Month": 11,
        "Day": 15,
        "WeekdayNumber": 0,
        "IsWeekend": False,
        "DemandLag1": 12.0,
        "DemandLag7": 8.0,
        "DemandLag14": 10.0,
        "DemandLag28": 9.0,
        "DemandRollingMean7": 11.2,
        "DemandRollingMean14": 10.5,
        "DemandRollingMean28": 9.8,
        "DemandTrend7_28": 1.4,
    }


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["model_loaded"] is True
    assert response.json["model_name"] == "Random Forest"


def test_products_endpoint_returns_strings(client):
    response = client.get("/api/products")
    assert response.status_code == 200
    assert response.json["count"] > 0
    assert all(isinstance(code, str) for code in response.json["products"])


def test_forecast_endpoint_returns_nonnegative_prediction(client):
    response = client.post("/api/forecast", json=valid_row())
    assert response.status_code == 200
    assert response.json["predicted_demand"] >= 0


def test_forecast_allows_negative_trend(client):
    row = valid_row()
    row["DemandTrend7_28"] = -1.4
    response = client.post("/api/forecast", json=row)
    assert response.status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"DemandLag7": None},
        {"DemandLag7": "not numeric"},
        {"DemandLag7": -1},
    ],
)
def test_forecast_rejects_invalid_features(client, change):
    row = valid_row()
    row.update(change)
    response = client.post("/api/forecast", json=row)
    assert response.status_code == 400


def test_forecast_rejects_missing_feature(client):
    row = valid_row()
    del row["DemandLag28"]
    response = client.post("/api/forecast", json=row)
    assert response.status_code == 400


def test_batch_preserves_order_and_reports_invalid_rows(client):
    invalid = valid_row()
    invalid["DemandLag1"] = "bad"
    response = client.post("/api/forecast/batch", json=[valid_row(), invalid])
    assert response.status_code == 200
    assert response.json["total"] == 2
    assert response.json["successful"] == 1
    assert response.json["failed"] == 1
    assert response.json["results"][0]["success"] is True
    assert response.json["results"][1]["success"] is False


def test_missing_model_returns_service_unavailable(tmp_path: Path):
    service = ModelService(tmp_path / "missing.joblib", tmp_path / "missing.json")
    app = create_app(model_service=service, product_codes={"10002"})
    response = app.test_client().get("/api/health")
    assert response.status_code == 503
    assert response.json["model_loaded"] is False


def test_dashboard_pages_load(client):
    for path in ("/", "/products", "/forecasting"):
        response = client.get(path)
        assert response.status_code == 200
        assert b"Smart Retail Intelligence" in response.data


def test_analytics_endpoints_return_data(client):
    summary = client.get("/api/analytics/summary")
    assert summary.status_code == 200
    assert summary.json["total_revenue"] > 0
    assert summary.json["monthly_sales"]

    top_products = client.get("/api/analytics/top-products")
    assert top_products.status_code == 200
    assert len(top_products.json["by_revenue"]) == 10

    countries = client.get("/api/analytics/countries")
    assert countries.status_code == 200
    assert countries.json["countries"]

    product = client.get("/api/analytics/product/10002")
    assert product.status_code == 200
    assert product.json["stock_code"] == "10002"
    assert product.json["daily"]


def test_forecast_rows_endpoint_returns_engineered_rows(client):
    response = client.get("/api/forecast-rows/10002")
    assert response.status_code == 200
    assert response.json["rows"]
    assert "DailyQuantityDemand" not in response.json["rows"][0]


def test_unknown_product_returns_not_found(client):
    assert client.get("/api/analytics/product/not-a-product").status_code == 404
    assert client.get("/api/forecast-rows/not-a-product").status_code == 404
