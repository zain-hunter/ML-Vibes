"""Focused Flask backend tests using the persisted final model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backend.app import create_app, load_products
from src.backend.config import CALENDAR_DATA_PATH, METADATA_PATH, MODEL_PATH
from src.backend.services.future_forecast_service import FutureForecastService
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


def test_product_catalog_returns_codes_and_descriptions(client):
    response = client.get("/api/product-catalog")
    assert response.status_code == 200
    assert response.json["count"] > 0
    assert {"StockCode", "Description"} <= set(response.json["products"][0])


def test_future_forecast_catalog_explains_product_readiness(client):
    response = client.get("/api/future-forecast-products")
    assert response.status_code == 200
    assert response.json["count"] > 0
    assert response.json["ready_count"] > 0
    product = response.json["products"][0]
    assert {"forecast_ready", "readiness_hint", "history_days"} <= set(product)


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


def test_forecasting_page_includes_prediction_chart(client):
    response = client.get("/forecasting")
    assert response.status_code == 200
    assert b"forecast-chart" in response.data
    assert b"historical lag inputs" in response.data


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


def test_future_forecast_endpoint_returns_recursive_prediction(client):
    response = client.post(
        "/api/forecast/future",
        json={"StockCode": "10002", "target_date": "2010-12-10"},
    )
    assert response.status_code == 200
    assert response.json["forecast_steps"] == 1
    assert response.json["predicted_daily_demand"] >= 0


def test_future_range_endpoint_returns_every_day(client):
    response = client.post(
        "/api/forecast/future/range",
        json={
            "StockCode": "10002",
            "start_date": "2010-12-10",
            "end_date": "2010-12-12",
        },
    )
    assert response.status_code == 200
    assert [row["date"] for row in response.json["predictions"]] == [
        "2010-12-10",
        "2010-12-11",
        "2010-12-12",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"StockCode": "10002", "target_date": "not-a-date"},
        {"StockCode": "10002", "target_date": "2010-12-09"},
        {"StockCode": "not-a-product", "target_date": "2010-12-10"},
    ],
)
def test_future_forecast_rejects_invalid_requests(client, payload):
    response = client.post("/api/forecast/future", json=payload)
    expected_status = 404 if payload["StockCode"] == "not-a-product" else 400
    assert response.status_code == expected_status


def test_future_forecast_rejects_excessive_horizon(client):
    response = client.post(
        "/api/forecast/future",
        json={"StockCode": "10002", "target_date": "2011-12-10"},
    )
    assert response.status_code == 400
    assert "too far" in response.json["error"]


def test_recursive_features_use_previous_prediction_only(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    calendar = pd.DataFrame(
        {
            "StockCode": ["X"] * 30,
            "Date": dates,
            "DailyQuantityDemand": list(range(30)),
        }
    )
    calendar_path = tmp_path / "calendar.csv"
    calendar.to_csv(calendar_path, index=False)

    class FakeModel:
        is_loaded = True

        def __init__(self):
            self.rows = []

        def predict(self, frame):
            self.rows.append(frame.iloc[0].to_dict())
            return [100.0]

    model = FakeModel()
    service = FutureForecastService(calendar_path, model)
    result = service.forecast("X", pd.Timestamp("2020-01-31"))

    assert result.predicted_demand == 100.0
    assert model.rows[0]["DemandLag1"] == 29.0
    assert model.rows[0]["DemandLag7"] == 23.0
    assert model.rows[0]["DemandRollingMean7"] == pytest.approx(26.0)
    assert model.rows[0]["DemandRollingMean28"] == pytest.approx(15.5)
    assert model.rows[0]["DemandTrend7_28"] == pytest.approx(10.5)
