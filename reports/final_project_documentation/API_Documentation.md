# API Documentation

Base URL: `http://127.0.0.1:5000`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | API and model status |
| GET | `/api/products` | String product codes |
| GET | `/api/product-catalog` | Product codes and descriptions |
| GET | `/api/future-forecast-products` | Forecast readiness and hints |
| POST | `/api/forecast` | One engineered-row prediction |
| POST | `/api/forecast/batch` | Ordered batch predictions |
| POST | `/api/forecast/future` | One recursive future prediction |
| POST | `/api/forecast/future/range` | Recursive prediction for every date in a range |
| GET | `/api/analytics/summary` | KPI and monthly sales data |
| GET | `/api/analytics/top-products` | Top products by revenue and quantity |
| GET | `/api/analytics/countries` | Country revenue |
| GET | `/api/analytics/product/<stock_code>` | Product history and totals |
| GET | `/api/forecast-rows/<stock_code>` | Historical rows with complete model features |

## Historical prediction

`POST /api/forecast` requires exactly the 14 model features: `StockCode`,
`Year`, `Month`, `Day`, `WeekdayNumber`, `IsWeekend`, four lags, three rolling
means, and `DemandTrend7_28`. Missing, non-numeric, or negative demand inputs
are rejected. Predictions are returned as `predicted_demand`.

## Future prediction

```json
{
  "StockCode": "10002",
  "target_date": "2010-12-10"
}
```

The response includes `predicted_daily_demand`, `historical_cutoff`,
`forecast_steps`, `forecast_method`, and an experimental warning. The range
endpoint accepts `StockCode`, `start_date`, and `end_date` and returns one
record per date.

HTTP behavior includes 200 for success, 400 for invalid input or unsupported
forecast conditions, 404 for unknown products, and 503 when the model is
unavailable. Internal stack traces are not exposed.

