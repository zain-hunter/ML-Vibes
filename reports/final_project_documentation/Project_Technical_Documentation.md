# Project Technical Documentation

## Architecture

```text
Raw Excel/CSV
  -> data understanding
  -> data cleaning and quality flags
  -> sales / returns outputs
  -> EDA
  -> product-day calendar
  -> date, lag, rolling, trend features
  -> chronological train/test split
  -> model pipelines and evaluation artifacts
  -> Flask services and dashboard
```

## Cleaning behavior

`src/data/clean_data.py` preserves every raw row in
`transactions_clean.csv`, standardizes types, flags missing IDs/descriptions,
duplicates, returns, cancellations, negative prices, and zero prices, and adds
signed and absolute revenue. Normal sales require a non-cancelled invoice,
positive quantity, and non-negative price. Returns and cancellations are
retained in a separate audit output; they are not silently deleted.

## Features

For each product and calendar date:

- `DemandLag1`, `DemandLag7`, `DemandLag14`, and `DemandLag28` are previous
  calendar-day demand values.
- Rolling means use the preceding 7, 14, or 28 values.
- `DemandTrend7_28 = DemandRollingMean7 - DemandRollingMean28`.
- `shift(1)` is applied before rolling so the current target cannot enter its
  own features.
- Year, month, day, weekday, weekend, and string `StockCode` are also used.

Initial unavailable windows remain missing. The modeling and evaluation code
does not invent missing historical values.

## Backend

`src/backend/app.py` creates the Flask application. `ModelService` loads the
saved joblib pipeline once and verifies its metadata feature list.
`PredictionService` builds feature-ordered DataFrames. `AnalyticsService` caches
prepared sales/modeling data. `FutureForecastService` caches the daily calendar
and recursively appends only prior predictions for future dates.

Recursive forecasting requires a product with consecutive history through the
global cutoff, at least 29 historical days, and a requested date after the
cutoff. Requests beyond 365 days are rejected explicitly.

## Frontend

Flask templates use Bootstrap 5 and Plotly.js from CDNs. JavaScript calls the
read-only analytics and forecasting endpoints, shows loading/error states, and
renders KPI and time-series charts. The future product selector includes
readiness hints and disables products that cannot safely support recursion.

## Operational commands

```powershell
.\venv\Scripts\python.exe -m src.backend.app
.\venv\Scripts\python.exe -m pytest .\tests\test_backend.py -q
```

