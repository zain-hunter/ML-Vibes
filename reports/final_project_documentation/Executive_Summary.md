# Executive Summary

## Project

Smart Retail Intelligence & Demand Forecasting System analyzes the Online Retail
II transaction dataset and provides product-level daily demand forecasting through
a saved Random Forest model, a Flask API, and a responsive browser dashboard.

## Delivered capabilities

- Raw CSV/Excel inspection without modifying source data.
- Audited transaction cleaning with explicit quality flags.
- Separate normal sales from returns and cancellations.
- Revenue calculation as `Quantity * Price`.
- Exploratory sales, product, customer, country, and return analysis.
- Product-calendar demand aggregation and leakage-safe time features.
- Seasonal-naive, Random Forest, Gradient Boosting, and optimized Random Forest
  experiments.
- Persisted final Random Forest pipeline and metadata.
- Flask health, product, single-row, batch, analytics, and recursive forecast APIs.
- Dashboard pages for overview analytics, product analytics, historical model
  demonstrations, and experimental recursive future forecasts.

## Verified result

The final Random Forest was selected because it outperformed the seasonal-naive
and Gradient Boosting comparisons on the common 209,641-row test set. Its
recorded metrics are MAE 8.1492, RMSE 60.8034, and WAPE 102.0622%.

The current automated backend/dashboard suite contains 24 tests and passed in
the audit run. Recursive forecasts are experimental and limited to 365 days
after the historical cutoff for responsiveness.

