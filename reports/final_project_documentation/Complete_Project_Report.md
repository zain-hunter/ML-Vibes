# Complete Project Report

## 1. Executive conclusion

Smart Retail Intelligence & Demand Forecasting System is an end-to-end
educational retail analytics application. It transforms Online Retail II
transactions into auditable sales data, product-day demand features, evaluated
machine-learning forecasts, and a Flask/Bootstrap/Plotly dashboard.

The implemented system is useful for inventory and sales decision support, but
its predictions are not guaranteed sales and should not be interpreted as
profit or real-time stock availability.

## 2. Dataset and data dictionary

The raw workbook is `data/raw/dataset.xlsx`. It contains 525,461 rows and the
following fields:

| Column | Meaning |
|---|---|
| Invoice | Invoice or cancellation identifier |
| StockCode | Product identifier |
| Description | Product description |
| Quantity | Transaction units; negative values represent returns |
| InvoiceDate | Transaction timestamp |
| Price | Unit price |
| Customer ID | Customer identifier, sometimes missing |
| Country | Customer country |

The audit found 107,927 missing customer IDs, 2,928 missing descriptions,
6,865 exact duplicates, 12,326 negative quantities, 10,206 cancelled invoices,
3 negative prices, and 3,687 zero prices. The observed unique counts recorded
during the data-understanding phase were 28,816 invoices, 4,632 products, and
4,383 customers. Product identifiers are explicitly loaded as strings to avoid
numeric/string identity drift.

## 3. Cleaning and EDA

The cleaning module preserves the raw file and writes a master audit dataset,
sales dataset, returns/cancellations dataset, and summary CSV. It does not
invent customer IDs or silently remove duplicates. Revenue is calculated as
`Quantity * Price`; absolute revenue is also retained for audit use.

EDA reads the prepared sales data and produces monthly, product, country,
customer, weekday, and return/cancellation analyses plus charts. Sales and
returns remain distinguishable, and December 2010 is a partial month.

## 4. Feature engineering

The feature pipeline aggregates daily product demand, builds active-window
calendar rows, adds calendar fields, creates grouped lags, and calculates
leakage-safe rolling means and trend. Chronological splitting keeps the future
test period after training. The current final modeling file has 930,130 rows.

## 5. Machine learning

The implemented comparisons include a seasonal-naive `DemandLag7` baseline,
Random Forest, Gradient Boosting, and a limited chronological Random Forest
optimization. The original Random Forest was retained as final because the
optimized variant did not improve the untouched test period.

The model metadata records 822,648 training observations, 209,641 common
evaluation observations, the 14 exact input features, and the target
`DailyQuantityDemand`. A fallback prediction artifact also records seasonal
naive and lag-1 fallback coverage for rows unavailable to the main model.

## 6. Backend and API

The Flask backend loads the final joblib model once through `ModelService`.
Validators enforce exact feature names, data types, valid product codes, and
non-negative lag/rolling demand values. Analytics data is cached to avoid
reading large CSVs for every request. See `API_Documentation.md` for all
routes and payloads.

## 7. Dashboard

The dashboard has an overview page with dynamic KPIs and Plotly charts, a
product analytics page with search and daily trends, and a forecasting page.
The forecasting page supports historical feature-row prediction and a separate
experimental future interface. The future product selector now supplies
readiness hints and disables products whose history cannot satisfy recursive
forecast requirements.

## 8. Recursive future forecasting

For a product with consecutive history through 2010-12-09, the service creates
calendar features for each requested day, derives lags and rolling means from
the historical sequence, predicts with the saved model, appends the prediction,
and repeats. It does not use unknown future actual demand. Intermediate dates
are generated even when the requested range begins later. A 365-day horizon
limit prevents an apparently frozen browser caused by thousands of model calls.

## 9. Testing and evidence

The audit executed the focused backend/dashboard suite and obtained 24 passing
tests. It also imported the Flask app successfully and smoke-tested the future
single-date and range endpoints. No browser screenshot was available in the
shared attachment, so no visual claims are made from screenshots.

## 10. Limitations and future work

The source dataset is historical and ends in 2010. The forecast target is
observed demand, not unconstrained customer demand; inventory availability is
not present. Missing customer IDs and returns affect interpretation. Recursive
errors can accumulate, particularly at longer horizons. The model requires
engineered history and cannot forecast a product from StockCode alone.

Recommended next steps are production WSGI deployment, CI, model/data versioning,
forecast monitoring, newer data, inventory-aware demand modeling, confidence
intervals, and broader data-pipeline tests.

