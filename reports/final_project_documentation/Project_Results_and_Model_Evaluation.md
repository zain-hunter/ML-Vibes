# Project Results and Model Evaluation

## Dataset audit

| Measure | Result |
|---|---:|
| Raw rows | 525,461 |
| Columns | 8 |
| Missing Customer IDs | 107,927 |
| Missing descriptions | 2,928 |
| Exact duplicate rows | 6,865 |
| Negative quantities | 12,326 |
| Cancelled invoices | 10,206 |
| Negative prices | 3 |
| Zero prices | 3,687 |
| Sales rows | 513,131 |
| Returns/cancellations rows | 12,327 |
| Input/master revenue | 9,539,484.63 |
| Normal sales revenue | 10,305,892.02 |
| Returns/cancellations revenue | -629,855.37 |

The figures are read from `data/processed/data_cleaning_summary.csv`.

## Modeling data

- Target: `DailyQuantityDemand`.
- Final modeling dataset: 930,130 rows.
- Training rows: 822,648, through 2010-09-25.
- Test rows: 215,239, from 2010-09-26 through 2010-12-09.
- Common evaluated observations: 209,641.
- Products are represented as string identifiers.

## Model comparison

| Model | MAE | RMSE | WAPE | Common observations |
|---|---:|---:|---:|---:|
| Seasonal Naive | 10.3211 | 83.6990 | 129.2642% | 209,641 |
| Random Forest | 8.1492 | 60.8034 | 102.0622% | 209,641 |
| Gradient Boosting | 8.4004 | 61.9711 | 105.2083% | 209,641 |
| Optimized Random Forest | 8.1715 | 60.8110 | 102.3420% | 209,641 |

The original Random Forest remained the final model because the optimized
configuration was slightly worse on the untouched test period. The saved
pipeline is `models/final_demand_forecasting_pipeline.joblib`; metadata is in
`models/final_model_metadata.json`.

## Final model

The persisted model is a `RandomForestRegressor` with 100 estimators, maximum
depth 15, minimum leaf size 5, `max_features=1.0`, `max_samples=0.5`, and
random state 42. The pipeline uses one-hot handling for `StockCode` and the
engineered numeric/calendar features.

## Interpretation and limitations

MAE measures average absolute unit error. RMSE penalizes large errors more
strongly. WAPE measures absolute error relative to total actual demand. The
WAPE above 100% demonstrates that the task is difficult and that predictions
should support decisions rather than be treated as exact commitments.

