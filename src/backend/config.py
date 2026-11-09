"""Backend configuration and project paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "final_demand_forecasting_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "final_model_metadata.json"
PRODUCT_DATA_PATH = PROJECT_ROOT / "data" / "features" / "final_modeling_dataset.csv"
SALES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sales_transactions.csv"
API_VERSION = "1.0.0"
