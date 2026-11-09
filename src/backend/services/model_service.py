"""Load and expose the persisted forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class ModelService:
    """Own one loaded model and its verified metadata for an app lifetime."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("final_model_name", "Unknown"))

    @property
    def feature_columns(self) -> list[str]:
        if self.model is not None and hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        return list(self.metadata.get("feature_columns", []))

    def _load(self) -> None:
        try:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model file was not found: {self.model_path}")
            if not self.metadata_path.is_file():
                raise FileNotFoundError(
                    f"Model metadata was not found: {self.metadata_path}"
                )
            self.model = joblib.load(self.model_path)
            self.metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            metadata_features = self.metadata.get("feature_columns", [])
            if list(metadata_features) != self.feature_columns:
                raise ValueError("Model features do not match model metadata.")
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            EOFError,
            ImportError,
            AttributeError,
        ) as error:
            self.model = None
            self.error = str(error)

    def predict(self, rows: Any) -> list[float]:
        if self.model is None:
            raise RuntimeError(self.error or "Forecasting model is unavailable.")
        predictions = self.model.predict(rows)
        return [max(0.0, float(value)) for value in predictions]
