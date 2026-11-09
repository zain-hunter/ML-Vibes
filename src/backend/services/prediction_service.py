"""Prediction orchestration for validated API rows."""

from __future__ import annotations

import pandas as pd

from src.backend.services.model_service import ModelService


class PredictionService:
    """Validate feature-shaped rows and call the loaded model."""

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service

    def predict(self, rows: list[dict[str, object]]) -> list[float]:
        dataframe = pd.DataFrame(rows, columns=self.model_service.feature_columns)
        return self.model_service.predict(dataframe)
