"""Recursive future demand forecasting using the persisted model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backend.services.model_service import ModelService


class FutureForecastError(ValueError):
    """Raised when a safe recursive forecast cannot be produced."""


@dataclass(frozen=True)
class FuturePrediction:
    stock_code: str
    target_date: pd.Timestamp
    predicted_demand: float
    historical_cutoff: pd.Timestamp
    forecast_steps: int


class FutureForecastService:
    """Build leakage-safe recursive forecasts from cached calendar demand."""

    MAX_FORECAST_HORIZON_DAYS = 365

    REQUIRED_FEATURES = [
        "StockCode",
        "Year",
        "Month",
        "Day",
        "WeekdayNumber",
        "IsWeekend",
        "DemandLag1",
        "DemandLag7",
        "DemandLag14",
        "DemandLag28",
        "DemandRollingMean7",
        "DemandRollingMean14",
        "DemandRollingMean28",
        "DemandTrend7_28",
    ]

    def __init__(self, calendar_path: Path, model_service: ModelService) -> None:
        self.calendar_path = calendar_path
        self.model_service = model_service
        self._calendar: pd.DataFrame | None = None
        self._readiness: dict[str, dict[str, Any]] | None = None

    def _load_calendar(self) -> pd.DataFrame:
        if self._calendar is None:
            if not self.calendar_path.is_file():
                raise FutureForecastError(
                    f"Historical demand dataset was not found: {self.calendar_path}"
                )
            frame = pd.read_csv(
                self.calendar_path,
                dtype={"StockCode": "string"},
                parse_dates=["Date"],
                usecols=["StockCode", "Date", "DailyQuantityDemand"],
            )
            if frame[["StockCode", "Date"]].duplicated().any():
                raise FutureForecastError(
                    "Historical demand contains duplicate product-date rows."
                )
            frame["DailyQuantityDemand"] = pd.to_numeric(
                frame["DailyQuantityDemand"], errors="coerce"
            )
            if frame["DailyQuantityDemand"].isna().any():
                raise FutureForecastError("Historical demand contains invalid values.")
            if frame["DailyQuantityDemand"].lt(0).any():
                raise FutureForecastError("Historical demand contains negative values.")
            self._calendar = frame.sort_values(["StockCode", "Date"])
        return self._calendar

    @property
    def historical_cutoff(self) -> pd.Timestamp:
        return self._load_calendar()["Date"].max()

    def product_readiness(self, stock_code: str) -> dict[str, Any]:
        """Explain whether a product can be used for recursive forecasting."""
        if self._readiness is None:
            self._readiness = {}
            frame = self._load_calendar()
            cutoff = self.historical_cutoff
            summary = frame.groupby("StockCode", sort=False).agg(
                history_days=("Date", "size"),
                first_date=("Date", "min"),
                last_date=("Date", "max"),
            )
            for code, row in summary.iterrows():
                days = int(row["history_days"])
                last_date = pd.Timestamp(row["last_date"])
                consecutive = (
                    last_date - pd.Timestamp(row["first_date"])
                ).days + 1 == days
                if last_date != cutoff:
                    hint = (
                        f"History ends on {last_date.strftime('%Y-%m-%d')}; "
                        f"it must reach {cutoff.strftime('%Y-%m-%d')}."
                    )
                    ready = False
                elif days < 29:
                    hint = "At least 29 consecutive historical days are required."
                    ready = False
                elif not consecutive:
                    hint = "Historical calendar days are not consecutive."
                    ready = False
                else:
                    hint = "Ready. Choose a date after the historical cutoff."
                    ready = True
                self._readiness[str(code)] = {
                    "forecast_ready": ready,
                    "history_days": days,
                    "historical_cutoff": last_date.strftime("%Y-%m-%d"),
                    "readiness_hint": hint,
                }
        return self._readiness.get(
            stock_code,
            {
                "forecast_ready": False,
                "history_days": 0,
                "historical_cutoff": None,
                "readiness_hint": "No daily demand history is available.",
            },
        )

    def _historical_sequence(self, stock_code: str) -> list[float]:
        frame = self._load_calendar()
        selected = frame[frame["StockCode"] == stock_code]
        if selected.empty:
            raise FutureForecastError(f"StockCode '{stock_code}' has no historical demand.")
        if selected["Date"].max() != self.historical_cutoff:
            raise FutureForecastError(
                f"StockCode '{stock_code}' has history only through "
                f"{selected['Date'].max().date()}; history through the global "
                f"cutoff {self.historical_cutoff.date()} is required."
            )
        dates = selected["Date"].sort_values()
        if len(dates) < 29:
            raise FutureForecastError(
                f"StockCode '{stock_code}' needs at least 29 historical days."
            )
        if dates.diff().dropna().ne(pd.Timedelta(days=1)).any():
            raise FutureForecastError(
                f"StockCode '{stock_code}' does not have consecutive historical dates."
            )
        return selected.sort_values("Date")["DailyQuantityDemand"].astype(float).tolist()

    @staticmethod
    def _features(stock_code: str, date: pd.Timestamp, sequence: list[float]) -> dict[str, Any]:
        if len(sequence) < 28:
            raise FutureForecastError("At least 28 previous daily values are required.")
        lag1 = sequence[-1]
        lag7 = sequence[-7]
        lag14 = sequence[-14]
        lag28 = sequence[-28]
        mean7 = sum(sequence[-7:]) / 7
        mean14 = sum(sequence[-14:]) / 14
        mean28 = sum(sequence[-28:]) / 28
        return {
            "StockCode": stock_code,
            "Year": int(date.year),
            "Month": int(date.month),
            "Day": int(date.day),
            "WeekdayNumber": int(date.day_of_week),
            "IsWeekend": bool(date.day_of_week >= 5),
            "DemandLag1": lag1,
            "DemandLag7": lag7,
            "DemandLag14": lag14,
            "DemandLag28": lag28,
            "DemandRollingMean7": mean7,
            "DemandRollingMean14": mean14,
            "DemandRollingMean28": mean28,
            "DemandTrend7_28": mean7 - mean28,
        }

    def forecast_range(
        self, stock_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[FuturePrediction]:
        if not self.model_service.is_loaded:
            raise FutureForecastError("Forecasting model is unavailable.")
        if start_date > end_date:
            raise FutureForecastError("start_date must be on or before end_date.")
        cutoff = self.historical_cutoff
        if start_date <= cutoff:
            raise FutureForecastError(
                f"Forecast dates must be after the historical cutoff {cutoff.date()}."
            )
        horizon = (end_date - cutoff).days
        if horizon > self.MAX_FORECAST_HORIZON_DAYS:
            raise FutureForecastError(
                "The requested forecast is too far from the historical cutoff. "
                f"Choose a date within {self.MAX_FORECAST_HORIZON_DAYS} days of "
                f"{cutoff.date()}."
            )
        sequence = self._historical_sequence(stock_code)
        predictions: list[FuturePrediction] = []
        current = start_date
        steps_before_start = (start_date - cutoff).days
        for step in range(1, steps_before_start):
            date = cutoff + pd.Timedelta(days=step)
            features = self._features(stock_code, date, sequence)
            predicted = self.model_service.predict(
                pd.DataFrame([features], columns=self.REQUIRED_FEATURES)
            )[0]
            sequence.append(max(0.0, predicted))
        while current <= end_date:
            features = self._features(stock_code, current, sequence)
            predicted = max(
                0.0,
                self.model_service.predict(
                    pd.DataFrame([features], columns=self.REQUIRED_FEATURES)
                )[0],
            )
            sequence.append(predicted)
            predictions.append(
                FuturePrediction(
                    stock_code=stock_code,
                    target_date=current,
                    predicted_demand=predicted,
                    historical_cutoff=cutoff,
                    forecast_steps=(current - cutoff).days,
                )
            )
            current += pd.Timedelta(days=1)
        return predictions

    def forecast(self, stock_code: str, target_date: pd.Timestamp) -> FuturePrediction:
        return self.forecast_range(stock_code, target_date, target_date)[0]
