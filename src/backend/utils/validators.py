"""Request validation for engineered forecasting features."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


CALENDAR_RANGES = {
    "Month": (1, 12),
    "Day": (1, 31),
    "WeekdayNumber": (0, 6),
}
NONNEGATIVE_FEATURES = {
    "DemandLag1",
    "DemandLag7",
    "DemandLag14",
    "DemandLag28",
    "DemandRollingMean7",
    "DemandRollingMean14",
    "DemandRollingMean28",
}


def validate_row(
    row: Any, required_features: list[str], available_products: set[str] | None = None
) -> list[str]:
    """Return clear validation errors for one engineered feature row."""
    if not isinstance(row, dict):
        return ["row must be a JSON object"]
    expected = set(required_features)
    supplied = set(row)
    errors: list[str] = []
    missing = expected - supplied
    extra = supplied - expected
    if missing:
        errors.append(f"missing required features: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"unexpected features: {', '.join(sorted(extra))}")
    if errors:
        return errors

    stock_code = row["StockCode"]
    if not isinstance(stock_code, str) or not stock_code.strip():
        errors.append("StockCode must be a non-empty string")
    elif available_products is not None and stock_code not in available_products:
        errors.append(f"StockCode '{stock_code}' is not available")

    for feature in required_features:
        if feature == "StockCode":
            continue
        value = row[feature]
        if feature == "IsWeekend":
            if not isinstance(value, bool):
                errors.append("IsWeekend must be a boolean")
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"{feature} must be numeric")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"{feature} must be finite")

    for feature, (minimum, maximum) in CALENDAR_RANGES.items():
        if feature in row and isinstance(row[feature], Real) and not isinstance(row[feature], bool):
            if not minimum <= row[feature] <= maximum:
                errors.append(f"{feature} must be between {minimum} and {maximum}")
    if "Year" in row and isinstance(row["Year"], Real) and not isinstance(row["Year"], bool):
        if row["Year"] < 1900:
            errors.append("Year must be a valid calendar year")

    for feature in NONNEGATIVE_FEATURES:
        if feature in row and isinstance(row[feature], Real) and not isinstance(row[feature], bool):
            if row[feature] < 0:
                errors.append(f"{feature} cannot be negative")
    return errors
