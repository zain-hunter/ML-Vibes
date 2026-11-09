"""Cached read-only analytics for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class AnalyticsService:
    """Load prepared sales and modeling data once and expose dashboard views."""

    def __init__(self, sales_path: Path, modeling_path: Path) -> None:
        self.sales_path = sales_path
        self.modeling_path = modeling_path
        self._sales: pd.DataFrame | None = None
        self._modeling: pd.DataFrame | None = None

    def sales(self) -> pd.DataFrame:
        if self._sales is None:
            if not self.sales_path.is_file():
                raise FileNotFoundError(f"Sales dataset was not found: {self.sales_path}")
            self._sales = pd.read_csv(
                self.sales_path,
                dtype={"StockCode": "string", "Description": "string", "Country": "string"},
                parse_dates=["InvoiceDate"],
            )
        return self._sales

    def modeling(self) -> pd.DataFrame:
        if self._modeling is None:
            if not self.modeling_path.is_file():
                raise FileNotFoundError(
                    f"Modeling dataset was not found: {self.modeling_path}"
                )
            self._modeling = pd.read_csv(
                self.modeling_path,
                dtype={"StockCode": "string"},
                parse_dates=["Date"],
            )
        return self._modeling

    def summary(self) -> dict[str, Any]:
        frame = self.sales()
        monthly = (
            frame.assign(Month=frame["InvoiceDate"].dt.to_period("M").astype(str))
            .groupby("Month", as_index=False)
            .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        )
        return {
            "total_revenue": float(frame["Revenue"].sum()),
            "total_quantity": int(frame["Quantity"].sum()),
            "total_invoices": int(frame["Invoice"].nunique()),
            "unique_products": int(frame["StockCode"].nunique()),
            "unique_customers": int(frame["Customer ID"].nunique()),
            "currency_note": "Revenue is reported in the dataset's original currency units; the source does not specify a currency.",
            "partial_month_note": "December 2010 is a partial month in this dataset.",
            "monthly_sales": monthly.to_dict("records"),
        }

    def top_products(self, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
        frame = self.sales()
        grouped = frame.groupby("StockCode", as_index=False).agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Description=("Description", "first"),
        )
        by_revenue = grouped.nlargest(limit, "Revenue").fillna({"Description": ""})
        by_quantity = grouped.nlargest(limit, "Quantity").fillna({"Description": ""})
        return {
            "by_revenue": by_revenue.to_dict("records"),
            "by_quantity": by_quantity.to_dict("records"),
        }

    def countries(self) -> list[dict[str, Any]]:
        frame = self.sales()
        return (
            frame.groupby("Country", as_index=False)
            .agg(Revenue=("Revenue", "sum"))
            .sort_values("Revenue", ascending=False)
            .fillna({"Country": "Unknown"})
            .to_dict("records")
        )

    def product(self, stock_code: str) -> dict[str, Any] | None:
        frame = self.sales()
        selected = frame[frame["StockCode"] == stock_code]
        if selected.empty:
            return None
        daily = (
            selected.assign(Date=selected["InvoiceDate"].dt.normalize())
            .groupby("Date", as_index=False)
            .agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"))
            .sort_values("Date")
        )
        description = selected["Description"].dropna()
        return {
            "stock_code": stock_code,
            "description": str(description.iloc[0]) if not description.empty else None,
            "total_quantity": int(selected["Quantity"].sum()),
            "total_revenue": float(selected["Revenue"].sum()),
            "daily": daily.to_dict("records"),
        }

    def forecast_rows(self, stock_code: str, limit: int = 20) -> list[dict[str, Any]]:
        frame = self.modeling()
        selected = frame[
            (frame["StockCode"] == stock_code)
            & frame[["DemandLag1", "DemandLag7", "DemandLag14", "DemandLag28",
                     "DemandRollingMean7", "DemandRollingMean14",
                     "DemandRollingMean28", "DemandTrend7_28"]].notna().all(axis=1)
        ].head(limit)
        feature_columns = [
            "StockCode", "Year", "Month", "Day", "WeekdayNumber", "IsWeekend",
            "DemandLag1", "DemandLag7", "DemandLag14", "DemandLag28",
            "DemandRollingMean7", "DemandRollingMean14",
            "DemandRollingMean28", "DemandTrend7_28",
        ]
        rows = selected[["Date", *feature_columns]].copy()
        rows["Date"] = rows["Date"].dt.strftime("%Y-%m-%d")
        return rows.to_dict("records")
