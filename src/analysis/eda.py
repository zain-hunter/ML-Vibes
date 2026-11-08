"""Exploratory data analysis for the cleaned Online Retail II datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_REPORT_DIR = Path("reports/eda")
CHART_DIR_NAME = "charts"

SALES_COLUMNS = {
    "Invoice",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "Revenue",
}
MASTER_COLUMNS = SALES_COLUMNS | {"IsReturn", "IsCancelledInvoice"}
RETURNS_COLUMNS = MASTER_COLUMNS


def load_processed_dataset(
    path: str | Path, required_columns: set[str]
) -> pd.DataFrame:
    """Load one processed CSV and validate its required columns."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset was not found: {dataset_path}")

    try:
        dataframe = pd.read_csv(dataset_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read dataset '{dataset_path}': {error}") from error

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_path.name} is missing columns: {missing}")

    dataframe["InvoiceDate"] = pd.to_datetime(
        dataframe["InvoiceDate"], errors="coerce"
    )
    if dataframe["InvoiceDate"].isna().any():
        raise ValueError(f"{dataset_path.name} contains invalid InvoiceDate values.")
    return dataframe


def load_datasets(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the master, sales, and returns/cancellations datasets."""
    directory = Path(data_dir)
    return {
        "master": load_processed_dataset(
            directory / "transactions_clean.csv", MASTER_COLUMNS
        ),
        "sales": load_processed_dataset(
            directory / "sales_transactions.csv", SALES_COLUMNS
        ),
        "returns": load_processed_dataset(
            directory / "returns_cancellations.csv", RETURNS_COLUMNS
        ),
    }


def calculate_sales_overview(sales: pd.DataFrame) -> dict[str, float | int]:
    """Calculate overall sales metrics from the sales-only dataset."""
    invoice_count = int(sales["Invoice"].nunique())
    total_revenue = float(sales["Revenue"].sum())
    return {
        "total_revenue": total_revenue,
        "total_quantity_sold": int(sales["Quantity"].sum()),
        "unique_invoices": invoice_count,
        "unique_products": int(sales["StockCode"].nunique()),
        "unique_customers": int(sales["Customer ID"].dropna().nunique()),
        "average_order_value": total_revenue / invoice_count
        if invoice_count
        else 0.0,
    }


def calculate_time_analysis(sales: pd.DataFrame) -> dict[str, pd.DataFrame | str]:
    """Calculate monthly, daily, weekday, and highest/lowest sales periods."""
    dated = sales.copy()
    dated["Month"] = dated["InvoiceDate"].dt.to_period("M").astype(str)
    dated["Date"] = dated["InvoiceDate"].dt.date.astype(str)
    dated["Weekday"] = dated["InvoiceDate"].dt.day_name()

    monthly = (
        dated.groupby("Month", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        .sort_values("Month")
    )
    daily = (
        dated.groupby("Date", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        .sort_values("Date")
    )
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday = (
        dated.groupby("Weekday", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        .assign(Weekday=lambda frame: pd.Categorical(
            frame["Weekday"], categories=weekday_order, ordered=True
        ))
        .sort_values("Weekday")
    )
    highest_month = monthly.loc[monthly["Revenue"].idxmax(), "Month"]
    lowest_month = monthly.loc[monthly["Revenue"].idxmin(), "Month"]
    return {
        "monthly": monthly,
        "daily": daily,
        "weekday": weekday,
        "highest_month": str(highest_month),
        "lowest_month": str(lowest_month),
    }


def calculate_product_analysis(sales: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate product-level quantity and revenue rankings."""
    summary = (
        sales.groupby("StockCode", as_index=False)
        .agg(
            QuantitySold=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            InvoiceCount=("Invoice", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )
    return {
        "summary": summary,
        "top_revenue": summary.head(10),
        "top_quantity": summary.sort_values("QuantitySold", ascending=False).head(10),
        "lowest_quantity": summary.sort_values("QuantitySold").head(10),
    }


def calculate_customer_analysis(sales: pd.DataFrame) -> dict[str, pd.DataFrame | int | float]:
    """Calculate customer metrics while excluding missing customer IDs."""
    customer_sales = sales.dropna(subset=["Customer ID"]).copy()
    summary = (
        customer_sales.groupby("Customer ID", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            QuantityPurchased=("Quantity", "sum"),
            PurchaseFrequency=("Invoice", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )
    return {
        "summary": summary,
        "top_revenue": summary.head(10),
        "unique_customers": int(summary["Customer ID"].nunique()),
        "average_customer_spending": float(summary["Revenue"].mean())
        if not summary.empty
        else 0.0,
    }


def calculate_country_analysis(sales: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate revenue and quantity sold by country."""
    summary = (
        sales.groupby("Country", as_index=False)
        .agg(Revenue=("Revenue", "sum"), QuantitySold=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
    )
    return {"summary": summary, "top_revenue": summary.head(10)}


def calculate_returns_analysis(returns: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    """Separate return and cancellation counts and summarize returned products."""
    return_records = returns.loc[returns["IsReturn"]].copy()
    cancellation_records = returns.loc[returns["IsCancelledInvoice"]].copy()
    by_product = (
        return_records.groupby("StockCode", as_index=False)
        .agg(ReturnQuantity=("Quantity", "sum"), ReturnRecordCount=("Invoice", "size"))
        .assign(ReturnedUnits=lambda frame: frame["ReturnQuantity"].abs())
        .sort_values("ReturnedUnits", ascending=False)
    )
    return {
        "return_records": int(len(return_records)),
        "cancellation_records": int(len(cancellation_records)),
        "return_products": by_product,
        "top_return_products": by_product.head(10),
    }


def validate_classifications(
    master: pd.DataFrame,
    sales: pd.DataFrame,
    returns: pd.DataFrame,
) -> None:
    """Check that the processed datasets agree on row classifications."""
    expected_sales = (
        ~master["IsCancelledInvoice"]
        & master["Quantity"].gt(0)
        & master["Price"].ge(0)
    )
    expected_returns = master["IsReturn"] | master["IsCancelledInvoice"]
    if int(expected_sales.sum()) != len(sales):
        raise ValueError("Sales dataset row count does not match master classifications.")
    if int(expected_returns.sum()) != len(returns):
        raise ValueError(
            "Returns/cancellations row count does not match master classifications."
        )
    if not sales["Revenue"].equals(sales["Quantity"] * sales["Price"]):
        raise ValueError("Sales revenue is inconsistent with Quantity * Price.")
    if not returns["Revenue"].equals(returns["Quantity"] * returns["Price"]):
        raise ValueError("Returns revenue is inconsistent with Quantity * Price.")


def save_chart(fig: plt.Figure, chart_dir: Path, filename: str) -> None:
    """Save and close a chart."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(chart_dir / filename, dpi=150)
    plt.close(fig)


def create_charts(
    time_analysis: dict[str, pd.DataFrame | str],
    products: dict[str, pd.DataFrame],
    customers: dict[str, pd.DataFrame | int | float],
    countries: dict[str, pd.DataFrame],
    returns_analysis: dict[str, pd.DataFrame | int],
    chart_dir: str | Path,
) -> None:
    """Create the requested labeled EDA charts."""
    destination = Path(chart_dir)
    monthly = time_analysis["monthly"]
    daily = time_analysis["daily"]
    weekday = time_analysis["weekday"]
    top_revenue_products = products["top_revenue"]
    top_quantity_products = products["top_quantity"]
    top_countries = countries["top_revenue"]
    top_customers = customers["top_revenue"]
    top_returns = returns_analysis["top_return_products"]

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(monthly["Month"], monthly["Revenue"], marker="o")
    axis.set(title="Monthly Revenue Trend", xlabel="Month", ylabel="Revenue")
    axis.tick_params(axis="x", rotation=45)
    save_chart(fig, destination, "monthly_revenue_trend.png")

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(daily["Date"], daily["Revenue"])
    axis.set(title="Daily Revenue Trend", xlabel="Date", ylabel="Revenue")
    axis.tick_params(axis="x", rotation=60)
    save_chart(fig, destination, "daily_revenue_trend.png")

    for frame, value, title, filename in [
        (
            top_revenue_products.sort_values("Revenue"),
            "Revenue",
            "Top 10 Products by Revenue",
            "top_10_products_by_revenue.png",
        ),
        (
            top_quantity_products.sort_values("QuantitySold"),
            "QuantitySold",
            "Top 10 Products by Quantity",
            "top_10_products_by_quantity.png",
        ),
        (
            top_countries.sort_values("Revenue"),
            "Revenue",
            "Top 10 Countries by Revenue",
            "top_10_countries_by_revenue.png",
        ),
        (
            top_customers.sort_values("Revenue"),
            "Revenue",
            "Top 10 Customers by Revenue",
            "top_10_customers_by_revenue.png",
        ),
        (
            top_returns.sort_values("ReturnedUnits"),
            "ReturnedUnits",
            "Returned Units by Product (Top 10)",
            "returns_by_product.png",
        ),
    ]:
        fig, axis = plt.subplots(figsize=(10, 6))
        labels = frame.iloc[:, 0].astype(str)
        axis.barh(labels, frame[value])
        axis.set(title=title, xlabel=value, ylabel=frame.columns[0])
        save_chart(fig, destination, filename)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(weekday["Weekday"].astype(str), weekday["Revenue"])
    axis.set(title="Sales by Weekday", xlabel="Weekday", ylabel="Revenue")
    axis.tick_params(axis="x", rotation=30)
    save_chart(fig, destination, "weekday_sales.png")


def write_monthly_sales(monthly: pd.DataFrame, report_dir: str | Path) -> None:
    """Save monthly revenue and quantity totals."""
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    monthly.assign(Revenue=monthly["Revenue"].round(2)).to_csv(
        destination / "monthly_sales.csv", index=False
    )


def write_summary(
    report_dir: str | Path,
    overview: dict[str, float | int],
    time_analysis: dict[str, pd.DataFrame | str],
    products: dict[str, pd.DataFrame],
    customers: dict[str, pd.DataFrame | int | float],
    countries: dict[str, pd.DataFrame],
    returns_analysis: dict[str, pd.DataFrame | int],
    master: pd.DataFrame,
    sales: pd.DataFrame,
) -> None:
    """Write key findings, assumptions, and exclusions to a text report."""
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    top_product = products["top_revenue"].iloc[0]
    top_country = countries["top_revenue"].iloc[0]
    top_customer = customers["top_revenue"].iloc[0]
    lines = [
        "Smart Retail Intelligence - Exploratory Data Analysis",
        "",
        "Data sources:",
        "- Sales analysis: data/processed/sales_transactions.csv",
        "- Returns/cancellations: data/processed/returns_cancellations.csv",
        "- Classification validation: data/processed/transactions_clean.csv",
        "",
        "Sales overview:",
        f"- Total revenue: {overview['total_revenue']:.2f}",
        f"- Total quantity sold: {overview['total_quantity_sold']}",
        f"- Unique invoices: {overview['unique_invoices']}",
        f"- Unique products: {overview['unique_products']}",
        f"- Unique customers: {overview['unique_customers']}",
        f"- Average order value: {overview['average_order_value']:.2f}",
        "",
        "Time analysis:",
        f"- Highest revenue month: {time_analysis['highest_month']}",
        f"- Lowest revenue month: {time_analysis['lowest_month']}",
        "",
        "Top findings:",
        f"- Top product by revenue: {top_product['StockCode']} ({top_product['Revenue']:.2f})",
        f"- Top product by quantity: {products['top_quantity'].iloc[0]['StockCode']}",
        f"- Top country by revenue: {top_country['Country']} ({top_country['Revenue']:.2f})",
        f"- Top customer by revenue: {top_customer['Customer ID']} ({top_customer['Revenue']:.2f})",
        "",
        "Returns and cancellations:",
        f"- Return records: {returns_analysis['return_records']}",
        f"- Cancellation records: {returns_analysis['cancellation_records']}",
        "",
        "Assumptions and exclusions:",
        "- Invoice is treated as an order identifier; unique invoices are counted, not rows.",
        "- Revenue is Quantity multiplied by Price, using the signed cleaned values.",
        "- Missing Customer IDs are excluded only from customer-specific analysis.",
        "- Sales metrics use sales_transactions.csv; return/cancellation metrics use the exception dataset.",
        "- Return and cancellation flags remain separate and may overlap.",
        f"- Master rows validated: {len(master)}; sales rows analyzed: {len(sales)}.",
        "- No raw or processed input dataset was modified.",
    ]
    (destination / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_eda(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> None:
    """Run all EDA calculations and save reports and charts."""
    datasets = load_datasets(data_dir)
    master, sales, returns = (
        datasets["master"],
        datasets["sales"],
        datasets["returns"],
    )
    validate_classifications(master, sales, returns)

    overview = calculate_sales_overview(sales)
    time_analysis = calculate_time_analysis(sales)
    products = calculate_product_analysis(sales)
    customers = calculate_customer_analysis(sales)
    countries = calculate_country_analysis(sales)
    returns_analysis = calculate_returns_analysis(returns)

    destination = Path(report_dir)
    write_monthly_sales(time_analysis["monthly"], destination)
    create_charts(
        time_analysis,
        products,
        customers,
        countries,
        returns_analysis,
        destination / CHART_DIR_NAME,
    )
    write_summary(
        destination,
        overview,
        time_analysis,
        products,
        customers,
        countries,
        returns_analysis,
        master,
        sales,
    )


def main() -> None:
    """Run EDA from the command line."""
    parser = argparse.ArgumentParser(
        description="Explore cleaned Online Retail II transaction data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    try:
        run_eda(args.data_dir, args.report_dir)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"EDA outputs saved to {args.report_dir}")


if __name__ == "__main__":
    main()
