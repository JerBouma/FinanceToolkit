"""Efficiency Module"""

__docformat__ = "google"

import pandas as pd


def get_asset_turnover_ratio(
    sales: pd.Series,
    average_total_assets: pd.Series,
) -> pd.Series:
    """
    Calculate the asset turnover ratio, an efficiency ratio that measures how
    efficiently a company uses its assets to generate sales.

    Args:
        sales (float or pd.Series): Total sales of the company.
        average_total_assets (float or pd.Series): Average total assets of the company.
        This is typically calculated as (beginning total assets + ending total assets) / 2.

    Returns:
        float | pd.Series: The asset turnover ratio value.
    """
    return sales / average_total_assets


def get_inventory_turnover_ratio(
    cost_of_goods_sold: pd.Series,
    average_inventory: pd.Series,
) -> pd.Series:
    """
    Calculate the inventory turnover ratio, an efficiency ratio that measures
    how quickly a company sells its inventory.

    Args:
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        average_inventory (float or pd.Series): Average inventory of the company.
        This is typically calculated as (beginning inventory + ending inventory) / 2.

    Returns:
        float | pd.Series: The inventory turnover ratio value.
    """
    return cost_of_goods_sold / average_inventory


def get_days_of_inventory_outstanding(
    average_inventory: pd.Series,
    cost_of_goods_sold: pd.Series,
    days: int | float = 365,
) -> pd.Series:
    """
    Calculate the days sales in inventory ratio, an efficiency ratio that measures
    how long it takes a company to sell its inventory.

    Args:
        average_inventory (float or pd.Series): Average inventory of the company.
        This is typically calculated as (beginning inventory + ending inventory) / 2.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days sales in inventory ratio value.
    """
    return average_inventory / cost_of_goods_sold * days


def get_days_of_sales_outstanding(
    average_accounts_receivable: pd.Series,
    net_credit_sales: pd.Series,
    days: int | float = 365,
) -> pd.Series:
    """
    Calculate the days of sales outstanding, an efficiency ratio that measures
    the average number of days it takes a company to collect payment on its
    credit sales.

    Args:
        average_accounts_receivable (float or pd.Series): Average accounts receivable of the company.
        This is typically calculated as (beginning accounts receivable + ending accounts receivable) / 2.
        net_credit_sales (float or pd.Series): Net credit sales of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days of sales outstanding value.
    """
    return average_accounts_receivable / net_credit_sales * days


def get_operating_cycle(
    days_of_inventory: pd.Series, days_of_sales_outstanding: pd.Series
) -> pd.Series:
    """
    Calculate the operating cycle, an efficiency ratio that measures the average
    number of days it takes a company to turn its inventory into cash.

    Args:
        days_of_inventory (float or pd.Series): Days of inventory of the company.
        days_of_sales_outstanding (float or pd.Series): Days of sales outstanding of the company.
        days_of_payables (float or pd.Series): Days of payables of the company.

    Returns:
        float | pd.Series: The operating cycle value.
    """
    return days_of_inventory + days_of_sales_outstanding


def get_accounts_payables_turnover_ratio(
    cost_of_goods_sold: pd.Series,
    average_accounts_payable: pd.Series,
) -> pd.Series:
    """
    Calculate the accounts payable turnover ratio is an efficiency ratio that measures how
    quickly a company pays its suppliers.

    Args:
        cost_of_goods_sold (float or pd.Series): Total Costs of Goods Sold of the company.
        average_accounts_payable (float or pd.Series): Average accounts payable of the company.

    Returns:
        float | pd.Series: The payables turnover ratio value.
    """
    return cost_of_goods_sold / average_accounts_payable


def get_days_of_accounts_payable_outstanding(
    cost_of_goods_sold: pd.Series,
    average_accounts_payable: pd.Series,
    days: int | float = 365,
) -> pd.Series:
    """
    Calculate the days payables outstanding, an efficiency ratio that measures the
    number of days it takes a company to pay its suppliers.

    Args:
        cost_of_goods_sold (float or pd.Series): Total Costs of Goods Sold of the company.
        accounts_payable_begin (float or pd.Series): Beginning accounts payable of the company.
        accounts_payable_end (float or pd.Series): Ending accounts payable of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days payables outstanding value.
    """
    return average_accounts_payable / cost_of_goods_sold * days


def get_cash_conversion_cycle(
    days_inventory: pd.Series,
    days_sales_outstanding: pd.Series,
    days_payables_outstanding: pd.Series,
) -> pd.Series:
    """
    Calculate the Cash Conversion Cycle, which measures the amount of time it takes for a company to convert
    its investments in inventory and accounts receivable into cash, while considering the time it takes to pay
    its accounts payable.

    Args:
        days_inventory (float or pd.Series): Average number of days that inventory is held before being sold.
        days_sales_outstanding (float or pd.Series): Average number of days it takes to collect accounts receivable.
        days_payables_outstanding (float or pd.Series): Average number of days it takes to pay accounts payable.

    Returns:
        float | pd.Series: The Cash Conversion Cycle value.
    """
    return days_inventory + days_sales_outstanding - days_payables_outstanding


def get_receivables_turnover(
    average_accounts_receivable: pd.Series,
    net_credit_sales: pd.Series,
) -> pd.Series:
    """
    Calculate the receivables turnover, a ratio that measures how efficiently a
    company uses its assets by comparing the amount of credit extended to customers to
    the amount of sales generated.

    Args:
        average_accounts_receivable (float or pd.Series): Average accounts receivable of the company.
        This is typically calculated as (beginning accounts receivable + ending accounts receivable) / 2.
        revenue (float or pd.Series): The total annual sales generated during the period.

    Returns:
        float | pd.Series: The receivables turnover value.
    """
    return average_accounts_receivable / net_credit_sales


def get_sga_to_revenue_ratio(sga_expenses: pd.Series, revenue: pd.Series) -> pd.Series:
    """
    Calculates the sales, general, and administrative (SG&A) expenses to revenue ratio,
    which measures the SG&A expenses relative to the revenue of the company.

    Args:
        sga_expenses (float or pd.Series): SG&A expenses of the company.
        revenue (float or pd.Series): Revenue of the company.

    Returns:
        float | pd.Series: The SG&A to revenue ratio.
    """
    return sga_expenses / revenue


def get_fixed_asset_turnover(
    net_sales: pd.Series,
    average_net_fixed_assets: pd.Series,
) -> pd.Series:
    """
    Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
    measures how efficiently a company uses its fixed assets to generate sales.

    Args:
        net_sales (float or pd.Series): Total sales of the company.
        average_net_fixed_assets (float or pd.Series): Average net fixed assets of the company.

    Returns:
        float | pd.Series: The Fixed Asset Turnover ratio value.
    """
    return net_sales / average_net_fixed_assets


def get_operating_margin(
    operating_income: pd.Series,
    revenue: pd.Series,
) -> pd.Series:
    """
    Calculate the operating margin, a financial metric that measures the
    percentage of revenue that remains after covering operating expenses.
    It is calculated as operating income divided by revenue.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float or pd.Series: The operating ratio value.
    """
    return operating_income / revenue


def get_operating_ratio(
    operating_expenses: pd.Series,
    cost_of_goods_sold: pd.Series,
    revenue: pd.Series,
) -> pd.Series:
    """
    Calculate the operating ratio, a financial metric that measures the efficiency
    of a company's operations by comparing its operating expenses to its revenue.

    Args:
        operating_expenses (float or pd.Series): Total operating expenses of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float or pd.Series: The operating ratio value.
    """
    return (operating_expenses + cost_of_goods_sold) / revenue


def get_cash_conversion_efficiency(
    operating_cash_flow: pd.Series,
    revenue: pd.Series,
):
    """
    Calculate the cash conversion efficiency, a financial metric that measures the efficiency
    of a company's operations by comparing its operating cash flow to its revenue.

    Args:
        operating_cash_flow (float or pd.Series): Total operating cash flow of the company.
        revenue (float or pd.Series): Total revenue of the company.

    Returns:
        float or pd.Series: The cash conversion efficiency value.
    """
    return operating_cash_flow / revenue
