"""Effiicency Module"""
__docformat__ = "numpy"

import pandas as pd


def get_asset_turnover_ratio(
    sales: float | pd.Series, total_assets: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the asset turnover ratio, an efficiency ratio that measures how
    efficiently a company uses its assets to generate sales.

    Args:
        sales (float or pd.Series): Total sales of the company.
        total_assets (float or pd.Series): Total assets of the company.

    Returns:
        float | pd.Series: The asset turnover ratio value.
    """
    return sales / total_assets


def get_inventory_turnover_ratio(
    cost_of_goods_sold: float | pd.Series, average_inventory: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the inventory turnover ratio, an efficiency ratio that measures
    how quickly a company sells its inventory.

    Args:
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        average_inventory (float or pd.Series): Average inventory of the company.

    Returns:
        float | pd.Series: The inventory turnover ratio value.
    """
    return cost_of_goods_sold / average_inventory


def get_days_of_inventory_outstanding(
    average_inventory: float | pd.Series,
    cost_of_goods_sold: float | pd.Series,
    days: int = 365,
) -> float | pd.Series:
    """
    Calculate the days sales in inventory ratio, an efficiency ratio that measures
    how long it takes a company to sell its inventory.

    Args:
        average_inventory (float or pd.Series): Average inventory of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.

    Returns:
        float | pd.Series: The days sales in inventory ratio value.
    """
    return days * average_inventory / cost_of_goods_sold


def get_days_of_sales_outstanding(
    accounts_receivable: float | pd.Series, revenue: float | pd.Series, days: int = 365
) -> float | pd.Series:
    """
    Calculate the days of sales outstanding, an efficiency ratio that measures
    the average number of days it takes a company to collect payment on its
    credit sales.

    Args:
        accounts_receivable (float or pd.Series): Accounts receivable of the company.
        revenue (float or pd.Series): Revenue of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days of sales outstanding value.
    """
    return (accounts_receivable / revenue) * days


def get_operating_cycle(
    days_of_inventory: float | pd.Series, days_of_sales_outstanding: float | pd.Series
) -> float | pd.Series:
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


def get_cash_conversion_cycle(
    days_inventory: float | pd.Series,
    days_sales_outstanding: float | pd.Series,
    days_payables_outstanding: float | pd.Series,
) -> float | pd.Series:
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


def get_payables_turnover_ratio(
    purchases: float | pd.Series, average_accounts_payable: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the payables turnover ratio, an efficiency ratio that measures how
    quickly a company pays its suppliers.

    Args:
        purchases (float or pd.Series): Total purchases of the company.
        average_accounts_payable (float or pd.Series): Average accounts payable of the company.

    Returns:
        float | pd.Series: The payables turnover ratio value.
    """
    return purchases / average_accounts_payable


def get_days_payables_outstanding(
    payables_turnover_ratio: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the days payables outstanding, an efficiency ratio that measures the
    number of days it takes a company to pay its suppliers.

    Args:
        payables_turnover_ratio (float or pd.Series): The payables turnover ratio of the company.

    Returns:
        float | pd.Series: The days payables outstanding value.
    """
    return 365 / payables_turnover_ratio


def get_receivables_turnover(
    average_receivables: float | pd.Series, annual_sales: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the receivables turnover, a ratio that measures how efficiently a
    company uses its assets by comparing the amount of credit extended to customers to
    the amount of sales generated.

    Args:
        average_receivables (float or pd.Series): The average amount of receivables during the period.
        annual_sales (float or pd.Series): The total annual sales generated during the period.

    Returns:
        float | pd.Series: The receivables turnover value.
    """
    return annual_sales / average_receivables


def get_sga_to_revenue_ratio(
    sga_expenses: float | pd.Series, revenue: float | pd.Series
) -> float | pd.Series:
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


def get_average_receivables(
    beginning_receivables: float | pd.Series,
    ending_receivables: float | pd.Series,
    num_periods: int,
) -> float | pd.Series:
    """
    Calculate the average receivables, which is the average amount of money owed to a company
    by its customers over a specified period of time.

    Args:
        beginning_receivables (float or pd.Series): The beginning balance of accounts receivable.
        ending_receivables (float or pd.Series): The ending balance of accounts receivable.
        num_periods (int): The number of periods over which to calculate the average receivables.

    Returns:
        float | pd.Series: The average receivables value.
    """
    return (beginning_receivables + ending_receivables) / 2 * num_periods


def get_fixed_asset_turnover(
    sales: float | pd.Series, net_fixed_assets: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
    measures how efficiently a company uses its fixed assets to generate sales.

    Args:
        sales (float or pd.Series): Total sales of the company.
        net_fixed_assets (float or pd.Series): Net fixed assets of the company.

    Returns:
        float | pd.Series: The Fixed Asset Turnover ratio value.
    """
    return sales / net_fixed_assets
