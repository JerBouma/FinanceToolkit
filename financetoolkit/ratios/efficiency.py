"""Efficiency Module"""
__docformat__ = "google"

import pandas as pd


def get_asset_turnover_ratio(
    sales: pd.Series, total_assets_begin: pd.Series, total_assets_end
) -> pd.Series:
    """
    Calculate the asset turnover ratio, an efficiency ratio that measures how
    efficiently a company uses its assets to generate sales.

    Args:
        sales (float or pd.Series): Total sales of the company.
        total_assets_begin (float or pd.Series): Total assets at the beginning of the period.
        total_assets_end (float or pd.Series): Total assets at the end of the period.

    Returns:
        float | pd.Series: The asset turnover ratio value.
    """
    return sales / ((total_assets_begin + total_assets_end) / 2)


def get_inventory_turnover_ratio(
    cost_of_goods_sold: pd.Series,
    inventory_begin: pd.Series,
    inventory_end: pd.Series,
) -> pd.Series:
    """
    Calculate the inventory turnover ratio, an efficiency ratio that measures
    how quickly a company sells its inventory.

    Args:
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        beginning_inventory (float or pd.Series): Beginning inventory of the company.
        ending_inventory (float or pd.Series): Ending inventory of the company.

    Returns:
        float | pd.Series: The inventory turnover ratio value.
    """
    return cost_of_goods_sold / ((inventory_begin + inventory_end) / 2)


def get_days_of_inventory_outstanding(
    inventory_begin: pd.Series,
    inventory_end: pd.Series,
    cost_of_goods_sold: pd.Series,
    days: int = 365,
) -> pd.Series:
    """
    Calculate the days sales in inventory ratio, an efficiency ratio that measures
    how long it takes a company to sell its inventory.

    Args:
        inventory_begin (float or pd.Series): Beginning inventory of the company.
        inventory_end (float or pd.Series): Ending inventory of the company.
        cost_of_goods_sold (float or pd.Series): Total cost of goods sold of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days sales in inventory ratio value.
    """
    return ((inventory_begin + inventory_end) / 2) / cost_of_goods_sold * days


def get_days_of_sales_outstanding(
    accounts_receivable_begin: pd.Series,
    accounts_receivable_end: pd.Series,
    net_credit_sales: pd.Series,
    days: int = 365,
) -> pd.Series:
    """
    Calculate the days of sales outstanding, an efficiency ratio that measures
    the average number of days it takes a company to collect payment on its
    credit sales.

    Args:
        accounts_receivable_begin (float or pd.Series): Beginning accounts receivable of the company.
        accounts_receivable_end (float or pd.Series): Ending accounts receivable of the company.
        net_credit_sales (float or pd.Series): Net credit sales of the company.
        days (int, optional): Number of days in the year. Defaults to 365.

    Returns:
        float | pd.Series: The days of sales outstanding value.
    """
    return (
        ((accounts_receivable_begin + accounts_receivable_end) / 2)
        / net_credit_sales
        * days
    )


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
    accounts_payable_begin: pd.Series,
    accounts_payable_end: pd.Series,
) -> pd.Series:
    """
    Calculate the accounts payable turnover ratio is an efficiency ratio that measures how
    quickly a company pays its suppliers.

    Args:
        cost_of_goods_sold (float or pd.Series): Total Costs of Goods Sold of the company.
        accounts_payable_begin (float or pd.Series): Beginning accounts payable of the company.
        accounts_payable_end (float or pd.Series): Ending accounts payable of the company.

    Returns:
        float | pd.Series: The payables turnover ratio value.
    """
    return cost_of_goods_sold / ((accounts_payable_begin + accounts_payable_end) / 2)


def get_days_of_accounts_payable_outstanding(
    cost_of_goods_sold: pd.Series,
    accounts_payable_begin: pd.Series,
    accounts_payable_end: pd.Series,
    days: int = 365,
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
    return (
        ((accounts_payable_begin + accounts_payable_end) / 2)
        / cost_of_goods_sold
        * days
    )


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
    accounts_receivable_begin: pd.Series,
    accounts_receivable_end: pd.Series,
    net_credit_sales: pd.Series,
) -> pd.Series:
    """
    Calculate the receivables turnover, a ratio that measures how efficiently a
    company uses its assets by comparing the amount of credit extended to customers to
    the amount of sales generated.

    Args:
        account_receivables (float or pd.Series): The average amount of receivables during the period.
        revenue (float or pd.Series): The total annual sales generated during the period.

    Returns:
        float | pd.Series: The receivables turnover value.
    """
    return (
        (accounts_receivable_begin + accounts_receivable_end) / 2
    ) / net_credit_sales


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
    net_fixed_assets_begin: pd.Series,
    net_fixed_assets_end: pd.Series,
) -> pd.Series:
    """
    Calculate the Fixed Asset Turnover ratio, an efficiency ratio that
    measures how efficiently a company uses its fixed assets to generate sales.

    Args:
        net_sales (float or pd.Series): Total sales of the company.
        net_fixed_assets_begin (float or pd.Series): Net fixed assets of the company.
        net_fixed_assets_end (float or pd.Series): Net fixed assets of the company.

    Returns:
        float | pd.Series: The Fixed Asset Turnover ratio value.
    """
    return net_sales / ((net_fixed_assets_begin + net_fixed_assets_end) / 2)


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
