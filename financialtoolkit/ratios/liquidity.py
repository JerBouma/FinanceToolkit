"""Liquidity Module"""
__docformat__ = "numpy"

import pandas as pd


def get_current_ratio(
    current_assets: float | pd.Series, current_liabilities: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the current ratio, a liquidity ratio that measures a company's ability
    to pay off its short-term liabilities with its current assets.

    Args:
        current_assets (float or pd.Series): Total current assets of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The current ratio value.
    """
    return current_assets / current_liabilities


def get_quick_ratio(
    current_assets: float | pd.Series,
    inventory: float | pd.Series,
    current_liabilities: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the quick ratio (also known as the acid-test ratio), a more stringent
    measure of liquidity that excludes inventory from current assets.

    This ratio is also referred to as the Acid Test Ratio.

    Args:
        current_assets (float or pd.Series): Total current assets of the company.
        inventory (float or pd.Series): Total inventory of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The quick ratio value.
    """
    return (current_assets - inventory) / current_liabilities


def get_cash_ratio(
    cash_and_equivalents: float | pd.Series, current_liabilities: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the cash ratio, a liquidity ratio that measures a company's ability
    to pay off its short-term liabilities with its cash and cash equivalents.

    Args:
        cash_and_equivalents (float or pd.Series): Total cash and cash equivalents of the company.
        current_liabilities (float or pd.Series): Total current liabilities of the company.

    Returns:
        float | pd.Series: The cash ratio value.
    """
    return cash_and_equivalents / current_liabilities


def get_working_capital(
    current_assets: float | pd.Series, current_liabilities: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the working capital, which is the difference between a company's current assets
    and current liabilities.

    Args:
        current_assets (float or pd.Series): The current assets of the company.
        current_liabilities (float or pd.Series): The current liabilities of the company.

    Returns:
        float | pd.Series: The working capital value.
    """
    return current_assets - current_liabilities


def get_working_capital_ratio(
    current_assets: float | pd.Series, current_liabilities: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the working capital ratio, a liquidity ratio that measures a company's
    ability to pay off its current liabilities with its current assets.

    Args:
        current_assets (float or pd.Series): Current assets of the company.
        current_liabilities (float or pd.Series): Current liabilities of the company.

    Returns:
        float | pd.Series: The working capital ratio value.
    """
    return current_assets / current_liabilities


def get_operating_cash_flow_ratio(
    operating_cash_flow: float | pd.Series, current_liabilities: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the operating cash flow ratio, a liquidity ratio that measures a company's
    ability to pay off its current liabilities with its operating cash flow.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        current_liabilities (float or pd.Series): Current liabilities of the company.

    Returns:
        float | pd.Series: The operating cash flow ratio value.
    """
    return operating_cash_flow / current_liabilities


def get_operating_cash_flow_sales_ratio(
    operating_cash_flow: float | pd.Series, revenue: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the operating cash flow to sales ratio, a liquidity ratio that measures the ability of a company to generate
    cash from its sales.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        revenue (float or pd.Series): Sales of the company.

    Returns:
        float | pd.Series: The operating cash flow to sales ratio value.
    """
    return operating_cash_flow / revenue


def get_short_term_coverage_ratio(
    operating_cash_flow: float | pd.Series,
    accounts_receivable: float | pd.Series,
    inventory: float | pd.Series,
    accounts_payable: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the short term coverage ratio, a liquidity ratio that measures a company's ability to pay off its
    short-term obligations with its operating cash flow.

    Args:
        operating_cash_flow (float or pd.Series): Operating cash flow of the company.
        accounts_receivable (float or pd.Series): Accounts receivable of the company.
        inventory (float or pd.Series): Inventory of the company.
        accounts_payable (float or pd.Series): Accounts payable of the company.

    Returns:
        float | pd.Series: The short term coverage ratio value.
    """
    return operating_cash_flow / (accounts_receivable + inventory - accounts_payable)
