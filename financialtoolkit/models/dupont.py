"""Dupont Module"""
__docformat__ = "numpy"

import pandas as pd

from financialtoolkit.ratios import (
    efficiency as _efficiency,
    profitability as _profitability,
    solvency as _solvency,
)


def get_dupont_analysis(
    net_income: float | pd.Series,
    total_revenue: float | pd.Series,
    total_assets: float | pd.Series,
    total_equity: float | pd.Series,
) -> pd.DataFrame:
    """
    Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

    Args:
        net_income (float or pd.Series): Net profit of the company.
        total_revenue (float or pd.Series): Total revenue of the company.
        total_assets (float or pd.Series): Total assets of the company.
        total_equity (float or pd.Series): Total equity of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.
    """
    # Calculate the net profit margin
    profit_margin = _profitability.get_net_profit_margin(net_income, total_revenue)

    # Calculate the asset turnover
    asset_turnover = _efficiency.get_asset_turnover_ratio(total_revenue, total_assets)

    # Calculate the financial leverage
    financial_leverage = _solvency.get_financial_leverage(total_assets, total_equity)

    # Calculate the return on equity
    return_on_equity = profit_margin * asset_turnover * financial_leverage

    # Create a dictionary with the Dupont analysis components
    components = {
        "Net Profit Margin": profit_margin,
        "Asset Turnover": asset_turnover,
        "Financial Leverage": financial_leverage,
        "Return on Equity": return_on_equity,
    }

    return pd.DataFrame.from_dict(components, orient="index")


def get_extended_dupont_analysis(
    income_before_tax: float | pd.Series,
    operating_income: float | pd.Series,
    net_income: float | pd.Series,
    total_revenue: float | pd.Series,
    total_assets: float | pd.Series,
    total_equity: float | pd.Series,
) -> pd.DataFrame:
    """
    Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.

    Args:
        income_before_tax (float or pd.Series): Income before taxes of the company.
        operating_income (float or pd.Series): Operating income of the company.
        net_income (float or pd.Series): Net profit of the company.
        total_revenue (float or pd.Series): Total revenue of the company.
        total_assets (float or pd.Series): Total assets of the company.
        total_equity (float or pd.Series): Total equity of the company.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.
    """
    # Calculate the interest burden ratio
    interest_burden_ratio = _profitability.get_interest_burden_ratio(
        income_before_tax, operating_income
    )

    # Calculate the tax burden ratio
    tax_burden_ratio = _profitability.get_tax_burden_ratio(
        net_income, income_before_tax
    )

    # Calculate the operating profit margin
    operating_profit_margin = _profitability.get_operating_margin(
        net_income, total_revenue
    )

    # Calculate the asset turnover
    asset_turnover = _efficiency.get_asset_turnover_ratio(total_revenue, total_assets)

    # Calculate the financial leverage
    financial_leverage = _solvency.get_financial_leverage(total_assets, total_equity)

    # Calculate the return on equity
    return_on_equity = (
        interest_burden_ratio
        * tax_burden_ratio
        * operating_profit_margin
        * asset_turnover
        * financial_leverage
    )

    # Create a dictionary with the Dupont analysis components
    components = {
        "Interest Burden Ratio": interest_burden_ratio,
        "Tax Burden Ratio": tax_burden_ratio,
        "Operating Profit Margin": operating_profit_margin,
        "Asset Turnover": asset_turnover,
        "Financial Leverage": financial_leverage,
        "Return on Equity": return_on_equity,
    }

    return pd.DataFrame.from_dict(components, orient="index")
