"""Dupont Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.ratios import efficiency_model, profitability_model, solvency_model


def get_dupont_analysis(
    net_income: pd.Series,
    total_revenue: pd.Series,
    average_total_assets: pd.Series,
    average_total_equity: pd.Series,
) -> pd.DataFrame:
    """
    Perform a Dupont analysis to breakdown the return on equity (ROE) into its components.

    Args:
        net_income (float or pd.Series): Net profit of the company.
        total_revenue (float or pd.Series): Total revenue of the company.
        total_assets_begin (float or pd.Series): Total assets of the company at the beginning of the period.
        total_assets_end (float or pd.Series): Total assets of the company at the end of the period.
        total_equity_begin (float or pd.Series): Total equity of the company at the beginning of the period.
        total_equity_end (float or pd.Series): Total equity of the company at the end of the period.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.
    """
    # Calculate the net profit margin
    profit_margin = profitability_model.get_net_profit_margin(net_income, total_revenue)

    # Calculate the asset turnover
    asset_turnover = efficiency_model.get_asset_turnover_ratio(
        total_revenue, average_total_assets
    )

    # Calculate the equity multiplier
    equity_multiplier = solvency_model.get_equity_multiplier(
        average_total_assets, average_total_equity
    )

    # Calculate the return on equity
    return_on_equity = profit_margin * asset_turnover * equity_multiplier

    # Create a dictionary with the Dupont analysis components
    components = {
        "Net Profit Margin": profit_margin,
        "Asset Turnover": asset_turnover,
        "Equity Multiplier": equity_multiplier,
        "Return on Equity": return_on_equity,
    }

    if isinstance(return_on_equity, pd.DataFrame):
        return (
            pd.concat(components)
            .swaplevel(1, 0)
            .sort_index(level=0, sort_remaining=False)
        )

    return pd.DataFrame.from_dict(components, orient="index")


def get_extended_dupont_analysis(
    operating_income: pd.Series,
    income_before_tax: pd.Series,
    net_income: pd.Series,
    total_revenue: pd.Series,
    average_total_assets: pd.Series,
    average_total_equity: pd.Series,
) -> pd.DataFrame:
    """
    Perform am Extended Dupont analysis to breakdown the return on equity (ROE) into its components.

    Args:
        operating_income (float or pd.Series): Operating income of the company.
        interest_expense (float or pd.Series): Interest expense of the company.
        income_before_tax (float or pd.Series): Income before taxes of the company.
        net_income (float or pd.Series): Net profit of the company.
        total_revenue (float or pd.Series): Total revenue of the company.
        total_assets_begin (float or pd.Series): Total assets of the company at the beginning of the period.
        total_assets_end (float or pd.Series): Total assets of the company at the end of the period.
        total_equity_begin (float or pd.Series): Total equity of the company at the beginning of the period.
        total_equity_end (float or pd.Series): Total equity of the company at the end of the period.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.
    """
    # Calculate the interest burden ratio
    interest_burden_ratio = profitability_model.get_interest_burden_ratio(
        income_before_tax, operating_income
    )

    # Calculate the tax burden ratio
    tax_burden_ratio = profitability_model.get_tax_burden_ratio(
        net_income, income_before_tax
    )

    # Calculate the operating profit margin
    operating_profit_margin = profitability_model.get_operating_margin(
        operating_income, total_revenue
    )

    # Calculate the asset turnover
    asset_turnover = efficiency_model.get_asset_turnover_ratio(
        total_revenue, average_total_assets
    )

    # Calculate the equity multiplier
    equity_multiplier = solvency_model.get_equity_multiplier(
        average_total_assets, average_total_equity
    )

    # Calculate the return on equity
    return_on_equity = (
        interest_burden_ratio
        * tax_burden_ratio
        * operating_profit_margin
        * asset_turnover
        * equity_multiplier
    )

    # Create a dictionary with the Dupont analysis components
    components = {
        "Interest Burden Ratio": interest_burden_ratio,
        "Tax Burden Ratio": tax_burden_ratio,
        "Operating Profit Margin": operating_profit_margin,
        "Asset Turnover": asset_turnover,
        "Equity Multiplier": equity_multiplier,
        "Return on Equity": return_on_equity,
    }

    if isinstance(return_on_equity, pd.DataFrame):
        return (
            pd.concat(components)
            .swaplevel(1, 0)
            .sort_index(level=0, sort_remaining=False)
        )

    return pd.DataFrame.from_dict(components, orient="index")
