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

    The formula is as follows:

        - Net Profit Margin = Net Income / Total Revenue
        - Asset Turnover = Total Revenue / Average Total Assets
        - Equity Multiplier = Average Total Assets / Average Total Equity
        - Return on Equity = Net Profit Margin * Asset Turnover * Equity Multiplier

    Also known as: DuPont, ROE decomposition, three-factor DuPont.

    Args:
        net_income (pd.Series): Net profit of the company.
        total_revenue (pd.Series): Total revenue of the company.
        average_total_assets (pd.Series): The average of the company's total assets over the
        beginning and the end of the period.
        average_total_equity (pd.Series): The average of the company's total equity over the
        beginning and the end of the period.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.

    Notes:
    - The three components multiply back to the Return on Equity exactly, since Total Revenue
    and Average Total Assets each cancel out of the product. This makes the decomposition a
    self-checking identity rather than an approximation.
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
    Perform an Extended Dupont analysis to breakdown the return on equity (ROE) into its components.

    The extended (five-factor) decomposition splits the three-factor version's Net Profit Margin
    into a Tax Burden, an Interest Burden and an Operating Profit Margin, isolating the effect of
    taxation, of debt financing and of operating performance on the return on equity.

    The formula is as follows:

        - Interest Burden Ratio = Income Before Tax / Operating Income
        - Tax Burden Ratio = Net Income / Income Before Tax
        - Operating Profit Margin = Operating Income / Total Revenue
        - Asset Turnover = Total Revenue / Average Total Assets
        - Equity Multiplier = Average Total Assets / Average Total Equity
        - Return on Equity = Interest Burden Ratio * Tax Burden Ratio * Operating Profit Margin *
        Asset Turnover * Equity Multiplier

    Also known as: extended DuPont, five-factor DuPont, ROE breakdown.

    Args:
        operating_income (pd.Series): Operating income (EBIT) of the company.
        income_before_tax (pd.Series): Income before taxes (EBT) of the company.
        net_income (pd.Series): Net profit of the company.
        total_revenue (pd.Series): Total revenue of the company.
        average_total_assets (pd.Series): The average of the company's total assets over the
        beginning and the end of the period.
        average_total_equity (pd.Series): The average of the company's total equity over the
        beginning and the end of the period.

    Returns:
        pd.DataFrame: A DataFrame containing the Dupont analysis components.

    Notes:
    - The five components multiply back to the Return on Equity exactly: Operating Income,
    Income Before Tax, Total Revenue and Average Total Assets each cancel out of the product,
    leaving Net Income / Average Total Equity.
    - The first three components multiply back to the Net Profit Margin reported by the
    three-factor `get_dupont_analysis`, so both decompositions resolve to the same ROE.
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
