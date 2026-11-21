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
        period ends that open and close the period over which the flows are measured.
        average_total_equity (pd.Series): The average of the company's total equity over the
        period ends that open and close the period over which the flows are measured.

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
        period ends that open and close the period over which the flows are measured.
        average_total_equity (pd.Series): The average of the company's total equity over the
        period ends that open and close the period over which the flows are measured.

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


def get_nissim_penman_decomposition(
    net_income: pd.Series | pd.DataFrame,
    net_financial_expense: pd.Series | pd.DataFrame,
    average_net_financial_obligations: pd.Series | pd.DataFrame,
    average_common_equity: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform the Nissim and Penman (2001) decomposition of the return on common equity (ROCE)
    into the return on the operations and the effect of how they are financed.

    The formula is as follows:

        - Operating Income = Net Income + Net Financial Expense
        - Average Net Operating Assets = Average Net Financial Obligations + Average Common Equity
        - Return on Net Operating Assets (RNOA) = Operating Income / Average Net Operating Assets
        - Net Borrowing Cost (NBC) = Net Financial Expense / Average Net Financial Obligations
        - Spread = RNOA - NBC
        - Financial Leverage (FLEV) = Average Net Financial Obligations / Average Common Equity
        - Leverage Effect = FLEV x Spread
        - Return on Common Equity (ROCE) = RNOA + Leverage Effect

    Also known as: Nissim-Penman decomposition, ROCE decomposition, advanced DuPont analysis.

    Args:
        net_income (pd.Series | pd.DataFrame): The net income attributable to the shareholders.
        net_financial_expense (pd.Series | pd.DataFrame): The after-tax net interest expense.
        average_net_financial_obligations (pd.Series | pd.DataFrame): The average financial
            obligations minus the average financial assets.
        average_common_equity (pd.Series | pd.DataFrame): The average common equity.

    Returns:
        pd.DataFrame: A DataFrame containing the decomposition components.

    Notes:
    - Since the net operating assets are the net financial obligations plus the common equity
    and the operating income is the net income plus the net financial expense, the components
    add up to Net Income / Average Common Equity exactly: RNOA + FLEV x (RNOA - NBC) =
    (Operating Income - Net Financial Expense) / Average Common Equity. Averages keep the
    identity, as the average of a sum is the sum of the averages.
    """
    return_on_net_operating_assets = (
        profitability_model.get_return_on_net_operating_assets(
            net_income,
            net_financial_expense,
            average_net_financial_obligations,
            average_common_equity,
        )
    )
    net_borrowing_cost = profitability_model.get_net_borrowing_cost(
        net_financial_expense, average_net_financial_obligations
    )
    financial_leverage = profitability_model.get_financial_leverage(
        average_net_financial_obligations, average_common_equity
    )
    spread = return_on_net_operating_assets - net_borrowing_cost
    leverage_effect = financial_leverage * spread

    components = {
        "Return on Net Operating Assets": return_on_net_operating_assets,
        "Net Borrowing Cost": net_borrowing_cost,
        "Spread": spread,
        "Financial Leverage": financial_leverage,
        "Leverage Effect": leverage_effect,
        "Return on Common Equity": return_on_net_operating_assets + leverage_effect,
    }

    if isinstance(return_on_net_operating_assets, pd.DataFrame):
        return (
            pd.concat(components)
            .swaplevel(1, 0)
            .sort_index(level=0, sort_remaining=False)
        )

    return pd.DataFrame.from_dict(components, orient="index")
