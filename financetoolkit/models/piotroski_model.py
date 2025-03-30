"""Altman Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit import helpers
from financetoolkit.ratios import (
    efficiency_model,
    liquidity_model,
    profitability_model,
    solvency_model,
)


def get_return_on_assets_criteria(
    net_income: float | pd.Series | pd.DataFrame,
    average_total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculates the Return on Assets (ROA) criteria for the Piotroski F-Score model.

    The Return on Assets (ROA) is a financial metric used to measure a company's profitability by
    showing how much profit a company generates from its total assets. The ROA is calculated by
    dividing the net income by the total assets.

    The formula is as follows:

        - Return on Assets (ROA) = (Net Income Beginning + Net Income End) /
        (Total Assets Beginning + Total Assets End)
        - Return on Assets Criteria = Return on Assets (ROA) > 0

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income of the company.
        average_total_assets (float | pd.Series | pd.DataFrame): The average total assets of the company.

    Returns:
        float | pd.Series | pd.DataFrame: A boolean value indicating whether the company meets the
        Return on Assets (ROA) criteria.

    Notes:
    - A positive Return on Assets (ROA) indicates that a company is generating profit from its assets.
    """

    return_on_assets = profitability_model.get_return_on_assets(
        net_income=net_income,
        average_total_assets=average_total_assets,
    )

    return_on_assets_criteria = return_on_assets > 0

    return return_on_assets_criteria


def get_operating_cashflow_criteria(
    operating_cashflow: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculates the operating cash flow criteria for the Piotroski F-Score model.

    The operating cash flow criteria is a binary metric that indicates whether a company's operating cash
    flow is positive or not. A positive operating cash flow is considered a good sign for a company's
    financial health.

    The formula is as follows:

            - Operating Cash Flow Criteria = Operating Cash Flow > 0

    Args:
        operating_cashflow (float | pd.Series | pd.DataFrame): The operating cash flow of a company.

    Returns:
        float | pd.Series | pd.DataFrame: A binary metric indicating whether the
        operating cash flow is positive (1) or not (0).
    """
    operating_cashflow_criteria = operating_cashflow > 0

    return operating_cashflow_criteria


def get_change_in_return_on_asset_criteria(
    net_income: float | pd.Series | pd.DataFrame,
    average_total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculates the change in the return on assets criteria for a company based o
    its net income and total assets.

    The function calculates the return on assets (ROA) for a company based on its
    net income and total assets at the beginning and end of a period. It then calculates
    the growth rate of the ROA over the period, and returns a boolean value indicating
    whether the growth rate has increased compared to the previous period.

    The formula is as follows:

        - Return on Assets (ROA) = (Net Income Beginning + Net Income End) /
        (Total Assets Beginning + Total Assets End)
        - Return on Assets Growth = Return on Assets (ROA) / Return on Assets (ROA) Shifted
        - Change in Return on Assets Criteria = Return on Assets Growth > 0

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income of the company for the period.
        average_total_assets (float | pd.Series | pd.DataFrame): The average total assets of the company
        for the period.

    Returns:
        float | pd.Series | pd.DataFrame: A boolean value indicating whether the growth rate of the ROA has increased
        compared to the previous period.

    """
    return_on_assets = profitability_model.get_return_on_assets(
        net_income=net_income, average_total_assets=average_total_assets
    )

    return_on_assets_growth = helpers.calculate_growth(return_on_assets)

    change_in_return_on_assets_criteria = (
        return_on_assets_growth > return_on_assets_growth.shift(1, axis=1)
    )

    return change_in_return_on_assets_criteria


def get_accruals_criteria(
    net_income: float | pd.Series | pd.DataFrame,
    average_total_assets: float | pd.Series | pd.DataFrame,
    operating_cashflow: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculates the accruals criteria for a company based on its net income, total assets, and operating cashflow.

    The accruals criteria is a financial metric used to measure a company's earnings quality. It represents the
    difference between a company's operating cashflow and its return on assets. A positive accruals criteria indicates
    that a company's earnings are of higher quality, while a negative accruals criteria indicates that a company's
    earnings are of lower quality.

    The formula is as follows:

        - Accruals Criteria = (Operating Cashflow / Total Assets) > Return on Assets

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income of the company.
        average_total_assets (float | pd.Series | pd.DataFrame): The average total assets of the company.
        operating_cashflow (float | pd.Series | pd.DataFrame): The operating cashflow of the company.
        total_assets (float | pd.Series | pd.DataFrame): The total assets of the company.

    Returns:
        float | pd.Series | pd.DataFrame: The accruals criteria for the company.
    """

    return_on_assets = profitability_model.get_return_on_assets(
        net_income=net_income, average_total_assets=average_total_assets
    )

    operating_cf_to_total_assets = operating_cashflow / total_assets

    accruals_criteria = operating_cf_to_total_assets > return_on_assets

    return accruals_criteria


def get_change_in_leverage_criteria(
    total_debt: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Calculate the criteria for evaluating changes in leverage by comparing the Debt to Assets Ratio over time.

    The Debt to Assets Ratio is a financial metric used to assess a company's solvency and leverage. It measures the
    proportion of a company's total assets that are financed by debt.

    The formula is as follows:

        - Debt to Assets Ratio = Total Debt / Total Assets

    Args:
        total_debt (float | pd.Series | pd.DataFrame): The total debt of a company, which can
        be a float or a time-series.
        total_assets (float | pd.Series | pd.DataFrame): The total assets of a company, which can
        be a float or a time-series.

    Returns:
        float | pd.Series | pd.DataFrame: A boolean criteria indicating whether the Debt to Assets
        Ratio decreased compared to the previous period (True for decrease, False for increase or no change).

    Notes:
    - A True value in the criteria indicates a decrease in leverage, while a False value suggests an
    increase or no change.
    - This function can be used to monitor changes in a company's leverage over time.
    """
    debt_ratio = solvency_model.get_debt_to_assets_ratio(
        total_debt=total_debt, total_assets=total_assets
    )

    debt_ratio_criteria = debt_ratio < debt_ratio.shift(1, axis=1)

    return debt_ratio_criteria


def get_change_in_current_ratio_criteria(
    current_assets: pd.Series | pd.DataFrame,
    current_liabilities: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate criteria for evaluating changes in the Current Ratio over time.

    The Current Ratio is a financial metric used to assess a company's short-term liquidity and its ability
    to meet short-term obligations. It measures the relationship between current assets and current liabilities.

    The formula is as follows:

        - Current Ratio = Current Assets / Current Liabilities
        - Current Ratio Criteria = Current Ratio > Current Ratio Shifted

    Args:
        current_assets (pd.Series | pd.DataFrame): The current assets of a company, which can
        be a time-series.
        current_liabilities (pd.Series | pd.DataFrame): The current liabilities of a company, which
        can be a time-series.

    Returns:
        pd.Series | pd.DataFrame: A boolean criteria indicating whether the Current Ratio increased compared
        to the previous period (True for increase, False for decrease or no change).

    Notes:
    - A True value in the criteria indicates an improvement in liquidity, while a False value
    suggests a decrease or no improvement.
    - This function can be used to monitor changes in a company's short-term liquidity over time.
    """
    current_ratio = liquidity_model.get_current_ratio(
        current_assets=current_assets, current_liabilities=current_liabilities
    )

    current_ratio_criteria = current_ratio > current_ratio.shift(1, axis=1)

    return current_ratio_criteria


def get_number_of_shares_criteria(
    common_stock_issued: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate criteria for evaluating the number of common shares issued by a company.

    This function assesses whether a company has issued common stock. It's a key indicator of
    the company's ownership structure and financing activities.

    Args:
        common_stock_issued (pd.Series | pd.DataFrame): A time-series or DataFrame representing the
        number of common shares issued by a company.

    Returns:
        pd.Series | pd.DataFrame: A boolean criteria indicating whether the company issued any
        common shares during the specified time period (True for issued shares, False
        for no shares issued).

    Notes:
    - A True value in the criteria indicates that common shares were issued, while a False value suggests
    no issuance of common shares.
    - This function can be used to monitor changes in the company's ownership structure and
    financing activities.
    """
    number_of_shares_criteria = common_stock_issued == 0

    return number_of_shares_criteria


def get_gross_margin_criteria(
    revenue: pd.Series | pd.DataFrame,
    cost_of_goods_sold: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate criteria for evaluating changes in the Gross Margin over time.

    The Gross Margin is a key profitability metric that measures the difference between a company's revenue and
    the cost of goods sold. It reflects the profitability of a company's core operations.

    Args:
        revenue (pd.Series | pd.DataFrame): The revenue generated by a company, which can be a time-series.
        cost_of_goods_sold (pd.Series | pd.DataFrame): The cost of goods sold by a company, which can be a time-series.

    Returns:
        pd.Series | pd.DataFrame: A boolean criteria indicating whether the Gross Margin increased compared to the
        previous period (True for increase, False for decrease or no change).

    Notes:
    - A True value in the criteria indicates an improvement in profitability, while a False value
    suggests a decrease or no improvement.
    - This function can be used to monitor changes in a company's core operational profitability over time.
    """
    gross_margin = profitability_model.get_gross_margin(
        revenue=revenue, cost_of_goods_sold=cost_of_goods_sold
    )

    gross_margin_criteria = gross_margin > gross_margin.shift(1, axis=1)

    return gross_margin_criteria


def get_asset_turnover_ratio_criteria(
    sales: pd.Series | pd.DataFrame,
    average_total_assets: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate criteria for evaluating changes in the Asset Turnover Ratio over time.

    The Asset Turnover Ratio is a fundamental efficiency metric that assesses how effectively a company generates sales
    from its total assets. It indicates the company's ability to efficiently utilize its assets to generate revenue.

    Args:
        sales (pd.Series | pd.DataFrame): The sales or revenue generated by a company, which can
        be a time-series or DataFrame.
        average_total_assets (pd.Series | pd.DataFrame): The average total assets of a company, which can
        be a time-series or DataFrame.

    Returns:
        pd.Series | pd.DataFrame: A boolean criteria indicating whether the Asset Turnover Ratio increased compared to the
        previous period (True for increase, False for decrease or no change).

    Notes:
    - A True value in the criteria suggests improved asset utilization and efficiency, while a False value
    indicates a decrease or no improvement.
    - This function can be used to monitor changes in a company's ability to generate revenue from its assets over time.
    """
    asset_turnover_ratio = efficiency_model.get_asset_turnover_ratio(
        sales=sales, average_total_assets=average_total_assets
    )

    asset_turnover_ratio_criteria = asset_turnover_ratio > asset_turnover_ratio.shift(
        1, axis=1
    )

    return asset_turnover_ratio_criteria


def get_piotroski_score(
    return_on_assets_criteria: float | pd.Series | pd.DataFrame,
    operating_cashflow_criteria: float | pd.Series | pd.DataFrame,
    change_in_return_on_asset_criteria: float | pd.Series | pd.DataFrame,
    accruals_criteria: float | pd.Series | pd.DataFrame,
    change_in_leverage_criteria: float | pd.Series | pd.DataFrame,
    change_in_current_ratio_criteria: float | pd.Series | pd.DataFrame,
    number_of_shares_criteria: float | pd.Series | pd.DataFrame,
    gross_margin_criteria: float | pd.Series | pd.DataFrame,
    asset_turnover_ratio_criteria: float | pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Piotroski Score, a comprehensive financial assessment tool that helps investors and analysts
    evaluate a company's financial health and fundamental strength.

    The Piotroski Score was developed by Joseph Piotroski and is based on a set of nine fundamental financial criteria.
    Each criterion is assigned a score of 0 or 1, and the scores are then summed to calculate the Piotroski Score.

    The nine criteria are categorized into three groups:

    1. Profitability:
        - Return on Assets (ROA) Criteria: Measures the profitability of the company.
        - Operating Cash Flow Criteria: Evaluates the company's ability to generate cash from its operations.
        - Change in ROA Criteria: Assesses the trend in ROA over time.
        - Accruals Criteria: Examines the quality of earnings.

    2. Leverage, Liquidity, and Operating Efficiency:
        - Change in Leverage Criteria: Analyzes changes in the company's leverage (debt).
        - Change in Current Ratio Criteria: Evaluates changes in the current ratio.
        - Number of Shares Criteria: Assesses the issuance of common shares.

    3. Operating Efficiency and Asset Utilization:
        - Gross Margin Criteria: Examines the company's gross margin, a measure of profitability.
        - Asset Turnover Ratio Criteria: Evaluates the efficiency of asset utilization and sales generation.

    The Piotroski Score is calculated by summing the scores assigned to each of the nine criteria.
    The maximum possible score is 9, indicating the highest financial strength, while the minimum score is 0,
    suggesting potential financial weaknesses.

    Args:
        return_on_assets_criteria (float | pd.Series | pd.DataFrame): Criteria indicating
        the company's return on assets.
        operating_cashflow_criteria (float | pd.Series | pd.DataFrame): Criteria indicating the company's
        operating cash flow.
        change_in_return_on_asset_criteria (float | pd.Series | pd.DataFrame): Criteria indicating changes
        in return on assets.
        accruals_criteria (float | pd.Series | pd.DataFrame): Criteria indicating
        the quality of earnings.
        change_in_leverage_criteria (float | pd.Series | pd.DataFrame): Criteria indicating changes
        in leverage.
        change_in_current_ratio_criteria (float | pd.Series | pd.DataFrame): Criteria indicating changes
        in the current ratio.
        number_of_shares_criteria (float | pd.Series | pd.DataFrame): Criteria indicating the issuance
        of common shares.
        gross_margin_criteria (float | pd.Series | pd.DataFrame): Criteria indicating the company's
        gross margin.
        asset_turnover_ratio_criteria (float | pd.Series | pd.DataFrame): Criteria indicating the company's
        asset turnover ratio.

    Returns:
        pd.Series | pd.DataFrame: The Piotroski Score, a numerical representation of a company's
        fundamental financial strength.
        A higher score indicates stronger financial health, while a lower score may suggest potential weaknesses.

    Notes:
    - A high Piotroski Score can be seen as a positive signal for investors, indicating that a company
    meets several favorable financial criteria.
    - It is essential to interpret the Piotroski Score in conjunction with other financial analysis and market research,
      as it provides insights into a company's financial condition but may not capture all aspects of its performance.

    References:
    - Piotroski, Joseph D. "Value Investing: The Use of Historical Financial Statement Information to
    Separate Winners from Losers." Journal of Accounting Research, Vol. 38, No. 3, 1999, pp. 1-41.
    """
    if isinstance(
        return_on_assets_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        return_on_assets_criteria = return_on_assets_criteria.astype(int)

    if isinstance(
        operating_cashflow_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        operating_cashflow_criteria = operating_cashflow_criteria.astype(int)

    if isinstance(
        change_in_return_on_asset_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        change_in_return_on_asset_criteria = change_in_return_on_asset_criteria.astype(
            int
        )

    if isinstance(
        accruals_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        accruals_criteria = accruals_criteria.astype(int)

    if isinstance(
        change_in_leverage_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        change_in_leverage_criteria = change_in_leverage_criteria.astype(int)

    if isinstance(
        change_in_current_ratio_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        change_in_current_ratio_criteria = change_in_current_ratio_criteria.astype(int)

    if isinstance(
        number_of_shares_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        number_of_shares_criteria = number_of_shares_criteria.astype(int)

    if isinstance(
        gross_margin_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        gross_margin_criteria = gross_margin_criteria.astype(int)

    if isinstance(
        asset_turnover_ratio_criteria,
        (
            pd.Series,
            pd.DataFrame,
        ),
    ):
        asset_turnover_ratio_criteria = asset_turnover_ratio_criteria.astype(int)

    piotroski_score = (
        return_on_assets_criteria
        + operating_cashflow_criteria
        + change_in_return_on_asset_criteria
        + accruals_criteria
        + change_in_leverage_criteria
        + change_in_current_ratio_criteria
        + number_of_shares_criteria
        + gross_margin_criteria
        + asset_turnover_ratio_criteria
    )

    return piotroski_score
