"""Altman Module"""

__docformat__ = "google"

import pandas as pd


def get_working_capital_to_total_assets_ratio(
    working_capital: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Working Capital to Total Assets Ratio is a financial metric used to measure a company's liquidity and
    ability to meet short-term obligations. It represents the proportion of a company's total assets that are
    financed by its working capital.

    The formula is as follows:

        - Working Capital to Total Assets Ratio = Working Capital / Total Assets

    Args:
        working_capital (float | pd.Series | pd.DataFrame): The difference between a company's current assets and
        its current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Working Capital to Total Assets Ratio.

    Notes:
    - A ratio of less than 1 indicates that a company may have difficulty meeting its short-term obligations.
    """

    return working_capital / total_assets


def get_retained_earnings_to_total_assets_ratio(
    retained_earnings: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Retained Earnings to Total Assets Ratio is a financial metric used to measure a company's profitability
    and the proportion of its assets that are financed by its retained earnings.

    The formula is as follows:

        - Retained Earnings to Total Assets Ratio = Retained Earnings / Total Assets

    Args:
        retained_earnings (float | pd.Series | pd.DataFrame): The portion of a company's net income that is
        not paid out as dividends.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Retained Earnings to Total Assets Ratio.

    Notes:
    - A high ratio indicates that a company has been profitable and has reinvested its earnings into the business.
    """

    return retained_earnings / total_assets


def get_earnings_before_interest_and_taxes_to_total_assets_ratio(
    ebit: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Earnings Before Interest and Taxes to Total Assets Ratio is a financial metric used to measure a company's
    profitability and the efficiency with which it uses its assets.

    The formula is as follows:

        - Earnings Before Interest and Taxes to Total Assets Ratio = EBIT / Total Assets

    Args:
        ebit (float | pd.Series | pd.DataFrame): Earnings before interest and taxes.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Earnings Before Interest and Taxes to Total Assets Ratio.

    Notes:
    - A high ratio indicates that a company is generating a high return on its assets.
    """

    return ebit / total_assets


def get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
    market_value_of_equity: float | pd.Series | pd.DataFrame,
    total_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Market Value of Equity to Book Value of Total Liabilities Ratio is a financial metric used to measure a
    company's solvency and the proportion of its liabilities that are covered by its equity.

    The formula is as follows:

        - Market Value of Equity to Book Value of Total Liabilities Ratio = Market Value of Equity / Total Liabilities

    Args:
        market_value_of_equity (float | pd.Series | pd.DataFrame): The market value of a company's equity.
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current liabilities.

    Returns:
        float | pd.Series | pd.DataFrame: The Market Value of Equity to Book Value of Total Liabilities Ratio.

    Notes:
    - A ratio of less than 1 indicates that a company may have difficulty meeting its long-term obligations.
    """

    return market_value_of_equity / total_liabilities


def get_sales_to_total_assets_ratio(
    sales: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Sales to Total Assets Ratio is a financial metric used to measure a company's efficiency and the
    proportion of its assets that are used to generate sales.

    The formula is as follows:

        - Sales to Total Assets Ratio = Sales / Total Assets

    Args:
        sales (float | pd.Series | pd.DataFrame): The market value of a company's equity.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current liabilities.

    Returns:
        float | pd.Series | pd.DataFrame: The Market Value of Equity to Book Value of Total Liabilities Ratio.

    Notes:
    - A ratio of less than 1 indicates that a company may have difficulty meeting its long-term obligations.
    """

    return sales / total_assets


def get_altman_z_score(
    working_capital_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    retained_earnings_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    earnings_before_interest_and_taxes_to_total_assets_ratio: (
        float | pd.Series | pd.DataFrame
    ),
    market_value_of_equity_to_book_value_of_total_liabilities_ratio: (
        float | pd.Series | pd.DataFrame
    ),
    sales_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
):
    """
    The Altman Z-Score is a financial metric used to measure a company's solvency and the likelihood that it will
    go bankrupt. It is calculated using a combination of five weighted business ratios.

    The formula is as follows:

    Altman Z-Score = 1.2 * Working Capital to Total Assets Ratio + 1.4 * Retained Earnings to Total Assets Ratio
    + 3.3 * Earnings Before Interest and Taxes to Total Assets Ratio
    + 0.6 * Market Value of Equity to Book Value of Total Liabilities Ratio + 1.0 * Sales to Total Assets Ratio

    Args:
        working_capital_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Working Capital to Total
        Assets Ratio.
        retained_earnings_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Retained Earnings to
        Total Assets Ratio.
        earnings_before_interest_and_taxes_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Earnings
        Before Interest and Taxes to Total Assets Ratio.
        market_value_of_equity_to_book_value_of_total_liabilities_ratio (float | pd.Series | pd.DataFrame): The Market
        Value of Equity to Book Value of Total Liabilities Ratio.
        sales_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Sales to Total Assets Ratio.

    Returns:
        float | pd.Series | pd.DataFrame: The Altman Z-Score.
    """
    return (
        1.2 * working_capital_to_total_assets_ratio
        + 1.4 * retained_earnings_to_total_assets_ratio
        + 3.3 * earnings_before_interest_and_taxes_to_total_assets_ratio
        + 0.6 * market_value_of_equity_to_book_value_of_total_liabilities_ratio
        + 1.0 * sales_to_total_assets_ratio
    )
