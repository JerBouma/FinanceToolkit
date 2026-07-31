"""Grover Module"""

__docformat__ = "google"

import pandas as pd


def get_working_capital_to_total_assets_ratio(
    working_capital: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Working Capital to Total Assets Ratio is a financial metric used to measure a company's liquidity and
    ability to meet short-term obligations. It represents the proportion of a company's total assets that are
    financed by its working capital. It is the liquidity component of the Grover Score.

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


def get_ebit_to_total_assets_ratio(
    ebit: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The EBIT to Total Assets Ratio is a financial metric used to measure a company's profitability and the
    efficiency with which it uses its assets. It is the profitability component of the Grover Score.

    The formula is as follows:

        - EBIT to Total Assets Ratio = EBIT / Total Assets

    Args:
        ebit (float | pd.Series | pd.DataFrame): Earnings before interest and taxes.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The EBIT to Total Assets Ratio.

    Notes:
    - A high ratio indicates that a company is generating a high return on its assets.
    """

    return ebit / total_assets


def get_return_on_assets_ratio(
    net_income: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Return on Assets (ROA) Ratio measures a company's profitability relative to the size of its balance
    sheet. It is the second profitability component of the Grover Score, alongside the EBIT to Total Assets
    Ratio.

    The formula is as follows:

        - Return on Assets Ratio = Net Income / Total Assets

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income (or loss) of a company for the period.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Return on Assets Ratio.

    Notes:
    - Unlike some other Return on Assets calculations elsewhere in this toolkit, this uses the point-in-time
    Total Assets balance rather than the average of the beginning and ending balance, matching the original
    Grover (2001) specification and the point-in-time convention used by the Altman Z-Score and Ohlson O-Score
    elsewhere in this module.
    """

    return net_income / total_assets


def get_grover_score(
    working_capital_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    ebit_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    return_on_assets_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Grover Score is a financial metric, developed by Jeffrey S. Grover in 2001, used to measure a
    company's solvency and the likelihood that it will go bankrupt. Grover built the model by adding a
    Return on Assets term to, and re-estimating the coefficients of, a reduced-form version of the Altman
    Z-Score, using a sample that paired each of the original Altman bankrupt firms with a matched
    non-bankrupt firm from the same industry and year.

    The formula is as follows:

        Grover Score = 1.65 * Working Capital to Total Assets Ratio + 3.404 * EBIT to Total Assets Ratio
        - 0.016 * Return on Assets Ratio + 0.057

    Also known as: Grover Score, G-Score, bankruptcy prediction, financial distress score.

    Args:
        working_capital_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Working Capital to Total
        Assets Ratio.
        ebit_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The EBIT to Total Assets Ratio.
        return_on_assets_ratio (float | pd.Series | pd.DataFrame): The Return on Assets Ratio (Net Income /
        Total Assets).

    Returns:
        float | pd.Series | pd.DataFrame: The Grover Score.

    Notes:
    - A Grover Score of -0.02 or lower indicates a high likelihood of bankruptcy (Grover's original
    "bankrupt" classification threshold).
    - Some secondary sources additionally report a Grover Score of 0.01 or higher as the "non-bankrupt"
    classification threshold, leaving a gray area in between, analogous to the gray area between Altman's
    1.81 and 2.99 cutoffs. This implementation reports the raw score only; interpret values between -0.02
    and 0.01 with additional caution.
    - As with the Altman Z-Score and Ohlson O-Score, this is a probabilistic, not a definitive, indicator and
    should be combined with further fundamental analysis.

    References:
    - Grover, Jeffrey S. "Validating the Grover Bankruptcy Model." Doctoral dissertation, University of North
    Texas, 2003.
    """
    if not isinstance(
        working_capital_to_total_assets_ratio, int | float | pd.Series | pd.DataFrame
    ):
        raise TypeError(
            "working_capital_to_total_assets_ratio must be a float, pd.Series or "
            f"pd.DataFrame, not {type(working_capital_to_total_assets_ratio)}."
        )

    return (
        1.65 * working_capital_to_total_assets_ratio
        + 3.404 * ebit_to_total_assets_ratio
        - 0.016 * return_on_assets_ratio
        + 0.057
    )
