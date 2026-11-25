"""Springate Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.utilities.statistics_model import drop_non_finite


def get_ebt_to_current_liabilities_ratio(
    ebt: float | pd.Series | pd.DataFrame,
    current_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The EBT to Current Liabilities Ratio is a financial metric used to measure a company's
    ability to cover its short-term obligations out of its pre-tax earnings.

    It is the solvency component of the Springate Score.

    The formula is as follows:

        - EBT to Current Liabilities Ratio = Earnings Before Taxes / Current Liabilities

    Args:
        ebt (float | pd.Series | pd.DataFrame): Earnings before taxes (income before tax).
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company due
        within one year.

    Returns:
        float | pd.Series | pd.DataFrame: The EBT to Current Liabilities Ratio.

    Notes:
    - A low (or negative) ratio indicates weak coverage of short-term liabilities out of pre-tax earnings.
    """

    return drop_non_finite(ebt / current_liabilities)


def get_springate_score(
    working_capital_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    ebit_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    ebt_to_current_liabilities_ratio: float | pd.Series | pd.DataFrame,
    sales_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Springate Score is a financial metric, developed by Gordon L.V. Springate in 1978,
    used to measure a company's solvency and the likelihood that it will go bankrupt.

    Springate built the model using a step-wise multiple discriminant analysis on 19
    possible financial ratios, of which four were ultimately selected. It follows the same
    discriminant-analysis methodology as the Altman Z-Score, using a smaller,
    Canadian-firm-calibrated set of ratios.

    The formula is as follows:

        Springate Score = 1.03 * Working Capital to Total Assets Ratio + 3.07 * EBIT to Total Assets Ratio +
          0.66 * EBT to Current Liabilities Ratio + 0.4 * Sales to Total Assets Ratio

    Also known as: Springate Score, S-Score, bankruptcy prediction, financial distress score.

    Args:
        working_capital_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Working Capital to Total
        Assets Ratio.
        ebit_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The EBIT to Total Assets Ratio.
        ebt_to_current_liabilities_ratio (float | pd.Series | pd.DataFrame): The EBT to Current Liabilities
        Ratio.
        sales_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Sales to Total Assets Ratio.

    Returns:
        float | pd.Series | pd.DataFrame: The Springate Score.

    Notes:
    - A Springate Score of less than 0.862 indicates a high likelihood of bankruptcy. In his original study,
    Springate found this cutoff correctly classified companies with over 92% accuracy on the sample used to
    build the model, though as with any model of this type, out-of-sample accuracy on other populations of
    companies (in particular companies outside of Canada or well outside the model's original size range)
    tends to be materially lower.
    - As with the Altman Z-Score, this is a probabilistic, not a definitive, indicator and should be combined
    with further fundamental analysis.

    References:
    - Springate, Gordon L.V. "Predicting the Possibility of Failure in a Canadian Firm." Unpublished M.B.A.
    Research Project, Simon Fraser University, 1978.
    """
    if not isinstance(
        working_capital_to_total_assets_ratio, int | float | pd.Series | pd.DataFrame
    ):
        raise TypeError(
            "working_capital_to_total_assets_ratio must be a float, pd.Series or "
            f"pd.DataFrame, not {type(working_capital_to_total_assets_ratio)}."
        )

    return (
        1.03 * working_capital_to_total_assets_ratio
        + 3.07 * ebit_to_total_assets_ratio
        + 0.66 * ebt_to_current_liabilities_ratio
        + 0.4 * sales_to_total_assets_ratio
    )
