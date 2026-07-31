"""Zmijewski Module"""

__docformat__ = "google"

import pandas as pd
from scipy.stats import norm


def get_net_income_to_total_assets_ratio(
    net_income: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Net Income to Total Assets Ratio measures a company's profitability relative to the size of its
    balance sheet. It is the profitability component of the Zmijewski Score.

    The formula is as follows:

        - Net Income to Total Assets Ratio = Net Income / Total Assets

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income (or loss) of a company for the
        period.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Net Income to Total Assets Ratio.

    Notes:
    - A lower (or negative) ratio indicates weaker profitability, which raises the Zmijewski Score and,
    in turn, the estimated probability of financial distress.
    """
    return net_income / total_assets


def get_total_liabilities_to_total_assets_ratio(
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Total Liabilities to Total Assets Ratio measures the proportion of a company's assets that are
    financed through debt rather than equity. It is the leverage component of the Zmijewski Score.

    The formula is as follows:

        - Total Liabilities to Total Assets Ratio = Total Liabilities / Total Assets

    Args:
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and
        non-current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Total Liabilities to Total Assets Ratio.

    Notes:
    - A higher ratio indicates higher leverage, which raises the Zmijewski Score and, in turn, the
    estimated probability of financial distress.
    """
    return total_liabilities / total_assets


def get_current_assets_to_current_liabilities_ratio(
    current_assets: float | pd.Series | pd.DataFrame,
    current_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Current Assets to Current Liabilities Ratio measures a company's short-term liquidity, i.e. its
    ability to cover short-term obligations with short-term resources. It is the liquidity component of
    the Zmijewski Score.

    The formula is as follows:

        - Current Assets to Current Liabilities Ratio = Current Assets / Current Liabilities

    Args:
        current_assets (float | pd.Series | pd.DataFrame): The short-term assets of a company that are
        expected to be converted to cash within one year.
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company
        due within one year.

    Returns:
        float | pd.Series | pd.DataFrame: The Current Assets to Current Liabilities Ratio.

    Notes:
    - This is the standard Current Ratio. Within the Zmijewski Score it carries a small negative
    coefficient, so a higher (stronger) liquidity ratio slightly lowers the score.
    """
    return current_assets / current_liabilities


def get_zmijewski_score(
    net_income_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    total_liabilities_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    current_assets_to_current_liabilities_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Zmijewski Score is a bankruptcy prediction model developed by Mark Zmijewski in 1984. It is a
    simpler, three-variable model compared to both the five-variable Altman Z-Score and the
    nine-variable Ohlson O-Score, and is commonly used as a robustness check when those two models
    disagree, since it is estimated on a different sample and with a different statistical technique
    (a probit model rather than discriminant analysis or logistic regression).

    The formula is as follows:

        X = -4.3 - 4.5 * (Net Income / Total Assets) + 5.7 * (Total Liabilities / Total Assets)
        - 0.004 * (Current Assets / Current Liabilities)

    Also known as: Zmijewski Score, ZFC score, bankruptcy prediction, financial distress score.

    Args:
        net_income_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Net Income to Total
        Assets Ratio.
        total_liabilities_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Total
        Liabilities to Total Assets Ratio.
        current_assets_to_current_liabilities_ratio (float | pd.Series | pd.DataFrame): The Current
        Assets to Current Liabilities Ratio.

    Returns:
        float | pd.Series | pd.DataFrame: The Zmijewski Score (X).

    Notes:
    - A higher Zmijewski Score X indicates a higher likelihood of financial distress.
    - The raw X value is not itself a probability. Use `get_zmijewski_bankruptcy_probability` to convert
    it into a probability of bankruptcy via the standard normal cumulative distribution function
    (probit link), NOT the logistic transform used for the Ohlson O-Score.

    References:
    - Zmijewski, Mark E. "Methodological Issues Related to the Estimation of Financial Distress
    Prediction Models." Journal of Accounting Research, Vol. 22, 1984, pp. 59-82.
    """
    return (
        -4.3
        - 4.5 * net_income_to_total_assets_ratio
        + 5.7 * total_liabilities_to_total_assets_ratio
        - 0.004 * current_assets_to_current_liabilities_ratio
    )


def get_zmijewski_bankruptcy_probability(
    zmijewski_score: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Converts the Zmijewski Score into a probability of bankruptcy using the cumulative distribution
    function (CDF) of the standard normal distribution, i.e. a probit link. This is deliberately
    different from the Ohlson O-Score's logistic link — Zmijewski's model is estimated with a probit
    regression, so applying a logistic transform here (a common mistake when the two models are
    implemented side by side) would produce the wrong probability.

    The formula is as follows:

        - P(bankruptcy) = Phi(X)

    Where Phi is the cumulative distribution function of the standard normal distribution and X is the
    Zmijewski Score.

    Args:
        zmijewski_score (float | pd.Series | pd.DataFrame): The Zmijewski Score (X).

    Returns:
        float | pd.Series | pd.DataFrame: The estimated probability of bankruptcy, bounded between 0
        and 1.

    References:
    - Zmijewski, Mark E. "Methodological Issues Related to the Estimation of Financial Distress
    Prediction Models." Journal of Accounting Research, Vol. 22, 1984, pp. 59-82.
    """
    probability = norm.cdf(zmijewski_score)

    if isinstance(zmijewski_score, pd.DataFrame):
        return pd.DataFrame(
            probability,
            index=zmijewski_score.index,
            columns=zmijewski_score.columns,
        )

    if isinstance(zmijewski_score, pd.Series):
        return pd.Series(probability, index=zmijewski_score.index)

    return probability
