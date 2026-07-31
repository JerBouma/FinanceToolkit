"""Ohlson Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# pylint: disable=too-many-arguments,too-many-positional-arguments


def get_log_of_total_assets(
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Log of Total Assets (SIZE) is a proxy for firm size used in the Ohlson O-Score model. Larger
    firms are, all else equal, considered less likely to go bankrupt, so this term enters the O-Score
    with a negative coefficient.

    The formula is as follows:

        - SIZE = ln(Total Assets / GNP Price-Level Index)

    Args:
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The (simplified) Log of Total Assets.

    Notes:
    - Ohlson's original 1980 specification deflates Total Assets by the US GNP implicit price-level
      deflator (rebased to 1968 = 100) before taking the natural log. That index is a US-specific,
      1970s-vintage macro series that is not available in this toolkit and is not meaningful outside
      the original US sample period, so it is simplified/omitted here — this function returns
      ln(Total Assets) in nominal terms instead of ln(Total Assets / GNP index).
    - Because the deflator is (approximately) constant across companies observed in the same period and
      currency, omitting it shifts the SIZE term, and therefore the resulting O-Score, by roughly the
      same constant amount for every observation in that period. This does not change the *ranking* of
      companies relative to one another, but it does mean the absolute O-Score (and the bankruptcy
      probability derived from it) is not an exact reproduction of Ohlson (1980) and should not be
      compared directly to probabilities reported in studies that do apply the deflator.
    """
    return np.log(total_assets)


def get_total_liabilities_to_total_assets_ratio(
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Total Liabilities to Total Assets Ratio (TLTA) measures the proportion of a company's assets
    that are financed through debt rather than equity. It is one of the leverage components of the
    Ohlson O-Score.

    The formula is as follows:

        - TLTA = Total Liabilities / Total Assets

    Args:
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and
        non-current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Total Liabilities to Total Assets Ratio.

    Notes:
    - A higher TLTA indicates higher leverage and, all else equal, a higher probability of bankruptcy.
    """
    return total_liabilities / total_assets


def get_working_capital_to_total_assets_ratio(
    working_capital: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Working Capital to Total Assets Ratio (WCTA) measures a company's short-term liquidity relative
    to the size of its balance sheet. It is one of the liquidity components of the Ohlson O-Score.

    The formula is as follows:

        - WCTA = Working Capital / Total Assets

    Args:
        working_capital (float | pd.Series | pd.DataFrame): The difference between a company's current
        assets and its current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Working Capital to Total Assets Ratio.

    Notes:
    - A lower (or negative) WCTA indicates weaker short-term liquidity, which raises the O-Score.
    """
    return working_capital / total_assets


def get_current_liabilities_to_current_assets_ratio(
    current_liabilities: float | pd.Series | pd.DataFrame,
    current_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Current Liabilities to Current Assets Ratio (CLCA) is the inverse framing of the current ratio
    and measures the extent to which short-term obligations exceed short-term resources. It is one of
    the liquidity components of the Ohlson O-Score.

    The formula is as follows:

        - CLCA = Current Liabilities / Current Assets

    Args:
        current_liabilities (float | pd.Series | pd.DataFrame): The short-term liabilities of a company
        due within one year.
        current_assets (float | pd.Series | pd.DataFrame): The short-term assets of a company that are
        expected to be converted to cash within one year.

    Returns:
        float | pd.Series | pd.DataFrame: The Current Liabilities to Current Assets Ratio.

    Notes:
    - A higher CLCA indicates weaker short-term liquidity, which raises the O-Score.
    """
    return current_liabilities / current_assets


def get_negative_equity_indicator(
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Negative Equity Indicator (OENEG) is a binary dummy variable used in the Ohlson O-Score that
    flags companies whose total liabilities exceed their total assets, i.e. companies with negative
    book equity.

    The formula is as follows:

        - OENEG = 1 if Total Liabilities > Total Assets else 0

    Args:
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and
        non-current liabilities.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: A binary indicator equal to 1 when book equity is negative,
        0 otherwise.

    Notes:
    - Negative book equity is a strong distress signal, hence the negative coefficient this term
      receives in the O-Score (it lowers, i.e. makes more negative, the O-Score contribution).
    """
    return total_liabilities > total_assets


def get_net_income_to_total_assets_ratio(
    net_income: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Net Income to Total Assets Ratio (NITA) measures a company's profitability relative to the size
    of its balance sheet. It is the profitability component of the Ohlson O-Score.

    The formula is as follows:

        - NITA = Net Income / Total Assets

    Args:
        net_income (float | pd.Series | pd.DataFrame): The net income (or loss) of a company for the
        period.
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
        assets.

    Returns:
        float | pd.Series | pd.DataFrame: The Net Income to Total Assets Ratio.

    Notes:
    - Unlike Altman's ROA-style component, NITA is not annualized/averaged across beginning and ending
      total assets — it uses the point-in-time Total Assets balance, matching Ohlson's (1980) original
      specification.
    """
    return net_income / total_assets


def get_funds_from_operations_to_total_liabilities_ratio(
    funds_from_operations: float | pd.Series | pd.DataFrame,
    total_liabilities: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Funds from Operations to Total Liabilities Ratio (FUTL) measures a company's ability to service
    its total debt load from the cash generated by its core operations.

    The formula is as follows:

        - FUTL = Funds from Operations / Total Liabilities

    Args:
        funds_from_operations (float | pd.Series | pd.DataFrame): The funds generated from a company's
        operations.
        total_liabilities (float | pd.Series | pd.DataFrame): The sum of a company's current and
        non-current liabilities.

    Returns:
        float | pd.Series | pd.DataFrame: The Funds from Operations to Total Liabilities Ratio.

    Notes:
    - Ohlson's (1980) original "Funds from Operations" is an accounting-flow concept from a period
      before cash flow statements were standardized. This toolkit does not have that exact historical
      line item available, so — as is standard practice in modern reproductions of the O-Score —
      Operating Cash Flow is used as the approximation for Funds from Operations. Operating Cash Flow is
      typically close to, but not identical to, the original FFO measure (it additionally reflects
      changes in working capital), so treat FUTL as an approximation rather than an exact reproduction.
    """
    return funds_from_operations / total_liabilities


def get_negative_income_indicator(
    net_income: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Negative Income Indicator (INTWO) is a binary dummy variable used in the Ohlson O-Score that
    flags companies that reported a net loss in both the current period and the prior period.

    The formula is as follows:

        - INTWO = 1 if Net Income < 0 for both the current and prior period else 0

    Args:
        net_income (pd.Series | pd.DataFrame): The net income (or loss) of a company, as a time-series.

    Returns:
        pd.Series | pd.DataFrame: A binary indicator equal to 1 when net income was negative in both
        the current and prior period, 0 otherwise.

    Notes:
    - The first period in any series will always evaluate to 0 (never 1) since there is no prior period
      to compare against.
    """
    current_period_loss = net_income < 0
    prior_period_loss = net_income.shift(1, axis=1) < 0

    return current_period_loss & prior_period_loss


def get_change_in_net_income_ratio(
    net_income: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    The Change in Net Income Ratio (CHIN) measures the direction and relative magnitude of the change in
    net income between the current and prior period, scaled by the combined absolute size of both
    periods' net income so that the measure stays bounded between -1 and 1.

    The formula is as follows:

        - CHIN = (Net Income (t) - Net Income (t-1)) / (|Net Income (t)| + |Net Income (t-1)|)

    Args:
        net_income (pd.Series | pd.DataFrame): The net income (or loss) of a company, as a time-series.

    Returns:
        pd.Series | pd.DataFrame: The Change in Net Income Ratio.

    Notes:
    - A positive CHIN indicates improving profitability compared to the prior period, which lowers the
      O-Score (i.e. lowers the probability of bankruptcy), reflected by CHIN's negative coefficient.
    - The first period in any series will always be NaN since there is no prior period to compare
      against.
    """
    net_income_prior = net_income.shift(1, axis=1)

    return (net_income - net_income_prior) / (net_income.abs() + net_income_prior.abs())


def get_ohlson_o_score(
    log_of_total_assets: float | pd.Series | pd.DataFrame,
    total_liabilities_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    working_capital_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    current_liabilities_to_current_assets_ratio: float | pd.Series | pd.DataFrame,
    negative_equity_indicator: float | pd.Series | pd.DataFrame,
    net_income_to_total_assets_ratio: float | pd.Series | pd.DataFrame,
    funds_from_operations_to_total_liabilities_ratio: float | pd.Series | pd.DataFrame,
    negative_income_indicator: float | pd.Series | pd.DataFrame,
    change_in_net_income_ratio: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Ohlson O-Score is a bankruptcy prediction model developed by James Ohlson in 1980. Unlike the
    Altman Z-Score, which is built with multiple discriminant analysis, the O-Score's coefficients come
    from a fitted logistic regression (logit) model. This is why the two models are usually reported
    side by side rather than one being treated as a replacement for the other: the Z-Score produces a
    score that is only meaningful when compared against Altman's empirically derived threshold bands,
    while the O-Score, run through the logistic transform (see `get_ohlson_bankruptcy_probability`),
    is directly interpretable as a probability of bankruptcy between 0 and 1.

    The formula is as follows:

        O-Score = -1.32 - 0.407 * SIZE + 6.03 * TLTA - 1.43 * WCTA + 0.0757 * CLCA - 1.72 * OENEG
        - 2.37 * NITA - 1.83 * FUTL + 0.285 * INTWO - 0.521 * CHIN

    The nine variables are:

        - SIZE: Log of Total Assets (deflator simplified/omitted, see `get_log_of_total_assets`)
        - TLTA: Total Liabilities to Total Assets Ratio
        - WCTA: Working Capital to Total Assets Ratio
        - CLCA: Current Liabilities to Current Assets Ratio
        - OENEG: Negative Equity Indicator (1 if Total Liabilities > Total Assets, else 0)
        - NITA: Net Income to Total Assets Ratio
        - FUTL: Funds from Operations to Total Liabilities Ratio (Operating Cash Flow used as the
          approximation for Funds from Operations, see `get_funds_from_operations_to_total_liabilities_ratio`)
        - INTWO: Negative Income Indicator (1 if Net Income was negative for the last two years, else 0)
        - CHIN: Change in Net Income Ratio

    Also known as: Ohlson O-Score, bankruptcy prediction, financial distress score.

    Args:
        log_of_total_assets (float | pd.Series | pd.DataFrame): The (simplified) Log of Total Assets (SIZE).
        total_liabilities_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Total Liabilities
        to Total Assets Ratio (TLTA).
        working_capital_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Working Capital to
        Total Assets Ratio (WCTA).
        current_liabilities_to_current_assets_ratio (float | pd.Series | pd.DataFrame): The Current
        Liabilities to Current Assets Ratio (CLCA).
        negative_equity_indicator (float | pd.Series | pd.DataFrame): The Negative Equity Indicator
        (OENEG).
        net_income_to_total_assets_ratio (float | pd.Series | pd.DataFrame): The Net Income to Total
        Assets Ratio (NITA).
        funds_from_operations_to_total_liabilities_ratio (float | pd.Series | pd.DataFrame): The Funds
        from Operations to Total Liabilities Ratio (FUTL).
        negative_income_indicator (float | pd.Series | pd.DataFrame): The Negative Income Indicator
        (INTWO).
        change_in_net_income_ratio (float | pd.Series | pd.DataFrame): The Change in Net Income Ratio
        (CHIN).

    Returns:
        float | pd.Series | pd.DataFrame: The Ohlson O-Score.

    Notes:
    - A higher O-Score indicates a higher likelihood of bankruptcy. Ohlson's (1980) original cutoff is
      approximately 0.38 (in probability terms, after the logistic transform), though as with the Altman
      Z-Score, thresholds should be interpreted with caution and alongside further fundamental analysis.
    - The GNP price-level deflator used in the original SIZE term and the Funds from Operations line item
      used in the original FUTL term are both simplified/approximated in this implementation, see
      `get_log_of_total_assets` and `get_funds_from_operations_to_total_liabilities_ratio` for details.
      This means the absolute O-Score (and derived probability) will not be an exact reproduction of
      Ohlson (1980), while the relative ranking of companies is preserved.

    References:
    - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
    Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.
    """
    if isinstance(negative_equity_indicator, pd.Series | pd.DataFrame):
        negative_equity_indicator = negative_equity_indicator.astype(int)

    if isinstance(negative_income_indicator, pd.Series | pd.DataFrame):
        negative_income_indicator = negative_income_indicator.astype(int)

    return (
        -1.32
        - 0.407 * log_of_total_assets
        + 6.03 * total_liabilities_to_total_assets_ratio
        - 1.43 * working_capital_to_total_assets_ratio
        + 0.0757 * current_liabilities_to_current_assets_ratio
        - 1.72 * negative_equity_indicator
        - 2.37 * net_income_to_total_assets_ratio
        - 1.83 * funds_from_operations_to_total_liabilities_ratio
        + 0.285 * negative_income_indicator
        - 0.521 * change_in_net_income_ratio
    )


def get_ohlson_bankruptcy_probability(
    ohlson_o_score: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    Converts the Ohlson O-Score into a probability of bankruptcy using the logistic (sigmoid) function.
    This is the standard way the O-Score is consumed in practice, unlike the Altman Z-Score which is
    typically compared against threshold bands directly, because the O-Score comes from a fitted
    logistic regression and is therefore only meaningful on the probability scale after this transform.

    The formula is as follows:

        - P(bankruptcy) = 1 / (1 + e^(-O-Score))

    Args:
        ohlson_o_score (float | pd.Series | pd.DataFrame): The Ohlson O-Score.

    Returns:
        float | pd.Series | pd.DataFrame: The estimated probability of bankruptcy, bounded between 0
        and 1.

    Notes:
    - Because this implementation simplifies/omits the GNP price-level deflator (see
      `get_log_of_total_assets`) and approximates Funds from Operations with Operating Cash Flow (see
      `get_funds_from_operations_to_total_liabilities_ratio`), the absolute probability produced here
      will differ somewhat from a probability computed with Ohlson's (1980) exact original inputs. The
      *ranking* of companies by this probability is unaffected by the deflator simplification.

    References:
    - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
    Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.
    """
    return 1 / (1 + np.exp(-ohlson_o_score))
