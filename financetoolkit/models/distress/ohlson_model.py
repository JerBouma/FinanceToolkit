"""Ohlson Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import drop_non_finite


def get_log_of_total_assets(
    total_assets: float | pd.Series | pd.DataFrame,
    price_level_index: float | pd.Series | pd.DataFrame | None = None,
) -> float | pd.Series | pd.DataFrame:
    """
    The Log of Total Assets (SIZE) is a proxy for firm size used in the Ohlson O-Score
    model.

    Larger firms are, all else equal, considered less likely to go bankrupt, so this term
    enters the O-Score with a negative coefficient.

    The formula is as follows:

        - SIZE = ln(Total Assets / GNP Price-Level Index)

    Args:
        total_assets (float | pd.Series | pd.DataFrame): The sum of a company's current and non-current
            assets.
        price_level_index (float | pd.Series | pd.DataFrame | None, optional): The US GNP implicit
            price deflator with 1968 = 100, of the period of the total assets (with a DataFrame of total
            assets, a Series indexed by its columns). Defaults to None, which leaves the total assets
            undeflated: ln(Total Assets).

    Returns:
        float | pd.Series | pd.DataFrame: The Log of Total Assets.

    Notes:
    - Ohlson (1980) deflates Total Assets by the US GNP implicit price-level deflator rebased to
      1968 = 100 (SIZE = ln(Total Assets / index)), which keeps SIZE on the scale the coefficients
      were estimated on. Without the deflator, SIZE is higher by ln(index) and the O-Score lower by
      0.407 times that: about 2.7 with the US price level of the 2020s at an index of about 700; the
      ranking of companies within a period is the same either way.
    - The total assets are taken in the units of the statements (currency units, not thousands or
      millions).
    """
    if price_level_index is not None:
        total_assets = total_assets / price_level_index

    # np.log keeps the pandas container at runtime, but the ufunc stubs describe an
    # ndarray; the ufunc fast path of `.apply` produces the same values with a typed result.
    if isinstance(total_assets, pd.Series | pd.DataFrame):
        return total_assets.apply(np.log)

    return float(np.log(total_assets))


def get_total_liabilities_to_total_assets_ratio(
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Total Liabilities to Total Assets Ratio (TLTA) measures the proportion of a
    company's assets that are financed through debt rather than equity.

    It is one of the leverage components of the Ohlson O-Score and the leverage component of the
    Zmijewski Score.

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
    - A higher TLTA indicates higher leverage and, all else equal, a higher probability of bankruptcy:
      it raises both the Ohlson O-Score and the Zmijewski Score.
    """
    return drop_non_finite(total_liabilities / total_assets)


def get_current_liabilities_to_current_assets_ratio(
    current_liabilities: float | pd.Series | pd.DataFrame,
    current_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Current Liabilities to Current Assets Ratio (CLCA) shows the extent to which
    short-term obligations exceed short-term resources.

    It is one of the liquidity components of the Ohlson O-Score.

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
    return drop_non_finite(current_liabilities / current_assets)


def get_negative_equity_indicator(
    total_liabilities: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Negative Equity Indicator (OENEG) flags companies whose total liabilities exceed
    their total assets.

    It is a binary dummy variable of the Ohlson O-Score for companies with negative book
    equity.

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
    - OENEG carries a *negative* coefficient (-1.72) in the O-Score, which reads backwards at first
      glance: negative book equity is plainly a distress signal, yet the term lowers the score and
      therefore the estimated probability of bankruptcy. Ohlson (1980) is explicit that this is
      deliberate — OENEG is a discontinuity correction on the leverage term, not a distress signal in
      its own right. A firm with negative equity necessarily has a Total Liabilities to Total Assets
      ratio above 1, so the 6.03 * TLTA term is already extreme for exactly these firms; OENEG damps
      that term back down rather than adding further distress on top of it. The two coefficients must
      therefore be read together, and OENEG must never be used as a standalone indicator.
    """
    return total_liabilities > total_assets


def get_net_income_to_total_assets_ratio(
    net_income: float | pd.Series | pd.DataFrame,
    total_assets: float | pd.Series | pd.DataFrame,
) -> float | pd.Series | pd.DataFrame:
    """
    The Net Income to Total Assets Ratio (NITA) measures a company's profitability relative
    to the size of its balance sheet.

    It is the profitability component of the Ohlson O-Score and of the Zmijewski Score.

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
    - A lower (or negative) ratio indicates weaker profitability, which raises the Zmijewski Score and,
      in turn, the estimated probability of financial distress.
    """
    return drop_non_finite(net_income / total_assets)


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
    return drop_non_finite(funds_from_operations / total_liabilities)


def get_negative_income_indicator(
    net_income: pd.Series | pd.DataFrame,
    lag: int = 1,
) -> pd.Series | pd.DataFrame:
    """
    The Negative Income Indicator (INTWO) flags companies that reported a net loss in both
    the current period and the prior period.

    The formula is as follows:

        - INTWO = 1 if Net Income < 0 for both the current and prior period else 0

    Args:
        net_income (pd.Series | pd.DataFrame): The net income (or loss) of a company, as a time-series.
        lag (int, optional): The number of periods between the current and the prior period, 4 for
            quarterly data so the comparison is year over year. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: A binary indicator equal to 1 when net income was negative in both
        the current and prior period, 0 otherwise.

    Notes:
    - Periods without a current or prior net income (such as the first `lag` periods) are NaN.
    """
    net_income_prior = net_income.shift(lag, axis=1)
    negative_income = (net_income < 0) & (net_income_prior < 0)

    return negative_income.astype(float).where(
        net_income.notna() & net_income_prior.notna()
    )


def get_change_in_net_income_ratio(
    net_income: pd.Series | pd.DataFrame,
    lag: int = 1,
) -> pd.Series | pd.DataFrame:
    """
    The Change in Net Income Ratio (CHIN) measures the direction and relative magnitude of
    the change in net income between the current and prior period.

    It is scaled by the combined absolute size of both periods' net income so that the
    measure stays bounded between -1 and 1.

    The formula is as follows:

        - CHIN = (Net Income (t) - Net Income (t-1)) / (|Net Income (t)| + |Net Income (t-1)|)

    Args:
        net_income (pd.Series | pd.DataFrame): The net income (or loss) of a company, as a time-series.
        lag (int, optional): The number of periods between the current and the prior period, 4 for
            quarterly data so the comparison is year over year. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: The Change in Net Income Ratio.

    Notes:
    - A positive CHIN indicates improving profitability compared to the prior period, which lowers the
      O-Score (i.e. lowers the probability of bankruptcy), reflected by CHIN's negative coefficient.
    - The first `lag` periods are NaN since there is no prior period to compare against.
    """
    net_income_prior = net_income.shift(lag, axis=1)

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
    The Ohlson O-Score is a bankruptcy prediction model developed by James Ohlson in 1980.

    Unlike the Altman Z-Score, which is built with multiple discriminant analysis, the
    O-Score's coefficients come from a fitted logistic regression (logit) model. This is why
    the two models are usually reported side by side rather than one being treated as a
    replacement for the other: the Z-Score produces a score that is only meaningful when
    compared against Altman's empirically derived threshold bands, while the O-Score, run
    through the logistic transform (see `get_ohlson_bankruptcy_probability`), is directly
    interpretable as a probability of bankruptcy between 0 and 1.

    The formula is as follows:

        O-Score = -1.32 - 0.407 * SIZE + 6.03 * TLTA - 1.43 * WCTA + 0.0757 * CLCA - 1.72 * OENEG -
          2.37 * NITA - 1.83 * FUTL + 0.285 * INTWO - 0.521 * CHIN

    The nine variables are:

        - SIZE: Log of Total Assets deflated by the US GNP price-level index (1968 = 100), see
          `get_log_of_total_assets`
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
        log_of_total_assets (float | pd.Series | pd.DataFrame): The Log of Total Assets (SIZE).
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
      0.038, i.e. 3.8% (in probability terms, after the logistic transform) — the threshold that
      minimized the sum of Type I and Type II misclassification errors on his sample, and deliberately
      far below the naive 0.50 midpoint because bankruptcy is a rare event. In raw O-Score terms that
      corresponds to ln(0.038 / 0.962), i.e. approximately -3.23. As with the Altman Z-Score, thresholds
      should be interpreted with caution and alongside further fundamental analysis.
    - The Funds from Operations line item of the original FUTL term is approximated with the Operating
      Cash Flow, see `get_funds_from_operations_to_total_liabilities_ratio`, so the absolute O-Score
      (and derived probability) is close to, but not an exact reproduction of, Ohlson (1980).

    References:
    - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
    Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.
    """
    if isinstance(negative_equity_indicator, pd.Series | pd.DataFrame):
        negative_equity_indicator = negative_equity_indicator.astype(float)

    if isinstance(negative_income_indicator, pd.Series | pd.DataFrame):
        negative_income_indicator = negative_income_indicator.astype(float)

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
    Converts the Ohlson O-Score into a probability of bankruptcy using the logistic
    (sigmoid) function.

    This is the standard way the O-Score is consumed in practice, unlike the Altman Z-Score
    which is typically compared against threshold bands directly, because the O-Score comes
    from a fitted logistic regression and is therefore only meaningful on the probability
    scale after this transform.

    The formula is as follows:

        - P(bankruptcy) = 1 / (1 + e^(-O-Score))

    Args:
        ohlson_o_score (float | pd.Series | pd.DataFrame): The Ohlson O-Score.

    Returns:
        float | pd.Series | pd.DataFrame: The estimated probability of bankruptcy, bounded between 0
        and 1.

    Notes:
    - Because this implementation approximates Funds from Operations with Operating Cash Flow (see
      `get_funds_from_operations_to_total_liabilities_ratio`), the absolute probability produced here
      will differ somewhat from a probability computed with Ohlson's (1980) exact original inputs.

    References:
    - Ohlson, James A. "Financial Ratios and the Probabilistic Prediction of Bankruptcy." Journal of
    Accounting Research, Vol. 18, No. 1, 1980, pp. 109-131.
    """
    return 1 / (1 + np.exp(-ohlson_o_score))
