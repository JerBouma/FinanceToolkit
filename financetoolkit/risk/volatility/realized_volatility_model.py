"""Realized (OHLC-based) Volatility Model"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from scipy import stats

from financetoolkit.risk.helpers import get_days_per_period
from financetoolkit.utilities.error_model import validate_alpha
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    by_own_trading_days,
    to_period_index,
    validate_period,
)

# The natural logarithm of 2, used by Parkinson, Garman-Klass and Yang-Zhang.
LN_2 = np.log(2)

# E|Z| and E|Z|^(4/3) of a standard normal Z, which scale the bipower variation and the
# tripower quarticity to the integrated variance and quarticity (Barndorff-Nielsen and Shephard, 2004).
MU_1 = math.sqrt(2 / math.pi)
MU_4_3 = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(1 / 2)

# The asymptotic variance factor of the relative jump, (pi / 2)^2 + pi - 5 = mu_1^-4 + 2 mu_1^-2 - 5
# (Barndorff-Nielsen and Shephard, 2006; Huang and Tauchen, 2005).
JUMP_TEST_VARIANCE_FACTOR = (math.pi / 2) ** 2 + math.pi - 5

# HAR-RV needs overlap between the monthly window, horizon and observations.
MINIMUM_HAR_RV_OBSERVATIONS = 10


def _group_and_scale(
    daily_term: pd.Series | pd.DataFrame,
    period: str,
    aggregation: str = "mean",
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Groups a daily estimator "term" (e.g. a squared log high/low ratio) by the given
    period and returns, per period, the daily Variance it implies: the mean of the
    term, or the Variance of a daily return series. It does not scale anything; the
    callers multiply the result by the number of daily observations in the period (e.g.
    252 / 52 for weekly, 365 / 52 for a ticker that trades on weekends, see
    `helpers.get_days_per_period`) and take the square root, in the same way as
    `risk_model.get_volatility` scales the daily Variance of close-to-close returns.

    Args:
        daily_term (pd.Series | pd.DataFrame): The daily estimator term to group.
        period (str): The period to group and scale for. Ignored when `groups` is
        provided.
        aggregation (str, optional): Either "mean" (the default, used when the term
        already represents a per-day variance contribution) or "var" (used when the
        term is itself a return series whose Variance needs to be taken, as is the case
        for the overnight and open-to-close components of the Yang-Zhang estimator).
        Defaults to "mean".
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `daily_term`, to group by instead of deriving calendar periods from
        `daily_term.index` via `.asfreq()`. Use this when `daily_term` does not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex on `daily_term`.

    Returns:
        pd.Series | pd.DataFrame: The daily Variance per period, not yet scaled to the
        period.
    """
    dates = (
        groups
        if groups is not None
        else to_period_index(daily_term.index).asfreq(PERIOD_TRANSLATION[period])
    )
    grouped = daily_term.groupby(dates)

    if aggregation == "var":
        return grouped.var()

    return grouped.mean()


@by_own_trading_days
def get_parkinson_volatility(
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Parkinson Volatility: how much an asset's price moves within each period,
    measured from the daily range between the High and the Low.

    The Parkinson estimator uses the daily trading range (High vs Low) instead of only
    the close-to-close return, which makes it considerably more efficient (i.e. it needs
    fewer observations to reach the same precision) than the standard close-to-close
    Volatility, at the cost of assuming that prices follow a continuous geometric
    Brownian motion with no drift and that the High and Low are true intraday extremes
    (i.e. no overnight jumps).

    The formula is as follows:

    - Parkinson Variance = (1 / (4 * N * ln(2))) * SUM[ln(High_t / Low_t)^2]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period by multiplying it with the number of daily observations
    within that period (252 / 52 for weekly, or 365 / 52 for a ticker that trades on
    weekends such as a cryptocurrency), exactly as `risk_model.get_volatility` scales
    the close-to-close Variance, and the square root is taken to obtain Volatility.

    Also known as: Parkinson's range-based Volatility, high-low Volatility.

    For more information about the method, see the following paper:

    - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
    Rate of Return." Journal of Business, 53(1), 61-65.

    Shares similar logic:
    - TTR (R), volatility (Parkinson): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        period (str): The period to calculate the Volatility for. Options are: "weekly",
        "monthly", "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `high_prices`/`low_prices`, to group by instead of deriving
        calendar periods from the index via `.asfreq()`. Use this when the prices do
        not have a DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Parkinson Volatility values with time (or `groups`)
        as the index.

    Notes:
    - A day on which a price the estimator uses is 0 or below (a bad bar) has no logarithm
    and is left out; the period is measured over the remaining days.
    """
    validate_period(period)

    if not isinstance(high_prices, pd.Series | pd.DataFrame) or not isinstance(
        low_prices, pd.Series | pd.DataFrame
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(high_prices, period, groups)

    # A bad bar (a price of 0 or below, as some feeds report for illiquid days) has no
    # logarithm: that day is left out instead of turning the period's mean into inf.
    high_prices, low_prices = (
        prices.where(prices > 0) for prices in (high_prices, low_prices)
    )

    log_high_low_squared = np.log(high_prices / low_prices) ** 2

    variance = _group_and_scale(log_high_low_squared, period, groups=groups) / (
        4 * LN_2
    )

    return np.sqrt(variance) * np.sqrt(days_per_period)


@by_own_trading_days
def get_garman_klass_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Garman-Klass Volatility: how much an asset's price moves within each period,
    measured from the daily Open, High, Low and Close.

    The Garman-Klass estimator extends Parkinson's range-based estimator by also
    incorporating the Open and Close, which allows it to account for the opening jump
    and makes it more efficient still (assuming, as Parkinson does, no drift and no
    overnight jumps beyond the modeled open).

    The formula is as follows:

    - Garman-Klass Variance = (1 / N) * SUM[0.5 * ln(High_t / Low_t)^2 -
      (2 * ln(2) - 1) * ln(Close_t / Open_t)^2]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period in the same way as `risk_model.get_volatility` and the
    square root is taken to obtain Volatility.

    Also known as: Garman-Klass range-based Volatility.

    For more information about the method, see the following paper:

    - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
    Volatilities from Historical Data." Journal of Business, 53(1), 67-78.

    Shares similar logic:
    - TTR (R), volatility (Garman-Klass): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Options are: "weekly",
        "monthly", "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Garman-Klass Volatility values with time (or
        `groups`) as the index.

    Notes:
    - A day on which a price the estimator uses is 0 or below (a bad bar) has no logarithm
    and is left out; the period is measured over the remaining days.
    """
    validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(close_prices, period, groups)

    # A bad bar (a price of 0 or below, as some feeds report for illiquid days) has no
    # logarithm: that day is left out instead of turning the period's mean into inf or NaN.
    open_prices, high_prices, low_prices, close_prices = (
        prices.where(prices > 0)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    )

    log_high_low = np.log(high_prices / low_prices)
    log_close_open = np.log(close_prices / open_prices)

    daily_term = 0.5 * log_high_low**2 - (2 * LN_2 - 1) * log_close_open**2

    variance = _group_and_scale(daily_term, period, groups=groups)

    return np.sqrt(variance) * np.sqrt(days_per_period)


@by_own_trading_days
def get_garman_klass_yang_zhang_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Garman-Klass Yang-Zhang Volatility: how much an asset's price moves within
    each period, from the daily Open, High, Low and Close, including the overnight jump.

    The Garman-Klass estimator assumes prices only move during the trading session, so it
    misses the jump from the previous Close to the Open. Adding the squared overnight return,
    as Yang and Zhang do for their own estimator, makes it account for that jump while
    keeping the Garman-Klass term for the session itself.

    The formula is as follows:

    - Variance = (1 / N) * SUM[ln(Open_t / Close_t-1)^2 + 0.5 * ln(High_t / Low_t)^2 -
      (2 * ln(2) - 1) * ln(Close_t / Open_t)^2]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period in the same way as `risk_model.get_volatility` and the
    square root is taken to obtain Volatility. Like Garman-Klass, the overnight term is a
    squared return rather than a Variance around the mean, so the estimator assumes no drift.

    Also known as: GK-YZ Volatility, Garman-Klass Yang-Zhang Volatility.

    For more information about the method, see the following papers:

    - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
    Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
    - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
    High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-492.

    Shares similar logic:
    - TTR (R), volatility (Garman-Klass Yang-Zhang): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Options are: "weekly",
        "monthly", "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Garman-Klass Yang-Zhang Volatility values with time (or
        `groups`) as the index.

    Notes:
    - Unlike TTR, which scales by a fixed number of days per year, the daily Variance is
    scaled by the number of trading days of the period.
    - A day on which a price the estimator uses is 0 or below (a bad bar) has no logarithm
    and is left out, and so is the first day, which has no previous Close; the period is
    measured over the remaining days.
    """
    validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(close_prices, period, groups)

    # A bad bar (a price of 0 or below, as some feeds report for illiquid days) has no
    # logarithm: that day is left out instead of turning the period's mean into inf or NaN.
    open_prices, high_prices, low_prices, close_prices = (
        prices.where(prices > 0)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    )

    # Computed before grouping so a period transition lands in the next period.
    overnight_returns = np.log(open_prices / close_prices.shift(1))
    log_high_low = np.log(high_prices / low_prices)
    log_close_open = np.log(close_prices / open_prices)

    daily_term = (
        overnight_returns**2
        + 0.5 * log_high_low**2
        - (2 * LN_2 - 1) * log_close_open**2
    )

    variance = _group_and_scale(daily_term, period, groups=groups)

    return np.sqrt(variance) * np.sqrt(days_per_period)


@by_own_trading_days
def get_rogers_satchell_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Rogers-Satchell Volatility: how much an asset's price moves within each
    period, from the daily Open, High, Low and Close, unaffected by a drift in the price.

    Unlike Parkinson and Garman-Klass, the Rogers-Satchell estimator is drift-independent,
    meaning it remains unbiased even when the underlying asset has a non-zero expected
    return over the period, at the cost of still assuming no overnight jumps.

    The formula is as follows:

    - Rogers-Satchell Variance = (1 / N) * SUM[ln(High_t / Close_t) * ln(High_t / Open_t) +
      ln(Low_t / Close_t) * ln(Low_t / Open_t)]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period in the same way as `risk_model.get_volatility` and the
    square root is taken to obtain Volatility.

    Also known as: Rogers-Satchell range-based Volatility, drift-independent Volatility.

    For more information about the method, see the following paper:

    - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
    Closing Prices." Annals of Applied Probability, 1(4), 504-512.

    Shares similar logic:
    - TTR (R), volatility (Rogers-Satchell): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Options are: "weekly",
        "monthly", "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Rogers-Satchell Volatility values with time (or
        `groups`) as the index.

    Notes:
    - A day on which a price the estimator uses is 0 or below (a bad bar) has no logarithm
    and is left out; the period is measured over the remaining days.
    """
    validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(close_prices, period, groups)

    # A bad bar (a price of 0 or below, as some feeds report for illiquid days) has no
    # logarithm: that day is left out instead of turning the period's mean into inf or NaN.
    open_prices, high_prices, low_prices, close_prices = (
        prices.where(prices > 0)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    )

    daily_term = np.log(high_prices / close_prices) * np.log(
        high_prices / open_prices
    ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

    variance = _group_and_scale(daily_term, period, groups=groups)

    return np.sqrt(variance) * np.sqrt(days_per_period)


@by_own_trading_days
def get_yang_zhang_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Yang-Zhang Volatility: how much an asset's price moves within each period,
    from the daily Open, High, Low and Close, including the overnight jump and any drift.

    The Yang-Zhang estimator is a weighted combination of the overnight (close-to-open)
    Variance, the open-to-close Variance and the Rogers-Satchell Variance. It is both
    drift-independent (like Rogers-Satchell) and accounts for overnight jumps (unlike
    Parkinson, Garman-Klass and Rogers-Satchell, which all assume prices only move during
    the trading session), which makes it the most statistically efficient of the four
    estimators implemented here (i.e. it has the lowest variance of the estimator itself
    across sub-samples).

    The formula is as follows:

    - Yang-Zhang Variance = Overnight Variance + k * Open-to-Close Variance +
      (1 - k) * Rogers-Satchell Variance

    Where:

    - Overnight Variance = Var(ln(Open_t / Close_t-1))
    - Open-to-Close Variance = Var(ln(Close_t / Open_t))
    - k = 0.34 / (1.34 + (N + 1) / (N - 1))

    And `N` is the number of trading days within the period. The daily Variance is scaled
    to the given period in the same way as `risk_model.get_volatility` and the square root
    is taken to obtain Volatility.

    Also known as: Yang-Zhang range-based Volatility, drift-independent overnight-aware
    Volatility.

    For more information about the method, see the following paper:

    - Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on
    High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-492.

    Shares similar logic:
    - TTR (R), volatility (Yang-Zhang): https://github.com/joshuaulrich/TTR/blob/v0.24.4/R/volatility.R#L139

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Options are: "weekly",
        "monthly", "quarterly", "yearly". Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Yang-Zhang Volatility values with time (or `groups`)
        as the index.

    Notes:
    - A day on which a price the estimator uses is 0 or below (a bad bar) has no logarithm
    and is left out; the period is measured over the remaining days.
    """
    validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    days_per_period = get_days_per_period(close_prices, period, groups)

    # A bad bar (a price of 0 or below, as some feeds report for illiquid days) has no
    # logarithm: that day is left out instead of turning the period's mean into inf or NaN.
    open_prices, high_prices, low_prices, close_prices = (
        prices.where(prices > 0)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    )

    # Computed before grouping so a period transition lands in the next period.
    previous_close = close_prices.shift(1)
    overnight_returns = np.log(open_prices / previous_close)
    open_to_close_returns = np.log(close_prices / open_prices)

    rogers_satchell_term = np.log(high_prices / close_prices) * np.log(
        high_prices / open_prices
    ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

    overnight_variance = _group_and_scale(
        overnight_returns, period, "var", groups=groups
    )
    open_to_close_variance = _group_and_scale(
        open_to_close_returns, period, "var", groups=groups
    )
    rogers_satchell_variance = _group_and_scale(
        rogers_satchell_term, period, groups=groups
    )

    dates = (
        groups
        if groups is not None
        else to_period_index(open_to_close_returns.index).asfreq(
            PERIOD_TRANSLATION[period]
        )
    )
    number_of_days = open_to_close_returns.groupby(dates).count()

    k = 0.34 / (1.34 + (number_of_days + 1) / (number_of_days - 1))

    variance = (
        overnight_variance
        + k * open_to_close_variance
        + (1 - k) * rogers_satchell_variance
    )

    return np.sqrt(variance) * np.sqrt(days_per_period)


@by_own_trading_days
def get_har_rv_forecast(
    realized_variance: pd.Series | pd.DataFrame,
    weekly_window: int = 5,
    monthly_window: int = 22,
    horizon: int = 1,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility (HAR-RV)
    forecast: the daily Realized Variance expected ahead, from its daily, weekly and monthly averages.

    Volatility clustering happens across multiple, overlapping time horizons at once
    -- traders reacting to today's move, this week's trend, and this month's regime
    all contribute to tomorrow's Variance. The HAR-RV model captures this cheaply
    (without the numerical optimization a GARCH-family fit requires, see
    `garch_model.get_garch`) by regressing future daily Realized Variance on three
    trailing average components, each a proxy for a different horizon of memory:

    - RV_(t+h) = b0 + bD * RV_t + bW * RV_t^(w) + bM * RV_t^(m) + e_(t+h)
    - RV_t^(w) = (1 / w) * SUM_(i=0)^(w-1) RV_(t-i)
    - RV_t^(m) = (1 / m) * SUM_(i=0)^(m-1) RV_(t-i)

    Where `RV_t` is the (daily) Realized Variance on day `t` -- typically the square
    of the daily return, or the (squared, pre-period-aggregation) daily term behind
    one of the OHLC range-based estimators above (e.g. `LN_2`-scaled
    `ln(High_t / Low_t) ** 2 / 4` for Parkinson), `w` and `m` are the `weekly_window`
    and `monthly_window` trailing windows (5 and 22 trading days by default), and `h`
    is the forecast `horizon`. The three components are fitted via Ordinary Least
    Squares over the whole available history, and this function returns the
    resulting (in-sample) fitted forecast path rather than the fitted coefficients
    themselves.

    Despite its simplicity (three regressors, no numerical optimization needed), HAR-RV
    is a well-established, highly competitive benchmark against fully parametric
    volatility models in the realized volatility forecasting literature, largely
    because the three components jointly approximate the long-memory (slowly-decaying
    autocorrelation) behavior that realized Volatility empirically exhibits.

    Also known as: HAR-RV model, Corsi's HAR model, Heterogeneous Autoregressive model.

    For more information about the method, see the following paper:

    - Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized
    Volatility." Journal of Financial Econometrics, 7(2), 174-196.

    Args:
        realized_variance (pd.Series | pd.DataFrame): A Series or Dataframe of daily
        Realized Variance (not Volatility -- i.e. not yet square-rooted), e.g. squared
        daily returns or the daily term of one of the OHLC estimators above.
        weekly_window (int, optional): The trailing window (in trading days) for the
        weekly RV component. Defaults to 5.
        monthly_window (int, optional): The trailing window (in trading days) for the
        monthly RV component. Defaults to 22.
        horizon (int, optional): The number of days ahead to forecast. Defaults to 1.

    Returns:
        pd.Series | pd.DataFrame: The fitted HAR-RV forecast of Realized Variance,
        indexed by the date the forecast was made (i.e. the forecast target is
        `horizon` days ahead of each index date). The first `monthly_window - 1`
        values (the rolling window burn-in) are NaN; the last `horizon` values are the
        forecasts of days that have not been observed yet.

    Raises:
        TypeError: If `realized_variance` is not a pd.Series or pd.DataFrame.
    """
    if isinstance(realized_variance, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_har_rv_forecast(
                    realized_variance[column], weekly_window, monthly_window, horizon
                )
                for column in realized_variance.columns
            }
        )
    if isinstance(realized_variance, pd.Series):
        rv = realized_variance.dropna()

        daily_component = rv
        weekly_component = rv.rolling(window=weekly_window).mean()
        monthly_component = rv.rolling(window=monthly_window).mean()
        target = rv.shift(-horizon)

        regression_frame = pd.concat(
            [target, daily_component, weekly_component, monthly_component], axis=1
        )
        regression_frame.columns = ["target", "daily", "weekly", "monthly"]
        estimation_sample = regression_frame.dropna()

        if len(estimation_sample) <= MINIMUM_HAR_RV_OBSERVATIONS:
            return pd.Series(np.nan, index=realized_variance.index)

        x = np.column_stack(
            [
                np.ones(len(estimation_sample)),
                estimation_sample["daily"].to_numpy(),
                estimation_sample["weekly"].to_numpy(),
                estimation_sample["monthly"].to_numpy(),
            ]
        )
        y = estimation_sample["target"].to_numpy()

        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

        # The coefficients are estimated where the target is known, and the forecast is
        # made wherever its inputs are, so the last `horizon` days get a forecast too.
        inputs = regression_frame[["daily", "weekly", "monthly"]].dropna()
        forecast = pd.Series(
            np.column_stack([np.ones(len(inputs)), inputs.to_numpy()]) @ coefficients,
            index=inputs.index,
        )

        return forecast.reindex(realized_variance.index)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_intraday_returns(open_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
    """
    Calculate the intraday log returns of every bar within its own day: the move each bar adds to the day,
    without the overnight gap.

    The formula is as follows:

    - r_(d,1) = ln(Close_(d,1) / Open_(d,1)), the first bar of day d
    - r_(d,j) = ln(Close_(d,j) / Close_(d,j-1)), the later bars of day d

    The first bar of a day starts from its own Open rather than the previous day's last Close,
    so the overnight return is left out, as the realized measures integrate the variance of the
    trading session (Andersen, Bollerslev, Diebold and Labys, 2003). Bars without a price, or with
    a price of 0 or below, are left out before the returns are taken.

    For more information, see the following paper:

    - Andersen, T.G., Bollerslev, T., Diebold, F.X. and Labys, P. (2003). "Modeling and Forecasting
      Realized Volatility." Econometrica, 71(2), 579-625.

    Args:
        open_prices (pd.Series): The Open of every intraday bar, indexed by its start time
            (a PeriodIndex).
        close_prices (pd.Series): The Close of every intraday bar, on the same index.

    Returns:
        pd.Series: The log return of every bar, on the bars with a price.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.risk.volatility import realized_volatility_model

    index = pd.PeriodIndex(
        ["2025-12-30 09:30", "2025-12-30 10:30", "2025-12-31 09:30", "2025-12-31 10:30"],
        freq="min",
    )
    open_prices = pd.Series([100.0, 101.0, 103.0, 102.0], index=index)
    close_prices = pd.Series([101.0, 102.0, 102.0, 104.0], index=index)

    realized_volatility_model.get_intraday_returns(open_prices, close_prices).round(4)
    ```

    Which returns:

    |                  |   Value |
    |:-----------------|--------:|
    | 2025-12-30 09:30 |  0.01   |
    | 2025-12-30 10:30 |  0.0099 |
    | 2025-12-31 09:30 | -0.0098 |
    | 2025-12-31 10:30 |  0.0194 |
    """
    traded = (close_prices > 0) & (open_prices > 0)
    close_prices = close_prices[traded]
    open_prices = open_prices[traded]
    days = to_period_index(close_prices.index).asfreq("D")

    log_close = np.log(close_prices)
    returns = log_close.groupby(days).diff()
    first_bar = returns.isna()

    returns[first_bar] = (log_close - np.log(open_prices))[first_bar]

    return returns


def get_realized_measures(intraday_returns: pd.Series) -> pd.DataFrame:
    """
    Calculate the realized measures of every day from its intraday returns: the realized variance, its split into
    upside and downside semivariance, and the bipower variation that leaves the jumps out.

    The realized variance sums the squared intraday returns and converges to the day's integrated variance plus
    its squared jumps as the bars get shorter (Andersen, Bollerslev, Diebold and Labys, 2003). The realized
    semivariances split that sum into the squared rises and the squared falls (Barndorff-Nielsen, Kinnebrock and
    Shephard, 2010): the downside semivariance carries most of the information about future volatility. The bipower
    variation multiplies neighbouring absolute returns, so a single jump enters only next to an ordinary return
    and the measure converges to the integrated variance without the jumps (Barndorff-Nielsen and Shephard, 2004).
    The difference is the jump variation. The tripower quarticity estimates the integrated quarticity that the
    jump test needs (see `get_jump_test`).

    The formula is as follows, with r_j the M intraday log returns of a day (see `get_intraday_returns`):

    - Realized Variance = SUM(r_j^2)
    - Realized Volatility = SQRT(Realized Variance)
    - Positive Semivariance = SUM(r_j^2 * 1(r_j > 0)) and Negative Semivariance = SUM(r_j^2 * 1(r_j < 0))
    - Bipower Variation = mu_1^-2 * (M / (M - 1)) * SUM_(j=2..M)(|r_j| * |r_(j-1)|)
    - Jump Variation = MAX(Realized Variance - Bipower Variation, 0)
    - Tripower Quarticity = M * mu_(4/3)^-3 * (M / (M - 2))
      * SUM_(j=3..M)(|r_j|^(4/3) * |r_(j-1)|^(4/3) * |r_(j-2)|^(4/3))

    Where mu_p = E|Z|^p of a standard normal Z: mu_1 = SQRT(2 / pi) and
    mu_(4/3) = 2^(2/3) * Gamma(7/6) / Gamma(1/2).
    The factors M / (M - 1) and M / (M - 2) make up for the products a day of M returns lacks (Huang and Tauchen,
    2005), which matters for hourly bars with M = 7.

    For more information, see the following papers:

    - Andersen, T.G., Bollerslev, T., Diebold, F.X. and Labys, P. (2003). "Modeling and Forecasting
      Realized Volatility." Econometrica, 71(2), 579-625.
    - Barndorff-Nielsen, O.E. and Shephard, N. (2004). "Power and Bipower Variation with Stochastic
      Volatility and Jumps." Journal of Financial Econometrics, 2(1), 1-37.
    - Barndorff-Nielsen, O.E., Kinnebrock, S. and Shephard, N. (2010). "Measuring Downside Risk:
      Realised Semivariance." In Volatility and Time Series Econometrics: Essays in Honor of Robert F.
      Engle, Oxford University Press, 117-136.
    - Huang, X. and Tauchen, G. (2005). "The Relative Contribution of Jumps to Total Price Variance."
      Journal of Financial Econometrics, 3(4), 456-499.

    Args:
        intraday_returns (pd.Series): The intraday log returns, indexed by the start time of their
            bar (a PeriodIndex), such as those of `get_intraday_returns`.

    Returns:
        pd.DataFrame: The Realized Variance, Realized Volatility, Positive Semivariance, Negative
        Semivariance, Bipower Variation, Jump Variation, Tripower Quarticity and Observations (M) per day,
        as daily (not annualized) decimals.

    Notes:
    - The Bipower Variation needs two returns in a day and the Tripower Quarticity three; with fewer
      they are NaN.
    - A day's measures cover the trading session only; the overnight return is not part of them.

    As an example:

    ```python
    import numpy as np
    import pandas as pd

    from financetoolkit.risk.volatility import realized_volatility_model

    index = pd.period_range("2025-12-31 09:30", periods=8, freq="min")
    returns = pd.Series([0.01, -0.02, 0.005, 0.0, -0.01, 0.03, -0.005, 0.01], index=index)

    realized_volatility_model.get_realized_measures(returns).T
    ```

    Which returns:

    |                       |   2025-12-31 |
    |:----------------------|-------------:|
    | Realized Variance     |  0.00165     |
    | Realized Volatility   |  0.0406202   |
    | Positive Semivariance |  0.001125    |
    | Negative Semivariance |  0.000525    |
    | Bipower Variation     |  0.00143616  |
    | Jump Variation        |  0.000213843 |
    | Tripower Quarticity   |  8.24619e-07 |
    | Observations          |  8           |
    """
    returns = intraday_returns.dropna()
    days = to_period_index(returns.index).asfreq("D")
    absolute = returns.abs()
    grouped = absolute.groupby(days)
    previous = grouped.shift(1)
    before_previous = grouped.shift(2)

    squared = returns**2
    observations = returns.groupby(days).count()
    realized_variance = squared.groupby(days).sum()

    bipower_sum = (absolute * previous).groupby(days).sum(min_count=1)
    tripower_sum = (
        ((absolute * previous * before_previous) ** (4 / 3))
        .groupby(days)
        .sum(min_count=1)
    )

    # A day with a single return has no product of neighbours (a NaN sum), and one with two
    # returns no product of three, so their measures are NaN rather than 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        bipower_variation = MU_1**-2 * observations / (observations - 1) * bipower_sum
        tripower_quarticity = (
            observations * MU_4_3**-3 * observations / (observations - 2) * tripower_sum
        )

    measures = pd.DataFrame(
        {
            "Realized Variance": realized_variance,
            "Realized Volatility": np.sqrt(realized_variance),
            "Positive Semivariance": squared.where(returns > 0, 0.0)
            .groupby(days)
            .sum(),
            "Negative Semivariance": squared.where(returns < 0, 0.0)
            .groupby(days)
            .sum(),
            "Bipower Variation": bipower_variation,
            "Jump Variation": (realized_variance - bipower_variation).clip(lower=0),
            "Tripower Quarticity": tripower_quarticity,
            "Observations": observations,
        }
    )
    measures.index.name = "Date"

    return measures


def get_jump_test(
    realized_variance: pd.Series,
    bipower_variation: pd.Series,
    tripower_quarticity: pd.Series,
    observations: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate the jump test of every day: whether the day's price path held a jump, from how far its realized
    variance exceeds its bipower variation.

    Without jumps the realized variance and the bipower variation estimate the same integrated variance, and
    the relative gap between them is asymptotically normal (Barndorff-Nielsen and Shephard, 2006). Huang and
    Tauchen (2005) found the ratio statistic with the tripower quarticity, bounded below as in the formula, to
    keep its size best on finite samples, and Andersen, Bollerslev and Diebold (2007) take the jump of a day
    with a significant statistic as its jump component. The test is one-sided: only a realized variance above
    the bipower variation points to a jump.

    The formula is as follows:

    - Relative Jump = (Realized Variance - Bipower Variation) / Realized Variance
    - z-Statistic = Relative Jump
      / SQRT(((pi / 2)^2 + pi - 5) * (1 / M) * MAX(1, Tripower Quarticity / Bipower Variation^2))
    - p-Value = 1 - N(z-Statistic)
    - Significant Jump Variation = MAX(Realized Variance - Bipower Variation, 0) when z-Statistic > N^-1(1 - alpha),
      else 0

    Where M is the number of intraday returns of the day and N the standard normal distribution. The measures
    come from `get_realized_measures`.

    For more information, see the following papers:

    - Barndorff-Nielsen, O.E. and Shephard, N. (2006). "Econometrics of Testing for Jumps in Financial
      Economics Using Bipower Variation." Journal of Financial Econometrics, 4(1), 1-30.
    - Huang, X. and Tauchen, G. (2005). "The Relative Contribution of Jumps to Total Price Variance."
      Journal of Financial Econometrics, 3(4), 456-499.
    - Andersen, T.G., Bollerslev, T. and Diebold, F.X. (2007). "Roughing It Up: Including Jump
      Components in the Measurement, Modeling, and Forecasting of Return Volatility." The Review of
      Economics and Statistics, 89(4), 701-720.

    Args:
        realized_variance (pd.Series): The realized variance per day.
        bipower_variation (pd.Series): The bipower variation per day.
        tripower_quarticity (pd.Series): The tripower quarticity per day.
        observations (pd.Series): The number of intraday returns M per day.
        alpha (float, optional): The significance level of the one-sided test. Defaults to 0.05.

    Returns:
        pd.DataFrame: The Relative Jump, z-Statistic, p-Value and Significant Jump Variation per day.

    Notes:
    - A day without movement (a realized variance of 0) or with fewer than three returns has no statistic.
    - With many days tested at once, about alpha of the days without a jump are flagged by chance; Andersen,
      Bollerslev and Diebold (2007) use alpha = 0.001 for that reason.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.risk.volatility import realized_volatility_model

    realized_volatility_model.get_jump_test(
        realized_variance=pd.Series([0.0001, 0.0004]),
        bipower_variation=pd.Series([0.000098, 0.0002]),
        tripower_quarticity=pd.Series([1e-08, 2e-08]),
        observations=pd.Series([78, 78]),
    ).round(4)
    ```

    Which returns:

    |    |   Relative Jump |   z-Statistic |   p-Value |   Significant Jump Variation |
    |---:|----------------:|--------------:|----------:|-----------------------------:|
    |  0 |            0.02 |        0.2218 |    0.4122 |                       0      |
    |  1 |            0.5  |        5.6586 |    0      |                       0.0002 |
    """
    validate_alpha(alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_jump = (
            realized_variance - bipower_variation
        ) / realized_variance.where(realized_variance > 0)
        quarticity_ratio = np.maximum(1, tripower_quarticity / bipower_variation**2)
        z_statistic = relative_jump / np.sqrt(
            JUMP_TEST_VARIANCE_FACTOR / observations * quarticity_ratio
        )
    z_statistic = z_statistic.where(tripower_quarticity.notna())
    jump_variation = (realized_variance - bipower_variation).clip(lower=0)

    return pd.DataFrame(
        {
            "Relative Jump": relative_jump,
            "z-Statistic": z_statistic,
            "p-Value": pd.Series(
                stats.norm.sf(z_statistic), index=z_statistic.index
            ).where(z_statistic.notna()),
            "Significant Jump Variation": jump_variation.where(
                z_statistic > stats.norm.ppf(1 - alpha), 0.0
            ).where(z_statistic.notna()),
        }
    )
