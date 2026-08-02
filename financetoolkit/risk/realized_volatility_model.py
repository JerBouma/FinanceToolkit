"""Realized (OHLC-based) Volatility Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    VOLATILITY_WINDOW_TRANSLATION,
)

# The natural logarithm of 2, used throughout the Parkinson, Garman-Klass and
# Yang-Zhang formulas below.
LN_2 = np.log(2)

# The HAR-RV regression needs enough overlap between the monthly rolling window, the
# forecast horizon and a handful of estimation observations to be meaningful.
MINIMUM_HAR_RV_OBSERVATIONS = 10


def _validate_period(period: str) -> None:
    if period not in PERIOD_TRANSLATION:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )


def _group_and_scale(
    daily_term: pd.Series | pd.DataFrame,
    period: str,
    aggregation: str = "mean",
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Groups a daily estimator "term" (e.g. a squared log high/low ratio) by the given
    period and scales the resulting (already period-average) variance to an annualized
    Volatility figure, in the same way as `risk_model.get_volatility` scales the daily
    Variance of close-to-close returns.

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
        pd.Series | pd.DataFrame: The period Variance (not yet annualized).
    """
    dates = (
        groups
        if groups is not None
        else daily_term.index.asfreq(PERIOD_TRANSLATION[period])
    )
    grouped = daily_term.groupby(dates)

    if aggregation == "var":
        return grouped.var()

    return grouped.mean()


def get_parkinson_volatility(
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Parkinson Volatility of an asset for a given period (weekly, monthly,
    quarterly or yearly) based on daily High and Low prices.

    The Parkinson estimator uses the daily trading range (High vs Low) instead of only
    the close-to-close return, which makes it considerably more efficient (i.e. it needs
    fewer observations to reach the same precision) than the standard close-to-close
    Volatility, at the cost of assuming that prices follow a continuous geometric
    Brownian motion with no drift and that the High and Low are true intraday extremes
    (i.e. no overnight jumps).

    The formula is as follows:

    - Parkinson Variance = (1 / (4 * N * ln(2))) * SUM[ln(High_t / Low_t)^2]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period by multiplying it with the number of trading days within
    that period (e.g. 252 / 52 for weekly), exactly as `risk_model.get_volatility` scales
    the close-to-close Variance, and the square root is taken to obtain Volatility.

    Also known as: Parkinson's range-based Volatility, high-low Volatility.

    For more information about the method, see the following paper:

    - Parkinson, M. (1980). "The Extreme Value Method for Estimating the Variance of the
    Rate of Return." Journal of Business, 53(1), 61-65.

    Args:
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        period (str): The period to calculate the Volatility for. Can be weekly, monthly,
        quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of `high_prices`/`low_prices`, to group by instead of deriving
        calendar periods from the index via `.asfreq()`. Use this when the prices do
        not have a DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Parkinson Volatility values with time (or `groups`)
        as the index.
    """
    _validate_period(period)

    if not isinstance(high_prices, pd.Series | pd.DataFrame) or not isinstance(
        low_prices, pd.Series | pd.DataFrame
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    log_high_low_squared = np.log(high_prices / low_prices) ** 2

    variance = _group_and_scale(log_high_low_squared, period, groups=groups) / (
        4 * LN_2
    )

    return np.sqrt(variance) * np.sqrt(volatility_window)


def get_garman_klass_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Garman-Klass Volatility of an asset for a given period (weekly,
    monthly, quarterly or yearly) based on daily Open, High, Low and Close prices.

    The Garman-Klass estimator extends Parkinson's range-based estimator by also
    incorporating the Open and Close, which allows it to account for the opening jump
    and makes it more efficient still (assuming, as Parkinson does, no drift and no
    overnight jumps beyond the modeled open).

    The formula is as follows:

    - Garman-Klass Variance = (1 / N) * SUM[0.5 * ln(High_t / Low_t)^2
    - (2 * ln(2) - 1) * ln(Close_t / Open_t)^2]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period in the same way as `risk_model.get_volatility` and the
    square root is taken to obtain Volatility.

    Also known as: Garman-Klass range-based Volatility.

    For more information about the method, see the following paper:

    - Garman, M.B., & Klass, M.J. (1980). "On the Estimation of Security Price
    Volatilities from Historical Data." Journal of Business, 53(1), 67-78.

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Can be weekly, monthly,
        quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Garman-Klass Volatility values with time (or
        `groups`) as the index.
    """
    _validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    log_high_low = np.log(high_prices / low_prices)
    log_close_open = np.log(close_prices / open_prices)

    daily_term = 0.5 * log_high_low**2 - (2 * LN_2 - 1) * log_close_open**2

    variance = _group_and_scale(daily_term, period, groups=groups)

    return np.sqrt(variance) * np.sqrt(volatility_window)


def get_rogers_satchell_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Rogers-Satchell Volatility of an asset for a given period (weekly,
    monthly, quarterly or yearly) based on daily Open, High, Low and Close prices.

    Unlike Parkinson and Garman-Klass, the Rogers-Satchell estimator is drift-independent,
    meaning it remains unbiased even when the underlying asset has a non-zero expected
    return over the period, at the cost of still assuming no overnight jumps.

    The formula is as follows:

    - Rogers-Satchell Variance = (1 / N) * SUM[ln(High_t / Close_t) * ln(High_t / Open_t)
    - + ln(Low_t / Close_t) * ln(Low_t / Open_t)]

    Where `N` is the number of trading days within the period. The daily Variance is
    scaled to the given period in the same way as `risk_model.get_volatility` and the
    square root is taken to obtain Volatility.

    Also known as: Rogers-Satchell range-based Volatility, drift-independent Volatility.

    For more information about the method, see the following paper:

    - Rogers, L.C.G., & Satchell, S.E. (1991). "Estimating Variance from High, Low and
    Close Prices." Annals of Applied Probability, 1(4), 504-512.

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Can be weekly, monthly,
        quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Rogers-Satchell Volatility values with time (or
        `groups`) as the index.
    """
    _validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    daily_term = np.log(high_prices / close_prices) * np.log(
        high_prices / open_prices
    ) + np.log(low_prices / close_prices) * np.log(low_prices / open_prices)

    variance = _group_and_scale(daily_term, period, groups=groups)

    return np.sqrt(variance) * np.sqrt(volatility_window)


def get_yang_zhang_volatility(
    open_prices: pd.Series | pd.DataFrame,
    high_prices: pd.Series | pd.DataFrame,
    low_prices: pd.Series | pd.DataFrame,
    close_prices: pd.Series | pd.DataFrame,
    period: str,
    groups: pd.Series | np.ndarray | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Yang-Zhang Volatility of an asset for a given period (weekly, monthly,
    quarterly or yearly) based on daily Open, High, Low and Close prices.

    The Yang-Zhang estimator is a weighted combination of the overnight (close-to-open)
    Variance, the open-to-close Variance and the Rogers-Satchell Variance. It is both
    drift-independent (like Rogers-Satchell) and accounts for overnight jumps (unlike
    Parkinson, Garman-Klass and Rogers-Satchell, which all assume prices only move during
    the trading session), which makes it the most statistically efficient of the four
    estimators implemented here (i.e. it has the lowest variance of the estimator itself
    across sub-samples).

    The formula is as follows:

    - Yang-Zhang Variance = Overnight Variance + k * Open-to-Close Variance
    - + (1 - k) * Rogers-Satchell Variance

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
    High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-491.

    Args:
        open_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Open prices.
        high_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily High prices.
        low_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Low prices.
        close_prices (pd.Series | pd.DataFrame): A Series or Dataframe of daily Close prices.
        period (str): The period to calculate the Volatility for. Can be weekly, monthly,
        quarterly or yearly. Ignored when `groups` is provided.
        groups (pd.Series | np.ndarray | None, optional): Explicit group labels, one
        per row of the price inputs, to group by instead of deriving calendar periods
        from the index via `.asfreq()`. Use this when the prices do not have a
        DatetimeIndex/PeriodIndex. Defaults to None, which requires a
        DatetimeIndex/PeriodIndex.

    Returns:
        pd.Series | pd.DataFrame: Yang-Zhang Volatility values with time (or `groups`)
        as the index.
    """
    _validate_period(period)

    if not all(
        isinstance(prices, pd.Series | pd.DataFrame)
        for prices in (open_prices, high_prices, low_prices, close_prices)
    ):
        raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")

    volatility_window = VOLATILITY_WINDOW_TRANSLATION[period]

    # The overnight (close-to-open) log-return is computed on the full series first,
    # before grouping by period, so that the transition from the last trading day of one
    # period into the first trading day of the next (e.g. across a weekend or a
    # month-end) is correctly captured as part of the following period's estimate.
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
        else open_to_close_returns.index.asfreq(PERIOD_TRANSLATION[period])
    )
    number_of_days = open_to_close_returns.groupby(dates).count()

    k = 0.34 / (1.34 + (number_of_days + 1) / (number_of_days - 1))

    variance = (
        overnight_variance
        + k * open_to_close_variance
        + (1 - k) * rogers_satchell_variance
    )

    return np.sqrt(variance) * np.sqrt(volatility_window)


def get_har_rv_forecast(
    realized_variance: pd.Series | pd.DataFrame,
    weekly_window: int = 5,
    monthly_window: int = 22,
    horizon: int = 1,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Corsi (2009) Heterogeneous Autoregressive Realized Volatility
    (HAR-RV) forecast of future daily Realized Variance.

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
        (rolling window burn-in) and last `horizon` (no future data available to
        forecast against) values are NaN.

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
        fitted_values = x @ coefficients

        forecast = pd.Series(fitted_values, index=estimation_sample.index)

        return forecast.reindex(realized_variance.index)

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
