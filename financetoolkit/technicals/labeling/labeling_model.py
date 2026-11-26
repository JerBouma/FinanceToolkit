"""Labeling Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# The bull flag templates of the matrix flag labels: rows from the highest price decile of the
# window (top) to the lowest, columns from the first tenth of the window (left) to the last.
# Leigh, Modani, Purvis and Roberts (2002) and Cervelló-Royo, Guijarro and Michniuk (2015). The
# bear templates are the bull templates upside down, the highest decile becoming the lowest.
LEIGH_BULL_TEMPLATE = np.array(
    [
        [0.5, 0, -1, -1, -1, -1, -1, -1, -1, 0],
        [1, 0.5, 0, -0.5, -1, -1, -1, -1, -0.5, 0],
        [1, 1, 0.5, 0, -0.5, -0.5, -0.5, -0.5, 0, 0.5],
        [0.5, 1, 1, 0.5, 0, -0.5, -0.5, -0.5, 0, 1],
        [0, 0.5, 1, 1, 0.5, 0, 0, 0, 0.5, 1],
        [0, 0, 0.5, 1, 1, 0.5, 0, 0, 1, 1],
        [-0.5, 0, 0, 0.5, 1, 1, 0.5, 0.5, 1, 1],
        [-0.5, -1, 0, 0, 0.5, 1, 1, 1, 1, 0],
        [-1, -1, -1, -0.5, 0, 0.5, 1, 1, 0, -2],
        [-1, -1, -1, -1, -1, 0, 0.5, 0.5, -2, -2.5],
    ]
)
CERVELLO_ROYO_BULL_TEMPLATE = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, -1, -2, -2, -2, -2, -2, -2],
        [0, 0, -1, -3, -3, -3, -3, -3, -3, -3],
        [0, -1, -3, -5, -5, -5, -5, -5, -5, -5],
        [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
        [0, -1, -5, -5, -5, -5, -5, -5, -5, -5],
        [5, -1, -5, -5, -5, -5, -5, -5, -5, -5],
    ]
)
MATRIX_FLAG_TEMPLATES = {
    "leigh_bull": LEIGH_BULL_TEMPLATE,
    "leigh_bear": LEIGH_BULL_TEMPLATE[::-1],
    "cervello_royo_bull": CERVELLO_ROYO_BULL_TEMPLATE,
    "cervello_royo_bear": CERVELLO_ROYO_BULL_TEMPLATE[::-1],
}


def get_daily_volatility(prices: pd.Series, span: int = 100) -> pd.Series:
    """
    Calculate the volatility target of the labels: the exponentially weighted standard deviation of the period
    returns, as López de Prado sets the profit-taking and stop-loss barriers.

    The formula is as follows:

    - r_t = p_t / p_t-1 - 1
    - sigma_t = EWM standard deviation of r up to and including t, with alpha = 2 / (span + 1)

    The weights are pandas' adjusted ones with the bias correction,
    so the value at t only uses returns up to t.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, Snippet 3.1.

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        span (int, optional): The span of the exponential weights. Defaults to 100.

    Returns:
        pd.Series: The volatility of the period returns at every date.

    Notes:
        Snippet 3.1 measures the return over the bar a calendar day earlier, written for
        intraday bars. This uses the return of each period, so on daily prices the daily return.
    """
    return prices.pct_change(fill_method=None).ewm(span=span).std()


def get_cusum_events(prices: pd.Series, threshold: float | pd.Series) -> pd.Series:
    """
    Calculate the events of the symmetric CUSUM filter: the dates the cumulative log return since the last event of
    its direction exceeds the threshold.

    The formula is as follows:

    - y_t = ln(p_t / p_t-1)
    - S+_t = max(0, S+_t-1 + y_t) and S-_t = min(0, S-_t-1 + y_t)
    - an event of -1 when S-_t < -h_t (S-_t resets to 0), else of 1 when S+_t > h_t (S+_t resets)

    For more information, see the following book and paper:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 2.5.2.1.
    - Lam, K. and Yam, H.C. (1997). "CUSUM Techniques for Technical Trading in Financial Markets."
      Financial Engineering and the Japanese Markets, 4(3), 257-274.

    Referenced against:
    - RiskLabAI, symmetric_cusum_filter: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L78

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        threshold (float | pd.Series): The threshold h in log return units, fixed or per date. No
            event occurs on a date without a threshold, while the sums keep accumulating.

    Returns:
        pd.Series: 1 for an upward event, -1 for a downward event and 0 otherwise, NaN on the first
        date and on dates without a return.
    """
    log_returns = np.log(prices.to_numpy(dtype=float))
    changes = np.diff(log_returns, prepend=np.nan)
    thresholds = (
        threshold.reindex(prices.index).to_numpy(dtype=float)
        if isinstance(threshold, pd.Series)
        else np.full(len(prices), float(threshold))
    )

    events = np.full(len(prices), np.nan)
    positive_sum, negative_sum = 0.0, 0.0
    for position, change in enumerate(changes):
        if np.isnan(change):
            continue
        positive_sum = max(0.0, positive_sum + change)
        negative_sum = min(0.0, negative_sum + change)
        events[position] = 0
        if negative_sum < -thresholds[position]:
            negative_sum = 0.0
            events[position] = -1
        elif positive_sum > thresholds[position]:
            positive_sum = 0.0
            events[position] = 1

    return pd.Series(events, index=prices.index)


def get_z_score_events(
    prices: pd.Series, window: int = 20, threshold: float = 3.0
) -> pd.Series:
    """
    Calculate the events of the z-score filter: the dates the price is at least `threshold` standard deviations away
    from its rolling mean.

    The formula is as follows:

    - z_t = (p_t - mean(p_t-window+1..t)) / std(p_t-window+1..t), with ddof=1
    - an event of 1 when z_t >= threshold, of -1 when z_t <= -threshold, else 0

    For more information, see the following source:

    - van Brakel, J.P.G. (2014). "Robust peak detection algorithm using z-scores." Stack
      Overflow (https://stackoverflow.com/questions/22583391), the peak detection this filter
      follows.

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        window (int, optional): The window of the rolling mean and standard deviation. Defaults to 20.
        threshold (float, optional): The number of standard deviations. Defaults to 3.0.

    Returns:
        pd.Series: 1 for an upward event, -1 for a downward event and 0 otherwise, NaN until the
        window is filled.

    Notes:
        A window of equal prices has no standard deviation and gives no event. The filter is
        symmetric, with events above and below the band, and takes one window for the mean and
        the standard deviation.
    """
    rolling = prices.rolling(window)
    z_scores = (prices - rolling.mean()) / rolling.std()

    events = pd.Series(
        np.where(
            z_scores >= threshold, 1.0, np.where(z_scores <= -threshold, -1.0, 0.0)
        ),
        index=prices.index,
    )
    # A window of equal prices has no deviation: no event rather than an undefined z-score.
    return events.where(rolling.std().notna())


def get_triple_barrier_labels(
    prices: pd.Series,
    target: pd.Series,
    horizon: int = 10,
    profit_taking: float = 1.0,
    stop_loss: float = 1.0,
    events: pd.Series | None = None,
    side: pd.Series | None = None,
    vertical_barrier_zero: bool = False,
    minimum_target: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the triple-barrier labels: for every event the first of three barriers the price reaches, a
    profit-taking barrier above, a stop-loss barrier below and a vertical barrier `horizon` periods later.

    Labels look forward by construction: the label of date t is known only once a barrier is reached.

    The formula is as follows:

    - R_t,k = (p_t+k / p_t - 1) x side_t, for k = 1..horizon
    - upper_t = profit_taking x target_t and lower_t = -stop_loss x target_t
    - k* = the first k with R_t,k > upper_t or R_t,k < lower_t, else horizon (the vertical barrier)
    - without a side: label = 1 at the upper barrier, -1 at the lower barrier and sign(R_t,k*) at
      the vertical barrier (0 with `vertical_barrier_zero`)
    - with a side (meta-labels): label = 1 when R_t,k* > 0, else 0 (also 0 at the vertical
      barrier with `vertical_barrier_zero`)

    A multiple of zero removes that barrier. An event without a barrier reached before the end of
    the data and whose vertical barrier lies beyond it gets no label (NaN): its outcome is not
    known yet.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 3
      (Snippets 3.2 to 3.7).

    Referenced against:
    - RiskLabAI, triple_barrier: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L193
    - RiskLabAI, meta_labeling: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/labeling.py#L402

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        target (pd.Series): The target return per date, the unit of the horizontal barriers.
        horizon (int, optional): The number of periods to the vertical barrier. Defaults to 10.
        profit_taking (float, optional): The multiple of the target of the upper barrier, 0 for
            none. Defaults to 1.0.
        stop_loss (float, optional): The multiple of the target of the lower barrier, 0 for none.
            Defaults to 1.0.
        events (pd.Series | None, optional): The dates to label, True or non-zero. Defaults to
            None, which labels every date.
        side (pd.Series | None, optional): The side of the bet per date, 1 (long) or -1 (short),
            for meta-labels; dates without a side get no label. Defaults to None.
        vertical_barrier_zero (bool, optional): Whether the vertical barrier labels 0. Defaults to
            False, which labels the sign of the return.
        minimum_target (float, optional): The target an event needs to exceed to be labeled.
            Defaults to 0.0.

    Returns:
        pd.DataFrame: The Label, Return (multiplied by the side for meta-labels), Barrier (1 upper,
        -1 lower, 0 vertical), Holding Period (the periods from the event to the barrier) and
        Target, with the Side for meta-labels.
    """
    values = prices.to_numpy(dtype=float)
    count = len(values)
    targets = target.reindex(prices.index).to_numpy(dtype=float)

    selected = np.isfinite(targets) & (targets > minimum_target)
    if events is not None:
        flags = events.reindex(prices.index)
        selected &= (flags.notna() & flags.astype(float).ne(0)).to_numpy()
    if side is None:
        sides = np.ones(count)
    else:
        sides = side.reindex(prices.index).to_numpy(dtype=float)
        selected &= np.isfinite(sides) & (sides != 0)

    # Row t holds the prices of t + 1 .. t + horizon, NaN beyond the data.
    future = sliding_window_view(
        np.concatenate([values[1:], np.full(horizon, np.nan)]), horizon
    )[:count]
    with np.errstate(invalid="ignore", divide="ignore"):
        path = (future / values[:, None] - 1) * sides[:, None]
    upper = profit_taking * targets if profit_taking > 0 else np.full(count, np.inf)
    lower = -stop_loss * targets if stop_loss > 0 else np.full(count, -np.inf)
    with np.errstate(invalid="ignore"):
        upper_reached = path > upper[:, None]
        lower_reached = path < lower[:, None]
    reached = upper_reached | lower_reached

    touched = reached.any(axis=1)
    first = reached.argmax(axis=1)
    rows = np.arange(count)
    complete = rows + horizon <= count - 1

    holding_period = np.where(touched, first + 1, np.where(complete, horizon, np.nan))
    barrier = np.where(
        touched,
        np.where(upper_reached[rows, first], 1.0, -1.0),
        np.where(complete, 0.0, np.nan),
    )
    known = selected & np.isfinite(holding_period)
    holding_period = np.where(known, holding_period, np.nan)
    barrier = np.where(known, barrier, np.nan)
    returns = np.full(count, np.nan)
    returns[known] = path[rows[known], holding_period[known].astype(int) - 1]

    label = np.sign(returns) if side is None else (returns > 0).astype(float)
    if vertical_barrier_zero:
        label = np.where(barrier == 0, 0.0, label)
    label = np.where(known, label, np.nan)

    labels = pd.DataFrame(
        {
            "Label": label,
            "Return": returns,
            "Barrier": barrier,
            "Holding Period": holding_period,
            "Target": np.where(known, targets, np.nan),
        },
        index=prices.index,
    )
    if side is not None:
        labels["Side"] = np.where(known, sides, np.nan)

    return labels


def get_volatility_triple_barrier_labels(
    prices: pd.Series,
    span: int = 100,
    horizon: int = 10,
    profit_taking: float = 1.0,
    stop_loss: float = 1.0,
    events: pd.DataFrame | None = None,
    side: pd.DataFrame | None = None,
    vertical_barrier_zero: bool = False,
    minimum_target: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the triple-barrier labels of one ticker with its daily volatility as the target,
    see get_triple_barrier_labels.

    Args:
        prices (pd.Series): The closes of the ticker, named after the ticker.
        span (int, optional): The span of the daily volatility. Defaults to 100.
        horizon (int, optional): The periods to the vertical barrier. Defaults to 10.
        profit_taking (float, optional): The multiple of the target of the profit-taking
            barrier. Defaults to 1.0.
        stop_loss (float, optional): The multiple of the target of the stop-loss barrier.
            Defaults to 1.0.
        events (pd.DataFrame | None, optional): The event dates per ticker. Defaults to None,
            every date.
        side (pd.DataFrame | None, optional): The side of the bets per ticker (meta-labeling).
            Defaults to None, no side.
        vertical_barrier_zero (bool, optional): Whether the vertical barrier labels 0.
            Defaults to False.
        minimum_target (float, optional): The smallest target an event needs. Defaults to 0.0.

    Returns:
        pd.DataFrame: The labels of the ticker, see get_triple_barrier_labels.
    """
    ticker = prices.name
    return get_triple_barrier_labels(
        prices,
        get_daily_volatility(prices, span),
        horizon=horizon,
        profit_taking=profit_taking,
        stop_loss=stop_loss,
        events=(None if events is None else events.get(ticker, pd.Series(dtype=float))),
        side=(
            None
            if side is None
            else side.get(ticker, pd.Series(dtype=float)).reindex(prices.index)
        ),
        vertical_barrier_zero=vertical_barrier_zero,
        minimum_target=minimum_target,
    )


def get_forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Calculate the forward returns: the return from each date to a number of periods later, the outcome a label
    describes.

    It looks forward by construction.

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        horizon (int, optional): The number of periods ahead. Defaults to 1.

    Returns:
        pd.Series: The forward return per date, NaN for the last `horizon` dates.
    """
    return prices.shift(-horizon) / prices - 1


def get_fixed_horizon_labels(
    forward_returns: pd.DataFrame,
    threshold: float | pd.DataFrame = 0.0,
    relative_to: str | None = None,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Calculate the fixed-horizon labels: the sign of the forward return, or of its excess over the cross-sectional
    mean or median or over a benchmark, with 0 within the threshold.

    The formula is as follows:

    - x_t = r_t,t+h, or r_t,t+h minus the mean or median of r_t,t+h across the tickers, or minus
      the benchmark's r_t,t+h
    - label = 1 when x_t > threshold, -1 when x_t < -threshold, else 0

    For more information, see the following book:

    - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge University
      Press, section 5.2.

    Args:
        forward_returns (pd.DataFrame): The forward returns, a column per ticker.
        threshold (float | pd.DataFrame, optional): The threshold, fixed or per date and ticker.
            Defaults to 0.0.
        relative_to (str | None, optional): What the return is measured against. Options are: "mean" (the excess
            over the tickers' mean return), "median" (the excess over the tickers' median return),
            "benchmark" (the excess over the benchmark returns). Defaults to None, the return itself.
        benchmark_returns (pd.Series | None, optional): The forward returns of the benchmark, for
            relative_to="benchmark". Defaults to None.

    Returns:
        pd.DataFrame: The Return (the excess return when relative) and the Label, with a column per
        measure and ticker.
    """
    if relative_to is None:
        returns = forward_returns
    elif relative_to == "mean":
        returns = forward_returns.sub(forward_returns.mean(axis=1), axis=0)
    elif relative_to == "median":
        returns = forward_returns.sub(forward_returns.median(axis=1), axis=0)
    elif relative_to == "benchmark":
        if benchmark_returns is None:
            raise ValueError("relative_to='benchmark' needs the benchmark returns.")
        returns = forward_returns.sub(benchmark_returns, axis=0)
    else:
        raise InvalidArgumentError(
            f"relative_to must be None, 'mean', 'median' or 'benchmark', not {relative_to!r}."
        )

    label = pd.DataFrame(
        np.where(
            returns.gt(threshold), 1.0, np.where(returns.lt(-threshold), -1.0, 0.0)
        ),
        index=returns.index,
        columns=returns.columns,
    ).where(returns.notna())
    if isinstance(threshold, pd.DataFrame):
        label = label.where(threshold.reindex_like(returns).notna())

    return pd.concat({"Return": returns, "Label": label}, axis=1)


def get_trend_scanning_labels(
    prices: pd.Series, minimum_window: int = 5, maximum_window: int = 20
) -> pd.DataFrame:
    """
    Calculate the trend-scanning labels: for every date the forward window, between the minimum and the maximum
    length, whose linear trend has the most significant slope, with the sign of that slope as the label.

    Labels look forward by construction.

    The formula is as follows:

    - for L = minimum_window..maximum_window: OLS of p_t+l = b0 + b1 x l + e, l = 0..L-1
    - t(b1) = b1 / sqrt(SSR / (L - 2) / SUM((l - mean(l))^2))
    - L* = the L with the largest |t(b1)| (the shortest on a tie)
    - label = sign(t(b1) at L*), Holding Period = L* - 1, Return = p_t+L*-1 / p_t - 1

    For more information, see the following book:

    - López de Prado, M. (2020). "Machine Learning for Asset Managers." Cambridge University
      Press, section 5.4.

    Referenced against:
    - RiskLabAI, find_trend_using_trend_scanning: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/labeling/financial_labels.py#L51

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        minimum_window (int, optional): The shortest window, at least 3. Defaults to 5.
        maximum_window (int, optional): The longest window. Defaults to 20.

    Returns:
        pd.DataFrame: The Label, t-Value, Return and Holding Period per date, NaN for the dates
        whose longest window runs beyond the data.
    """
    if minimum_window < 3 or maximum_window < minimum_window:  # noqa: PLR2004
        raise ValueError(
            "The windows need 3 <= minimum_window <= maximum_window, as the t-value of the slope "
            "needs at least one degree of freedom."
        )
    values = prices.to_numpy(dtype=float)
    count = len(values)
    # No date has a complete longest window, and scanning a window longer than the data costs
    # its length squared for nothing.
    if maximum_window > count:
        return pd.DataFrame(
            np.nan,
            index=prices.index,
            columns=["Label", "t-Value", "Return", "Holding Period"],
        )
    padded = np.concatenate([values, np.full(maximum_window - 1, np.nan)])

    best_absolute = np.full(count, -np.inf)
    best_t_value = np.full(count, np.nan)
    best_window = np.full(count, np.nan)
    for window in range(minimum_window, maximum_window + 1):
        windows = sliding_window_view(padded, window)[:count]
        steps = np.arange(window) - (window - 1) / 2
        steps_squared = steps @ steps
        centered = windows - windows.mean(axis=1, keepdims=True)
        slope = centered @ steps / steps_squared
        # The residual sum of squares from the centered prices, SSR = SUM(c^2) - b1^2 x SUM(l^2), clipped at zero for a perfect line.  # noqa: E501
        squared_residuals = np.maximum(
            np.einsum("ij,ij->i", centered, centered) - slope**2 * steps_squared, 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t_value = slope / np.sqrt(squared_residuals / (window - 2) / steps_squared)
        better = np.abs(t_value) > best_absolute
        best_absolute = np.where(better, np.abs(t_value), best_absolute)
        best_t_value = np.where(better, t_value, best_t_value)
        best_window = np.where(better, window, best_window)

    complete = np.arange(count) + maximum_window - 1 <= count - 1
    best_t_value = np.where(complete, best_t_value, np.nan)
    holding_period = np.where(
        complete & np.isfinite(best_window), best_window - 1, np.nan
    )
    returns = np.full(count, np.nan)
    known = np.isfinite(holding_period)
    end = (np.arange(count) + np.nan_to_num(holding_period)).astype(int)
    returns[known] = values[end[known]] / values[known] - 1

    return pd.DataFrame(
        {
            "Label": np.sign(best_t_value),
            "t-Value": best_t_value,
            "Return": returns,
            "Holding Period": holding_period,
        },
        index=prices.index,
    )


def get_tail_set_labels(scores: pd.DataFrame, quantile: float = 0.1) -> pd.DataFrame:
    """
    Calculate the tail-set labels: per date, 1 for the tickers in the top `quantile` of the scores (such as
    volatility-adjusted forward returns), -1 for those in the bottom `quantile` and 0 for the rest.

    The formula is as follows:

    - q_i = (rank_i - 1) / (N - 1), the percentile rank among the N tickers with a score
    - label = 1 when q_i >= 1 - quantile, -1 when q_i <= quantile, else 0

    The tail sets follow Huerta, R., Corbacho, F. and Elkan, C. (2013). "Nonlinear Support
    Vector Machines Can Systematically Identify Stocks with High and Low Future Returns."
    Algorithmic Finance, 2(1), 45-58, who label the stocks with the highest and lowest
    volatility-adjusted returns and leave the middle out.

    Args:
        scores (pd.DataFrame): The scores, a column per ticker.
        quantile (float, optional): The share of tickers in each tail, between 0 and 0.5.
            Defaults to 0.1.

    Returns:
        pd.DataFrame: The labels per date and ticker, NaN without a score or on dates with fewer
        than two scores.
    """
    if not 0 < quantile <= 0.5:  # noqa: PLR2004
        raise ValueError(f"The quantile must lie in (0, 0.5], not {quantile}.")
    ranks = scores.rank(axis=1, method="average")
    counts = scores.count(axis=1)
    spread = (counts - 1).where(counts > 1)
    # Compared in ranks with a little room, so a percentile on the boundary (such as 0.9 for the
    # tenth of eleven tickers) counts as inside the tail whatever its rounding.
    upper = ranks.sub(1).ge(spread.mul(1 - quantile) - 1e-9, axis=0)
    lower = ranks.sub(1).le(spread.mul(quantile) + 1e-9, axis=0)

    labels = pd.DataFrame(
        np.where(upper, 1.0, np.where(lower, -1.0, 0.0)),
        index=scores.index,
        columns=scores.columns,
    )
    return labels.where(scores.notna()).where(spread.notna(), axis=0)


def get_matrix_flag_labels(
    prices: pd.Series,
    window: int = 40,
    template: str | np.ndarray = "leigh_bull",
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Calculate the matrix flag scores: how closely the prices of the window ending at each date fit a flag pattern,
    by laying a 10 x 10 template over the deciles of the window's prices.

    The formula is as follows:

    - split the window's prices into ten consecutive parts (the first parts one longer when the
      window is not a multiple of ten)
    - D[d, j] = the share of the prices of part j in price decile d of the whole window, with
      the highest decile in the top row
    - Score = SUM(W x D), W the template; Label = 1 when Score >= threshold, else 0

    The window ends at the date itself, so the score uses no later prices.

    For more information, see the following papers:

    - Leigh, W., Modani, N., Purvis, R. and Roberts, T. (2002). "Stock Market Trading Rule
      Discovery Using Technical Charting Heuristics." Expert Systems with Applications, 23(2),
      155-159.
    - Cervelló-Royo, R., Guijarro, F. and Michniuk, K. (2015). "Stock Market Trading Rule Based on
      Pattern Recognition and Technical Analysis: Forecasting the DJIA Index with Intraday Data."
      Expert Systems with Applications, 42(14), 5963-5975.

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        window (int, optional): The number of prices of the window, at least 10. Defaults to 40.
        template (str | np.ndarray, optional): The template of weights. Options are: "leigh_bull" (the bull flag
            of Leigh, Modani, Purvis and Roberts, 2002), "leigh_bear" (the Leigh bull flag upside down),
            "cervello_royo_bull" (the bull flag of Cervelló-Royo, Guijarro and Michniuk, 2015),
            "cervello_royo_bear" (the Cervelló-Royo bull flag upside down), or a 10 x 10 array of weights.
            Defaults to "leigh_bull".
        threshold (float | None, optional): The score from which the label is 1. Defaults to None,
            which returns the score only.

    Returns:
        pd.DataFrame: The Score per date, with the Label when a threshold is given.
    """
    if window < 10:  # noqa: PLR2004
        raise ValueError(f"The window needs at least 10 prices, not {window}.")
    if isinstance(template, str):
        check_option("template", template, MATRIX_FLAG_TEMPLATES)
        weights = MATRIX_FLAG_TEMPLATES[template]
    else:
        weights = np.asarray(template, dtype=float)
        if weights.shape != (10, 10):
            raise ValueError("A template needs 10 x 10 weights.")

    values = prices.to_numpy(dtype=float)
    count = len(values)
    scores = np.full(count, np.nan)
    if count >= window:
        windows = sliding_window_view(values, window)
        cutoffs = np.percentile(windows, np.arange(10, 101, 10), axis=1).T
        # The decile of a price: the number of cutoffs below it, so a price on a cutoff belongs
        # to the decile the cutoff closes.
        deciles = (windows[:, :, None] > cutoffs[:, None, :]).sum(axis=2).clip(max=9)
        sizes = np.array([len(part) for part in np.array_split(np.arange(window), 10)])
        parts = np.repeat(np.arange(10), sizes)
        contributions = weights[9 - deciles, parts] / sizes[parts]
        scores[window - 1 :] = contributions.sum(axis=1)
        scores[window - 1 :][np.isnan(windows).any(axis=1)] = np.nan

    flags = pd.DataFrame({"Score": scores}, index=prices.index)
    if threshold is not None:
        flags["Label"] = np.where(scores >= threshold, 1.0, 0.0)
        flags["Label"] = flags["Label"].where(flags["Score"].notna())

    return flags


def get_label_concurrency(holding_periods: pd.Series) -> pd.Series:
    """
    Calculate the concurrency of labels: per date, the number of labels whose span, from the label's date to its
    barrier, includes the date.

    The formula is as follows:

    - c_t = SUM_i 1[t0_i <= t <= t1_i], t1_i the date `Holding Period` periods after t0_i

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 4.3
      (Snippet 4.1).

    Args:
        holding_periods (pd.Series): The holding period of the label at each date in periods, NaN
            for dates without a label, over the ticker's own trading days.

    Returns:
        pd.Series: The number of concurrent labels per date.
    """
    count = len(holding_periods)
    periods = holding_periods.to_numpy(dtype=float)
    starts = np.flatnonzero(np.isfinite(periods))
    ends = np.minimum(starts + periods[starts].astype(int), count - 1)

    changes = np.zeros(count + 1)
    np.add.at(changes, starts, 1)
    np.add.at(changes, ends + 1, -1)

    return pd.Series(np.cumsum(changes[:count]), index=holding_periods.index)


def get_average_uniqueness(holding_periods: pd.Series) -> pd.Series:
    """
    Calculate the average uniqueness of every label: the mean over its span of one over the number of concurrent
    labels.

    A label that overlaps no other has a uniqueness of 1.

    The formula is as follows:

    - u_i = mean of 1 / c_t over t0_i <= t <= t1_i

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 4.4
      (Snippet 4.2).

    Referenced against:
    - RiskLabAI, calculate_average_uniqueness: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L71

    Args:
        holding_periods (pd.Series): The holding period of the label at each date in periods, NaN
            for dates without a label, over the ticker's own trading days.

    Returns:
        pd.Series: The average uniqueness of the label at each date, NaN without a label.
    """
    concurrency = get_label_concurrency(holding_periods).to_numpy()
    with np.errstate(divide="ignore"):
        inverse = np.where(concurrency > 0, 1 / concurrency, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])

    periods = holding_periods.to_numpy(dtype=float)
    starts = np.flatnonzero(np.isfinite(periods))
    ends = np.minimum(starts + periods[starts].astype(int), len(periods) - 1)
    uniqueness = np.full(len(periods), np.nan)
    uniqueness[starts] = (cumulative[ends + 1] - cumulative[starts]) / (
        ends - starts + 1
    )

    return pd.Series(uniqueness, index=holding_periods.index)


def get_label_uniqueness(holding_periods: pd.Series) -> pd.DataFrame:
    """
    Calculate the concurrency and the average uniqueness of the labels, see
    get_label_concurrency and get_average_uniqueness.

    Args:
        holding_periods (pd.Series): The holding period of the label of every date.

    Returns:
        pd.DataFrame: The Concurrency and the Average Uniqueness per date.
    """
    return pd.DataFrame(
        {
            "Concurrency": get_label_concurrency(holding_periods),
            "Average Uniqueness": get_average_uniqueness(holding_periods),
        }
    )


def get_return_attribution_weights(
    prices: pd.Series, holding_periods: pd.Series
) -> pd.Series:
    """
    Calculate the sample weights by return attribution: the absolute sum of the log returns over a label's span,
    each divided by the number of labels sharing it, scaled so the weights add up to the number of labels.

    The formula is as follows:

    - r_t = ln(p_t / p_t-1)
    - w~_i = |SUM_{t0_i < t <= t1_i} r_t / c_t|
    - w_i = w~_i x N / SUM_j w~_j

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 4.6
      (Snippet 4.10).

    Referenced against:
    - RiskLabAI, sample_weight_absolute_return_meta_labeling: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L200

    Args:
        prices (pd.Series): The prices of one ticker over its own trading days.
        holding_periods (pd.Series): The holding period of the label at each date in periods, NaN
            for dates without a label, on the dates of the prices.

    Returns:
        pd.Series: The weight of the label at each date, NaN without a label.

    Notes:
        Snippet 4.10 also adds the return into t0_i, the period before the label starts; this sum
        starts after it, with the returns the label earns.
    """
    holding_periods = holding_periods.reindex(prices.index)
    concurrency = get_label_concurrency(holding_periods).to_numpy()
    log_returns = np.diff(np.log(prices.to_numpy(dtype=float)), prepend=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        attributed = np.where(concurrency > 0, log_returns / concurrency, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.nan_to_num(attributed))])

    periods = holding_periods.to_numpy(dtype=float)
    starts = np.flatnonzero(np.isfinite(periods))
    ends = np.minimum(starts + periods[starts].astype(int), len(periods) - 1)
    weights = np.full(len(periods), np.nan)
    weights[starts] = np.abs(cumulative[ends + 1] - cumulative[starts + 1])
    total = np.nansum(weights)
    if total > 0:
        weights *= len(starts) / total

    return pd.Series(weights, index=prices.index)


def get_time_decay_weights(
    average_uniqueness: pd.Series, decay: float = 1.0
) -> pd.Series:
    """
    Calculate the sample weights by time decay, so that older labels count less without counting overlapping labels
    twice.

    The weights fall linearly with the cumulative uniqueness from 1 for the newest label to `decay` for the oldest.

    The formula is as follows:

    - x_i = the cumulative average uniqueness up to label i, X the total
    - slope = (1 - decay) / X when decay >= 0, else 1 / ((decay + 1) x X)
    - w_i = max(0, 1 - slope x X + slope x x_i)

    A decay of 1 gives every label a weight of 1, 0 lets the weights fall to 0 for the oldest
    label, and a negative decay gives the oldest share (-decay) of the uniqueness a weight of 0.

    For more information, see the following book:

    - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, section 4.7
      (Snippet 4.11).

    Referenced against:
    - RiskLabAI, calculate_time_decay: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/weights/sample_weights.py#L279

    Args:
        average_uniqueness (pd.Series): The average uniqueness of the label at each date, NaN
            without a label.
        decay (float, optional): The weight of the oldest label, in (-1, 1]. Defaults to 1.0.

    Returns:
        pd.Series: The weight of the label at each date, NaN without a label.
    """
    if not -1 < decay <= 1:
        raise ValueError(f"The decay must lie in (-1, 1], not {decay}.")
    cumulative = average_uniqueness.dropna().sort_index().cumsum()
    if cumulative.empty:
        return average_uniqueness * np.nan
    total = cumulative.iloc[-1]
    slope = (1 - decay) / total if decay >= 0 else 1 / ((decay + 1) * total)
    weights = (1 - slope * total + slope * cumulative).clip(lower=0)

    return weights.reindex(average_uniqueness.index)


def get_sample_weights(
    prices: pd.Series,
    holding_periods: pd.Series,
    method: str = "return",
    decay: float = 1.0,
) -> pd.Series:
    """
    Calculate the sample weights of the labels: by return attribution
    (get_return_attribution_weights) or by time decay of the average uniqueness
    (get_time_decay_weights).

    Args:
        prices (pd.Series): The closes.
        holding_periods (pd.Series): The holding period of the label of every date.
        method (str, optional): "return" or "time_decay". Defaults to "return".
        decay (float, optional): The weight of the oldest label of the time decay.
            Defaults to 1.0.

    Returns:
        pd.Series: The weight per date.
    """
    if method == "return":
        return get_return_attribution_weights(prices, holding_periods)
    return get_time_decay_weights(get_average_uniqueness(holding_periods), decay)
