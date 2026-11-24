"""Drawdown Model: drawdown measures of each sample of returns, compounded or not"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from financetoolkit.risk import risk_model
from financetoolkit.risk.value_at_risk import evar_model
from financetoolkit.utilities.error_model import check_option
from financetoolkit.utilities.statistics_model import by_own_trading_days

AVERAGE_DRAWDOWN_METHODS = ["episode", "observation"]

# A single return already has a drawdown (a loss from the starting wealth).
MINIMUM_DRAWDOWN_OBSERVATIONS = 1


def get_drawdown_samples(
    samples: np.ndarray, compound_returns: bool = True
) -> np.ndarray:
    """
    Calculate the drawdowns of every row of returns without missing values: how far the wealth
    path stands below its running peak at every observation.

    The peak starts before the first return (a wealth of 1, or a cumulative return of 0 without
    compounding), so a loss on the first return is a drawdown.

    The formula is as follows:

        Compounded: Drawdown_t = Wealth_t / max(1, max(Wealth_s for s <= t)) - 1, Wealth_t = Π (1 + Return_s)
        Uncompounded: Drawdown_t = Sum_t - max(0, max(Sum_s for s <= t)), Sum_t = Σ Return_s

    The uncompounded drawdown is the absolute drawdown of Riskfolio-Lib (MDD_Abs, DaR_Abs) and
    the uncompounded drawdown of skfolio: the decline of the summed returns from their peak.

    Shares similar logic:
    - skfolio, get_drawdowns: https://skfolio.org/generated/skfolio.measures.get_drawdowns.html
    - PerformanceAnalytics (R), Drawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/Drawdowns.R#L3

    Args:
        samples (np.ndarray): The returns, one sample per row.
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.

    Returns:
        np.ndarray: The drawdowns, of the same shape, 0 at a peak (within 1e-12, the
            rounding of the compounding) and negative below it.
    """
    if compound_returns:
        wealth = np.cumprod(1 + samples, axis=1)
        drawdowns = wealth / np.maximum(np.maximum.accumulate(wealth, axis=1), 1) - 1
    else:
        cumulative = np.cumsum(samples, axis=1)
        drawdowns = cumulative - np.maximum(
            np.maximum.accumulate(cumulative, axis=1), 0
        )

    # A path back at its old high lands within rounding of it, see get_wealth_and_peak.
    return np.where(drawdowns > -risk_model.PEAK_TOLERANCE, 0.0, drawdowns)


def get_maximum_drawdown(
    returns: pd.Series | pd.DataFrame,
    compound_returns: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Maximum Drawdown of returns: the largest decline from a peak, the lowest
    drawdown of `get_drawdown_samples`.

    It is taken over the whole sample, per period or over every rolling window (each window
    starting from a fresh wealth path).

    Referenced against:
    - skfolio, max_drawdown: https://skfolio.org/generated/skfolio.measures.max_drawdown.html
    - Riskfolio-Lib, MDD_Rel (compounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.MDD_Rel
    - Riskfolio-Lib, MDD_Abs (uncompounded): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.MDD_Abs

    Shares similar logic:
    - PerformanceAnalytics (R), maxDrawdown: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L44
    - empyrical, max_drawdown: https://github.com/quantopian/empyrical/blob/0.5.5/empyrical/stats.py#L352
    - quantstats, max_drawdown: https://github.com/ranaroussi/quantstats/blob/v0.0.86/quantstats/stats.py#L2677

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Maximum Drawdown, negative (0 without a decline).
    """
    return risk_model.apply_to_observations(
        returns,
        lambda samples: get_drawdown_samples(samples, compound_returns).min(axis=1),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
    compound_returns: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Drawdown at Risk (DaR) of returns: the drawdown that is only exceeded in a
    fraction alpha of the observations, the alpha quantile of the drawdowns (Chekhlov, Uryasev
    and Zabarankin 2005).

    The formula is as follows:

        DaR = Quantile(Drawdown_t, alpha)

    The quantile is interpolated linearly, as in the historic Value at Risk and the
    Conditional Drawdown at Risk of the Toolkit.

    Shares similar logic:
    - skfolio, drawdown_at_risk: https://skfolio.org/generated/skfolio.measures.drawdown_at_risk.html
    - Riskfolio-Lib, DaR_Rel: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.DaR_Rel

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The DaR, negative (0 without a decline).
    """
    evar_model.validate_alpha(alpha)

    return risk_model.apply_to_observations(
        returns,
        lambda samples: np.quantile(
            get_drawdown_samples(samples, compound_returns), alpha, axis=1
        ),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_drawdown_troughs(drawdowns: np.ndarray) -> np.ndarray:
    """
    Calculate the depth of every drawdown episode in a series of drawdowns: the lowest drawdown
    of each run of observations below the peak.

    Shares similar logic:
    - PerformanceAnalytics (R), findDrawdowns: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/findDrawdowns.R#L45

    Args:
        drawdowns (np.ndarray): The drawdowns of one sample.

    Returns:
        np.ndarray: The trough of every episode, in the order they occur.
    """
    below_peak = drawdowns < 0
    starts = np.flatnonzero(below_peak & ~np.concatenate([[False], below_peak[:-1]]))

    if len(starts) == 0:
        return np.array([])

    # Each segment runs from the start of an episode to the start of the next, so it holds the
    # episode and the observations at a peak after it, which are 0 and not below its trough.
    return np.minimum.reduceat(drawdowns, starts)


def get_average_drawdown(
    returns: pd.Series | pd.DataFrame,
    method: str = "episode",
    compound_returns: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Average Drawdown of returns: how deep the declines from a peak typically are,
    either the average depth of the drawdown episodes (Bacon 2008) or the average drawdown of
    every observation (Chekhlov, Uryasev and Zabarankin 2005).

    The formula is as follows:

        Episode: Average Drawdown = Σ_j Trough_j / d
        Observation: Average Drawdown = Σ_t Drawdown_t / n

    Where Trough_j is the lowest drawdown of the j-th of the d episodes below the peak (an
    episode that has not recovered by the end included) and n the number of observations,
    those at a peak counting as 0.

    Referenced against:
    - skfolio, average_drawdown (observation): https://skfolio.org/generated/skfolio.measures.average_drawdown.html
    - Riskfolio-Lib, ADD_Rel (observation): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.ADD_Rel

    Shares similar logic:
    - PerformanceAnalytics (R), AverageDrawdown (episode): https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L178

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        method (str, optional): The average. Options are: "episode" (the average depth of the
            drawdown episodes), "observation" (the average drawdown of every observation).
            Defaults to "episode".
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Average Drawdown, negative (0 without a decline).
    """
    check_option("method", method, AVERAGE_DRAWDOWN_METHODS)

    def average_drawdown(samples: np.ndarray) -> np.ndarray:
        drawdowns = get_drawdown_samples(samples, compound_returns)

        if method == "observation":
            return drawdowns.mean(axis=1)

        troughs = [get_drawdown_troughs(row) for row in drawdowns]

        return np.array([row.mean() if len(row) else 0.0 for row in troughs])

    return risk_model.apply_to_observations(
        returns, average_drawdown, window_size, MINIMUM_DRAWDOWN_OBSERVATIONS
    )


def get_entropic_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
    compound_returns: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Entropic Drawdown at Risk (EDaR) of returns: the tightest entropic bound on
    the deep drawdowns, the historical Entropic Value at Risk of the drawdowns of every
    observation (Cajas 2021).

    The formula is as follows:

        EDaR = -inf over z > 0 of z * ln(mean(exp(-Drawdown_t / z)) / alpha)

    It lies between the Conditional Drawdown at Risk and the Maximum Drawdown, see
    `evar_model.get_evar_samples` for the minimization.

    Referenced against:
    - skfolio, edar: https://skfolio.org/generated/skfolio.measures.edar.html
    - Riskfolio-Lib, EDaR_Rel: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EDaR_Rel

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The EDaR, negative (0 without a decline).
    """
    evar_model.validate_alpha(alpha)

    return risk_model.apply_to_observations(
        returns,
        lambda samples: evar_model.get_evar_samples(
            get_drawdown_samples(samples, compound_returns), alpha
        ),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_conditional_expected_drawdown(
    returns: pd.Series | pd.DataFrame,
    window: int = 21,
    alpha: float = 0.05,
    compound_returns: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Conditional Expected Drawdown (CED) of returns: how deep the worst declines
    within a path of a given length get, the tail average of the Maximum Drawdowns over every
    path of `window` returns (Goldberg and Mahmoud 2017).

    The formula is as follows:

        MDD_s = min over t in [s, s + window) of Drawdown_t, each path starting
        from a fresh wealth of 1
        CED = (Σ_{i <= k} MDD_(i) + (m * alpha - k) * MDD_(k + 1)) / (m * alpha)

    Where MDD_(1) <= MDD_(2) <= ... are the m = n - window + 1 Maximum Drawdowns in ascending
    order and k = floor(m * alpha): the paths overlap, one starting at every return, and the
    tail average is the exact one of Rockafellar and Uryasev (2002), the path at the boundary
    counted with the fraction of it inside the tail. Where the Conditional
    Drawdown at Risk averages the drawdowns of single observations, the CED averages the
    worst decline of whole paths, so it answers how deep a bad month (a window of 21 daily
    returns) gets.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        window (int, optional): The length of each path in returns. Defaults to 21.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        compound_returns (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window the CED is
            calculated over. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The CED, negative (0 without a decline).
    """
    evar_model.validate_alpha(alpha)

    if window < 1:
        raise ValueError("The window must be at least 1.")

    def conditional_expected_drawdown(samples: np.ndarray) -> np.ndarray:
        paths = sliding_window_view(samples, window, axis=1)
        maximum_drawdowns = (
            get_drawdown_samples(paths.reshape(-1, window), compound_returns)
            .min(axis=1)
            .reshape(paths.shape[:2])
        )
        # The tail average of the deepest alpha share, the path at the boundary counted with
        # its fraction: overlapping paths often share their Maximum Drawdown up to rounding, and
        # a cut at a quantile would take or leave such ties on the last digit.
        deepest = np.sort(maximum_drawdowns, axis=1)
        tail = deepest.shape[1] * alpha
        whole = int(np.floor(tail))
        boundary = deepest[:, min(whole, deepest.shape[1] - 1)]

        return (deepest[:, :whole].sum(axis=1) + (tail - whole) * boundary) / tail

    return risk_model.apply_to_observations(
        returns, conditional_expected_drawdown, window_size, window
    )


@by_own_trading_days
def get_drawdown_state(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the share of the time spent below the peak and the state of the drawdown that is
    still open at the end of the data: how often and how far an asset is under water, per
    column, on the drawdowns of `risk_model.get_drawdown_series`.

    The formula is as follows:

        Coverage = the number of periods below the peak / n
        Active Drawdown = Drawdown_T
        Active Duration = T - t(peak)
        Active Recovery Duration = T - t(trough)
        Active Recovery Return = (1 + Drawdown_T) / (1 + Drawdown_trough) - 1
        Active Recovery = 1 - Drawdown_T / Drawdown_trough

    Where T is the last period, n the number of returns, peak the last period at the peak and
    trough the lowest drawdown since. Active Recovery is the share of the decline from the peak
    to the trough that has been made good.

    Args:
        returns (pd.DataFrame): The returns, one column per asset.

    Returns:
        pd.DataFrame: One row per column with the Coverage, the Active Drawdown and Active
            Duration (0 at a peak) and the Active Recovery Duration, Active Recovery Return
            and Active Recovery (NaN at a peak, as there is no trough).
    """
    drawdown_series = risk_model.get_drawdown_series(returns)
    rows = {}

    for column in drawdown_series.columns:
        observed = np.flatnonzero(returns[column].notna().to_numpy())

        if len(observed) == 0:
            continue

        # From the close the first return is measured from to the last return, so the rows
        # after a delisting are not counted as time below the peak.
        drawdowns = drawdown_series[column].to_numpy()[
            max(observed[0] - 1, 0) : observed[-1] + 1
        ]
        last = len(drawdowns) - 1
        at_peak = np.flatnonzero(drawdowns >= 0)
        # The starting wealth before the first row is the peak when no row is at it.
        peak = at_peak[-1] if len(at_peak) else -1
        state = {
            "Coverage": np.count_nonzero(drawdowns < 0) / len(observed),
            "Active Drawdown": drawdowns[last],
            "Active Duration": float(last - peak),
            "Active Recovery Duration": np.nan,
            "Active Recovery Return": np.nan,
            "Active Recovery": np.nan,
        }

        if peak < last:
            trough = peak + 1 + int(np.argmin(drawdowns[peak + 1 :]))
            state["Active Recovery Duration"] = float(last - trough)
            state["Active Recovery Return"] = (1 + drawdowns[last]) / (
                1 + drawdowns[trough]
            ) - 1
            state["Active Recovery"] = 1 - drawdowns[last] / drawdowns[trough]

        rows[column] = state

    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=[
            "Coverage",
            "Active Drawdown",
            "Active Duration",
            "Active Recovery Duration",
            "Active Recovery Return",
            "Active Recovery",
        ],
    )


def get_drawdown_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the drawdown episodes of every column: how many there were, how deep and long they
    were and how they recovered.

    The summary (see `risk_model.get_drawdown_table`) holds the number of drawdowns, their
    average depth, the drawdown deviation, their average lengths and recoveries, the share of
    the time below the peak and the state of the open drawdown (see `get_drawdown_state`).

    The formula is as follows:

        Average Drawdown = Σ_j Drawdown_j / d
        Drawdown Deviation = √(Σ_j Drawdown_j^2 / n)
        Average To Trough, Average To Recovery, Average Length = the mean of each over the episodes
        Average Recovery Ratio = the mean of To Recovery_j / To Trough_j over the recovered episodes
        Average Recovery Return = the mean of Recovery Return_j over the recovered episodes

    Where Drawdown_j is the depth of the j-th of the d episodes and n the number of returns.
    The lengths count periods, Average Length (Days) calendar days. An episode that has not
    recovered by the end of the data counts towards the depths and To Trough, not towards To
    Recovery, the Lengths and the recovery measures. The Recovery Return of an episode runs from
    the trough to the close of the recovery, which may lie above the old peak.

    Shares similar logic:
    - PerformanceAnalytics (R), DrawdownDeviation: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L147
    - PerformanceAnalytics (R), AverageRecovery: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L212
    - PerformanceAnalytics (R), AverageLength: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/maxDrawdown.R#L240

    Args:
        returns (pd.DataFrame): The returns, one column per asset.

    Returns:
        pd.DataFrame: One row per column with the Drawdowns, Average Drawdown (negative),
            Drawdown Deviation (positive), Average To Trough, Average To Recovery, Average
            Length, Average Length (Days), Average Recovery Ratio, Average Recovery Return and
            the columns of `get_drawdown_state`. NaN for a column without returns, 0 drawdowns,
            0 Coverage and NaN averages for one that never fell below its peak.
    """
    # A column with dates without a return and no drawdown gives an empty part, which leaves the
    # combined table with object columns.
    table = risk_model.get_drawdown_table(returns, top=len(returns)).infer_objects()
    grouped = table.groupby(level=0, sort=False)
    recovered = table[table["Recovery"].notna()].groupby(level=0, sort=False)
    observations = returns.count()

    statistics = pd.DataFrame(
        {
            "Drawdowns": grouped.size(),
            "Average Drawdown": grouped["Drawdown"].mean(),
            "Drawdown Deviation": np.sqrt(
                (table["Drawdown"] ** 2).groupby(level=0, sort=False).sum()
                / observations
            ),
            "Average To Trough": grouped["To Trough"].mean(),
            "Average To Recovery": grouped["To Recovery"].mean(),
            "Average Length": grouped["Length"].mean(),
            "Average Length (Days)": grouped["Length (Days)"].mean(),
            "Average Recovery Ratio": recovered["Recovery Ratio"].mean(),
            "Average Recovery Return": recovered["Recovery Return"].mean(),
        }
    ).reindex(returns.columns)

    statistics["Drawdowns"] = statistics["Drawdowns"].fillna(0)
    statistics["Drawdown Deviation"] = statistics["Drawdown Deviation"].fillna(0)
    statistics = statistics.join(get_drawdown_state(returns))

    return statistics.where(observations > 0, axis=0)
