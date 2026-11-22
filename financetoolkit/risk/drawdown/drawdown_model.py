"""Drawdown Model: drawdown measures of each sample of returns, compounded or not"""

import numpy as np
import pandas as pd

from financetoolkit.risk import risk_model
from financetoolkit.risk.value_at_risk import evar_model

AVERAGE_DRAWDOWN_METHODS = ["episode", "observation"]

# A single return already has a drawdown (a loss from the starting wealth).
MINIMUM_DRAWDOWN_OBSERVATIONS = 1


def get_drawdown_samples(samples: np.ndarray, compounded: bool = True) -> np.ndarray:
    """
    Calculate the drawdowns of every row of returns without missing values, measured from the
    running peak of the wealth path, which starts before the first return (a wealth of 1, or a
    cumulative return of 0 without compounding), so a loss on the first return is a drawdown.

    The formula is as follows:

        Compounded: Drawdown_t = Wealth_t / max(1, max(Wealth_s for s <= t)) - 1, Wealth_t = Π (1 + Return_s)
        Uncompounded: Drawdown_t = Sum_t - max(0, max(Sum_s for s <= t)), Sum_t = Σ Return_s

    The uncompounded drawdown is the absolute drawdown of Riskfolio-Lib (MDD_Abs, DaR_Abs) and
    skfolio (get_drawdowns(compounded=False)): the decline of the summed returns from their peak.

    Args:
        samples (np.ndarray): The returns, one sample per row.
        compounded (bool, optional): Whether to compound the returns. Defaults to True.

    Returns:
        np.ndarray: The drawdowns, of the same shape, 0 at a peak and negative below it.
    """
    if compounded:
        wealth = np.cumprod(1 + samples, axis=1)

        return wealth / np.maximum(np.maximum.accumulate(wealth, axis=1), 1) - 1

    cumulative = np.cumsum(samples, axis=1)

    return cumulative - np.maximum(np.maximum.accumulate(cumulative, axis=1), 0)


def get_maximum_drawdown(
    returns: pd.Series | pd.DataFrame,
    compounded: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Maximum Drawdown of returns, the lowest drawdown of `get_drawdown_samples`,
    over the whole sample, per period or over every rolling window (each window starting from
    a fresh wealth path).

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        compounded (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Maximum Drawdown, negative (0 without a decline).
    """
    return risk_model.apply_to_observations(
        returns,
        lambda samples: get_drawdown_samples(samples, compounded).min(axis=1),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_drawdown_at_risk(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
    compounded: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Drawdown at Risk (DaR) of returns, the alpha quantile of the drawdowns of
    every observation (Chekhlov, Uryasev and Zabarankin 2005):

        DaR = Quantile(Drawdown_t, alpha)

    The quantile is interpolated linearly, as in the historic Value at Risk and the
    Conditional Drawdown at Risk of the Toolkit.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        compounded (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The DaR, negative (0 without a decline).
    """
    evar_model.validate_alpha(alpha)

    return risk_model.apply_to_observations(
        returns,
        lambda samples: np.quantile(
            get_drawdown_samples(samples, compounded), alpha, axis=1
        ),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_drawdown_troughs(drawdowns: np.ndarray) -> np.ndarray:
    """
    The depth of every drawdown episode in a series of drawdowns: the lowest drawdown of each
    run of observations below the peak.

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
    compounded: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Average Drawdown of returns, either the average depth of the drawdown
    episodes (Bacon 2008) or the average drawdown of every observation (Chekhlov, Uryasev and
    Zabarankin 2005):

        Episode: Average Drawdown = Σ_j Trough_j / d
        Observation: Average Drawdown = Σ_t Drawdown_t / n

    Where Trough_j is the lowest drawdown of the j-th of the d episodes below the peak (an
    episode that has not recovered by the end included) and n the number of observations,
    those at a peak counting as 0.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        method (str, optional): "episode" or "observation". Defaults to "episode".
        compounded (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Average Drawdown, negative (0 without a decline).
    """
    if method not in AVERAGE_DRAWDOWN_METHODS:
        raise ValueError("Method must be episode or observation.")

    def average_drawdown(samples: np.ndarray) -> np.ndarray:
        drawdowns = get_drawdown_samples(samples, compounded)

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
    compounded: bool = True,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Entropic Drawdown at Risk (EDaR) of returns, the historical Entropic Value at
    Risk of the drawdowns of every observation (Cajas 2021):

        EDaR = -inf over z > 0 of z * ln(mean(exp(-Drawdown_t / z)) / alpha)

    It lies between the Conditional Drawdown at Risk and the Maximum Drawdown, see
    `evar_model.get_evar_samples` for the minimization.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        compounded (bool, optional): Whether to compound the returns. Defaults to True.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The EDaR, negative (0 without a decline).
    """
    evar_model.validate_alpha(alpha)

    return risk_model.apply_to_observations(
        returns,
        lambda samples: evar_model.get_evar_samples(
            get_drawdown_samples(samples, compounded), alpha
        ),
        window_size,
        MINIMUM_DRAWDOWN_OBSERVATIONS,
    )


def get_drawdown_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the drawdown episodes of every column (see `risk_model.get_drawdown_table`): the
    number of drawdowns, their average depth, the drawdown deviation and their average lengths.

    The formula is as follows:

        Average Drawdown = Σ_j Drawdown_j / d
        Drawdown Deviation = √(Σ_j Drawdown_j^2 / n)
        Average To Trough, Average To Recovery, Average Length = the mean of each over the episodes

    Where Drawdown_j is the depth of the j-th of the d episodes and n the number of returns.
    The lengths count periods, Average Length (Days) calendar days. An episode that has not
    recovered by the end of the data counts towards the depths and To Trough, not towards To
    Recovery and the Lengths.

    Args:
        returns (pd.DataFrame): The returns, one column per asset.

    Returns:
        pd.DataFrame: One row per column with the Drawdowns, Average Drawdown (negative),
            Drawdown Deviation (positive), Average To Trough, Average To Recovery, Average
            Length and Average Length (Days). NaN for a column without returns, 0 drawdowns
            and NaN averages for one that never fell below its peak.
    """
    table = risk_model.get_drawdown_table(returns, top=len(returns))
    grouped = table.groupby(level=0, sort=False)
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
        }
    ).reindex(returns.columns)

    statistics["Drawdowns"] = statistics["Drawdowns"].fillna(0)
    statistics["Drawdown Deviation"] = statistics["Drawdown Deviation"].fillna(0)

    return statistics.where(observations > 0, axis=0)
