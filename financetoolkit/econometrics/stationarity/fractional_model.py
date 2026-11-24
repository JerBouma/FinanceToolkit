"""Fractional Differentiation Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.econometrics.stationarity.unitroot_model import (
    MINIMUM_TEST_OBSERVATIONS,
    get_augmented_dickey_fuller,
)
from financetoolkit.utilities.error_model import check_option

# The weight thresholds of López de Prado (2018): the expanding window skips the first values
# until the relative weight loss is at most 1%, the fixed-width window drops the weights
# smaller than 1e-5 in absolute value.
EXPANDING_THRESHOLD = 0.01
FIXED_WIDTH_THRESHOLD = 1e-5

# The fixed-width threshold of the search for the minimum d, 0.01 as in López de Prado (2018,
# Snippet 5.4). At 1e-5 the window of d = 0.1 spans about 4,000 observations and at 1e-4 about
# 500, so on a few years of data every small d would come out empty and fail the test by
# default; at 0.01 the windows span 6 to 12 observations.
MINIMUM_D_THRESHOLD = 0.01


def get_fractional_weights(
    d: float, size: int | None = None, threshold: float | None = None
) -> np.ndarray:
    """
    Calculate the weights of the fractional difference operator (1 - L)^d: how much every
    past value counts in a fractional difference.

    The weights are the binomial series of Hosking (1981) and López de Prado (2018, ch. 5):

    - ω₀ = 1
    - ωₖ = -ωₖ₋₁·(d - k + 1) / k = Π(i=1..k) (i - 1 - d) / i

    For a whole number d the weights end after d + 1 terms (d = 1 gives 1, -1, the first
    difference); for any other d they decay in absolute value without ending.

    Referenced against:
    - RiskLabAI, calculate_weights_std: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L23
    - RiskLabAI, calculate_weights_ffd: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L51

    Args:
        d (float): The order of differentiation, at least 0.
        size (int | None, optional): The number of weights, ω₀ to ωₛᵢᵤₑ₋₁. Defaults to None,
            which requires a threshold.
        threshold (float | None, optional): Stop before the first weight whose absolute value is
            below the threshold, the fixed-width window. With a size as well, at most `size`
            weights are returned. Defaults to None.

    Returns:
        np.ndarray: The weights ω₀, ω₁, ... in the order of the lags they multiply.

    Raises:
        ValueError: If d is negative, or neither a size nor a threshold is given.
    """
    if d < 0:
        raise ValueError("The order of differentiation d must be at least 0.")
    if size is None and threshold is None:
        raise ValueError("Pass the number of weights (size) or a threshold.")

    if threshold is None and size is not None:
        return np.cumprod(
            np.concatenate(([1.0], (np.arange(size - 1) - d) / np.arange(1, size)))
        )

    # The weights decay like k^(-1-d), so the window grows quickly for a small d: build it in
    # blocks and cut it at the first weight below the threshold.
    weights = np.array([1.0])
    block = 1024
    while size is None or len(weights) < size:
        k = np.arange(len(weights), len(weights) + block)
        new = weights[-1] * np.cumprod((k - 1 - d) / k)
        below = np.flatnonzero(np.abs(new) < threshold)
        if below.size:
            new = new[: below[0]]
        weights = np.concatenate((weights, new))
        if below.size:
            break

    return weights if size is None else weights[:size]


def get_fractional_difference(
    series: pd.Series,
    d: float = 0.4,
    method: str = "fixed_width",
    threshold: float | None = None,
) -> pd.Series:
    """
    Calculate the fractional difference of a series, (1 - L)^d applied to its values, which
    removes as much of the trend as needed for a stationary series while keeping the memory
    a first difference (d = 1) throws away (López de Prado, 2018, ch. 5).

    The formula is as follows, with Xₜ the series and ωₖ the weights of
    `get_fractional_weights`:

    - X̃ₜ = Σ(k=0..t) ωₖ·Xₜ₋ₖ for the expanding window, kept from the first t at which the
      share of the weight left out, λₜ = Σ(j>t) |ωⱼ| / Σ(j=0..n-1) |ωⱼ|, is at most the threshold
      (0.01 by default): the values before it miss too much of their weight to compare
    - X̃ₜ = Σ(k=0..l*) ωₖ·Xₜ₋ₖ for the fixed-width window (FFD), with l* the last lag whose
      weight is at least the threshold (1e-5 by default) in absolute value: every value has
      the same weights, so the series has no drift from a growing window

    Referenced against:
    - RiskLabAI, fractional_difference_std: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L85
    - RiskLabAI, fractional_difference_fixed: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L142

    Args:
        series (pd.Series): The values, such as log prices. Missing values are left out.
        d (float, optional): The order of differentiation, at least 0. Defaults to 0.4.
        method (str, optional): The window. Options are: "fixed_width" (fixed-width window,
            FFD), "expanding" (expanding window). Defaults to "fixed_width".
        threshold (float | None, optional): The weight threshold. Defaults to None, which is
            1e-5 for the fixed-width window and 0.01 for the expanding window.

    Returns:
        pd.Series: The fractionally differentiated series on the dates of the series, NaN for
        the first values without a full window, and NaN throughout for the fixed-width window
        when the window is longer than the series.

    Raises:
        ValueError: If the method is unknown or d is negative.
    """
    check_option("method", method, ("fixed_width", "expanding"))

    values = series.dropna()
    x = values.to_numpy(dtype=float)
    result = np.full(len(x), np.nan)

    if method == "fixed_width":
        # One weight more than the series holds tells a window that fits from one that is cut
        # short at the length of the series: a cut window never reached the threshold, so the
        # series is too short for it and the result is all NaN.
        weights = get_fractional_weights(
            d,
            size=len(x) + 1,
            threshold=FIXED_WIDTH_THRESHOLD if threshold is None else threshold,
        )
        width = len(weights)
        if width <= len(x):
            # Each value is the dot product of the weights with the window ending at it.
            result[width - 1 :] = np.convolve(x, weights, mode="valid")
    elif len(x):
        weights = get_fractional_weights(d, size=len(x))
        absolute = np.abs(weights)
        # The share of the total weight left out by a value at position l uses lags 0..l; the
        # values are skipped while the weight from lag l on is above the threshold (AFML 5.5).
        weight_from = (
            1 - np.concatenate(([0.0], np.cumsum(absolute)[:-1])) / absolute.sum()
        )
        skip = int(
            np.sum(
                weight_from > (EXPANDING_THRESHOLD if threshold is None else threshold)
            )
        )
        result[skip:] = np.convolve(x, weights)[: len(x)][skip:]

    return pd.Series(result, index=values.index, name=series.name).reindex(series.index)


def get_minimum_fractional_difference(
    series: pd.Series,
    confidence: float = 0.95,
    step: float = 0.1,
    threshold: float = MINIMUM_D_THRESHOLD,
    max_lag: int | None = 1,
) -> tuple[pd.DataFrame, float]:
    """
    Find the smallest order of fractional differentiation d at which the fixed-width
    fractional difference of a series passes the Augmented Dickey-Fuller test, the d that
    makes it stationary while keeping the most memory (López de Prado, 2018, section 5.6).

    The formula is as follows:

    - For d = 0, step, 2·step, ..., 1: X̃ = FFD(X, d) with the weight threshold, the ADF test
      on X̃ (constant, lags by AIC up to `max_lag`, 1 by default as in Snippet 5.4) and the correlation of X̃ with X
    - d* = the smallest d with an ADF p-value below 1 - confidence, refined in steps of
      step / 10 between the last d that fails and the first that passes

    Referenced against:
    - RiskLabAI, find_optimal_ffd_simple: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/differentiation/differentiation.py#L273

    Args:
        series (pd.Series): The values, such as log prices.
        confidence (float, optional): The confidence of the ADF test, the p-value has to be
            below 1 - confidence. Defaults to 0.95.
        step (float, optional): The step of the grid of d from 0 to 1. Defaults to 0.1.
        threshold (float, optional): The weight threshold of the fixed-width window. Defaults to
            0.01.
        max_lag (int | None, optional): The largest lag of the ADF lag search. Defaults to 1, as
            López de Prado (2018) uses; None searches up to the default of
            `get_augmented_dickey_fuller`, where the many lags of daily data weaken the test and
            push d* towards 1.

    Returns:
        tuple[pd.DataFrame, float]: The ADF statistic, p-value, lags, observations, 5% critical
        value, correlation with the series, window width and whether the series is stationary
        for every d tried (in the index), and d*, NaN when no d up to 1 passes.

    Raises:
        ValueError: If confidence is not between 0 and 1 or the step not between 0 and 1.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1, e.g. 0.95.")
    if not 0 < step <= 1:
        raise ValueError("step must be above 0 and at most 1, e.g. 0.1.")

    values = series.dropna()
    significance = 1 - confidence

    def evaluate(d: float) -> pd.Series:
        width = len(get_fractional_weights(d, size=len(values), threshold=threshold))
        differenced = get_fractional_difference(
            values, d=d, method="fixed_width", threshold=threshold
        ).dropna()
        if len(differenced) < MINIMUM_TEST_OBSERVATIONS:
            adf = get_augmented_dickey_fuller(differenced.iloc[:0])
            correlation = np.nan
        else:
            adf = get_augmented_dickey_fuller(differenced, max_lag=max_lag)
            correlation = (
                np.corrcoef(differenced, values.loc[differenced.index])[0, 1]
                if differenced.std() > 0
                else np.nan
            )

        return pd.Series(
            {
                "ADF Statistic": adf["ADF Statistic"],
                "P-Value": adf["P-Value"],
                "Lags Used": adf["Lags Used"],
                "Observations": adf["Observations"],
                "Critical Value 5%": adf["Critical Value 5%"],
                "Correlation": correlation,
                "Window Width": width,
                "Stationary": bool(adf["P-Value"] < significance),
            }
        )

    grid = np.round(np.arange(0, 1 + step / 2, step), 10)
    table = {}
    minimum_d = np.nan
    for d in grid:
        table[d] = evaluate(d)
        if table[d]["Stationary"]:
            minimum_d = d
            break

    if not np.isnan(minimum_d) and minimum_d > 0:
        # Refine between the last d that failed and the first that passed.
        fine_step = step / 10
        for d in np.round(
            np.arange(minimum_d - step + fine_step, minimum_d, fine_step), 10
        ):
            table[d] = evaluate(d)
            if table[d]["Stationary"]:
                minimum_d = d
                break

    # The rest of the grid shows how the statistic and the correlation move beyond d*.
    for d in grid:
        if d not in table:
            table[d] = evaluate(d)

    result = pd.DataFrame(table).T.sort_index()
    result.index.name = "d"
    result["Stationary"] = result["Stationary"].astype(bool)

    return result, float(minimum_d)
