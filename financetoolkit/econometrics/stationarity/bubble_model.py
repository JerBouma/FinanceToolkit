"""Bubble Model"""

__docformat__ = "google"

from functools import lru_cache

import numpy as np
import pandas as pd

from financetoolkit.utilities.error_model import check_option

# The critical value levels every test reports, as confidence levels (right-tailed tests).
CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)

# The number of window statistics computed at once. Arrays of 50,000 windows (400 KB) stay in
# the CPU cache: for ten years of daily prices a block of 10,000 or 250,000 windows takes 20%
# to 50% longer, and the memory stays bounded however long the series is.
WINDOWS_PER_BLOCK = 50_000


def get_minimum_window(observations: int) -> int:
    """
    Calculate the smallest window of the recursive ADF regressions with the rule of Phillips,
    Shi and Yu (2015).

    The rule is r₀ = 0.01 + 1.8 / √T of the T observations, floor(r₀·T) observations.

    Args:
        observations (int): The number of observations T of the series.

    Returns:
        int: The minimum window in observations.

    Raises:
        ValueError: If the series has no observations.
    """
    if observations < 1:
        raise ValueError(
            "The series has no observations in the date range to test for a bubble."
        )
    # Rounded first, so 0.19 of 100 observations is 19 and not 18 by floating point.
    return int(np.floor(round((0.01 + 1.8 / np.sqrt(observations)) * observations, 9)))


def _prefix_sums(
    levels: np.ndarray, lags: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The cumulative cross products of the ADF regression rows, so the sums over any window are
    a difference of two of them.

    Row t (t = p + 1, ..., T - 1, p = lags) regresses Δyₜ on [1, Δyₜ₋₁, ..., Δyₜ₋ₚ, yₜ₋₁], the lagged
    level last; entry j of each array holds the sum over the first j rows, so the window of
    levels [a, b] (its rows a + lags + 1 to b) is entry b - lags minus entry a.
    """
    # The ADF regression with a constant does not change when the levels shift, and centring
    # them keeps the differences of the prefix sums accurate for prices far from zero.
    levels = levels - levels.mean()
    changes = np.diff(levels)
    rows = len(levels) - lags - 1
    regressors = np.column_stack(
        [np.ones(rows)]
        + [changes[lags - lag : lags - lag + rows] for lag in range(1, lags + 1)]
        + [levels[lags : lags + rows]]
    )
    target = changes[lags:]

    def cumulative(values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0))
        )

    return (
        cumulative(regressors[:, :, None] * regressors[:, None, :]),
        cumulative(regressors * target[:, None]),
        cumulative(target**2),
    )


class _Workspace:
    """
    The planes of `_window_statistics`, kept from one block of windows to the next. A new
    array per operation costs more than the operation: arrays of this size come fresh from
    the operating system, and filling their pages took more than half of the time.
    """

    def __init__(self) -> None:
        self.buffers: dict[tuple[str, int], np.ndarray] = {}

    def plane(
        self, name: str, shape: tuple[int, int], dtype: type = float
    ) -> np.ndarray:
        """A (shape) view of the buffer `name`, which grows when a block needs more room."""
        size = shape[0] * shape[1]
        key = (name, np.dtype(dtype).num)
        if key not in self.buffers or self.buffers[key].size < size:
            # Room for a block of the budget at once, so blocks of slightly different sizes
            # do not each allocate again.
            self.buffers[key] = np.empty(max(size, WINDOWS_PER_BLOCK), dtype=dtype)
        return self.buffers[key][:size].reshape(shape)


def _window_statistics(
    sums: tuple[np.ndarray, np.ndarray, np.ndarray],
    lags: int,
    ends: np.ndarray,
    starts: np.ndarray,
    minimum_rows: int,
    workspace: _Workspace | None = None,
) -> np.ndarray:
    """
    The ADF statistic t = β̂ / SE(β̂) of the lagged level for every window [start, end] of the
    levels, as an (ends × starts) array, minus infinity where the window has no variation or
    fewer than `minimum_rows` regression rows (at least the regressors plus one), so that the
    largest statistic per end is a plain maximum.

    With the Cholesky factor X'X = LL' and Lz = X'y, the coefficient of the last regressor (the
    lagged level) is β̂ = z_k / L_kk and its variance s²·(X'X)⁻¹_kk = s² / L_kk², so
    t = z_k / s with s² = (y'y - z'z) / (m - k): a forward substitution per window, done for
    all windows at once over the (few) regressors. Every step writes into a plane of the
    `workspace`, so the returned array is only valid until the next call with it.
    """
    workspace = workspace if workspace is not None else _Workspace()
    cross, joint, target = sums
    high, low = ends - lags, starts
    shape = (len(ends), len(starts))
    size = lags + 2

    count = workspace.plane("count", shape)
    # The rows are whole numbers, so subtracting them as floats is exact.
    np.subtract(high[:, None], low[None, :], out=count, dtype=float)
    scratch = workspace.plane("scratch", shape)

    def window_sum(prefix: np.ndarray, name: str) -> np.ndarray:
        return np.subtract(
            prefix[high][:, None],
            prefix[low][None, :],
            out=workspace.plane(name, shape),
        )

    def subtract_product(
        plane: np.ndarray, first: np.ndarray, second: np.ndarray
    ) -> None:
        np.multiply(first, second, out=scratch)
        plane -= scratch

    # The factor and the solution as one (ends × starts) plane per element, so every step is
    # a contiguous array operation over all windows.
    factor: dict[tuple[int, int], np.ndarray] = {}
    solution: list[np.ndarray] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for column in range(size):
            # The first regressor is the constant, whose sum of squares over a window is its
            # number of rows (the prefix sums of ones are exact).
            if column == 0:
                diagonal = workspace.plane("factor 0 0", shape)
                np.copyto(diagonal, count)
            else:
                diagonal = window_sum(
                    cross[:, column, column], f"factor {column} {column}"
                )
            for inner in range(column):
                subtract_product(diagonal, factor[column, inner], factor[column, inner])
            factor[column, column] = np.sqrt(diagonal, out=diagonal)
            for row in range(column + 1, size):
                element = window_sum(cross[:, row, column], f"factor {row} {column}")
                for inner in range(column):
                    subtract_product(element, factor[row, inner], factor[column, inner])
                element /= factor[column, column]
                factor[row, column] = element
            value = window_sum(joint[:, column], f"solution {column}")
            for inner in range(column):
                subtract_product(value, factor[column, inner], solution[inner])
            value /= factor[column, column]
            solution.append(value)

        residual = window_sum(target, "residual")
        for value in solution:
            subtract_product(residual, value, value)
        np.subtract(count, size, out=scratch)
        residual /= scratch
        statistic = np.divide(
            solution[-1], np.sqrt(residual, out=residual), out=residual
        )

    invalid = workspace.plane("invalid", shape, bool)
    finite = workspace.plane("finite", shape, bool)
    np.less(count, max(minimum_rows, size + 1), out=invalid)
    np.isfinite(statistic, out=finite)
    np.logical_not(finite, out=finite)
    np.logical_or(invalid, finite, out=invalid)
    np.copyto(statistic, -np.inf, where=invalid)

    return statistic


def get_recursive_adf_statistics(
    values: np.ndarray, lags: int = 0, minimum_window: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the forward and the backward recursive ADF statistics of a series, the sequences
    behind the SADF test of Phillips, Wu and Yu (2011) and the GSADF test of Phillips, Shi and
    Yu (2015).

    The formula is as follows, for the ADF regression on the window of observations r₁ to r₂:

    - Δyₜ = α + β·yₜ₋₁ + Σ(i=1..lags) γᵢ·Δyₜ₋ᵢ + εₜ, ADF(r₁, r₂) = β̂ / SE(β̂)
    - Forward ADF(r₂) = ADF(0, r₂), the window from the first observation
    - BSADF(r₂) = max over r₁ from 0 to r₂ - w + 1 of ADF(r₁, r₂), every window of at least w
      observations that ends at r₂

    Referenced against:
    - statsmodels, adfuller: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
    - RiskLabAI, get_bsadf_sequence: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L434

    Args:
        values (np.ndarray): The levels of the series (such as log prices), without NaN.
        lags (int, optional): The number of lagged differences in the regression. Defaults to 0.
        minimum_window (int | None, optional): The smallest window w in observations. Defaults to
            None, the rule of `get_minimum_window`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The forward ADF and the BSADF statistic per observation,
        NaN before the first full window.

    Raises:
        ValueError: If the window cannot hold the regression or exceeds the series.
    """
    observations = len(values)
    window = (
        get_minimum_window(observations)
        if minimum_window is None
        else int(minimum_window)
    )

    if window < 2 * lags + 5:
        raise ValueError(
            f"The minimum window of {window} observations cannot hold an ADF regression with {lags} "
            f"lags; it needs at least {2 * lags + 5}."
        )
    if window > observations:
        raise ValueError(
            f"The series has {observations} observations, fewer than the minimum window of {window}."
        )

    sums = _prefix_sums(np.asarray(values, dtype=float), lags)
    workspace = _Workspace()
    forward = np.full(observations, np.nan)
    backward = np.full(observations, np.nan)

    end = window - 1
    while end < observations:
        # A block of ends shares the starts up to the last end's, about block·(end + block)
        # windows: the largest block within the budget solves block² + end·block = budget.
        block = max(
            1,
            min(
                observations - end,
                int((np.sqrt(end**2 + 4 * WINDOWS_PER_BLOCK) - end) / 2),
            ),
        )
        ends = np.arange(end, end + block)
        starts = np.arange(ends[-1] - window + 2)
        # A window from start to end holds end - lags - start regression rows, so the
        # windows of at least w observations are those with at least w - 1 - lags rows.
        statistic = _window_statistics(
            sums, lags, ends, starts, window - 1 - lags, workspace
        )
        largest = statistic.max(axis=1)
        forward[ends] = np.where(statistic[:, 0] == -np.inf, np.nan, statistic[:, 0])
        backward[ends] = np.where(largest == -np.inf, np.nan, largest)
        end += block

    return forward, backward


@lru_cache(maxsize=8)
def simulate_bubble_critical_values(
    observations: int,
    minimum_window: int,
    lags: int = 0,
    simulations: int = 2000,
    seed: int = 0,
    levels: tuple[float, ...] = CONFIDENCE_LEVELS,
) -> dict[str, np.ndarray]:
    """
    Simulate the critical values of the SADF and GSADF tests and of the forward ADF and BSADF
    sequences under the null hypothesis of Phillips, Shi and Yu (2015), a random walk with an
    asymptotically negligible drift:

    - yₜ = yₜ₋₁ + T⁻¹ + εₜ, εₜ ~ N(0, 1), y₀ = 0
    - SADF = max forward ADF, GSADF = max BSADF; the critical value at confidence c is the c
      quantile over the simulations, per observation for the sequences

    The statistics do not depend on the level or the scale of the series, so one simulation
    serves every series of the same length, window and lags; the results are kept for the
    session.

    Referenced against:
    - RiskLabAI, simulate_psy_critical_values: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L607

    Args:
        observations (int): The number of observations T.
        minimum_window (int): The smallest window in observations.
        lags (int, optional): The number of lagged differences. Defaults to 0.
        simulations (int, optional): The number of simulated paths. Defaults to 2,000.
        seed (int, optional): The seed of the random numbers. Defaults to 0.
        levels (tuple[float, ...], optional): The confidence levels. Defaults to (0.90, 0.95, 0.99).

    Returns:
        dict[str, np.ndarray]: "SADF" and "GSADF" (sorted simulated statistics), "SADF Critical
        Values" and "GSADF Critical Values" (one per level), and "ADF Sequence" and "BSADF
        Sequence" (levels × observations).
    """
    generator = np.random.default_rng(seed)
    forward = np.empty((simulations, observations))
    backward = np.empty((simulations, observations))

    for path in range(simulations):
        levels_path = np.cumsum(
            1 / observations + generator.standard_normal(observations)
        )
        forward[path], backward[path] = get_recursive_adf_statistics(
            levels_path, lags, minimum_window
        )

    with np.errstate(invalid="ignore"):
        sadf = np.sort(np.nanmax(forward, axis=1))
        gsadf = np.sort(np.nanmax(backward, axis=1))

    return {
        "SADF": sadf,
        "GSADF": gsadf,
        "SADF Critical Values": np.quantile(sadf, levels),
        "GSADF Critical Values": np.quantile(gsadf, levels),
        "ADF Sequence": np.quantile(forward, levels, axis=0),
        "BSADF Sequence": np.quantile(backward, levels, axis=0),
    }


def get_bubble_test(
    series: pd.Series,
    method: str = "gsadf",
    lags: int = 0,
    minimum_window: int | None = None,
    simulations: int = 2000,
    seed: int = 0,
) -> pd.Series:
    """
    Calculate the SADF test of Phillips, Wu and Yu (2011) or the GSADF test of Phillips, Shi and
    Yu (2015): whether a series shows explosive behaviour (a bubble), with critical values
    simulated under a random walk.

    The formula is as follows, with the forward ADF and BSADF sequences of
    `get_recursive_adf_statistics`:

    - SADF = max over r₂ from w - 1 to T - 1 of ADF(0, r₂)
    - GSADF = max over r₂ of BSADF(r₂), every window of at least w observations
    - P-value = the share of the simulated statistics at least as large

    Referenced against:
    - RiskLabAI, get_gsadf_statistic: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L486
    - RiskLabAI, simulate_psy_critical_values: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L607

    Args:
        series (pd.Series): The levels of the series, such as log prices. Missing values are
            left out.
        method (str, optional): The test. Options are: "gsadf" (GSADF test), "sadf" (SADF
            test). Defaults to "gsadf".
        lags (int, optional): The number of lagged differences in the ADF regressions. Defaults
            to 0.
        minimum_window (int | None, optional): The smallest window in observations. Defaults to
            None, floor((0.01 + 1.8 / √T)·T).
        simulations (int, optional): The number of simulated paths for the critical values.
            Defaults to 2,000, as in Phillips, Shi and Yu (2015): the simulated critical values
            serve every series of the same length, so with 500 paths the size of the test at 5%
            ranges from about 3.5% to 7% depending on the seed.
        seed (int, optional): The seed of the simulation. Defaults to 0.

    Returns:
        pd.Series: The statistic, its simulated p-value, the 90%, 95% and 99% critical values,
        the observations, minimum window, lags and replications (the number of simulations), and whether the series is
        explosive at the 5% level.

    Raises:
        ValueError: If the method is unknown or the series too short for the window.
    """
    check_option("method", method, ("gsadf", "sadf"))

    values = series.dropna().to_numpy(dtype=float)
    window = (
        get_minimum_window(len(values))
        if minimum_window is None
        else int(minimum_window)
    )
    forward, backward = get_recursive_adf_statistics(values, lags, window)
    statistic = float(np.nanmax(forward if method == "sadf" else backward))
    simulated = simulate_bubble_critical_values(
        len(values), window, lags, simulations, seed, CONFIDENCE_LEVELS
    )
    name = method.upper()
    critical_values = simulated[f"{name} Critical Values"]
    # A series without variation (or without values) has no statistic, and no p-value.
    p_value = (
        float(np.mean(simulated[name] >= statistic))
        if np.isfinite(statistic)
        else np.nan
    )

    return pd.Series(
        {
            f"{name} Statistic": statistic,
            "P-Value": p_value,
            "Critical Value 10%": critical_values[0],
            "Critical Value 5%": critical_values[1],
            "Critical Value 1%": critical_values[2],
            "Observations": len(values),
            "Minimum Window": window,
            "Lags": lags,
            "Replications": simulations,
            "Explosive (5%)": bool(statistic > critical_values[1]),
        }
    )


def get_bubble_episodes(
    series: pd.Series,
    method: str = "gsadf",
    lags: int = 0,
    minimum_window: int | None = None,
    confidence: float = 0.95,
    minimum_duration: int | None = None,
    simulations: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Date-stamp the explosive episodes (bubbles) of a series with the BSADF sequence of Phillips,
    Shi and Yu (2015), or the forward ADF sequence of Phillips, Wu and Yu (2011), against their
    simulated critical value per date.

    The formula is as follows:

    - Bubble at r₂ when BSADF(r₂) > cv(r₂), the confidence quantile of the simulated BSADF at
      r₂ (GSADF), or when ADF(0, r₂) exceeds its own (SADF)
    - An episode starts at the first date above the critical value and ends at the last date
      before it falls back; episodes shorter than the minimum duration, by default
      round(log(T)) observations (δ = 1 in the δ·log(T) rule of PSY), are left out

    Referenced against:
    - RiskLabAI, get_bsadf_sequence: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L434
    - RiskLabAI, get_bubble_episodes: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/structural_breaks.py#L544

    Args:
        series (pd.Series): The levels of the series, such as log prices.
        method (str, optional): The sequence. Options are: "gsadf" (BSADF sequence), "sadf"
            (forward ADF sequence). Defaults to "gsadf".
        lags (int, optional): The number of lagged differences. Defaults to 0.
        minimum_window (int | None, optional): The smallest window. Defaults to None, the PSY rule.
        confidence (float, optional): The confidence of the critical values. Defaults to 0.95.
        minimum_duration (int | None, optional): The fewest observations of an episode. Defaults
            to None, round(log(T)).
        simulations (int, optional): The number of simulated paths. Defaults to 2,000.
        seed (int, optional): The seed of the simulation. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The episodes (Start, End, Duration in observations,
        Peak Date and Peak Statistic, the largest statistic of the episode) and, per date, the
        statistic, its critical value and whether it is above it.

    Raises:
        ValueError: If the method is unknown, the confidence not between 0 and 1 or the series
        too short for the window.
    """
    check_option("method", method, ("gsadf", "sadf"))
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1, e.g. 0.95.")

    data = series.dropna()
    values = data.to_numpy(dtype=float)
    window = (
        get_minimum_window(len(values))
        if minimum_window is None
        else int(minimum_window)
    )
    forward, backward = get_recursive_adf_statistics(values, lags, window)
    levels = tuple(sorted({*CONFIDENCE_LEVELS, confidence}))
    simulated = simulate_bubble_critical_values(
        len(values), window, lags, simulations, seed, levels
    )

    statistic = backward if method == "gsadf" else forward
    critical = simulated["BSADF Sequence" if method == "gsadf" else "ADF Sequence"][
        levels.index(confidence)
    ]
    above = statistic > critical
    name = "BSADF" if method == "gsadf" else "ADF"

    sequence = pd.DataFrame(
        {name: statistic, "Critical Value": critical, "Bubble": above}, index=data.index
    )

    duration = (
        max(1, round(np.log(len(values))))
        if minimum_duration is None
        else int(minimum_duration)
    )
    # Runs of consecutive dates above the critical value: a start where it turns on, an end
    # where it turns off.
    edges = np.diff(np.concatenate(([0], above.astype(int), [0])))
    episodes = []
    for start, stop in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    ):
        if stop - start >= duration:
            peak = start + int(np.argmax(statistic[start:stop]))
            episodes.append(
                {
                    "Start": data.index[start],
                    "End": data.index[stop - 1],
                    "Duration": stop - start,
                    "Peak Date": data.index[peak],
                    "Peak Statistic": statistic[peak],
                }
            )

    return (
        pd.DataFrame(
            episodes,
            columns=["Start", "End", "Duration", "Peak Date", "Peak Statistic"],
        ),
        sequence,
    )


def get_chow_dickey_fuller_statistics(
    series: pd.Series, trim: float = 0.15
) -> pd.Series:
    """
    Calculate the Chow-type Dickey-Fuller statistic for a switch from a random walk to an
    explosive process after every candidate break date (López de Prado, 2018, section 17.3.3;
    Homm and Breitung, 2012).

    The formula is as follows, with yₜ the series minus its first value, so the test does not
    depend on the level of the series:

    - Δyₜ = δ·yₜ₋₁·Dₜ(τ) + εₜ, Dₜ(τ) = 1 after the break date τ and 0 up to it
    - DFC(τ) = δ̂ / SE(δ̂), with SE from the residuals of the regression over all T - 1 changes

    Args:
        series (pd.Series): The levels of the series, such as log prices.
        trim (float, optional): The share of observations left out at either end of the break
            dates. Defaults to 0.15.

    Returns:
        pd.Series: DFC(τ) on the candidate break dates, NaN on the other dates.

    Raises:
        ValueError: If trim is not in [0, 0.5).
    """
    if not 0 <= trim < 0.5:  # noqa: PLR2004
        raise ValueError("trim must be in [0, 0.5), e.g. 0.15.")

    data = series.dropna()
    statistics = _chow_dickey_fuller(data.to_numpy(dtype=float)[None, :], trim)[0]

    return pd.Series(statistics, index=data.index, name=series.name).reindex(
        series.index
    )


def _chow_dickey_fuller(paths: np.ndarray, trim: float) -> np.ndarray:
    """
    DFC(τ) for every path (row) and every break position τ, with suffix sums: the regressor is
    zero up to τ, so only the changes after τ enter δ̂.
    """
    observations = paths.shape[1]
    levels = paths - paths[:, :1]
    lagged, changes = levels[:, :-1], np.diff(levels, axis=1)
    # Suffix sums over the rows after each break position.
    sxx = np.cumsum((lagged**2)[:, ::-1], axis=1)[:, ::-1]
    sxy = np.cumsum((lagged * changes)[:, ::-1], axis=1)[:, ::-1]
    syy = np.sum(changes**2, axis=1, keepdims=True)

    statistics = np.full(paths.shape, np.nan)
    first = int(np.floor(trim * observations))
    last = observations - 1 - first
    # Break at position τ: rows from τ + 1 on, which are rows τ onwards of the lagged arrays.
    positions = np.arange(first, min(last, observations - 2) + 1)
    if positions.size:
        delta = sxy[:, positions] / sxx[:, positions]
        variance = (
            (syy - delta * sxy[:, positions]) / (observations - 2) / sxx[:, positions]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics[:, positions] = delta / np.sqrt(variance)
    statistics[~np.isfinite(statistics)] = np.nan

    return statistics


@lru_cache(maxsize=8)
def simulate_chow_dickey_fuller_critical_values(
    observations: int, trim: float = 0.15, simulations: int = 1000, seed: int = 0
) -> np.ndarray:
    """
    Simulate the maximum Chow-type Dickey-Fuller statistic (SDFC) under a Gaussian random walk
    started at zero, sorted, so its quantiles are the critical values.

    Args:
        observations (int): The number of observations T.
        trim (float, optional): The share of observations left out at either end. Defaults to 0.15.
        simulations (int, optional): The number of simulated paths. Defaults to 1000.
        seed (int, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        np.ndarray: The simulated SDFC statistics, sorted.
    """
    generator = np.random.default_rng(seed)
    maxima = np.empty(simulations)
    # Blocks of paths keep the arrays to a few megabytes.
    block = max(1, 500_000 // observations)
    for first in range(0, simulations, block):
        count = min(block, simulations - first)
        paths = np.cumsum(generator.standard_normal((count, observations)), axis=1)
        with np.errstate(invalid="ignore"):
            maxima[first : first + count] = np.nanmax(
                _chow_dickey_fuller(paths, trim), axis=1
            )

    return np.sort(maxima)


def get_chow_dickey_fuller_test(
    series: pd.Series, trim: float = 0.15, simulations: int = 1000, seed: int = 0
) -> pd.Series:
    """
    Calculate the supremum Chow-type Dickey-Fuller test (SDFC) for a switch from a random walk to
    an explosive process at an unknown date (López de Prado, 2018, section 17.3.3; Homm and
    Breitung, 2012; the supremum over break dates as in Andrews, 1993), with critical values
    simulated under a random walk.

    The formula is as follows, with DFC(τ) of `get_chow_dickey_fuller_statistics`:

    - SDFC = max over τ from floor(trim·T) to T - 1 - floor(trim·T) of DFC(τ)
    - The break date is the τ of the maximum; the p-value is the share of the simulated SDFC
      statistics at least as large

    Args:
        series (pd.Series): The levels of the series, such as log prices.
        trim (float, optional): The share of observations left out at either end. Defaults to 0.15.
        simulations (int, optional): The number of simulated paths. Defaults to 1000.
        seed (int, optional): The seed of the simulation. Defaults to 0.

    Returns:
        pd.Series: The SDFC statistic, the break date, the simulated p-value, the 90%, 95% and
        99% critical values, the observations and whether the series turns explosive at the 5%
        level.
    """
    data = series.dropna()
    first = int(np.floor(trim * len(data)))

    # The regression over the T - 1 changes needs a degree of freedom besides δ.
    too_short = len(data) <= 2  # noqa: PLR2004
    if too_short or min(len(data) - 1 - first, len(data) - 2) < first:
        raise ValueError("The series is too short for the trimmed break dates.")

    statistics = get_chow_dickey_fuller_statistics(series, trim).dropna()

    simulated = simulate_chow_dickey_fuller_critical_values(
        len(data), trim, simulations, seed
    )
    critical_values = np.quantile(simulated, CONFIDENCE_LEVELS)
    # A series without variation has no statistic on any break date, and no p-value.
    statistic = float(statistics.max()) if not statistics.empty else np.nan

    return pd.Series(
        {
            "SDFC Statistic": statistic,
            "Break Date": statistics.idxmax() if not statistics.empty else np.nan,
            "P-Value": (
                float(np.mean(simulated >= statistic))
                if np.isfinite(statistic)
                else np.nan
            ),
            "Critical Value 10%": critical_values[0],
            "Critical Value 5%": critical_values[1],
            "Critical Value 1%": critical_values[2],
            "Observations": len(data),
            "Replications": simulations,
            "Explosive (5%)": bool(statistic > critical_values[1]),
        }
    )
