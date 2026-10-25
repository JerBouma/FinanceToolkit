"""Regime Switching and Structural Break Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Returns are scaled up before the Markov model is fitted, as the optimizer handles a
# daily variance of 1e-4 poorly, and scaled back afterwards.
SCALE = 100.0
# The seed of the random search for starting values, so a fit is reproducible.
SEARCH_SEED = 0


def get_markov_regime_switching(
    series: pd.Series, regimes: int = 2, switching_variance: bool = True
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Fit a Markov regime switching model (Hamilton, 1989) in which the mean (and the
    variance) of a series depend on a regime that is not observed and switches with fixed
    probabilities. With returns, two regimes typically separate calm markets (a higher
    mean and a low volatility) from turbulent ones (a lower mean and a high volatility).
    The regimes are ordered from the lowest to the highest volatility (or mean, without a
    switching variance).

    The random search for starting values runs from a fixed seed, so the same series always
    gives the same fit; the caller's random state is restored afterwards.

    For more information, see the following paper:

    - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time
      Series and the Business Cycle." Econometrica, 57(2), 357-384.

    Args:
        series (pd.Series): The series, e.g. returns.
        regimes (int, optional): The number of regimes. Defaults to 2.
        switching_variance (bool, optional): Whether the variance switches with the regime
            as well as the mean. Defaults to True.

    Returns:
        tuple[dict[str, float], pd.DataFrame]: Per regime its Mean, Volatility, Probability
        of Staying, Expected Duration (in periods) and Share of the periods it most likely
        held, then the Current Regime and its probability; and the smoothed probability of
        every regime per date.
    """
    values = series.dropna()
    if len(values) < 30 * regimes:  # noqa: PLR2004
        raise ValueError(
            f"At least {30 * regimes} observations are needed for {regimes} regimes."
        )

    model = MarkovRegression(
        values.to_numpy(dtype=float) * SCALE,
        k_regimes=regimes,
        trend="c",
        switching_variance=switching_variance,
    )
    # Several starting points, as the likelihood of a mixture has local optima. The
    # statsmodels search draws from NumPy's global random state and takes no seed, so it
    # runs from a fixed seed (the same series gives the same fit) and the caller's state
    # is restored afterwards.
    fits = []
    random_state = np.random.get_state()
    np.random.seed(SEARCH_SEED)
    try:
        for search in (0, 20):
            # A starting point that does not converge is one of several, so its warning
            # is not passed on; the best fit of all of them is used.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fits.append(model.fit(search_reps=search, disp=False))
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    continue
    finally:
        np.random.set_state(random_state)
    if not fits:
        raise ValueError("The Markov regime switching model did not converge.")
    fit = max(fits, key=lambda result: result.llf)

    names = fit.model.param_names
    parameters = dict(zip(names, np.asarray(fit.params), strict=True))
    means = np.array([parameters[f"const[{regime}]"] for regime in range(regimes)])
    variances = (
        np.array([parameters[f"sigma2[{regime}]"] for regime in range(regimes)])
        if switching_variance
        else np.full(regimes, parameters["sigma2"])
    )
    transition = np.asarray(fit.regime_transition)[:, :, 0]
    probabilities = np.asarray(fit.smoothed_marginal_probabilities)

    order = np.argsort(variances if switching_variance else means)
    most_likely = probabilities[:, order].argmax(axis=1)

    summary: dict[str, float] = {}
    for number, regime in enumerate(order, start=1):
        staying = float(transition[regime, regime])
        summary[f"Regime {number} Mean"] = means[regime] / SCALE
        summary[f"Regime {number} Volatility"] = np.sqrt(variances[regime]) / SCALE
        summary[f"Regime {number} Probability of Staying"] = staying
        summary[f"Regime {number} Expected Duration"] = (
            1 / (1 - staying) if staying < 1 else np.inf
        )
        summary[f"Regime {number} Share"] = float(np.mean(most_likely == number - 1))
    summary["Current Regime"] = float(most_likely[-1] + 1)
    summary["Current Regime Probability"] = float(
        probabilities[-1, order[most_likely[-1]]]
    )

    smoothed = pd.DataFrame(
        probabilities[:, order],
        index=values.index,
        columns=[f"Regime {number}" for number in range(1, regimes + 1)],
    )
    return summary, smoothed


def _segment_costs(
    cumulative: np.ndarray,
    squares: np.ndarray,
    start: int | np.ndarray,
    ends: int | np.ndarray,
) -> np.ndarray:
    """The sum of squared deviations from the mean of the segments start..end (exclusive)."""
    length = ends - start
    total = cumulative[ends] - cumulative[start]
    return squares[ends] - squares[start] - total**2 / length


def get_structural_breaks(
    series: pd.Series,
    max_breaks: int = 5,
    minimum_segment: float = 0.15,
    breaks: int | None = None,
) -> pd.DataFrame:
    """
    Find the dates at which the mean of a series shifts (Bai and Perron, 1998, 2003): for
    every number of breaks the break dates that minimize the sum of squared residuals of a
    mean per segment are found exactly by dynamic programming, and the number of breaks is
    chosen by the Bayesian information criterion (unless given). Every segment covers at
    least a share of the observations (the trimming h = [εT], the integer part as in Bai
    and Perron), so a break is not fitted to a few outliers.

    The BIC suits series without serial correlation; with serially correlated series,
    such as the level of inflation, it tends to choose too many breaks (Bai and Perron,
    2003), so the segments for a few fixed numbers of breaks are worth comparing then.

    For more information, see the following papers:

    - Bai, J. and Perron, P. (1998). "Estimating and Testing Linear Models with Multiple
      Structural Changes." Econometrica, 66(1), 47-78.
    - Bai, J. and Perron, P. (2003). "Computation and Analysis of Multiple Structural Change
      Models." Journal of Applied Econometrics, 18(1), 1-22.

    Args:
        series (pd.Series): The series.
        max_breaks (int, optional): The most breaks considered. Defaults to 5.
        minimum_segment (float, optional): The smallest segment as a share of the
            observations. Defaults to 0.15.
        breaks (int | None, optional): The number of breaks to find, instead of choosing it
            by the BIC. Defaults to None.

    Returns:
        pd.DataFrame: Per segment its Start, End, Observations, Mean and Standard Deviation,
        and the BIC of the chosen number of breaks.
    """
    values = series.dropna()
    data = values.to_numpy(dtype=float)
    n = len(data)
    # The trimming h = [εT] of Bai and Perron: the integer part, with at least two
    # observations so every segment has a standard deviation. Rounding to nine decimals
    # first keeps 0.29 * 100 (28.999999999999996 in floating point) at 29.
    h = max(int(round(minimum_segment * n, 9)), 2)
    most = min(max_breaks if breaks is None else breaks, n // h - 1)
    if most < 0 or (breaks is not None and breaks > most):
        raise ValueError(
            f"{n} observations with segments of at least {h} leave room for at most "
            f"{max(n // h - 1, 0)} breaks."
        )

    cumulative = np.concatenate([[0.0], np.cumsum(data)])
    squares = np.concatenate([[0.0], np.cumsum(data**2)])

    # cost[m][j]: the smallest sum of squared residuals of the first j observations with m
    # breaks; last[m][j]: where its last segment starts.
    cost = np.full((most + 1, n + 1), np.inf)
    last = np.zeros((most + 1, n + 1), dtype=int)
    ends = np.arange(n + 1)
    valid = ends >= h
    cost[0, valid] = _segment_costs(cumulative, squares, 0, ends[valid])

    for m in range(1, most + 1):
        for j in range((m + 1) * h, n + 1):
            starts = np.arange(m * h, j - h + 1)
            candidates = cost[m - 1, starts] + _segment_costs(
                cumulative, squares, starts, j
            )
            best = int(np.argmin(candidates))
            cost[m, j] = candidates[best]
            last[m, j] = starts[best]

    # Every segment has a mean and the breaks are estimated too; the variance is common.
    criteria = {
        m: n * np.log(cost[m, n] / n) + (2 * m + 1) * np.log(n)
        for m in range(most + 1)
        if np.isfinite(cost[m, n]) and cost[m, n] > 0
    }
    if not criteria:
        raise ValueError("The series is constant, so it has no breaks to find.")
    chosen = breaks if breaks is not None else min(criteria, key=criteria.get)

    boundaries = [n]
    for m in range(chosen, 0, -1):
        boundaries.append(last[m, boundaries[-1]])
    boundaries.append(0)
    boundaries = boundaries[::-1]

    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment = data[start:end]
        rows.append(
            {
                "Start": str(values.index[start]),
                "End": str(values.index[end - 1]),
                "Observations": end - start,
                "Mean": float(segment.mean()),
                "Standard Deviation": float(segment.std(ddof=1)),
            }
        )

    result = pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="Segment"))
    result["BIC"] = criteria.get(chosen, np.nan)
    return result
