"""Regime Switching and Structural Break Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd

from financetoolkit.econometrics.helpers import import_optional
from financetoolkit.utilities.error_model import check_option

# Returns are scaled up before the Markov model is fitted, as the optimizer handles a
# daily variance of 1e-4 poorly, and scaled back afterwards.
SCALE = 100.0
# The seed of the random search for starting values, so a fit is reproducible.
SEARCH_SEED = 0


def get_markov_regime_switching(
    series: pd.Series, regimes: int = 2, switching_variance: bool = True
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Fit a Markov regime switching model (Hamilton, 1989): the hidden regimes, such as calm and
    turbulent markets, that set the mean (and the variance) of a series.

    The regime is not observed and switches with fixed probabilities, and the model is
    fitted with statsmodels' MarkovRegression. With returns, two regimes typically separate calm markets (a higher
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

    markov_regression = import_optional(
        "statsmodels.tsa.regime_switching.markov_regression",
        "The Markov regime switching model",
    )
    model = markov_regression.MarkovRegression(
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
    Find the structural breaks of a series (Bai and Perron, 1998, 2003): the dates at which
    its mean shifts.

    For every number of breaks the break dates that minimize the sum of squared residuals of a
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


# The parameters a segment of each PELT cost fits: its mean, its variance or both.
CHANGE_POINT_COST_PARAMETERS = {"mean": 1, "variance": 1, "mean_variance": 2}

# The variance of a segment is floored at this share of the variance of the whole series, so
# a segment of equal values (a variance of 0) has a finite cost, as ruptures adds 1e-6.
VARIANCE_FLOOR = 1e-6

# The median absolute deviation of a normal distribution is 0.6745 standard deviations.
NORMAL_MEDIAN_ABSOLUTE_DEVIATION = 0.6744897501960817


def get_change_points(
    series: pd.Series,
    cost: str = "mean_variance",
    penalty: float | None = None,
    minimum_segment: int = 5,
) -> pd.DataFrame:
    """
    Find the change points of a series with the Pruned Exact Linear Time (PELT) method of
    Killick, Fearnhead and Eckley (2012): the dates at which its mean or volatility changes.

    PELT finds the segmentation that minimizes the cost of every segment, minus twice its
    Gaussian log-likelihood, plus a penalty per change point, exactly. Unlike the Bai-Perron
    breaks the number of change points has no upper bound, and with the variance or the mean
    and variance cost a change in the volatility counts as well.

    The formula is as follows, with yₜ the series, n observations and a segment s of nₛ
    observations:

    - Mean cost: C(s) = Σ(t in s) (yₜ - ȳₛ)² / σ̂², with σ̂ = MAD(Δy) / (0.6745·√2) the robust
      standard deviation of the series from its changes, which a shift in the mean hardly moves
    - Variance cost: C(s) = nₛ·ln(Σ(t in s) (yₜ - ȳ)² / nₛ), with ȳ the mean of the whole series
    - Mean and variance cost: C(s) = nₛ·ln(Σ(t in s) (yₜ - ȳₛ)² / nₛ)
    - F(t) = min over τ of F(τ) + C(y_(τ+1..t)) + β, with F(0) = -β, every segment at least
      `minimum_segment` = m observations long (5 by default); once F(τ) + C(y_(τ+1..t)) > F(t), τ is
      dropped from the candidates from t + m on, which never drops the optimum (K = 0 for these costs;
      the segment after t needs m observations, so with m = 1 this is the pruning of PELT itself)
    - β = (p + 1)·ln(n), the BIC with p the parameters per segment (1 for the mean and the variance
      cost, 2 for the mean and variance cost) plus the change point itself, unless `penalty` is given

    Constant terms of the log-likelihood are left out, as they add the same n·(1 + ln 2π) to
    every segmentation.

    For more information, see the following papers:

    - Killick, R., Fearnhead, P. and Eckley, I.A. (2012). "Optimal Detection of Changepoints
    With a Linear Computational Cost." Journal of the American Statistical Association,
    107(500), 1590-1598.
    - Jackson, B. et al. (2005). "An Algorithm for Optimal Partitioning of Data on an
    Interval." IEEE Signal Processing Letters, 12(2), 105-108.

    Referenced against:
    - ruptures, Pelt: https://github.com/deepcharles/ruptures/blob/v1.1.10/src/ruptures/detection/pelt.py#L11
    - changepoint (R), penalty_decision: https://github.com/cran/changepoint/blob/2.3/R/penalty_decision.R#L1

    Shares similar logic:
    - RiskLabAI, pelt_change_points: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/structural_breaks/pelt_change_points.py#L34

    Args:
        series (pd.Series): The series.
        cost (str, optional): The segment cost. Options are: "mean" (a change in the mean),
            "variance" (a change in the variance), "mean_variance" (a change in either).
            Defaults to "mean_variance".
        penalty (float | None, optional): The penalty β per change point. Defaults to None,
            the BIC (p + 1)·ln(n).
        minimum_segment (int, optional): The fewest observations of a segment. A segment of
            two nearly equal values has a variance close to zero, which the Gaussian
            likelihood rewards without bound, so the variance costs need a few more.
            Defaults to 5.

    Returns:
        pd.DataFrame: Per segment the Start, End, Observations, Mean and Standard Deviation
        (ddof=1), with the Penalty.

    Notes:
        ruptures' Pelt with the L2 cost on the series divided by σ̂ and the normal cost finds
        the same change points when it checks every observation, and the penalty follows the
        convention of Killick's changepoint package. Unlike ruptures, which adds 1e-6 to every
        segment's variance, a variance here is floored at 1e-6 times that of the whole series,
        which keeps the cost independent of the units.
    """
    check_option("cost", cost, CHANGE_POINT_COST_PARAMETERS)
    if minimum_segment < 1 or (cost != "mean" and minimum_segment < 2):  # noqa: PLR2004
        raise ValueError(
            "A segment needs at least 1 observation for a mean and 2 for a variance."
        )

    values = series.dropna()
    data = values.to_numpy(dtype=float)
    n = len(data)

    if n < minimum_segment:
        raise ValueError(
            f"{n} observations are fewer than a segment of {minimum_segment}."
        )
    if np.ptp(data) == 0:
        raise ValueError("The series is constant, so it has no change points to find.")

    penalty = (
        float(penalty)
        if penalty is not None
        else (CHANGE_POINT_COST_PARAMETERS[cost] + 1) * np.log(n)
    )

    if cost == "mean":
        changes = np.diff(data)
        scale = (
            np.median(np.abs(changes - np.median(changes)))
            / NORMAL_MEDIAN_ABSOLUTE_DEVIATION
            / np.sqrt(2)
        )
        # A series whose changes are mostly 0 has no robust scale; the standard deviation
        # of the changes takes over.
        scale = scale if scale > 0 else np.std(changes, ddof=1) / np.sqrt(2)
        data = data / scale
    elif cost == "variance":
        data = data - data.mean()

    cumulative = np.concatenate([[0.0], np.cumsum(data)])
    squares = np.concatenate([[0.0], np.cumsum(data**2)])
    floor = VARIANCE_FLOOR * np.var(data)

    def segment_costs(starts: np.ndarray, end: int) -> np.ndarray:
        """The cost of the segments from every start up to (not including) end."""
        lengths = end - starts
        sums = cumulative[end] - cumulative[starts]
        sum_of_squares = squares[end] - squares[starts]
        if cost == "mean":
            return sum_of_squares - sums**2 / lengths
        if cost == "variance":
            return lengths * np.log(np.maximum(sum_of_squares / lengths, floor))
        variances = (sum_of_squares - sums**2 / lengths) / lengths
        return lengths * np.log(np.maximum(variances, floor))

    optimum = np.full(n + 1, np.inf)
    optimum[0] = -penalty
    last_change = np.zeros(n + 1, dtype=int)
    # The end before which a pruned start still counts: with segments of at least m
    # observations, a start beaten at t by a split at t is only beaten from t + m on, as
    # the segment after t needs m observations (with m = 1 this is the pruning of Killick,
    # Fearnhead and Eckley, 2012).
    expires = np.full(n + 1, n + 1)
    candidates = np.array([0])

    for end in range(minimum_segment, n + 1):
        candidates = candidates[expires[candidates] > end]
        eligible = candidates[end - candidates >= minimum_segment]
        costs = optimum[eligible] + segment_costs(eligible, end)
        best = int(np.argmin(costs))
        optimum[end] = costs[best] + penalty
        last_change[end] = eligible[best]

        beaten = eligible[costs > optimum[end]]
        expires[beaten] = np.minimum(expires[beaten], end + minimum_segment)
        if end <= n - minimum_segment:
            candidates = np.append(candidates, end)

    boundaries = [n]
    while boundaries[-1] > 0:
        boundaries.append(int(last_change[boundaries[-1]]))
    boundaries = boundaries[::-1]

    original = values.to_numpy(dtype=float)
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        segment = original[start:end]
        rows.append(
            {
                "Start": str(values.index[start]),
                "End": str(values.index[end - 1]),
                "Observations": end - start,
                "Mean": float(segment.mean()),
                "Standard Deviation": (
                    float(segment.std(ddof=1)) if len(segment) > 1 else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="Segment"))
    result["Penalty"] = penalty
    return result
