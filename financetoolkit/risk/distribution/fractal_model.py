"""Fractal Model: memory, fractal and chaos measures of a series"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.special import gammaln

from financetoolkit.helpers import delay_embedding

# The most pairwise distances held in memory at once (32 MB of float64).
DISTANCE_BLOCK = 4_000_000

# The relative difference below which two distances of the k-d tree count as equal, see get_largest_lyapunov_exponent.  # noqa: E501
TIE_TOLERANCE = 1e-9

# A series whose range is within rounding of its size (1e-14 per observation) has no
# variation: the windows would hold rounding noise and give a spurious exponent.
ROUNDING_PER_OBSERVATION = 1e-14


def get_generalized_hurst_exponent(
    data: pd.Series | np.ndarray, max_lag: int = 20, q: float = 2.0
) -> float:
    """
    Calculate the generalized Hurst exponent H(q) of a series: its long-term memory, which
    shows whether the series is mean-reverting, trending, or a random walk.

    The Hurst Exponent (H) is interpreted as follows:

    - H < 0.5: the series is mean-reverting (anti-persistent).
    - H = 0.5: the series is a random walk (no memory).
    - H > 0.5: the series is trending (persistent).

    For a self-affine process (e.g. fractional Brownian motion) the q-th moment of the
    lagged differences scales as a power law of the lag, E[abs(X_(t+lag) - X_t)^q] ~
    lag^(q * H(q)), so H(q) is the slope of a regression of the log of that moment on the
    log of the lag, divided by q. A process whose H(q) changes with q is multifractal: for
    returns with fat tails H(1) usually exceeds H(2).

    The formula is as follows:

    - d(tau) = X_(t+tau) - X_t, every overlapping difference over tau periods
    - K_q(tau) = Mean(abs(d(tau) - Mean(d(tau)))^q)
    - ln K_q(tau) = c + q * H(q) * ln tau, fitted by least squares over tau = 2, ..., max_lag

    With q = 2, K_2 is the population variance (with n in the denominator) of the differences, so H(2) is the
    slope of the log standard deviation on the log lag.

    Di Matteo et al. (2005) apply it to the logarithm of the price, X_t = ln(P_t), so the
    lagged differences are log returns over the lag and their dispersion does not depend
    on the price level; pass log prices for that estimate.

    Also known as: GHE, structure-function Hurst estimator, multifractal Hurst exponent.

    For more information, see the following sources:

    - Di Matteo, T., Aste, T., & Dacorogna, M.M. (2005). "Long-Term Memories of Developed and
      Emerging Markets: Using the Scaling Analysis to Characterize Their Stage of Development."
      Journal of Banking & Finance, 29(4), 827-851.
    - Di Matteo, T. (2007). "Multi-scaling in finance." Quantitative Finance, 7(1), 21-36.
    - Barabási, A.-L. & Vicsek, T. (1991). "Multifractality of self-affine fractals."
      Physical Review A, 44(4), 2730-2733.

    Shares similar logic:
    - nolds, mfhurst_dm: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1695

    Notes:
        - The differences are taken from their mean, which removes the drift of a trending
          price, the effect nolds' mfhurst_dm removes by detrending the series. nolds averages
          the exponent over the largest lags 5 to 19 and samples the differences without
          overlap; this regression uses every overlapping difference and one largest lag.

    Args:
        data (pd.Series | np.ndarray): The levels (e.g. log prices) to calculate the exponent for.
        max_lag (int, optional): The maximum lag, included, to use when estimating the
            exponent; the regression runs over the lags 2 up to and including `max_lag`.
            Capped so that each lag has at least two differences. Defaults to 20.
        q (float, optional): The order of the moment. Defaults to 2.0.

    Returns:
        float: The estimated exponent H(q), or NaN when fewer than two lags have a positive,
        finite moment.
    """
    if not isinstance(data, pd.Series | np.ndarray):
        raise TypeError("Expects pd.Series or np.ndarray, no other value.")
    if q <= 0:
        raise ValueError("The order q should be positive.")

    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    # Each lag needs at least two differences to estimate their dispersion.
    lags = np.arange(2, min(max_lag, len(values) - 2) + 1)
    moments = np.array(
        [
            np.mean(np.abs(differences - differences.mean()) ** q)
            for differences in (values[lag:] - values[:-lag] for lag in lags)
        ]
    )
    usable = np.isfinite(moments) & (moments > 0)
    if usable.sum() < 2:  # noqa: PLR2004
        return np.nan

    slope = np.polyfit(np.log(lags[usable]), np.log(moments[usable]), 1)[0]

    return float(slope / q)


def get_window_sizes(
    length: int, minimum: int = 10, share: float = 0.1, factor: float = 1.2
) -> np.ndarray:
    """
    Calculate the window sizes of the rescaled range and the detrended fluctuation analysis:
    the scales, evenly spread on a log scale, at which the memory of a series is measured.

    They are whole numbers that grow by `factor` from `minimum` up to `share` of the
    observations, so every size is covered by at least 1 / share windows.

    Shares similar logic:
    - nolds, logarithmic_n: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L874

    Args:
        length (int): The number of observations.
        minimum (int, optional): The smallest window. Defaults to 10.
        share (float, optional): The largest window as a share of the observations.
            Defaults to 0.1.
        factor (float, optional): The growth from one size to the next. Defaults to 1.2.

    Returns:
        np.ndarray: The window sizes, empty when the series is too short for one.
    """
    largest = int(length * share)
    if largest < minimum:
        return np.array([], dtype=np.int64)

    steps = int(math.floor(math.log(largest / minimum) / math.log(factor)))

    return np.unique(
        np.floor(minimum * factor ** np.arange(steps + 1)).astype(np.int64)
    )


def get_expected_rescaled_range(window_sizes: np.ndarray) -> np.ndarray:
    """
    Calculate the expected rescaled range of independent, normally distributed observations
    per window size n: the rescaled range a series without memory is expected to have (Anis
    and Lloyd, 1976, with the small-sample correction of Peters, 1994).

    The formula is as follows:

    - E[R/S](n) = ((n - 0.5) / n) * Gamma((n - 1) / 2) / (Sqrt(pi) * Gamma(n / 2)) * Sum(Sqrt((n - i) / i))

    with the sum over i = 1, ..., n - 1 and, for n above 340, where the gamma functions are
    large, 1 / Sqrt(n * pi / 2) in place of the ratio of gamma functions.

    Shares similar logic:
    - nolds, expected_rs: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L969

    Args:
        window_sizes (np.ndarray): The window sizes n.

    Returns:
        np.ndarray: The expected rescaled range per window size.
    """
    expected = []
    for size in np.asarray(window_sizes, dtype=np.int64):
        steps = np.arange(1, size)
        total = np.sum(np.sqrt((size - steps) / steps))
        front = (
            np.exp(gammaln((size - 1) / 2) - gammaln(size / 2)) / np.sqrt(np.pi)
            if size <= 340  # noqa: PLR2004
            else 1 / np.sqrt(size * np.pi / 2)
        )
        expected.append((size - 0.5) / size * front * total)

    return np.array(expected)


def get_rescaled_range_hurst_exponent(
    data: pd.Series | np.ndarray,
    window_sizes: list[int] | np.ndarray | None = None,
    corrected: bool = True,
) -> float:
    """
    Calculate the Hurst Exponent of a series of changes (e.g. returns) by rescaled range
    (R/S) analysis: its long-term memory, which shows whether it is persistent, anti-persistent
    or without memory.

    The exponent can be corrected with the Anis-Lloyd-Peters correction for small windows.

    The series is cut into windows of n observations. In every window the cumulative
    deviations from the window mean span a range R, which is divided by the standard
    deviation S of the window. For a series with long memory the average R/S grows as n^H.
    Over short windows R/S grows faster than n^0.5 even without memory, so the corrected
    exponent measures the growth in excess of the R/S that independent observations are
    expected to have, plus 0.5.

    The formula is as follows:

    - Y_k = Sum(x_i - Mean(x), i = 1, ..., k), k = 1, ..., n, within each window
    - R/S = (Max(Y) - Min(Y)) / Std(x), the population standard deviation (with n in the denominator)
    - (R/S)(n) = the average R/S over the floor(N / n) windows of n observations
    - H = slope of ln (R/S)(n) on ln n
    - Corrected H = 0.5 + slope of (ln (R/S)(n) - ln E[R/S](n)) on ln n

    with E[R/S](n) the expected rescaled range of `get_expected_rescaled_range`.

    Also known as: R/S analysis, rescaled range, Hurst-Mandelbrot exponent.

    For more information, see the following sources:

    - Hurst, H.E. (1951). "Long-term storage capacity of reservoirs." Transactions of the
      American Society of Civil Engineers, 116, 770-799.
    - Mandelbrot, B.B. & Wallis, J.R. (1969). "Robustness of the rescaled range R/S in the
      measurement of noncyclic long run statistical dependence." Water Resources Research,
      5(5), 967-988.
    - Anis, A.A. & Lloyd, E.H. (1976). "The expected value of the adjusted rescaled Hurst
      range of independent normal summands." Biometrika, 63(1), 111-116.
    - Peters, E.E. (1994). "Fractal Market Analysis." Wiley.

    Referenced against:
    - nolds, hurst_rs: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1203

    Notes:
        - S is the population standard deviation (with n in the denominator), as in Hurst's and Peters'
          definition, which is nolds' hurst_rs without its unbiased standard deviation. nolds
          fits by RANSAC unless asked for a polynomial fit; this is a least squares fit.
        - Windows without variation are left out of the average.

    Args:
        data (pd.Series | np.ndarray): The changes (e.g. returns).
        window_sizes (list[int] | np.ndarray | None, optional): The window sizes n. Defaults
            to None, which uses `get_window_sizes`: 10 up to a tenth of the observations,
            growing by 20%.
        corrected (bool, optional): Whether to apply the Anis-Lloyd-Peters correction.
            Defaults to True.

    Returns:
        float: The Hurst exponent, NaN when fewer than two window sizes have a value or the
        changes do not vary.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.ptp(values) <= len(
        values
    ) * ROUNDING_PER_OBSERVATION * np.abs(values).max(initial=0.0):
        return np.nan
    sizes = np.asarray(
        get_window_sizes(len(values)) if window_sizes is None else window_sizes,
        dtype=np.int64,
    )
    sizes = sizes[(sizes >= 2) & (sizes <= len(values))]  # noqa: PLR2004

    averages = []
    for size in sizes:
        windows = values[: len(values) // size * size].reshape(-1, size)
        deviations = np.cumsum(windows - windows.mean(axis=1, keepdims=True), axis=1)
        ranges = deviations.max(axis=1) - deviations.min(axis=1)
        deviation = windows.std(axis=1)
        usable = deviation > 0
        averages.append(
            np.mean(ranges[usable] / deviation[usable]) if usable.any() else np.nan
        )

    averages = np.array(averages)
    usable = np.isfinite(averages) & (averages > 0)
    if usable.sum() < 2:  # noqa: PLR2004
        return np.nan

    log_sizes = np.log(sizes[usable])
    log_ranges = np.log(averages[usable])
    if corrected:
        slope = np.polyfit(
            log_sizes,
            log_ranges - np.log(get_expected_rescaled_range(sizes[usable])),
            1,
        )[0]
        return float(0.5 + slope)

    return float(np.polyfit(log_sizes, log_ranges, 1)[0])


def get_detrended_fluctuation(
    data: pd.Series | np.ndarray,
    window_sizes: list[int] | np.ndarray | None = None,
    order: int = 1,
) -> float:
    """
    Calculate the scaling exponent alpha of the Detrended Fluctuation Analysis (DFA) of a
    series of changes (e.g. returns): its long-term memory, robust to slow shifts in the mean.

    The changes are summed into a profile, which is cut into windows of n observations. In
    every window a polynomial trend is fitted and removed, and the fluctuation F(n) is the
    root mean square of what is left. For a series with long memory F(n) grows as n^alpha,
    where alpha equals the Hurst exponent for fractional Gaussian noise: 0.5 for independent
    changes, above 0.5 for persistent and below for anti-persistent ones. Removing the trend
    of each window makes it robust to slow shifts in the mean.

    The formula is as follows:

    - Y_k = Sum(x_i - Mean(x), i = 1, ..., k), the profile
    - Y_n(k) = the least squares polynomial of degree `order` through Y in the window of k
    - F(n) = Sqrt(Mean((Y_k - Y_n(k))^2)) over the floor(N / n) windows of n observations
    - alpha = slope of ln F(n) on ln n

    Also known as: DFA, detrended fluctuation analysis exponent.

    For more information, see the following sources:

    - Peng, C.-K., Buldyrev, S.V., Havlin, S., Simons, M., Stanley, H.E. & Goldberger, A.L.
      (1994). "Mosaic organization of DNA nucleotides." Physical Review E, 49(2), 1685-1689.

    Referenced against:
    - nolds, dfa: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1997

    Shares similar logic:
    - antropy, detrended_fluctuation: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L369

    Notes:
        - The windows do not overlap, as in Peng et al.; nolds' dfa lets them overlap by half
          unless asked not to and fits by RANSAC unless asked for a polynomial fit. antropy
          fixes its own window sizes and the first order.

    Args:
        data (pd.Series | np.ndarray): The changes (e.g. returns).
        window_sizes (list[int] | np.ndarray | None, optional): The window sizes n. Defaults
            to None, which uses `get_window_sizes`: 10 up to a tenth of the observations,
            growing by 20%.
        order (int, optional): The degree of the polynomial trend removed from every window.
            Defaults to 1.

    Returns:
        float: The scaling exponent alpha, NaN when fewer than two window sizes have a value or
        the changes do not vary.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.ptp(values) <= len(
        values
    ) * ROUNDING_PER_OBSERVATION * np.abs(values).max(initial=0.0):
        return np.nan
    profile = np.cumsum(values - values.mean())
    sizes = np.asarray(
        get_window_sizes(len(values)) if window_sizes is None else window_sizes,
        dtype=np.int64,
    )
    sizes = sizes[(sizes > order + 1) & (sizes <= len(values))]

    fluctuations = []
    for size in sizes:
        windows = profile[: len(profile) // size * size].reshape(-1, size)
        design = np.vander(np.arange(size, dtype=float), order + 1)
        coefficients = np.linalg.lstsq(design, windows.T, rcond=None)[0]
        residuals = windows.T - design @ coefficients
        fluctuations.append(np.sqrt(np.mean(residuals**2)))

    fluctuations = np.array(fluctuations)
    usable = np.isfinite(fluctuations) & (fluctuations > 0)
    if usable.sum() < 2:  # noqa: PLR2004
        return np.nan

    return float(np.polyfit(np.log(sizes[usable]), np.log(fluctuations[usable]), 1)[0])


def get_higuchi_fractal_dimension(
    data: pd.Series | np.ndarray, kmax: int = 10
) -> float:
    """
    Calculate the Higuchi Fractal Dimension of a curve, such as a log price path: how fast
    its length grows as it is measured with a finer step.

    A smooth line has a dimension of 1 and a curve that fills the plane a dimension of 2.
    A random walk has 1.5, which is 2 - H for a self-affine path with Hurst exponent H, so a
    trending path has a lower and a mean-reverting path a higher dimension.

    The formula is as follows:

    - L_m(k) = Sum(abs(x_(m + i*k) - x_(m + (i - 1)*k))) * (N - 1) / (floor((N - m) / k) * k) / k
    - L(k) = Mean(L_m(k)) over m = 1, ..., k
    - FD = -slope of ln L(k) on ln k, k = 1, ..., kmax

    with the sum over i = 1, ..., floor((N - m) / k).

    For more information, see the following sources:

    - Higuchi, T. (1988). "Approach to an irregular time series on the basis of the fractal
      theory." Physica D, 31(2), 277-283.

    Referenced against:
    - antropy, higuchi_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L232

    Args:
        data (pd.Series | np.ndarray): The curve, e.g. log prices.
        kmax (int, optional): The largest step k. Defaults to 10.

    Returns:
        float: The fractal dimension.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    number = len(values)
    steps = np.arange(1, kmax + 1)
    if number <= kmax + 1:
        return np.nan

    lengths = []
    for step in steps:
        curve_lengths = []
        for start in range(step):
            points = values[start::step]
            intervals = len(points) - 1
            if intervals < 1:
                continue
            curve_lengths.append(
                np.abs(np.diff(points)).sum() * (number - 1) / (intervals * step) / step
            )
        lengths.append(np.mean(curve_lengths))

    lengths = np.array(lengths)
    if not (lengths > 0).all():
        return np.nan

    return float(-np.polyfit(np.log(steps), np.log(lengths), 1)[0])


def get_katz_fractal_dimension(data: pd.Series | np.ndarray) -> float:
    """
    Calculate the Katz Fractal Dimension of a curve: how much the curve doubles back on
    itself, from the number of steps and how far it gets from its start for the distance it travels.

    A straight line has a dimension of 1; the more the curve doubles back on itself, the
    higher the dimension.

    The formula is as follows:

    - L = Sum(abs(x_(i+1) - x_i)), the length of the curve
    - d = Max(abs(x_i - x_1)), the largest distance from the first point
    - n = N - 1, the number of steps
    - FD = log10(n) / (log10(n) + log10(d / L))

    For more information, see the following sources:

    - Katz, M.J. (1988). "Fractals and the analysis of waveforms." Computers in Biology and
      Medicine, 18(3), 145-156.

    Referenced against:
    - antropy, katz_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L104

    Args:
        data (pd.Series | np.ndarray): The curve, e.g. log prices.

    Returns:
        float: The fractal dimension.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:  # noqa: PLR2004
        return np.nan

    length = np.abs(np.diff(values)).sum()
    distance = np.abs(values - values[0]).max()
    if length <= 0 or distance <= 0:
        return np.nan

    steps = np.log10(len(values) - 1)

    return float(steps / (steps + np.log10(distance / length)))


def get_petrosian_fractal_dimension(data: pd.Series | np.ndarray) -> float:
    """
    Calculate the Petrosian Fractal Dimension of a curve: how irregular the curve is, from the
    number of times its direction changes.

    A curve that keeps its direction has a dimension of 1; the more often it turns, the
    higher the dimension.

    The formula is as follows:

    - N_delta = the number of sign changes of the differences x_(i+1) - x_i
    - FD = log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta)))

    with N the number of observations. A difference of exactly 0 interrupts a sign change.

    For more information, see the following sources:

    - Petrosian, A. (1995). "Kolmogorov complexity of finite sequences and recognition of
      different preictal EEG patterns." Proceedings of the Eighth IEEE Symposium on
      Computer-Based Medical Systems, 212-217.

    Referenced against:
    - antropy, petrosian_fd: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/fractal.py#L14

    Args:
        data (pd.Series | np.ndarray): The curve, e.g. log prices.

    Returns:
        float: The fractal dimension.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    number = len(values)
    if number < 3:  # noqa: PLR2004
        return np.nan

    changes = np.diff(values)
    turns = np.sum(changes[1:] * changes[:-1] < 0)

    return float(
        np.log10(number)
        / (np.log10(number) + np.log10(number / (number + 0.4 * turns)))
    )


def get_correlation_dimension(
    data: pd.Series | np.ndarray,
    embedding: int = 10,
    delay: int = 1,
    radii: list[float] | np.ndarray | None = None,
) -> float:
    """
    Calculate the Correlation Dimension of a series by the Grassberger-Procaccia algorithm:
    the dimension of the set its delay embedding fills, low for a deterministic process.

    The series is embedded into vectors of `embedding` consecutive values and the
    correlation sum C(r), the share of pairs of vectors closer than r, is counted for a
    range of radii. For a set of dimension D the correlation sum grows as r^D. A
    deterministic (chaotic) process fills a set of a low dimension; for noise the dimension
    keeps growing with the embedding dimension.

    The formula is as follows:

    - X_i = [x_i, x_(i + tau), ..., x_(i + (m - 1) * tau)], the delay embedding
    - C(r) = 2 / (M * (M - 1)) * Sum(1[||X_i - X_j|| < r], i < j), M vectors
    - D2 = slope of ln C(r) on ln r

    with ||.|| the Euclidean norm. The radii are multiples of the sample standard deviation
    (with n - 1 in the denominator) of the series when given; by default they are the radii, 3% apart, at which
    between 0.01% and 1% of the pairs are closer than r, which places the fit at the small
    distances a long enough series still has pairs for, whatever the embedding dimension.

    Also known as: D2, Grassberger-Procaccia dimension.

    For more information, see the following sources:

    - Grassberger, P. & Procaccia, I. (1983). "Characterization of strange attractors."
      Physical Review Letters, 50(5), 346-349.

    Shares similar logic:
    - nolds, corr_dim: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L1848

    Notes:
        - The pairs are those of different vectors. nolds' corr_dim also counts every vector
          as a pair with itself (a distance of 0), which keeps C(r) above 1 / (M - 1) and pulls
          the slope towards 0 where few pairs are close; it fits by RANSAC unless asked for a
          polynomial fit and takes the radii 0.1 to 0.5 times the standard deviation, at which an embedding
          of 10 daily returns has hardly any pairs.
        - Every pair of vectors is compared, so the time grows with the square of the
          length of the series (about 12.5 million distances for 5,000 observations); the
          distances are counted in blocks, so the memory stays bounded for long series.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.
        embedding (int, optional): The embedding dimension m. Defaults to 10.
        delay (int, optional): The number of observations between the values of a vector.
            Defaults to 1.
        radii (list[float] | np.ndarray | None, optional): The radii as multiples of the
            standard deviation. Defaults to None, which uses the radii at which C(r) lies
            between 0.0001 and 0.01.

    Returns:
        float: The correlation dimension, NaN when fewer than two radii have a pair.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    vectors = delay_embedding(values, embedding, delay)
    number = len(vectors)
    if number < 3 or np.ptp(values) <= 0:  # noqa: PLR2004
        return np.nan

    if radii is None:
        # No two vectors are further apart than the range of the series times the square root of the dimension, so a grid 3% apart down from there covers every distance that matters.  # noqa: E501
        largest = np.ptp(values) * np.sqrt(embedding) * 1.03
        thresholds = largest * 1.03 ** -np.arange(
            int(np.log(1e6) / np.log(1.03)), -1, -1
        )
    else:
        thresholds = np.sort(np.asarray(radii, dtype=float) * np.std(values, ddof=1))

    # The number of pairs closer than every radius: a distance below the k-th radius counts for that radius and every larger one.  # noqa: E501
    counts = np.zeros(len(thresholds) + 1, dtype=np.int64)
    step = max(1, DISTANCE_BLOCK // (number * embedding))
    for start in range(0, number, step):
        block = vectors[start : start + step]
        distances = np.sqrt(
            ((block[:, None, :] - vectors[None, start + 1 :, :]) ** 2).sum(axis=2)
        )
        later = (
            np.arange(start + 1, number)[None, :]
            > np.arange(start, start + len(block))[:, None]
        )
        counts += np.bincount(
            np.searchsorted(thresholds, distances[later], side="right"),
            minlength=len(thresholds) + 1,
        )

    correlation_sums = np.cumsum(counts)[:-1] * 2 / (number * (number - 1))
    usable = (
        correlation_sums > 0
        if radii is not None
        else (correlation_sums >= 0.0001) & (correlation_sums <= 0.01)  # noqa: PLR2004
    )
    if usable.sum() < 2:  # noqa: PLR2004
        return np.nan

    return float(
        np.polyfit(np.log(thresholds[usable]), np.log(correlation_sums[usable]), 1)[0]
    )


def get_mean_period(data: pd.Series | np.ndarray) -> int:
    """
    Calculate the mean period of a series: the typical length of its cycles, in observations.

    It is one over the power-weighted mean frequency of its periodogram, rounded up.
    Rosenstein et al. (1993) keep the nearest neighbours of the largest Lyapunov exponent at
    least this far apart in time.

    Args:
        data (pd.Series | np.ndarray): The series.

    Returns:
        int: The mean period, at least 1.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    # A series of one value or none has no frequencies.
    if len(values) < 2:  # noqa: PLR2004
        return 1
    power = np.abs(np.fft.rfft(values - values.mean())) ** 2
    frequencies = np.fft.rfftfreq(len(values))
    if power[1:].sum() <= 0:
        return 1

    mean_frequency = np.sum(frequencies[1:] * power[1:]) / power[1:].sum()

    return max(1, math.ceil(1 / mean_frequency))


def get_largest_lyapunov_exponent(
    data: pd.Series | np.ndarray,
    embedding: int = 10,
    delay: int = 1,
    min_separation: int | None = None,
    trajectory_length: int = 20,
) -> float:
    """
    Calculate the largest Lyapunov exponent of a series by the algorithm of Rosenstein,
    Collins and De Luca: the average rate, per period, at which nearby trajectories of its
    delay embedding move apart, positive for a chaotic process.

    Every embedded vector is paired with its nearest neighbour that is at least
    `min_separation` periods away in time, and both are followed for `trajectory_length`
    periods. In a chaotic system the distance between them grows exponentially, so the
    average log distance rises linearly with the number of periods, at the largest Lyapunov
    exponent. A positive exponent points to chaos (sensitive dependence on the starting
    point), a value around 0 or below to a regular or noisy process.

    The formula is as follows:

    - X_i = [x_i, x_(i + tau), ..., x_(i + (m - 1) * tau)], the delay embedding
    - j(i) = argmin ||X_i - X_j|| over abs(i - j) > min_separation, the nearest neighbour
    - y(k) = Mean(ln ||X_(i+k) - X_(j(i)+k)||) over i, k = 0, ..., trajectory_length - 1
    - lambda_1 = slope of y(k) on k

    with ||.|| the Euclidean norm. A distance of 0 is left out of the mean.

    For more information, see the following sources:

    - Rosenstein, M.T., Collins, J.J. & De Luca, C.J. (1993). "A practical method for
      calculating largest Lyapunov exponents from small data sets." Physica D, 65(1-2),
      117-134.

    Referenced against:
    - nolds, lyap_r: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L117

    Notes:
        - nolds chooses the delay from the autocorrelation function and fits by RANSAC unless
          given a delay and asked for a polynomial fit; this uses the `delay` as given and a
          least squares fit.
        - The nearest neighbours are found with a k-d tree, so the time grows with about
          n log n rather than with the square of the length of the series.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.
        embedding (int, optional): The embedding dimension m. Defaults to 10.
        delay (int, optional): The number of observations between the values of a vector.
            Defaults to 1.
        min_separation (int | None, optional): The smallest number of periods between a
            vector and its neighbour. Defaults to None, which uses the mean period of the
            series (see `get_mean_period`), as Rosenstein et al. suggest.
        trajectory_length (int, optional): The number of periods the pairs are followed.
            Defaults to 20.

    Returns:
        float: The largest Lyapunov exponent per period, NaN for a series too short to follow
        its pairs (or without values).
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    vectors = delay_embedding(values, embedding, delay)
    separation = get_mean_period(values) if min_separation is None else min_separation
    # Only vectors that can be followed for the whole trajectory are paired.
    starts = len(vectors) - trajectory_length + 1
    if starts <= 2 * separation + 2:
        return np.nan

    followed = vectors[:starts]
    origins = np.arange(starts)

    # At most 2 * separation + 1 vectors (the vector itself and those within the separation)
    # are excluded, so the nearest vector allowed is among the 2 * separation + 2 nearest a
    # k-d tree finds, without the distances between every pair.
    nearest = min(starts, 2 * separation + 2)
    candidates = np.sort(
        np.reshape(KDTree(followed).query(followed, k=nearest)[1], (starts, -1)),
        axis=1,
    )
    # The squared distances are taken as below for every vector, and of equally near vectors
    # the earliest is the neighbour, so the pairs do not depend on the order of the tree.
    distances = ((followed[:, None, :] - followed[candidates]) ** 2).sum(axis=2)
    furthest = distances.max(axis=1)
    distances[np.abs(candidates - origins[:, None]) <= separation] = np.inf
    neighbours = candidates[origins, distances.argmin(axis=1)]

    # A neighbour as far away as the furthest vector the tree returned can have an equally
    # near, earlier vector beyond them (repeated values give many vectors at one distance),
    # so it is taken from the distances to every vector instead.
    ties = (nearest < starts) & (
        distances.min(axis=1) >= furthest * (1 - TIE_TOLERANCE)
    )
    for origin in np.flatnonzero(ties):
        every_distance = ((followed[origin] - followed) ** 2).sum(axis=1)
        every_distance[np.abs(origins - origin) <= separation] = np.inf
        neighbours[origin] = every_distance.argmin()

    divergence = []
    for offset in range(trajectory_length):
        distances = np.linalg.norm(
            vectors[origins + offset] - vectors[neighbours + offset], axis=1
        )
        positive = distances[distances > 0]
        divergence.append(np.mean(np.log(positive)) if len(positive) else np.nan)

    divergence = np.array(divergence)
    usable = np.isfinite(divergence)
    if usable.sum() < 2:  # noqa: PLR2004
        return np.nan

    return float(
        np.polyfit(np.arange(trajectory_length)[usable], divergence[usable], 1)[0]
    )


def _solve_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Solve a stack of least squares problems min ||A_i x_i - y_i|| at once, with the minimum
    norm solution x_i = V S⁺ U' y_i of the singular value decomposition A_i = U S V', as
    np.linalg.lstsq solves one: singular values up to the machine precision times
    max(rows, columns) of the largest count as zero.

    Args:
        design (np.ndarray): The designs A_i, shape (problems, rows, columns).
        target (np.ndarray): The targets y_i, shape (problems, rows).

    Returns:
        np.ndarray: The solutions x_i, shape (problems, columns).
    """
    left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    cutoff = np.finfo(float).eps * max(design.shape[1:]) * singular_values[:, :1]
    inverse = np.where(
        singular_values > cutoff,
        1 / np.where(singular_values > 0, singular_values, 1),
        0,
    )
    projected = np.einsum("pij,pi->pj", left, target) * inverse

    return np.einsum("pji,pj->pi", right, projected)


def get_lyapunov_spectrum(
    data: pd.Series | np.ndarray,
    embedding: int = 10,
    matrix_dimension: int = 4,
    min_neighbors: int | None = None,
    min_separation: int = 0,
) -> pd.Series:
    """
    Calculate the Lyapunov spectrum of a series by the algorithm of Eckmann, Kamphorst,
    Ruelle and Ciliberto: the largest rates, per period, at which the reconstructed dynamics
    stretch or shrink small perturbations.

    The local dynamics are reconstructed from the delay embedding: around every vector a
    linear map (the Jacobian) is fitted by least squares on the neighbours, sending each
    neighbour's offset to its offset m periods later. Multiplying these maps along the
    trajectory, with a QR decomposition at every step, gives the exponents as the average
    log growth along each direction. A positive largest exponent points to chaos.

    The formula is as follows:

    - m = (embedding - 1) / (matrix_dimension - 1), the step between the d = matrix_dimension
      coordinates of X_i = [x_i, x_(i + m), ..., x_(i + (d - 1) * m)]
    - Neighbours of i: the vectors j with abs(i - j) > min_separation within the distance of
      the min_neighbors-th nearest one (Chebyshev norm over the full embedding)
    - a_i = the least squares solution of x_(j + d*m) - x_(i + d*m) = a_i . (X_j - X_i)
    - T_i = the companion matrix that shifts X_i to X_(i+m), with a_i as its last row
    - T_i Q_(i-m) = Q_i R_i, the QR decomposition, along every chain i, i + m, i + 2m, ...
    - lambda_k = Mean(ln abs(R_i,kk)) / m

    For more information, see the following sources:

    - Eckmann, J.-P., Kamphorst, S.O., Ruelle, D. & Ciliberto, S. (1986). "Liapunov exponents
      from time series." Physical Review A, 34(6), 4971-4979.

    Referenced against:
    - nolds, lyap_e: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L400

    Notes:
        - nolds is matched with as many exponents as the embedding dimension (a step of one),
          where both multiply the same maps.
        - The map T_i moves a perturbation from i to i + m, so the decompositions are chained
          over i, i + m, i + 2m, ... (m interleaved chains that together use every vector).
          nolds' lyap_e chains the maps of consecutive vectors, which mixes perturbations at
          different times; on maps with known exponents (logistic, Hénon) the chained
          products here recover them more closely.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.
        embedding (int, optional): The embedding dimension; embedding - 1 must be a multiple
            of matrix_dimension - 1. Defaults to 10.
        matrix_dimension (int, optional): The number of exponents d. Defaults to 4.
        min_neighbors (int | None, optional): The number of neighbours of every vector.
            Defaults to None, which uses min(2 * d, d + 4), as Eckmann et al. suggest.
        min_separation (int, optional): The smallest number of periods between a vector and
            a neighbour. Defaults to 0.

    Returns:
        pd.Series: The exponents per period from the largest to the smallest, "Exponent 1"
        to "Exponent d".
    """
    names = [f"Exponent {number}" for number in range(1, matrix_dimension + 1)]
    too_small = matrix_dimension < 2  # noqa: PLR2004
    if too_small or (embedding - 1) % (matrix_dimension - 1):
        raise ValueError(
            "The embedding - 1 should be a multiple of the matrix_dimension - 1."
        )

    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    step = (embedding - 1) // (matrix_dimension - 1)
    neighbours_needed = (
        min(2 * matrix_dimension, matrix_dimension + 4)
        if min_neighbors is None
        else min_neighbors
    )
    # Every vector needs the value m periods after its last one, the new coordinate of X_(i+m).  # noqa: E501
    vectors = delay_embedding(values[:-step], embedding)
    number = len(vectors)
    if number <= neighbours_needed + 2 * min_separation + 1:
        return pd.Series(np.nan, index=names)

    coordinates = vectors[:, ::step]
    following = values[embedding - 1 + step : embedding - 1 + step + number]

    jacobians = np.empty((number, matrix_dimension, matrix_dimension))
    shift = np.eye(matrix_dimension, k=1)
    block_size = max(1, DISTANCE_BLOCK // (number * embedding))
    embedded = values[: number + embedding - 1]
    for start in range(0, number, block_size):
        rows = np.arange(start, min(start + block_size, number))
        # The Chebyshev distance of X_i and X_j is the largest of abs(x_(i+k) - x_(j+k)) over k,
        # so the differences of the single values are taken once and the distances follow as
        # the maximum along the diagonals.
        differences = np.abs(
            embedded[start : start + len(rows) + embedding - 1, None]
            - embedded[None, :]
        )
        distances = differences[: len(rows), :number].copy()
        for offset in range(1, embedding):
            np.maximum(
                distances,
                differences[offset : offset + len(rows), offset : offset + number],
                out=distances,
            )
        too_close = np.abs(rows[:, None] - np.arange(number)[None, :]) <= min_separation
        distances[too_close] = np.inf
        radii = np.partition(distances, neighbours_needed - 1, axis=1)[
            :, neighbours_needed - 1
        ]
        within = distances <= radii[:, None]
        jacobians[rows] = shift

        # A vector without ties at its radius has exactly min_neighbors neighbours, so the
        # least squares problems of those vectors have the same size and are solved at once
        # (in the order of the neighbours in time, as one problem at a time would be).
        exact = within.sum(axis=1) == neighbours_needed
        if exact.any():
            neighbours = np.sort(
                np.argpartition(distances[exact], neighbours_needed - 1, axis=1)[
                    :, :neighbours_needed
                ],
                axis=1,
            )
            centres = rows[exact]
            jacobians[centres, -1] = _solve_least_squares(
                coordinates[neighbours] - coordinates[centres][:, None, :],
                following[neighbours] - following[centres][:, None],
            )

        for row in np.flatnonzero(~exact):
            neighbours = np.flatnonzero(within[row])
            jacobians[rows[row], -1] = np.linalg.lstsq(
                coordinates[neighbours] - coordinates[rows[row]],
                following[neighbours] - following[rows[row]],
                rcond=None,
            )[0]

    # The m chains run side by side: step p decomposes the maps of i = chain + p * m of every
    # chain that is that long, one stacked QR decomposition per step.
    totals = np.zeros(matrix_dimension)
    count = 0
    orthonormal = np.broadcast_to(
        np.eye(matrix_dimension), (step, matrix_dimension, matrix_dimension)
    ).copy()
    for first in range(0, number, step):
        chains = min(step, number - first)
        orthonormal[:chains], upper = np.linalg.qr(
            jacobians[first : first + chains] @ orthonormal[:chains]
        )
        diagonal = np.abs(np.diagonal(upper, axis1=1, axis2=2))
        valid = (diagonal > 0).all(axis=1)
        totals += np.log(diagonal[valid]).sum(axis=0)
        count += int(valid.sum())

    if count == 0:
        return pd.Series(np.nan, index=names)

    return pd.Series(np.sort(totals / count / step)[::-1], index=names)
