"""Entropy Model: the entropies and complexity measures of a series"""

__docformat__ = "google"

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from financetoolkit.helpers import delay_embedding
from financetoolkit.utilities.error_model import InvalidArgumentError

# The most pairwise comparisons made at once when templates are compared with every other
# template. Blocks of half a megabyte of float64 stay in the processor's cache, which makes
# them about three times as fast as one large block for a window of a few hundred values.
DISTANCE_BLOCK = 65_536

# The most cells of the grid of windows by radii, and of the pairs of templates, that the
# rolling sample entropy holds at once (see get_rolling_pair_counts): 16 MB of integers each.
ENTROPY_GRID_CELLS = 2_097_152


def count_template_matches(
    values: np.ndarray, length: int, radius: float, number: int | None = None
) -> np.ndarray:
    """
    Count the template matches of a series: for each template of consecutive values, the
    templates within a Chebyshev (maximum) distance of the radius, itself included.

    The count runs over the first `number` templates of `length` consecutive values.

    Two templates are within r when every pair of their values is, so the comparison of the
    single values, abs(x_i - x_j) <= r, is made once and the templates follow from it along
    the diagonals. The comparisons are made block by block, so the memory stays bounded for
    long series.

    Args:
        values (np.ndarray): The series.
        length (int): The number of values per template.
        radius (float): The tolerance r.
        number (int | None, optional): The number of templates, from the start. Defaults to
            None, which uses every template, N - length + 1.

    Returns:
        np.ndarray: The number of templates within r of each template, itself included.
    """
    number = len(values) - length + 1 if number is None else number
    counts = np.zeros(max(number, 0), dtype=np.int64)
    columns = values[: number + length - 1]
    step = max(1, DISTANCE_BLOCK // max(len(columns), 1) - length)

    for start in range(0, number, step):
        rows = min(step, number - start)
        close = (
            np.abs(values[start : start + rows + length - 1, None] - columns[None, :])
            <= radius
        )
        matches = close[:rows, :number].copy()
        for offset in range(1, length):
            matches &= close[offset : offset + rows, offset : offset + number]
        counts[start : start + rows] = matches.sum(axis=1)

    return counts


def get_sample_entropy(
    data: pd.Series | np.ndarray, order: int = 2, tolerance: float = 0.2
) -> float:
    """
    Calculate the Sample Entropy (SampEn) of a series: the negative logarithm of the
    conditional probability that two stretches of `order` observations that are alike stay
    alike at the next observation.

    A regular, predictable series has a low sample entropy and an irregular one a high
    value. Unlike the approximate entropy, a template is not counted as a match of itself,
    which removes the bias towards regularity of short series.

    The formula is as follows:

    - x_m(i) = [x_i, ..., x_(i + m - 1)], i = 1, ..., N - m, the templates of length m
    - r = tolerance * Std(x), the sample standard deviation (ddof=1)
    - B = the number of pairs (i, j), i != j, with max_k abs(x_m(i)_k - x_m(j)_k) <= r
    - A = the same for the templates of length m + 1
    - SampEn = -ln(A / B)

    Both counts run over the same N - m templates, so A and B compare the same stretches.

    Also known as: SampEn.

    Reference: Richman, J.S. & Moorman, J.R. (2000). "Physiological time-series analysis
    using approximate entropy and sample entropy." American Journal of Physiology - Heart and
    Circulatory Physiology, 278(6), H2039-H2049.

    Referenced against:
    - nolds, sampen: https://github.com/CSchoel/nolds/blob/0.6.3/nolds/measures.py#L699
    - antropy, sample_entropy: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L740

    Notes:
        - The tolerance uses the sample standard deviation (ddof=1) and the matches include a
          distance of exactly r, as in Richman and Moorman. Unlike this, nolds' default
          tolerance is corrected for the order and counts distances strictly below r unless
          asked to include them, and antropy uses the population standard deviation (ddof=0).
        - Without a match of length m + 1 (or m) the entropy is undefined and NaN, where nolds
          and antropy give infinity.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.
        order (int, optional): The template length m. Defaults to 2.
        tolerance (float, optional): The tolerance r as a multiple of the standard
            deviation. Defaults to 0.2.

    Returns:
        float: The sample entropy, NaN when it is undefined.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= order + 1:
        return np.nan

    radius = tolerance * np.std(values, ddof=1)
    number = len(values) - order

    # Every count includes the template itself once, which a sample entropy leaves out.
    matches = count_template_matches(values, order, radius, number) - 1
    longer_matches = count_template_matches(values, order + 1, radius, number) - 1
    total, longer_total = matches.sum(), longer_matches.sum()

    if total == 0 or longer_total == 0:
        return np.nan

    return float(-np.log(longer_total / total))


def get_approximate_entropy(
    data: pd.Series | np.ndarray, order: int = 2, tolerance: float = 0.2
) -> float:
    """
    Calculate the Approximate Entropy (ApEn) of a series: the average logarithmic likelihood
    that stretches of `order` observations that are alike stay alike at the next observation.

    A regular, predictable series has a low approximate entropy and an irregular one a high
    value. Every template counts itself as a match, which keeps the logarithms finite but
    biases the measure towards regularity for short series (see the sample entropy).

    The formula is as follows:

    - x_m(i) = [x_i, ..., x_(i + m - 1)], i = 1, ..., N - m + 1, the templates of length m
    - r = tolerance * Std(x), the sample standard deviation (ddof=1)
    - C_i(m) = the share of templates j with max_k abs(x_m(i)_k - x_m(j)_k) <= r, j = i included
    - Phi(m) = Mean(ln C_i(m))
    - ApEn = Phi(m) - Phi(m + 1)

    Also known as: ApEn.

    Reference: Pincus, S.M. (1991). "Approximate entropy as a measure of system complexity."
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.

    Referenced against:
    - antropy, app_entropy: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L638

    Notes:
        - The tolerance uses the sample standard deviation (ddof=1); antropy uses the population
          standard deviation (ddof=0), which gives the same value for the same r.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.
        order (int, optional): The template length m. Defaults to 2.
        tolerance (float, optional): The tolerance r as a multiple of the standard
            deviation. Defaults to 0.2.

    Returns:
        float: The approximate entropy.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= order + 1:
        return np.nan

    radius = tolerance * np.std(values, ddof=1)

    phi = []
    for length in (order, order + 1):
        counts = count_template_matches(values, length, radius)
        phi.append(np.mean(np.log(counts / len(counts))))

    return float(phi[0] - phi[1])


def get_rolling_pair_counts(
    values: np.ndarray, window: int, length: int, number: int, radii: np.ndarray
) -> np.ndarray:
    """
    Count the matching template pairs of every trailing window of a series: the pairs of
    templates within a Chebyshev distance of that window's radius.

    The pairs i < j run over the first `number` templates of `length` consecutive values:
    the totals of `count_template_matches` without the matches of a template with itself,
    halved.

    A pair of templates (i, j) belongs to the windows that start from j - number + 1 to i,
    and matches in those whose radius is at least its distance. Every pair is therefore an
    event over a range of windows and a range of radii, and the counts of all windows follow
    from one two-dimensional cumulative sum over the windows and the ranks of their radii,
    rather than from comparing the templates of every window again. The distances are taken
    gap by gap (j - i), as the largest abs(x_(i+k) - x_(j+k)) over k, in blocks that keep
    the memory bounded.

    Args:
        values (np.ndarray): The series, without missing values.
        window (int): The number of values per window.
        length (int): The number of values per template.
        number (int): The number of templates per window, from the start of the window.
        radii (np.ndarray): The tolerance r of every window, len(values) - window + 1 of them.

    Returns:
        np.ndarray: The number of pairs within r per window.
    """
    windows = len(radii)
    templates = windows - 1 + number
    sorted_radii = np.sort(radii)
    # A pair matches in a window when the radii below its distance are at most the radii below the window's radius.  # noqa: E501
    radius_ranks = np.searchsorted(sorted_radii, radii, "left")
    counts = np.zeros(windows, dtype=np.int64)
    chunk = max(1, min(windows, ENTROPY_GRID_CELLS // (windows + 1)))
    gap_block = max(1, ENTROPY_GRID_CELLS // (chunk + number))

    for first in range(0, windows, chunk):
        last = min(windows, first + chunk)
        rows = last - first + 1
        grid = np.zeros(rows * (windows + 1), dtype=np.int64)
        # The pairs of these windows start at a template from `first` and end before the last window's end.  # noqa: E501
        starts = np.arange(first, min(last + number - 1, templates))

        for gap_start in range(1, number, gap_block):
            gaps = np.arange(gap_start, min(number, gap_start + gap_block))[:, None]
            ends = starts[None, :] + gaps
            valid = ends <= min(last + number - 2, templates - 1)
            beginnings = np.broadcast_to(starts[None, :], ends.shape)[valid]
            ends = ends[valid]

            distances = np.abs(values[beginnings] - values[ends])
            for offset in range(1, length):
                np.maximum(
                    distances,
                    np.abs(values[beginnings + offset] - values[ends + offset]),
                    out=distances,
                )
            keys = np.searchsorted(sorted_radii, distances, "left")
            entering = np.maximum(ends - number + 1, first) - first
            leaving = np.minimum(beginnings, last - 1) + 1 - first
            grid += np.bincount(
                entering * (windows + 1) + keys, minlength=grid.size
            ) - np.bincount(leaving * (windows + 1) + keys, minlength=grid.size)

        grid = grid.reshape(rows, windows + 1).cumsum(axis=0).cumsum(axis=1)
        counts[first:last] = grid[np.arange(last - first), radius_ranks[first:last]]

    return counts


def get_rolling_template_matches(
    values: np.ndarray,
    window: int,
    length: int,
    number: int,
    radii: np.ndarray,
    per_template: bool = False,
) -> np.ndarray:
    """
    Count the template matches of every trailing window of a series at once, as
    `count_template_matches` does for one window.

    For each window, the counts cover the first `number` templates of `length` consecutive
    values within a Chebyshev distance of that window's radius.

    The distance of two templates i and j depends on the values only, not on the window,
    so the distances of all pairs are taken once for a block of windows and every window
    counts the pairs of its own templates against its own radius. The distance of the
    templates starting at i and j is the largest of abs(x_(i+k) - x_(j+k)), which follows
    from the differences of the single values along the diagonals; a pair is counted with
    i < j only (the distance is symmetric), the matches of a template with itself added
    back where needed.

    Args:
        values (np.ndarray): The series, without missing values.
        window (int): The number of values per window.
        length (int): The number of values per template.
        number (int): The number of templates per window, from the start of the window.
        radii (np.ndarray): The tolerance r of every window, len(values) - window + 1 of them.
        per_template (bool, optional): Whether to return the count of every template (itself
            included) rather than the number of matching pairs. Defaults to False.

    Returns:
        np.ndarray: The number of pairs i < j within r per window, or (with per_template)
        the number of templates within r of each template per window, a row per window.
    """
    windows = len(values) - window + 1
    counts = (
        np.zeros((windows, number), dtype=np.int64)
        if per_template
        else np.zeros(windows, dtype=np.int64)
    )
    block = max(1, number)

    for first in range(0, windows, block):
        size = min(block, windows - first)
        templates = size - 1 + number
        stretch = values[first : first + templates + length - 1]
        differences = np.abs(stretch[:, None] - stretch[None, :])
        distances = differences[:templates, :templates].copy()
        for offset in range(1, length):
            np.maximum(
                distances,
                differences[offset : offset + templates, offset : offset + templates],
                out=distances,
            )
        # Each pair once, with i < j: the diagonal and the lower triangle never match.
        distances[np.tril_indices(templates)] = np.inf

        for position in range(size):
            close = (
                distances[position : position + number, position : position + number]
                <= radii[first + position]
            )
            if per_template:
                counts[first + position] = 1 + close.sum(axis=1) + close.sum(axis=0)
            else:
                counts[first + position] = np.count_nonzero(close)

    return counts


def get_rolling_sample_entropy(
    data: pd.Series | np.ndarray, window: int, order: int = 2, tolerance: float = 0.2
) -> np.ndarray:
    """
    Calculate the rolling Sample Entropy of a series: how irregular every trailing window is
    (see `get_sample_entropy`).

    The template matches of all windows counted at once (see
    `get_rolling_template_matches`). It gives the same values as `get_sample_entropy` on
    each window.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns, without missing values.
        window (int): The number of observations per window.
        order (int, optional): The template length m. Defaults to 2.
        tolerance (float, optional): The tolerance r as a multiple of the standard
            deviation of the window. Defaults to 0.2.

    Returns:
        np.ndarray: The sample entropy of every window, from the window ending at
        observation `window` to the one ending at the last, NaN where it is undefined.
    """
    values = np.asarray(data, dtype=float)
    windows = len(values) - window + 1
    if windows < 1:
        return np.array([])
    if window <= order + 1 or not np.all(np.isfinite(values)):
        return np.array(
            [
                get_sample_entropy(values[end - window : end], order, tolerance)
                for end in range(window, len(values) + 1)
            ]
        )

    radii = np.array(
        [
            tolerance * np.std(values[end - window : end], ddof=1)
            for end in range(window, len(values) + 1)
        ]
    )
    number = window - order

    # Every pair i < j counts twice in the totals of get_sample_entropy (once from each template).  # noqa: E501
    total = 2 * get_rolling_pair_counts(values, window, order, number, radii)
    longer_total = 2 * get_rolling_pair_counts(values, window, order + 1, number, radii)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.log(longer_total / total)

    return np.where((total == 0) | (longer_total == 0), np.nan, entropy)


def get_rolling_approximate_entropy(
    data: pd.Series | np.ndarray, window: int, order: int = 2, tolerance: float = 0.2
) -> np.ndarray:
    """
    Calculate the rolling Approximate Entropy of a series: how irregular every trailing
    window is (see `get_approximate_entropy`).

    The template matches of all windows are counted at once (see
    `get_rolling_template_matches`). It gives the same values as `get_approximate_entropy`
    on each window.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns, without missing values.
        window (int): The number of observations per window.
        order (int, optional): The template length m. Defaults to 2.
        tolerance (float, optional): The tolerance r as a multiple of the standard
            deviation of the window. Defaults to 0.2.

    Returns:
        np.ndarray: The approximate entropy of every window, from the window ending at
        observation `window` to the one ending at the last.
    """
    values = np.asarray(data, dtype=float)
    windows = len(values) - window + 1
    if windows < 1:
        return np.array([])
    if window <= order + 1 or not np.all(np.isfinite(values)):
        return np.array(
            [
                get_approximate_entropy(values[end - window : end], order, tolerance)
                for end in range(window, len(values) + 1)
            ]
        )

    radii = np.array(
        [
            tolerance * np.std(values[end - window : end], ddof=1)
            for end in range(window, len(values) + 1)
        ]
    )

    phi = []
    for length in (order, order + 1):
        counts = get_rolling_template_matches(
            values, window, length, window - length + 1, radii, per_template=True
        )
        phi.append(np.mean(np.log(counts / counts.shape[1]), axis=1))

    return phi[0] - phi[1]


def get_ordinal_patterns(
    data: pd.Series | np.ndarray, order: int = 3, delay: int = 1
) -> np.ndarray:
    """
    Calculate the ordinal pattern of every stretch of observations: the order in which the
    values of the stretch rank, as one number per stretch.

    The stretches hold `order` observations, `delay` apart. Equal values rank
    in the order they occur (Bandt and Pompe, 2002).

    Args:
        data (pd.Series | np.ndarray): The series.
        order (int, optional): The number of observations per pattern. Defaults to 3.
        delay (int, optional): The number of observations between the values of a pattern.
            Defaults to 1.

    Returns:
        np.ndarray: A pattern number per stretch, from 0 to order^order - 1.
    """
    embedded = delay_embedding(np.asarray(data, dtype=float), order, delay)
    ranks = np.argsort(embedded, axis=1, kind="stable")

    return ranks @ (order ** np.arange(order))


def get_permutation_entropy(
    data: pd.Series | np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Calculate the Permutation Entropy of a series: the Shannon entropy of the ordinal
    patterns, the order in which `order` consecutive observations rank.

    It only looks at the order of the values, so it is robust to outliers and to any
    monotonic transformation of the series. A trending or repeating series has few patterns
    and a low entropy; white noise has every pattern equally often and the maximum entropy,
    log2(order!), or 1 when normalized.

    The formula is as follows:

    - pi(i) = the ordinal pattern of [x_i, x_(i + tau), ..., x_(i + (n - 1) * tau)]
    - p(pi) = the share of the N - (n - 1) * tau stretches with pattern pi
    - H = -Sum(p(pi) * log2(p(pi)))
    - Normalized H = H / log2(n!)

    with n the order and tau the delay.

    Reference: Bandt, C. & Pompe, B. (2002). "Permutation entropy: a natural complexity
    measure for time series." Physical Review Letters, 88(17), 174102.

    Referenced against:
    - antropy, perm_entropy: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L143

    Args:
        data (pd.Series | np.ndarray): The series.
        order (int, optional): The number of observations per pattern n. Defaults to 3.
        delay (int, optional): The number of observations between the values of a pattern.
            Defaults to 1.
        normalize (bool, optional): Whether to divide by log2(order!), so the entropy lies
            between 0 and 1. Defaults to True.

    Returns:
        float: The permutation entropy in bits, or normalized.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    patterns = get_ordinal_patterns(values, order, delay)
    if len(patterns) == 0:
        return np.nan

    shares = np.unique(patterns, return_counts=True)[1] / len(patterns)
    entropy = -np.sum(shares * np.log2(shares))

    return float(entropy / np.log2(math.factorial(order)) if normalize else entropy)


def get_spectral_entropy(
    data: pd.Series | np.ndarray,
    estimator: str = "periodogram",
    normalize: bool = True,
) -> float:
    """
    Calculate the Spectral Entropy of a series: the Shannon entropy of its power spectrum
    taken as a probability distribution over the frequencies.

    A series whose variance sits in a few cycles has a low spectral entropy; white noise,
    with the same power at every frequency, has the maximum, log2 of the number of
    frequencies, or 1 when normalized.

    The formula is as follows:

    - P(f) = the one-sided power spectral density of the demeaned series at frequency f
    - p(f) = P(f) / Sum(P)
    - H = -Sum(p(f) * log2(p(f)))
    - Normalized H = H / log2(number of frequencies)

    The density is the periodogram, abs(FFT)^2, or Welch's average of the periodograms of
    overlapping segments of 256 observations with a Hann window.

    Reference: Inouye, T., Shinosaki, K., Sakamoto, H. et al. (1991). "Quantification of EEG
    irregularity by use of the entropy of the power spectrum." Electroencephalography and
    Clinical Neurophysiology, 79(3), 204-210.

    Referenced against:
    - antropy, spectral_entropy: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L315

    Args:
        data (pd.Series | np.ndarray): The series.
        estimator (str, optional): The power spectrum. Options are: "periodogram" (the
            periodogram), "welch" (Welch's average of periodograms). Defaults to "periodogram".
        normalize (bool, optional): Whether to divide by log2 of the number of frequencies,
            so the entropy lies between 0 and 1. Defaults to True.

    Returns:
        float: The spectral entropy in bits, or normalized.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:  # noqa: PLR2004
        return np.nan

    if estimator == "periodogram":
        power = signal.periodogram(values)[1]
    elif estimator == "welch":
        power = signal.welch(values, nperseg=min(256, len(values)))[1]
    else:
        raise InvalidArgumentError(
            f"estimator must be one of 'periodogram' or 'welch', "
            f"not {estimator!r}. Please pass one of these as estimator."
        )

    if power.sum() <= 0:
        return np.nan

    shares = power / power.sum()
    positive = shares[shares > 0]
    entropy = -np.sum(positive * np.log2(positive))

    return float(entropy / np.log2(len(shares)) if normalize else entropy)


def get_svd_entropy(
    data: pd.Series | np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Calculate the Singular Value Decomposition (SVD) Entropy of a series: the Shannon
    entropy of the normalized singular values of its delay embedding.

    It counts how many independent directions the stretches of `order` observations need:
    a series driven by one smooth component has one dominant singular value and a low
    entropy, white noise spreads them evenly, up to log2(order), or 1 when normalized.

    The formula is as follows:

    - Y = the delay embedding, rows [x_i, x_(i + tau), ..., x_(i + (n - 1) * tau)]
    - sigma_k = the singular values of Y, k = 1, ..., n
    - s_k = sigma_k / Sum(sigma)
    - H = -Sum(s_k * log2(s_k))
    - Normalized H = H / log2(n)

    Reference: Roberts, S.J., Penny, W. & Rezek, I. (1999). "Temporal and spatial complexity
    measures for electroencephalogram based brain-computer interfacing." Medical & Biological
    Engineering & Computing, 37(1), 93-98.

    Referenced against:
    - antropy, svd_entropy: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L437

    Args:
        data (pd.Series | np.ndarray): The series.
        order (int, optional): The embedding dimension n. Defaults to 3.
        delay (int, optional): The number of observations between the values of a row.
            Defaults to 1.
        normalize (bool, optional): Whether to divide by log2(order), so the entropy lies
            between 0 and 1. Defaults to True.

    Returns:
        float: The SVD entropy in bits, or normalized.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    embedded = delay_embedding(values, order, delay)
    if len(embedded) == 0:
        return np.nan

    singular_values = np.linalg.svd(embedded, compute_uv=False)
    if singular_values.sum() <= 0:
        return np.nan

    shares = singular_values / singular_values.sum()
    positive = shares[shares > 0]
    entropy = -np.sum(positive * np.log2(positive))

    return float(entropy / np.log2(order) if normalize else entropy)


def encode_series(
    data: pd.Series | np.ndarray, encoding: str = "binary", bins: int = 10
) -> np.ndarray:
    """
    Encode a series of returns into a message of discrete symbols (López de Prado, 2018,
    section 18.6): the direction or size of every return as a symbol.

    The message is the input of the Shannon, Lempel-Ziv and Kontoyiannis entropies and the
    Lempel-Ziv complexity. The encodings are:

    - Binary: 1 for a positive and 0 for a negative return; returns of exactly zero are left
      out, as they carry no direction.
    - Quantile: the quantile of the series the return falls in, 0 to bins - 1, so every
      symbol is about equally frequent.
    - Sigma: the bucket of one standard deviation (ddof=1) the return falls in, counted from
      the lowest return, so the symbols keep the size of the moves.

    Args:
        data (pd.Series | np.ndarray): The returns.
        encoding (str, optional): The encoding. Options are: "binary" (the direction),
            "quantile" (the quantile of the return), "sigma" (the standard deviation bucket of
            the return). Defaults to "binary".
        bins (int, optional): The number of quantiles of the quantile encoding. Defaults
            to 10.

    Returns:
        np.ndarray: The symbols as integers.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]

    if encoding == "binary":
        values = values[values != 0]
        return (values > 0).astype(np.int64)
    if encoding == "quantile":
        if bins < 2:  # noqa: PLR2004
            raise ValueError("The quantile encoding needs at least 2 bins.")
        edges = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
        return np.searchsorted(edges, values, side="right").astype(np.int64)
    if encoding == "sigma":
        deviation = np.std(values, ddof=1) if len(values) > 1 else 0.0
        if not deviation > 0:
            return np.zeros(len(values), dtype=np.int64)
        return np.floor((values - values.min()) / deviation).astype(np.int64)

    raise InvalidArgumentError(
        f"encoding must be one of 'binary', 'quantile' or 'sigma', "
        f"not {encoding!r}. Please pass one of these as encoding."
    )


def get_shannon_entropy(symbols: np.ndarray, word_length: int = 1) -> float:
    """
    Calculate the plug-in (maximum likelihood) estimate of the Shannon entropy rate of a
    message: the entropy of the words of `word_length` symbols, per symbol.

    With a word length of 1 it is the Shannon entropy of the symbols. Longer words also
    capture the dependence between consecutive symbols, but need a long message: the
    estimate is biased downwards when the words are not each seen many times.

    The formula is as follows:

    - p(w) = the share of the N - L + 1 overlapping words of L symbols that equal w
    - H = -Sum(p(w) * log2(p(w))) / L

    Reference: López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley,
    chapter 18.3 (Snippet 18.1). Shannon, C.E. (1948). "A Mathematical Theory of
    Communication." Bell System Technical Journal, 27(3), 379-423.

    Shares similar logic:
    - RiskLabAI, plug_in_entropy_estimator: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/entropy_features/plug_in.py#L10

    Notes:
        - Every overlapping word is counted, the last one included, unlike the loop of López
          de Prado's Snippet 18.1, which stops one word short.

    Args:
        symbols (np.ndarray): The message, see `encode_series`.
        word_length (int, optional): The word length L. Defaults to 1.

    Returns:
        float: The entropy in bits per symbol.
    """
    words = delay_embedding(np.asarray(symbols, dtype=float), word_length)
    if len(words) == 0:
        return np.nan

    shares = np.unique(words, axis=0, return_counts=True)[1] / len(words)

    return float(-np.sum(shares * np.log2(shares)) / word_length)


def get_lempel_ziv_entropy(symbols: np.ndarray) -> float:
    """
    Calculate the Lempel-Ziv entropy of a message: the number of distinct phrases of its
    Lempel-Ziv (1978) parsing divided by its length.

    The message is cut into the shortest phrases not seen before; a redundant message is
    built from few, long phrases and has a low value, a random one from many short phrases.

    The formula is as follows:

    - Phrases: from the start, each phrase is the shortest substring that is not yet a phrase
    - H = number of phrases / N

    Reference: López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley,
    chapter 18.4 (Snippet 18.2). Ziv, J. & Lempel, A. (1978). "Compression of individual
    sequences via variable-rate coding." IEEE Transactions on Information Theory, 24(5),
    530-536.

    Shares similar logic:
    - RiskLabAI, lempel_ziv_entropy: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/entropy_features/lempel_ziv.py#L6

    Args:
        symbols (np.ndarray): The message, see `encode_series`.

    Returns:
        float: The number of phrases per symbol.
    """
    message = "".join(map(chr, np.asarray(symbols, dtype=np.int64) + 0x100))
    if not message:
        return np.nan

    phrases = {message[0]}
    start = 1
    while start < len(message):
        end = start + 1
        while end <= len(message) and message[start:end] in phrases:
            end += 1
        if end <= len(message):
            phrases.add(message[start:end])
        start = end

    return len(phrases) / len(message)


def get_kontoyiannis_entropy(symbols: np.ndarray) -> float:
    """
    Calculate the Kontoyiannis entropy of a message: the increasing-window estimate of its
    entropy rate from the lengths of the longest matches with the past.

    At every position the estimator looks for the longest stretch that already occurred in
    the message before: in a redundant message these matches are long and the entropy low,
    in a random message short.

    The formula is as follows:

    - L_i = 1 + the longest l <= i such that the l symbols from position i also start at a
      position j < i, i = 1, ..., N / 2 (positions counted from 0)
    - H = Mean(log2(i + 1) / L_i)

    Every position i is compared with the i symbols before it and looks i symbols ahead, so
    the estimate runs over the first half of the message.

    Reference: Kontoyiannis, I., Algoet, P.H., Suhov, Y.M. & Wyner, A.J. (1998).
    "Nonparametric entropy estimation for stationary processes and random fields, with
    applications to English text." IEEE Transactions on Information Theory, 44(3), 1319-1327.
    López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley, chapter 18.4
    (Snippets 18.3 and 18.4).

    Shares similar logic:
    - RiskLabAI, kontoyiannis_entropy: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/features/entropy_features/kontoyiannis.py#L57

    Args:
        symbols (np.ndarray): The message, see `encode_series`.

    Returns:
        float: The entropy in bits per symbol.
    """
    message = "".join(map(chr, np.asarray(symbols, dtype=np.int64) + 0x100))
    if len(message) < 2:  # noqa: PLR2004
        return np.nan

    ratios = []
    for position in range(1, len(message) // 2 + 1):
        length = 0
        # A match of l symbols from position starts before position, so it lies within the first position + l - 1 symbols.  # noqa: E501
        while length < position and (
            message.find(
                message[position : position + length + 1], 0, position + length
            )
            != -1
        ):
            length += 1
        ratios.append(np.log2(position + 1) / (length + 1))

    return float(np.mean(ratios))


def get_lempel_ziv_complexity(
    symbols: np.ndarray, normalize: bool = True, alphabet: int | None = None
) -> float:
    """
    Calculate the Lempel-Ziv complexity of a message: the number of phrases of its
    Lempel-Ziv (1976) parsing, in which every phrase is the longest stretch that can be
    copied from earlier in the message plus one new symbol.

    A periodic message needs few phrases and a random one many. Normalized, the count
    approaches 1 for a random message of independent, equally likely symbols, and falls
    towards 0 for a regular one.

    The formula is as follows:

    - c(N) = the number of phrases
    - Normalized c(N) = c(N) * log_k(N) / N

    with k the number of distinct symbols.

    Reference: Lempel, A. & Ziv, J. (1976). "On the complexity of finite sequences." IEEE
    Transactions on Information Theory, 22(1), 75-81. Kaspar, F. & Schuster, H.G. (1987).
    "Easily calculable measure for the complexity of spatiotemporal patterns." Physical Review
    A, 36(2), 842-848.

    Referenced against:
    - antropy, lziv_complexity: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L915

    Args:
        symbols (np.ndarray): The message, see `encode_series`.
        normalize (bool, optional): Whether to normalize by N / log_k(N). Defaults to True.
        alphabet (int | None, optional): The number of possible symbols k. Defaults to
            None, which uses the number of distinct symbols of the message.

    Returns:
        float: The (normalized) number of phrases.
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    message = "".join(map(chr, symbols + 0x100))
    number = len(message)
    if number == 0:
        return np.nan

    phrases, start = 0, 0
    while start < number:
        length = 1
        # The phrase copies the longest stretch that starts earlier (it may overlap itself) and adds one symbol.  # noqa: E501
        while (
            start + length <= number
            and message.find(message[start : start + length], 0, start + length - 1)
            != -1
        ):
            length += 1
        phrases += 1
        start += length

    if not normalize:
        return float(phrases)

    alphabet = alphabet if alphabet is not None else len(np.unique(symbols))
    if alphabet < 2 or number < 2:  # noqa: PLR2004
        return np.nan

    return float(phrases * np.log(number) / np.log(alphabet) / number)


def get_hjorth_parameters(data: pd.Series | np.ndarray) -> pd.Series:
    """
    Calculate the Hjorth parameters of a series, its Activity, Mobility and Complexity, and
    the number of times it crosses zero.

    The Activity is the variance; the Mobility the standard deviation of the changes relative
    to that of the series, a mean frequency; the Complexity how much the Mobility of the
    changes exceeds that of the series, which is 1 for a pure sine wave and grows as the
    series looks less like one.

    The formula is as follows:

    - Activity = Var(x)
    - Mobility = Sqrt(Var(dx) / Var(x))
    - Complexity = Mobility(dx) / Mobility(x) = Sqrt(Var(d2x) / Var(dx)) / Mobility(x)
    - Zero Crossings = the number of sign changes of x, values of exactly 0 skipped

    with dx the first and d2x the second differences and Var the sample variance (ddof=1).

    Reference: Hjorth, B. (1970). "EEG analysis based on time domain properties."
    Electroencephalography and Clinical Neurophysiology, 29(3), 306-310.

    Referenced against:
    - antropy, hjorth_params: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L1126
    - antropy, num_zerocross: https://github.com/raphaelvallat/antropy/blob/v0.2.3/src/antropy/entropy.py#L1033

    Notes:
        - Unlike this, antropy uses the population variance (ddof=0), so its Mobility and
        Complexity differ from these by the ratio of the numbers of observations of x, dx
        and d2x (less than 0.1% for 1,000 observations).
        - antropy counts a value of 0 as positive, so a return of 0 between two negative ones
          adds two crossings there and none here.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. returns.

    Returns:
        pd.Series: Activity, Mobility, Complexity and Zero Crossings.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    names = ["Activity", "Mobility", "Complexity", "Zero Crossings"]
    if len(values) < 4:  # noqa: PLR2004
        return pd.Series(np.nan, index=names)

    changes = np.diff(values)
    variance = np.var(values, ddof=1)
    change_variance = np.var(changes, ddof=1)
    second_variance = np.var(np.diff(changes), ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        mobility = np.sqrt(change_variance / variance)
        complexity = np.sqrt(second_variance / change_variance) / mobility

    signs = np.sign(values[values != 0])
    crossings = int(np.sum(signs[1:] != signs[:-1]))

    return pd.Series(
        [variance, mobility, complexity, crossings], index=names, dtype=float
    )


def get_share_entropy(
    data: pd.Series | np.ndarray,
    window: int | None = None,
    base: float = 2.0,
    normalize: bool = False,
) -> np.ndarray | float:
    """
    Calculate the Share Entropy of a series: the Shannon entropy of its values taken as shares
    of their sum, which is highest when the values are equal and falls as they spread out.

    For prices, a lower entropy points to a larger move within the window. The entropy is at
    its maximum, log(n), when all n values are equal; in base 2 it is measured in bits.

    The formula is as follows:

    - P(i) = x(i) / Sum(x), over the n values of the window
    - Entropy = -Sum(P * log_base(P))
    - Normalized Entropy = -Sum(P * ln(P)) / ln(n)

    Also known as: ENTP, Shannon entropy of the prices, information entropy.

    Reference: Shannon, C.E. (1948). "A Mathematical Theory of Communication." Bell System
    Technical Journal, 27(3), 379-423.

    Referenced against:
    - pandas-ta-classic, entropy: https://github.com/xgboosted/pandas-ta-classic/blob/0.9.0/pandas_ta_classic/statistics/entropy.py#L13

    Notes:
        - The rolling entropy equals pandas-ta-classic's entropy (window 10, base 2).
        - The shares are only probabilities for positive values, such as prices: a window
          with a negative value or a sum of 0 has no entropy.

    Args:
        data (pd.Series | np.ndarray): The series, e.g. prices.
        window (int | None): The number of values of a rolling window. Defaults to None,
            which takes every value and returns one number.
        base (float): The base of the logarithm. Defaults to 2.0, in bits.
        normalize (bool): Whether to divide the entropy by its maximum, log(n), so it lies
            between 0 and 1 whatever the base. Defaults to False.

    Returns:
        np.ndarray | float: The entropy of every window, the first on the window's last
        value, or one number without a window.
    """
    values = np.asarray(data, dtype=float)
    length = window if window is not None else len(values)

    if not 0 < length <= len(values):
        return np.array([]) if window is not None else np.nan

    windows = sliding_window_view(values, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = windows / windows.sum(axis=1, keepdims=True)
        terms = np.where(shares == 0, 0.0, -shares * np.log(shares))
        entropy = terms.sum(axis=1) / np.log(length if normalize else base)
    entropy = np.where(
        (shares < 0).any(axis=1) | ~np.isfinite(entropy), np.nan, entropy
    )

    return entropy if window is not None else float(entropy[-1])
