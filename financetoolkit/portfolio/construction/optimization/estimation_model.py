"""Portfolio Estimation Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from financetoolkit.risk.dependence import copula_model

# pylint: disable=too-many-arguments,too-many-locals

# The smallest eigenvalue a repaired correlation matrix keeps, as in skfolio's cov_nearest.
MINIMUM_EIGENVALUE = 1e-13

# The lowest codependence that the tail distance -ln(λ) is taken of, as in Riskfolio-Lib, so
# that two assets that never fall into their tails together get a large but finite distance.
MINIMUM_TAIL_DEPENDENCE = 1e-8

GERBER_STATISTICS = (0, 1, 2)
COMOMENT_METHODS = ("sample", "exponential", "semi")
BIN_METHODS = ("freedman", "hacine_gharbi")
BOOTSTRAP_METHODS = ("stationary", "circular", "moving", "normal")


def get_james_stein_expected_returns(
    returns: pd.DataFrame,
) -> tuple[pd.Series, float]:
    """
    The James-Stein estimator of the expected returns: every average return pulled
    towards the grand mean, the average of the average returns, by a shrinkage intensity
    that grows as the averages differ less compared to their estimation error.

    With μ̂ the average returns of the T periods of the N assets, μ̄ their average and
    λ_1, ..., λ_N the eigenvalues of the sample covariance matrix S (ddof=1), the estimate
    is μ = (1 - β) μ̂ + β μ̄ 1 with β = (Σ λ_k - 2 max λ_k) / (T ‖μ̂ - μ̄ 1‖²), clipped to
    [0, 1] (the positive-part James-Stein estimator). This is the form of Meucci (2005)
    for an unknown covariance matrix, which skfolio's ShrunkMu (JAMES_STEIN) and
    Riskfolio-Lib's mean_vector (method="JS") use; they do not clip β, which only differs
    when the averages lie so close together that β leaves [0, 1]. With two assets
    Σ λ_k - 2 max λ_k is negative, so the estimator needs at least three assets to shrink.

    See James, W. and Stein, C. (1961), "Estimation with Quadratic Loss", Proceedings of
    the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1, 361-379,
    and Meucci, A. (2005), Risk and Asset Allocation, Springer, chapter 4.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.Series, float]: The expected return per asset per period and the
            shrinkage intensity β between 0 (the averages) and 1 (the grand mean).
    """
    average = returns.mean().to_numpy(dtype=float)
    grand_mean = average.mean()
    eigenvalues = np.linalg.eigvalsh(returns.cov().to_numpy(dtype=float))
    dispersion = float(np.sum((average - grand_mean) ** 2))
    if dispersion == 0:
        return pd.Series(average, index=returns.columns), 1.0
    shrinkage = (eigenvalues.sum() - 2 * eigenvalues.max()) / (
        len(returns) * dispersion
    )
    shrinkage = float(min(max(shrinkage, 0.0), 1.0))
    return (
        pd.Series(
            (1 - shrinkage) * average + shrinkage * grand_mean, index=returns.columns
        ),
        shrinkage,
    )


def get_bodnar_okhrin_parolya_expected_returns(
    returns: pd.DataFrame,
) -> tuple[pd.Series, float, float]:
    """
    The shrinkage estimator of the expected returns of Bodnar, Okhrin and Parolya (2019):
    μ = α μ̂ + β b, a combination of the average returns μ̂ and a target b with the
    weights α and β that are optimal in the sense of the quadratic loss when the number
    of assets grows with the number of periods. The target is the grand mean, the average
    of the average returns; since α and β scale with b, any multiple of a vector of ones
    gives the same estimate.

    With S the sample covariance matrix (ddof=1), T periods and N assets, u = μ̂'S⁻¹μ̂,
    v = μ̂'S⁻¹b and w = b'S⁻¹b:

    - α = ((u - N / (T - N)) w - v²) / (u w - v²)
    - β = (1 - α) v / w

    This equals skfolio's ShrunkMu (BODNAR_OKHRIN) and Riskfolio-Lib's mean_vector
    (method="BOP") with the grand mean as the target.

    See Bodnar, T., Okhrin, O. and Parolya, N. (2019), "Optimal Shrinkage Estimator for
    High-Dimensional Mean Vector", Journal of Multivariate Analysis, 170, 63-79.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.Series, float, float]: The expected return per asset per period and the
            weights α of the averages and β of the target.
    """
    periods, assets = returns.shape
    if periods <= assets:
        raise ValueError(
            "The Bodnar-Okhrin-Parolya expected returns need more periods than assets."
        )
    average = returns.mean().to_numpy(dtype=float)
    target = np.full(assets, average.mean())
    inverse = np.linalg.inv(returns.cov().to_numpy(dtype=float))
    u = float(average @ inverse @ average)
    v = float(average @ inverse @ target)
    w = float(target @ inverse @ target)
    alpha = ((u - assets / (periods - assets)) * w - v**2) / (u * w - v**2)
    beta = (1 - alpha) * v / w
    return (
        pd.Series(alpha * average + beta * target, index=returns.columns),
        float(alpha),
        float(beta),
    )


def get_shrunk_covariance(
    returns: pd.DataFrame, shrinkage: float = 0.1
) -> pd.DataFrame:
    """
    The covariance matrix shrunk with a fixed intensity towards the identity matrix scaled
    by the average variance: Σ = (1 - δ) S + δ (tr(S) / N) I, with S the sample covariance
    matrix (ddof=1) of the N assets. Unlike Ledoit-Wolf and OAS, the intensity δ is
    chosen rather than estimated.

    This equals PyPortfolioOpt's CovarianceShrinkage.shrunk_covariance, which defaults to
    δ = 0.2. scikit-learn's ShrunkCovariance, skfolio's ShrunkCovariance and
    Riskfolio-Lib's covar_matrix (method="shrunk") default to δ = 0.1, the default here,
    and shrink the maximum likelihood covariance (divided by T rather than T - 1), which
    is T / (T - 1) times smaller.

    See Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Journal of Multivariate Analysis, 88(2), 365-411, for the
    target.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        shrinkage (float, optional): The shrinkage intensity δ between 0 (the sample
            covariance) and 1 (the scaled identity). Defaults to 0.1.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError("The shrinkage intensity must lie between 0 and 1.")
    sample = returns.cov().to_numpy(dtype=float)
    target = np.trace(sample) / len(sample) * np.eye(len(sample))
    return pd.DataFrame(
        (1 - shrinkage) * sample + shrinkage * target,
        index=returns.columns,
        columns=returns.columns,
    )


def get_nearest_correlation(correlation: np.ndarray) -> np.ndarray:
    """
    The correlation matrix itself when it is positive definite, otherwise the nearest one
    by eigenvalue clipping: the eigenvalues below 1e-13 are raised to it and the matrix is
    rescaled to a unit diagonal, as skfolio's cov_nearest does without Higham's algorithm.

    See Higham, N. J. (2002), "Computing the Nearest Correlation Matrix - A Problem from
    Finance", IMA Journal of Numerical Analysis, 22(3), 329-343.

    Args:
        correlation (np.ndarray): A symmetric matrix with a unit diagonal.

    Returns:
        np.ndarray: A positive definite correlation matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues.min() >= MINIMUM_EIGENVALUE:
        return correlation
    repaired = (
        eigenvectors @ np.diag(np.clip(eigenvalues, MINIMUM_EIGENVALUE, None))
    ) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def get_gerber_covariance(
    returns: pd.DataFrame, threshold: float = 0.5, statistic: int = 1
) -> pd.DataFrame:
    """
    The Gerber covariance matrix: Σ = D G D, with D the standard deviations (ddof=1) and G
    the Gerber statistic, a correlation that counts only the periods in which returns move
    beyond a threshold. A return r_ti is up when r_ti ≥ c σ_i and down when r_ti ≤ -c σ_i,
    with c the threshold; small moves, which are mostly noise, do not count, and neither
    does the size of large ones, so a few extreme returns do not dominate it.

    Per pair of assets, n_c counts the periods in which both moved beyond the threshold in
    the same direction (concordant), n_d those in opposite directions (discordant) and n_n
    those in which neither did. With U and D the T × N indicators of up and down moves:

    - Statistic 0: g_ij = (n_c - n_d) / (n_c + n_d), the original pairwise statistic,
      which need not give a positive semidefinite matrix.
    - Statistic 1: g_ij = (n_c - n_d) / (T - n_n), which is skfolio's GerberCovariance
      (psd_variant=True, its default) and Riskfolio-Lib's gerber_cov_stat1.
    - Statistic 2: G = H / √(diag(H) diag(H)'), with H = U'U + D'D - U'D - D'U, the
      matrix form that is positive semidefinite by construction, Riskfolio-Lib's
      gerber_cov_stat2 and skfolio's GerberCovariance with psd_variant=False.

    A matrix that is not positive definite is replaced by the nearest one by clipping the
    eigenvalues of G (see get_nearest_correlation), as skfolio does. The thresholds apply
    to the returns themselves, not to their deviations from the mean, as in the paper and
    both libraries. Both libraries take the standard deviations with ddof=0; here they
    take ddof=1, which equals their result with the threshold multiplied by √(T / (T - 1))
    and the covariance by T / (T - 1).

    See Gerber, S., Markowitz, H. M., Ernst, P. A., Miao, Y., Javid, B. and Sargen, P.
    (2022), "The Gerber Statistic: A Robust Co-Movement Measure for Portfolio
    Optimization", The Journal of Portfolio Management, 48(3), 87-102.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        threshold (float, optional): The threshold c, in standard deviations, between 0
            and 1. Defaults to 0.5.
        statistic (int, optional): The Gerber statistic: 0, 1 or 2. Defaults to 1.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    if not 0 < threshold < 1:
        raise ValueError("The Gerber threshold must lie between 0 and 1.")
    if statistic not in GERBER_STATISTICS:
        raise ValueError("The Gerber statistic must be 0, 1 or 2.")
    values = returns.to_numpy(dtype=float)
    volatility = values.std(axis=0, ddof=1)
    if np.any(volatility <= 0):
        raise ValueError("Every asset needs returns that vary.")
    up = (values >= threshold * volatility).astype(float)
    down = (values <= -threshold * volatility).astype(float)
    concordant = up.T @ up + down.T @ down
    discordant = up.T @ down + down.T @ up
    difference = concordant - discordant
    if statistic == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            gerber = difference / (concordant + discordant)
    elif statistic == 1:
        neither = 1 - up - down
        with np.errstate(divide="ignore", invalid="ignore"):
            gerber = difference / (len(values) - neither.T @ neither)
    else:
        scale = np.sqrt(np.diag(difference))
        with np.errstate(divide="ignore", invalid="ignore"):
            gerber = difference / np.outer(scale, scale)
    if np.isfinite(gerber).all():
        gerber = get_nearest_correlation(gerber)
    return pd.DataFrame(
        gerber * np.outer(volatility, volatility),
        index=returns.columns,
        columns=returns.columns,
    )


def get_detoned_covariance(returns: pd.DataFrame, components: int = 1) -> pd.DataFrame:
    """
    The detoned covariance matrix: the sample covariance (ddof=1) with the first principal
    components of its correlation matrix, the market, removed. With λ_k and v_k the
    eigenvalues and eigenvectors of the correlation matrix C in descending order, the
    detoned correlation is C - Σ_(k ≤ m) λ_k v_k v_k' for the m components, rescaled to a
    unit diagonal and turned back into a covariance with the sample standard deviations.

    The market component makes all assets look alike; without it, the clusters of assets
    that move together for other reasons stand out. The detoned matrix has rank N - m, so
    it is singular: it suits clustering (codependence="detoned") but cannot be inverted
    by an optimizer.

    This equals skfolio's DetoneCovariance on the sample covariance (which also replaces
    the zero eigenvalues by 1e-13) and López de Prado's detoning applied to the
    correlation matrix itself rather than to a denoised one.

    See López de Prado, M. (2020), Machine Learning for Asset Managers, Cambridge
    University Press, chapter 2.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        components (int, optional): The number m of principal components removed.
            Defaults to 1.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    assets = returns.shape[1]
    if not 1 <= components < assets:
        raise ValueError(
            f"The number of components removed must lie between 1 and {assets - 1}."
        )
    sample = returns.cov().to_numpy(dtype=float)
    volatility = np.sqrt(np.diag(sample))
    correlation = sample / np.outer(volatility, volatility)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    market = eigenvectors[:, :components]
    correlation = correlation - market @ np.diag(eigenvalues[:components]) @ market.T
    scale = np.sqrt(np.diag(correlation))
    correlation = correlation / np.outer(scale, scale)
    return pd.DataFrame(
        correlation * np.outer(volatility, volatility),
        index=returns.columns,
        columns=returns.columns,
    )


def get_distance_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    The distance correlation of every pair of assets, which is zero only when their
    returns are independent, so it also captures non-linear co-movement that the Pearson
    correlation misses. It lies between 0 and 1.

    With a_st = |x_s - x_t| the distances between the returns of one asset in periods s
    and t, A the doubly centred distances (A_st = a_st - ā_s. - ā_.t + ā_..), and B the
    same for the other asset:

    - dCov²(x, y) = (1 / T²) Σ A_st B_st
    - dCor(x, y) = √(dCov²(x, y) / √(dCov²(x, x) dCov²(y, y)))

    This equals skfolio's DistanceCorrelation and Riskfolio-Lib's dcorr_matrix. It builds
    a T × T matrix per asset, so it grows with the square of the number of periods.

    See Székely, G. J., Rizzo, M. L. and Bakirov, N. K. (2007), "Measuring and Testing
    Dependence by Correlation of Distances", The Annals of Statistics, 35(6), 2769-2794.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        pd.DataFrame: The distance correlation matrix, with ones on the diagonal.
    """
    values = returns.to_numpy(dtype=float)
    assets = values.shape[1]

    def centred(column: np.ndarray) -> np.ndarray:
        distance = squareform(pdist(column[:, None]))
        return (
            distance
            - distance.mean(axis=0)[None, :]
            - distance.mean(axis=1)[:, None]
            + distance.mean()
        )

    variance = np.zeros(assets)
    for i in range(assets):
        first = centred(values[:, i])
        variance[i] = (first**2).sum()
    matrix = np.eye(assets)
    for i in range(assets):
        first = centred(values[:, i])
        for j in range(i + 1, assets):
            covariance = (first * centred(values[:, j])).sum()
            matrix[i, j] = matrix[j, i] = np.sqrt(
                max(covariance, 0.0) / np.sqrt(variance[i] * variance[j])
            )
    return pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)


def get_number_of_bins(
    first: np.ndarray, second: np.ndarray | None = None, method: str = "freedman"
) -> int:
    """
    The number of bins of the histograms that the mutual information is estimated from.

    - "freedman": the rule of Freedman and Diaconis (1981), ⌈(max - min) / (2 IQR /
      T^(1/3))⌉ per series (5 for a series without an interquartile range), and the
      larger of the two for a pair, as skfolio's MutualInformation does by default.
    - "hacine_gharbi": the rule of Hacine-Gharbi et al. (2012) for one series,
      round(z / 6 + 2 / (3 z) + 1 / 3) with z = (8 + 324 T + 12 √(36 T + 729 T²))^(1/3),
      and of Hacine-Gharbi and Ravier (2018) for a pair with correlation ρ,
      round(√(1 + √(1 + 24 T / (1 - ρ²))) / √2), Riskfolio-Lib's numBins and the choice
      of López de Prado (2020).

    See Freedman, D. and Diaconis, P. (1981), "On the Histogram as a Density Estimator:
    L2 Theory", Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 57,
    453-476, and Hacine-Gharbi, A. and Ravier, P. (2018), "A Binning Formula of
    Bi-histogram for Joint Entropy Estimation Using Mean Square Error Minimization",
    Pattern Recognition Letters, 101, 21-28.

    Args:
        first (np.ndarray): The returns of the first asset.
        second (np.ndarray | None, optional): The returns of the second asset, for a
            pair. Defaults to None.
        method (str, optional): "freedman" or "hacine_gharbi". Defaults to "freedman".

    Returns:
        int: The number of bins.
    """
    if method not in BIN_METHODS:
        raise ValueError(
            f"The bins must be a number or one of {', '.join(BIN_METHODS)}, not "
            f"'{method}'."
        )
    periods = len(first)
    if method == "hacine_gharbi":
        correlation = 1.0 if second is None else float(np.corrcoef(first, second)[0, 1])
        if correlation >= 1:
            z = (
                8 + 324 * periods + 12 * (36 * periods + 729 * periods**2) ** 0.5
            ) ** (1 / 3)
            return int(np.round(z / 6 + 2 / (3 * z) + 1 / 3))
        return int(
            np.round(
                2**-0.5 * (1 + (1 + 24 * periods / (1 - correlation**2)) ** 0.5) ** 0.5
            )
        )

    bins = []
    for series in (first,) if second is None else (first, second):
        lower, upper = np.percentile(series, [25, 75])
        width = 2 * (upper - lower) / periods ** (1 / 3)
        bins.append(
            5
            if width == 0
            else round(max(1, np.ceil((series.max() - series.min()) / width)))
        )
    return int(max(bins))


def get_mutual_information_matrix(
    returns: pd.DataFrame, bins: str | int = "freedman", normalize: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The mutual information and the variation of information of every pair of assets,
    estimated from histograms of their returns. The mutual information I(X, Y) =
    H(X) + H(Y) - H(X, Y) measures how much knowing one return tells about the other,
    linear or not; the variation of information VI(X, Y) = H(X) + H(Y) - 2 I(X, Y) is the
    information they do not share, a metric distance between the assets.

    With p_ab the share of the periods in bin a of X and bin b of Y of the joint
    histogram, and p_a, p_b its margins (natural logarithms):

    - H(X) = -Σ p_a ln p_a, H(Y) likewise
    - I(X, Y) = Σ p_ab ln(p_ab / (p_a p_b))
    - normalized: I(X, Y) / min(H(X), H(Y)) between 0 and 1, and VI(X, Y) / H(X, Y)
      between 0 (identical) and 1 (independent)

    This equals skfolio's MutualInformation (codependence and distance) and
    Riskfolio-Lib's mutual_info_matrix and var_info_matrix with the same number of bins.

    See Meilă, M. (2007), "Comparing Clusterings - An Information Based Distance",
    Journal of Multivariate Analysis, 98(5), 873-895, and López de Prado, M. (2020),
    Machine Learning for Asset Managers, Cambridge University Press, chapter 3.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        bins (str | int, optional): The number of bins of every histogram, or the rule
            that sets it per pair: "freedman" or "hacine_gharbi" (see
            get_number_of_bins). Defaults to "freedman".
        normalize (bool, optional): Whether to normalize the mutual information by the
            smaller entropy and the variation of information by the joint entropy.
            Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The mutual information matrix and the
            variation of information matrix.
    """
    values = returns.to_numpy(dtype=float)
    assets = values.shape[1]
    information = np.zeros((assets, assets))
    variation = np.zeros((assets, assets))
    for i in range(assets):
        for j in range(i, assets):
            count = (
                bins
                if isinstance(bins, int)
                else get_number_of_bins(
                    values[:, i], None if i == j else values[:, j], bins
                )
            )
            joint = np.histogram2d(values[:, i], values[:, j], count)[0]
            joint = joint / joint.sum()
            first, second = joint.sum(axis=1), joint.sum(axis=0)
            entropy_first = -np.sum(first[first > 0] * np.log(first[first > 0]))
            entropy_second = -np.sum(second[second > 0] * np.log(second[second > 0]))
            filled = joint > 0
            mutual = float(
                np.sum(
                    joint[filled]
                    * np.log(joint[filled] / np.outer(first, second)[filled])
                )
            )
            distance = entropy_first + entropy_second - 2 * mutual
            if normalize:
                joint_entropy = entropy_first + entropy_second - mutual
                distance = distance / joint_entropy if joint_entropy > 0 else 0.0
                smallest = min(entropy_first, entropy_second)
                mutual = mutual / smallest if smallest > 0 else 1.0
            information[i, j] = information[j, i] = mutual
            variation[i, j] = variation[j, i] = max(distance, 0.0)
    return (
        pd.DataFrame(information, index=returns.columns, columns=returns.columns),
        pd.DataFrame(variation, index=returns.columns, columns=returns.columns),
    )


def get_lower_tail_dependence_matrix(
    returns: pd.DataFrame, tail_level: float = 0.05
) -> pd.DataFrame:
    """
    The lower tail dependence of every pair of assets: the share of the periods in which
    one asset fell into its lowest tail_level of returns in which the other did too,
    λ_ij = P(r_j ≤ q_j(α) | r_i ≤ q_i(α)) at α = tail_level, with q(α) the empirical
    quantile. It measures how often assets crash together, which the correlation, an
    average over all periods, does not show.

    Each pair comes from the empirical estimator of get_tail_dependence_coefficient
    (quantiles with linear interpolation). Riskfolio-Lib's ltdi_matrix takes the k-th
    lowest return as the quantile, with k = ⌈T α⌉, and divides by k; both count the same
    periods whenever the k-th lowest return is the last one at or below the interpolated
    quantile.

    See Embrechts, P., McNeil, A. and Straumann, D. (2002), "Correlation and Dependence
    in Risk Management: Properties and Pitfalls", in Risk Management: Value at Risk and
    Beyond, Cambridge University Press, 176-223.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        tail_level (float, optional): The share α of the lowest returns that forms the
            tail, between 0 and 0.5. Defaults to 0.05.

    Returns:
        pd.DataFrame: The lower tail dependence matrix, with ones on the diagonal.
    """
    if not 0 < tail_level < 0.5:  # noqa: PLR2004
        raise ValueError("The tail level must lie between 0 and 0.5.")
    columns = returns.columns
    matrix = np.eye(len(columns))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            matrix[i, j] = matrix[j, i] = copula_model.get_tail_dependence_coefficient(
                returns.iloc[:, i], returns.iloc[:, j], q=1 - tail_level
            )["Lower Tail Dependence"]
    return pd.DataFrame(matrix, index=columns, columns=columns)


def get_fast_threshold_clusters(
    correlation: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    """
    The Fast Threshold Clustering Algorithm (FTCA) of Varadi (2013): a greedy clustering
    on a correlation threshold, which needs no number of clusters.

    Among the assets not yet clustered, take the one with the highest average correlation
    to the others (H) and the one with the lowest (L). If the correlation of H and L is
    above the threshold, they form a cluster with every remaining asset whose average
    correlation to H and L is above it. Otherwise H forms a cluster with every remaining
    asset correlated above the threshold with H, and then L with every asset still
    remaining that is correlated above it with L. This repeats until every asset belongs
    to a cluster; a last single asset forms its own.

    This equals ffn's calc_ftca, which clusters on the Pearson correlation of the
    returns; any codependence matrix can be passed here.

    See Varadi, D. (2013), "The Fast Threshold Clustering Algorithm (FTCA)", CSS
    Analytics.

    Args:
        correlation (pd.DataFrame): The correlation (or another codependence) matrix of
            the assets.
        threshold (float, optional): The correlation above which assets share a cluster.
            Defaults to 0.5.

    Returns:
        np.ndarray: The cluster of every asset, numbered from 0 in the order the clusters
            are formed, in the order of the correlation matrix.
    """
    matrix = correlation.to_numpy(dtype=float)
    remaining = list(range(len(matrix)))
    labels = np.full(len(matrix), -1)
    cluster = 0

    while remaining:
        if len(remaining) == 1:
            labels[remaining.pop()] = cluster
            break
        average = matrix[np.ix_(remaining, remaining)].mean(axis=0)
        # A stable sort keeps the first asset on ties, as ffn's pandas sort does.
        order = np.argsort(average, kind="stable")
        low, high = remaining[order[0]], remaining[order[-1]]
        remaining.remove(low)
        remaining.remove(high)
        if matrix[high, low] > threshold:
            members = [low, high] + [
                asset
                for asset in remaining
                if (matrix[asset, high] + matrix[asset, low]) / 2 > threshold
            ]
            labels[members] = cluster
            remaining = [asset for asset in remaining if asset not in members]
            cluster += 1
        else:
            for anchor in (high, low):
                members = [anchor] + [
                    asset for asset in remaining if matrix[asset, anchor] > threshold
                ]
                labels[members] = cluster
                remaining = [asset for asset in remaining if asset not in members]
                cluster += 1
    return labels


def get_comoment_weights(periods: int, method: str, span: int) -> np.ndarray:
    """
    The weights of the periods in a co-moment: 1 / T each for "sample" and "semi", and
    (1 - α)^age normalized to sum to one, with α = 2 / (span + 1), for "exponential".

    Args:
        periods (int): The number of periods T.
        method (str): "sample", "exponential" or "semi".
        span (int): The span in periods of the "exponential" weights.

    Returns:
        np.ndarray: The weight of every period, oldest first.
    """
    if method not in COMOMENT_METHODS:
        raise ValueError(
            f"The method must be one of {', '.join(COMOMENT_METHODS)}, not '{method}'."
        )
    if method != "exponential":
        return np.full(periods, 1 / periods)
    if span < 1:
        raise ValueError("The span must be at least 1.")
    weights = (1 - 2 / (span + 1)) ** np.arange(periods - 1, -1, -1)
    return weights / weights.sum()


def get_coskewness_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    span: int = 180,
    standardize: bool = False,
) -> pd.DataFrame:
    """
    The coskewness matrix M3, the third co-moments of the returns: N rows and N² columns
    with m_ijk = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k), with μ the average returns
    and w_t the weight of period t. The skewness of a portfolio with weights w is
    w'M3(w ⊗ w) / σ_p³, so M3 is what a mean-variance-skewness optimization or a modified
    VaR of a portfolio needs.

    - "sample": w_t = 1 / T, PerformanceAnalytics' M3.MM (unbiased=FALSE) and
      CoSkewnessMatrix.
    - "exponential": w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent
      periods weigh most. PerformanceAnalytics' M3.ewma uses the same weights (its
      default λ = 0.97 is a span of about 65.7) on returns it assumes to have a zero
      mean; here the deviations are taken from the average return, as in the
      exponential covariance, so it equals M3.ewma of the demeaned returns.
    - "semi": the lower semi co-moment, with min(r_ti - μ_i, 0) in place of the
      deviations, so only returns below the average count (Cajas, Riskfolio-Lib).

    With standardize, every m_ijk is divided by σ_i σ_j σ_k, with σ² = Σ_t w_t
    (r_t - μ)² the weighted second moment of the full deviations, so the diagonal m_iii is
    the moment skewness of each asset.

    See Jondeau, E. and Rockinger, M. (2006), "Optimal Portfolio Allocation under Higher
    Moments", European Financial Management, 12(1), 29-55, and Martellini, L. and
    Ziemann, V. (2010), "Improved Estimates of Higher-Order Comoments and Implications for
    Portfolio Selection", The Review of Financial Studies, 23(4), 1467-1502.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): "sample", "exponential" or "semi". Defaults to "sample".
        span (int, optional): The span in periods of the "exponential" weights.
            Defaults to 180.
        standardize (bool, optional): Whether to divide by the product of the standard
            deviations. Defaults to False.

    Returns:
        pd.DataFrame: The coskewness matrix, one row per asset i and one column per pair
            of assets (j, k).
    """
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    weights = get_comoment_weights(periods, method, span)
    deviations = values - values.mean(axis=0)
    volatility = np.sqrt(weights @ deviations**2)
    if method == "semi":
        deviations = np.minimum(deviations, 0)
    pairs = (deviations[:, :, None] * deviations[:, None, :]).reshape(periods, -1)
    matrix = (deviations * weights[:, None]).T @ pairs
    if standardize:
        matrix = matrix / np.einsum("i,j,k->ijk", *[volatility] * 3).reshape(assets, -1)
    return pd.DataFrame(
        matrix,
        index=returns.columns,
        columns=pd.MultiIndex.from_product([returns.columns, returns.columns]),
    )


def get_cokurtosis_matrix(
    returns: pd.DataFrame,
    method: str = "sample",
    span: int = 180,
    standardize: bool = False,
) -> pd.DataFrame:
    """
    The cokurtosis matrix M4, the fourth co-moments of the returns: N rows and N³ columns
    with m_ijkl = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k)(r_tl - μ_l), with μ the
    average returns and w_t the weight of period t. The kurtosis of a portfolio with
    weights w is w'M4(w ⊗ w ⊗ w) / σ_p⁴.

    - "sample": w_t = 1 / T, PerformanceAnalytics' M4.MM and CoKurtosisMatrix; its
      N² × N² rearrangement is Riskfolio-Lib's cokurt_matrix (method="hist").
    - "exponential": w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent
      periods weigh most; PerformanceAnalytics' M4.ewma of the demeaned returns.
    - "semi": the lower semi cokurtosis, with min(r_ti - μ_i, 0) in place of the
      deviations, Riskfolio-Lib's cokurt_matrix (method="semi") rearranged.

    With standardize, every m_ijkl is divided by σ_i σ_j σ_k σ_l, with σ² = Σ_t w_t
    (r_t - μ)² the weighted second moment of the full deviations, so the diagonal m_iiii
    is the moment kurtosis of each asset (3 for a normal distribution, not the excess).

    See Jondeau, E. and Rockinger, M. (2006), "Optimal Portfolio Allocation under Higher
    Moments", European Financial Management, 12(1), 29-55, and Cajas, D. (2023),
    "Kurtosis and Semi-Kurtosis Portfolio Optimization", SSRN 4482613.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): "sample", "exponential" or "semi". Defaults to "sample".
        span (int, optional): The span in periods of the "exponential" weights.
            Defaults to 180.
        standardize (bool, optional): Whether to divide by the product of the standard
            deviations. Defaults to False.

    Returns:
        pd.DataFrame: The cokurtosis matrix, one row per asset i and one column per
            triple of assets (j, k, l).
    """
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    weights = get_comoment_weights(periods, method, span)
    deviations = values - values.mean(axis=0)
    volatility = np.sqrt(weights @ deviations**2)
    if method == "semi":
        deviations = np.minimum(deviations, 0)
    pairs = (deviations[:, :, None] * deviations[:, None, :]).reshape(periods, -1)
    # The N² × N² square form, whose rows (i, j) and columns (k, l) reshape to N × N³.
    matrix = ((pairs * weights[:, None]).T @ pairs).reshape(assets, -1)
    if standardize:
        matrix = matrix / np.einsum("i,j,k,l->ijkl", *[volatility] * 4).reshape(
            assets, -1
        )
    return pd.DataFrame(
        matrix,
        index=returns.columns,
        columns=pd.MultiIndex.from_product([returns.columns] * 3),
    )


def get_bootstrap_indices(
    periods: int,
    method: str = "stationary",
    block_size: int = 3,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """
    The periods of one bootstrapped history of the same length, drawn in blocks so that
    the dependence between consecutive returns (volatility clustering) survives.

    - "stationary": the stationary bootstrap of Politis and Romano (1994), blocks of a
      random length with a geometric distribution of mean block_size: every period
      continues the block with the next period (after the last, the first) with
      probability 1 - 1 / block_size and starts a new block at a random period otherwise.
    - "circular": blocks of block_size periods that start at a random period and wrap
      from the last period to the first (Politis and Romano, 1992).
    - "moving": blocks of block_size periods that start at a random period from which a
      whole block fits (Künsch, 1989).

    These are the schemes of the arch package that Riskfolio-Lib's bootstrapping uses.

    See Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap", Journal of
    the American Statistical Association, 89(428), 1303-1313, and Künsch, H. R. (1989),
    "The Jackknife and the Bootstrap for General Stationary Observations", The Annals of
    Statistics, 17(3), 1217-1241.

    Args:
        periods (int): The number of periods T of the history.
        method (str, optional): "stationary", "circular" or "moving". Defaults to
            "stationary".
        block_size (int, optional): The (average) number of periods in a block. Defaults
            to 3.
        generator (np.random.Generator | None, optional): The random number generator.
            Defaults to None, a new unseeded one.

    Returns:
        np.ndarray: The T periods, numbered from 0, of the bootstrapped history.
    """
    if not 1 <= block_size < periods:
        raise ValueError(
            f"The block size must lie between 1 and {periods - 1} for {periods} periods."
        )
    generator = generator if generator is not None else np.random.default_rng()
    if method == "stationary":
        starts = generator.random(periods) < 1 / block_size
        starts[0] = True
        indices = np.empty(periods, dtype=int)
        random = generator.integers(0, periods, size=periods)
        for t in range(periods):
            indices[t] = random[t] if starts[t] else (indices[t - 1] + 1) % periods
        return indices
    blocks = -(-periods // block_size)
    highest = periods if method == "circular" else periods - block_size + 1
    starts = generator.integers(0, highest, size=blocks)
    indices = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:periods]
    return indices % periods


def get_bootstrap_estimates(
    returns: pd.DataFrame,
    method: str = "stationary",
    simulations: int = 1000,
    block_size: int = 3,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    How uncertain the average returns and the covariance matrix are: both are estimated
    again on many simulated histories of the same length, and the spread of those
    estimates gives a confidence interval (box) for every average return and every
    covariance, and a standard error for every average return.

    - "stationary", "circular", "moving": the histories are block bootstraps of the
      returns (see get_bootstrap_indices), which keep the dependence between consecutive
      returns.
    - "normal": the histories are drawn from a multivariate normal distribution with the
      average returns and the sample covariance matrix.

    The bounds are the (1 - c) / 2 and (1 + c) / 2 percentiles of the estimates at
    confidence level c, and the standard error is their standard deviation (ddof=1).
    The averages and covariances are those of the returns per period (ddof=1). This
    follows Riskfolio-Lib's bootstrapping (q = 1 - c, n_sim, window) and
    normal_simulation, whose box and ellipsoid uncertainty sets come from these
    estimates; the draws differ from its arch-based ones, so results agree only in
    distribution. The bounds of the covariances are per entry and are not adjusted to
    form positive semidefinite matrices.

    See Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap", Journal of
    the American Statistical Association, 89(428), 1303-1313, and Tütüncü, R. H. and
    Koenig, M. (2004), "Robust Asset Allocation", Annals of Operations Research, 132,
    157-187.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): "stationary", "circular", "moving" or "normal". Defaults
            to "stationary".
        simulations (int, optional): The number of simulated histories. Defaults to 1000,
            as skfolio's bootstrap uncertainty sets (Riskfolio-Lib draws 6000).
        block_size (int, optional): The (average) number of periods in a block. Defaults
            to 3.
        confidence_level (float, optional): The confidence level of the bounds. Defaults
            to 0.95.
        seed (int | None, optional): The seed of the random number generator, so the
            result repeats. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Per asset the Expected Return
            (the average), its Lower and Upper bound and its Standard Error; the lower
            bounds of the covariance matrix; and its upper bounds, all per period.
    """
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"The method must be one of {', '.join(BOOTSTRAP_METHODS)}, not '{method}'."
        )
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    if simulations < 2:  # noqa: PLR2004
        raise ValueError("At least two simulations are needed.")
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    generator = np.random.default_rng(seed)
    average = values.mean(axis=0)
    sample = np.cov(values, rowvar=False)

    means = np.empty((simulations, assets))
    covariances = np.empty((simulations, assets, assets))
    for simulation in range(simulations):
        history = (
            generator.multivariate_normal(average, sample, size=periods)
            if method == "normal"
            else values[get_bootstrap_indices(periods, method, block_size, generator)]
        )
        means[simulation] = history.mean(axis=0)
        covariances[simulation] = np.cov(history, rowvar=False)

    percentiles = [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]
    lower, upper = np.percentile(means, percentiles, axis=0)
    covariance_lower, covariance_upper = np.percentile(covariances, percentiles, axis=0)
    columns = returns.columns
    return (
        pd.DataFrame(
            {
                "Expected Return": average,
                "Lower": lower,
                "Upper": upper,
                "Standard Error": means.std(axis=0, ddof=1),
            },
            index=columns,
        ),
        pd.DataFrame(covariance_lower, index=columns, columns=columns),
        pd.DataFrame(covariance_upper, index=columns, columns=columns),
    )
