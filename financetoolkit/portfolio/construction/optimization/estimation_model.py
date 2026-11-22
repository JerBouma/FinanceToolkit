"""Portfolio Estimation Model"""

__docformat__ = "google"

import itertools

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp

from financetoolkit.portfolio.helpers import get_linear_regression, get_view_vector
from financetoolkit.risk.dependence import copula_model

# pylint: disable=too-many-arguments,too-many-locals

# The smallest eigenvalue a repaired correlation matrix keeps, a floor of 1e-13 as in skfolio's cov_nearest.
MINIMUM_EIGENVALUE = 1e-13

# The lowest codependence that the tail distance -ln(λ) is taken of, a floor of 1e-8 as in
# Riskfolio-Lib, so that two assets that never fall into their tails together get a large but
# finite distance.
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
    for an unknown covariance matrix. Referenced against skfolio's ShrunkMu (JAMES_STEIN)
    and Riskfolio-Lib's mean_vector (method="JS"), which do not clip β; that only differs
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

    Referenced against skfolio's ShrunkMu (BODNAR_OKHRIN) and Riskfolio-Lib's mean_vector
    (method="BOP"), whose values it matches with the grand mean as the target.

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

    Referenced against PyPortfolioOpt's CovarianceShrinkage.shrunk_covariance, whose
    values it matches; that defaults to δ = 0.2. scikit-learn's ShrunkCovariance, skfolio's ShrunkCovariance and
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
    rescaled to a unit diagonal. Shares similar logic as skfolio's cov_nearest
    (higham=False).

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
    - Statistic 1: g_ij = (n_c - n_d) / (T - n_n), referenced against skfolio's
      GerberCovariance (psd_variant=True, its default) and Riskfolio-Lib's
      gerber_cov_stat1.
    - Statistic 2: G = H / √(diag(H) diag(H)'), with H = U'U + D'D - U'D - D'U, the
      matrix form that is positive semidefinite by construction, referenced against
      Riskfolio-Lib's gerber_cov_stat2 and skfolio's GerberCovariance with
      psd_variant=False.

    A matrix that is not positive definite is replaced by the nearest one by clipping the
    eigenvalues of G (see get_nearest_correlation), as in skfolio. The thresholds apply
    to the returns themselves, not to their deviations from the mean, as in the paper and
    both libraries. Unlike the libraries, which take the standard deviations with ddof=0,
    they take ddof=1 here, which gives their result with the threshold multiplied by
    √(T / (T - 1)) and the covariance by T / (T - 1).

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

    This is the detoning of López de Prado (2020), applied to the correlation matrix
    itself rather than to a denoised one. Referenced against skfolio's DetoneCovariance on
    the sample covariance, whose values it matches (it also raises the zero eigenvalues to
    1e-13).

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


def get_minimum_covariance_determinant(
    returns: pd.DataFrame,
    support_fraction: float | None = None,
    reweight: bool = True,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    The Minimum Covariance Determinant (MCD) covariance matrix: the covariance of the h
    periods whose covariance matrix has the smallest determinant, so that the periods far
    from the bulk of the returns (crashes, squeezes, data errors) do not inflate it. With
    T periods and N assets, h = ⌊(T + N + 1) / 2⌋ by default, the subset size with the
    highest breakdown point: the estimate withstands up to ⌊(T - N + 1) / 2⌋ outlying
    periods (Rousseeuw, 1985).

    The subset is found with the FAST-MCD algorithm of Rousseeuw and Van Driessen (1999).
    Its building block is the C-step (concentration step): from the mean μ and covariance
    S of a subset, the next subset is the h periods with the smallest Mahalanobis distance
    d²_t = (r_t - μ)' S⁻¹ (r_t - μ), whose covariance never has a larger determinant.

    - Up to 600 periods: 500 random subsets of N + 1 periods (extended with further random
      periods while their covariance is singular) each give a first subset of h periods,
      followed by two C-steps. The 50 with the smallest determinant are iterated until
      the determinant no longer falls, and the best of them is the raw MCD.
    - Above 600 periods: the same search runs within k = min(5, ⌊T / 300⌋) disjoint random
      groups (of all periods, or of 1500 drawn at random above 1500 periods), with 500 / k
      starts and a subset size scaled to the group. The 10 best of each group take two
      C-steps on the merged groups and are then iterated to convergence on all periods.

    The paper iterates the 10 best starts to convergence; iterating 50 costs little more
    and, on returns without outliers, where many subsets have nearly the same
    determinant, more often finds the smallest one.

    The raw estimate is then made consistent at the normal distribution and reweighted
    (Rousseeuw and Van Driessen, 1999): the distances are divided by the consistency
    factor c(N, h / T), with c(N, α) = α / F_(N+2)(χ²_N(α)) (Croux and Haesbroeck, 1999),
    the periods with a distance below the 97.5% quantile of χ²_N are kept, and their
    covariance is multiplied by c(N, 0.975). The covariances divide by the number of
    periods (ddof=0), on which the consistency factors are based.

    Unlike the median rule of Rousseeuw and Van Driessen (1999), med d²_t / χ²_N(0.5), the
    consistency factors are the exact ones at the normal distribution, and the reweighted
    covariance is made consistent as well, as in robustbase's covMcd and scikit-learn's
    MinCovDet. Referenced against scikit-learn's MinCovDet, which starts from 30 random
    subsets of h periods: the random subsets differ, so the two can end in different
    local minima of the determinant, which on small samples moves the estimate by a few
    percent; periods far from the bulk are flagged by both. With reweight=False it is the
    raw covariance of the best subset, without the consistency factor, as
    PyPortfolioOpt's risk_models.min_cov_determinant returns it.

    See Rousseeuw, P. J. (1985), "Multivariate Estimation with High Breakdown Point", in
    Grossmann, W., Pflug, G., Vincze, I. and Wertz, W. (eds.), Mathematical Statistics and
    Applications, Vol. B, Reidel, 283-297; Rousseeuw, P. J. and Van Driessen, K. (1999), "A
    Fast Algorithm for the Minimum Covariance Determinant Estimator", Technometrics, 41(3),
    212-223; and Croux, C. and Haesbroeck, G. (1999), "Influence Function and Efficiency of
    the Minimum Covariance Determinant Scatter Matrix Estimator", Journal of Multivariate
    Analysis, 71(2), 161-190.

    Args:
        returns (pd.DataFrame): The returns per period, at least two assets, without missing
            values.
        support_fraction (float | None, optional): The share of the periods h / T in the
            subset, between 0 and 1. Defaults to None, ⌊(T + N + 1) / 2⌋ periods.
        reweight (bool, optional): Whether to make the estimate consistent and reweight
            it. Defaults to True.
        seed (int | None, optional): The seed of the random subsets (numpy's default
            generator), so the same returns give the same estimate. Defaults to 0.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    data = returns.to_numpy(dtype=float)
    periods, assets = data.shape
    if assets < 2:  # noqa: PLR2004
        raise ValueError(
            "The minimum covariance determinant needs at least two assets."
        )
    if support_fraction is None:
        size = min((periods + assets + 1) // 2, periods)
    elif 0 < support_fraction <= 1:
        size = int(support_fraction * periods)
    else:
        raise ValueError("The support fraction must lie between 0 and 1.")
    if size <= assets:
        raise ValueError(
            "The minimum covariance determinant needs more periods than assets."
        )
    rng = np.random.default_rng(seed)

    def mahalanobis(sample: np.ndarray, centres: np.ndarray, scatters: np.ndarray):
        # The squared distance of every period to every (centre, scatter) pair, one row per
        # pair, in blocks of 50 pairs to bound the memory; the pseudo-inverse keeps an
        # exact fit (a singular scatter) usable.
        inverses = np.linalg.pinv(scatters, hermitian=True)
        distances = np.empty((len(centres), len(sample)))
        for block in range(0, len(centres), 50):
            deviations = sample[None] - centres[block : block + 50, None, :]
            distances[block : block + 50] = (
                (deviations @ inverses[block : block + 50]) * deviations
            ).sum(axis=2)
        return distances

    def concentrate(sample: np.ndarray, h: int, centres, scatters, steps: int):
        # Up to `steps` C-steps for every (centre, scatter) pair; a pair stops once its
        # determinant no longer falls, which includes a subset that no longer changes.
        signs, logarithms = np.linalg.slogdet(scatters)
        logarithms = np.where(signs > 0, logarithms, -np.inf)
        active = np.ones(len(centres), dtype=bool)
        first = True
        for _ in range(steps):
            if not active.any():
                break
            closest = np.argpartition(
                mahalanobis(sample, centres[active], scatters[active]), h - 1, axis=1
            )[:, :h]
            chosen = sample[closest]
            centre = chosen.mean(axis=1)
            deviations = chosen - centre[:, None, :]
            scatter = np.einsum("mti,mtj->mij", deviations, deviations) / h
            signs, logarithm = np.linalg.slogdet(scatter)
            logarithm = np.where(signs > 0, logarithm, -np.inf)
            # The first step leaves the starting scatter (of another subset size), so it
            # always counts; after that a step that does not lower the determinant ends.
            improved = (
                np.ones(len(logarithm), dtype=bool)
                if first
                else (logarithm < logarithms[active] - 1e-12)
            )
            indices = np.flatnonzero(active)
            centres[indices[improved]] = centre[improved]
            scatters[indices[improved]] = scatter[improved]
            logarithms[indices[improved]] = logarithm[improved]
            active[indices[~improved | ~np.isfinite(logarithm)]] = False
            first = False
        return centres, scatters, logarithms

    def best(centres, scatters, logarithms, count: int):
        order = np.argsort(logarithms, kind="stable")[:count]
        return centres[order], scatters[order]

    def random_starts(sample: np.ndarray, h: int, starts: int, count: int):
        # Random (N + 1)-subsets, each extended with the next periods of its random order
        # while its covariance is singular; the first of three steps turns each into an
        # h-subset and two C-steps follow.
        orders = rng.permuted(np.tile(np.arange(len(sample)), (starts, 1)), axis=1)
        centres, scatters = [], []
        for order in orders:
            members = assets + 1
            scatter = np.cov(sample[order[:members]], rowvar=False, ddof=0)
            while np.linalg.slogdet(scatter)[0] <= 0 and members < len(sample):
                members += 1
                scatter = np.cov(sample[order[:members]], rowvar=False, ddof=0)
            centres.append(sample[order[:members]].mean(axis=0))
            scatters.append(scatter)
        return best(
            *concentrate(sample, h, np.array(centres), np.array(scatters), 3), count
        )

    if periods <= 600:  # noqa: PLR2004
        centres, scatters = random_starts(data, size, 500, 50)
    else:
        groups = min(5, periods // 300)
        merged = rng.permutation(periods)[: min(periods, 1500)]
        candidates = [
            random_starts(
                data[group],
                int(np.ceil(len(group) * size / periods)),
                500 // groups,
                10,
            )
            for group in np.array_split(merged, groups)
        ]
        centres, scatters = best(
            *concentrate(
                data[merged],
                int(np.ceil(len(merged) * size / periods)),
                np.concatenate([centre for centre, _ in candidates]),
                np.concatenate([scatter for _, scatter in candidates]),
                2,
            ),
            50,
        )
    centres, scatters = best(*concentrate(data, size, centres, scatters, 100), 1)
    location, covariance = centres[0], scatters[0]

    if reweight:

        def consistency(alpha: float) -> float:
            return alpha / stats.chi2.cdf(stats.chi2.ppf(alpha, assets), assets + 2)

        distance = mahalanobis(data, location[None], covariance[None])[0]
        kept = distance / consistency(size / periods) < stats.chi2.ppf(0.975, assets)
        covariance = np.cov(data[kept], rowvar=False, ddof=0) * consistency(0.975)
    return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)


def get_triangulated_maximally_filtered_graph(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The Triangulated Maximally Filtered Graph (TMFG) of a matrix of non-negative weights
    (such as similarities between assets): a planar graph that keeps the strongest links,
    3(N - 2) of the N(N - 1) / 2, built from 4-cliques (tetrahedra) joined by 3-cliques
    (triangles, the separators).

    The graph grows by T2 moves (Massara, Di Matteo and Aste, 2017), each of which
    inserts a vertex into a triangular face:

    - The first 4-clique joins the four vertices with the largest strength, the sum of
      their weights above the average weight: a quick stand-in for the tetrahedron with
      the largest total weight, the choice of Aste's MATLAB code for the TMFG (DBHT,
      MATLAB Central File Exchange 46750).
    - The gain of a vertex v not yet in the graph and a face {a, b, c} is the weight of the
      three links it would add, w_va + w_vb + w_vc. Every step inserts the vertex and face
      with the largest gain: the face becomes a separator, the face and the vertex form a
      new 4-clique, and the face splits into three new faces that hold the vertex.

    Ties in the gain go to the earliest face and then to the lowest vertex index. Shares
    similar logic as Riskfolio-Lib's PMFG_T2s, which gives the same graph on the test
    data; it compares the gains rounded to six decimals, so two gains within 1e-6 of each
    other can make it choose another face.

    See Massara, G. P., Di Matteo, T. and Aste, T. (2017), "Network Filtering for Big Data:
    Triangulated Maximally Filtered Graph", Journal of Complex Networks, 5(2), 161-178.

    Args:
        weights (np.ndarray): A symmetric N × N matrix of non-negative weights, N ≥ 4.

    Returns:
        tuple[np.ndarray, np.ndarray]: The N - 3 4-cliques (rows of four vertex indices)
        and the N - 4 separators (rows of three vertex indices).
    """
    vertices = len(weights)
    if vertices < 4:  # noqa: PLR2004
        raise ValueError("The TMFG needs at least four vertices.")
    strength = (weights * (weights > weights.mean())).sum(axis=1)
    tetrahedron = np.argsort(strength)[::-1][:4]
    faces = [list(face) for face in itertools.combinations(tetrahedron, 3)]
    outside = np.setdiff1d(np.arange(vertices), tetrahedron)
    cliques, separators = [list(tetrahedron)], []

    while len(outside):
        # The gain of every face (rows) and vertex outside the graph (columns); the first
        # largest entry in row order is the earliest face and the lowest vertex.
        gain = weights[np.ix_(np.ravel(faces), outside)].reshape(len(faces), 3, -1)
        face, column = divmod(int(np.argmax(gain.sum(axis=1))), len(outside))
        vertex = outside[column]
        first, second, third = faces[face]
        separators.append([first, second, third])
        cliques.append([first, second, third, vertex])
        faces[face] = [first, second, vertex]
        faces += [[first, third, vertex], [second, third, vertex]]
        outside = np.delete(outside, column)

    return np.array(cliques, dtype=int), np.array(separators, dtype=int).reshape(-1, 3)


def get_logo_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    The j-LoGo covariance matrix: the inverse of the sparse inverse covariance (precision
    matrix) that the Local-Global (LoGo) method of Barfuss, Massara, Di Matteo and Aste
    (2016) builds on the TMFG of the correlations. Two assets that the planar graph does
    not link are conditionally independent given the others, so most of the precision
    matrix is zero and far fewer parameters are estimated than in the full matrix.

    The TMFG is a tree of 4-cliques joined by separators, and for such a graph the
    precision matrix of a Gaussian follows from the local covariances alone. With S the
    sample covariance (ddof=1) and the TMFG of the similarities (1 + ρ_ij) / 2 (see
    get_triangulated_maximally_filtered_graph):

    - J = Σ_C [S_CC]⁻¹ - Σ_S [S_SS]⁻¹, each local inverse added to the rows and columns of
      its vertices, over the 4-cliques C and the separators S.
    - Σ = J⁻¹.

    The similarity (1 + ρ_ij) / 2 = 1 - d²_ij, with d_ij = √((1 - ρ_ij) / 2) the
    correlation distance, is the choice of Riskfolio-Lib's covar_matrix(method="jlogo");
    referenced against it, the two agree to 1e-10. With four or fewer assets the graph
    links every pair and the estimate is the sample covariance.

    See Barfuss, W., Massara, G. P., Di Matteo, T. and Aste, T. (2016), "Parsimonious
    Modeling with Information Filtering Networks", Physical Review E, 94(6), 062306, and
    Massara, G. P., Di Matteo, T. and Aste, T. (2017), "Network Filtering for Big Data:
    Triangulated Maximally Filtered Graph", Journal of Complex Networks, 5(2), 161-178.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    sample = returns.cov().to_numpy(dtype=float)
    if len(sample) <= 4:  # noqa: PLR2004
        return pd.DataFrame(sample, index=returns.columns, columns=returns.columns)
    correlation = np.corrcoef(returns.to_numpy(dtype=float), rowvar=False)
    cliques, separators = get_triangulated_maximally_filtered_graph(
        np.clip((1 + correlation) / 2, 0.0, 1.0)
    )
    precision = np.zeros_like(sample)
    for groups, sign in ((cliques, 1.0), (separators, -1.0)):
        rows, columns = groups[:, :, None], groups[:, None, :]
        np.add.at(
            precision, (rows, columns), sign * np.linalg.inv(sample[rows, columns])
        )
    return pd.DataFrame(
        np.linalg.inv(precision), index=returns.columns, columns=returns.columns
    )


def get_geodesic_shrinkage_covariance(
    returns: pd.DataFrame, shrinkage: float = 0.1, target: str = "scaled_identity"
) -> pd.DataFrame:
    """
    The covariance matrix shrunk towards a target along the geodesic of the positive
    definite matrices rather than along a straight line. With S the sample covariance
    (ddof=1), T the target and α the shrinkage intensity:

    - Σ(α) = S^(1/2) (S^(-1/2) T S^(-1/2))^α S^(1/2), so Σ(0) = S and Σ(1) = T.
    - For the scaled identity T = (tr(S) / N) I, the eigenvectors of S are kept and its
      eigenvalues interpolated geometrically, λ_i^(1 - α) (tr(S) / N)^α, where linear
      shrinkage (see get_shrunk_covariance) gives (1 - α) λ_i + α tr(S) / N. The condition
      number becomes κ(S)^(1 - α), and the estimate stays positive definite for every α.
    - For the diagonal target T = diag(S), the correlations shrink towards zero, and Σ(α)
      = B Λ^(1 - α) B' with S V = T V Λ, V' T V = I and B = T V.

    Unlike linear shrinkage, the trace is not kept at intermediate α. A sample covariance
    that is not positive definite is first replaced by the nearest one (see
    get_nearest_correlation).

    Referenced against skfolio's GeodesicShrinkageCovariance, whose values it matches on
    its default EmpiricalCovariance (ddof=1), with the same default intensity of 0.1.

    See Bhatia, R. (2007), Positive Definite Matrices, Princeton University Press, and
    Musolas, A., Smith, S. T. and Marzouk, Y. (2021), "Geodesically Parameterized
    Covariance Estimation", SIAM Journal on Matrix Analysis and Applications, 42(2),
    528-556.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        shrinkage (float, optional): The shrinkage intensity α between 0 (the sample
            covariance) and 1 (the target). Defaults to 0.1.
        target (str, optional): The target: "scaled_identity" or "diagonal". Defaults to
            "scaled_identity".

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError("The shrinkage intensity must lie between 0 and 1.")
    if target not in ("scaled_identity", "diagonal"):
        raise ValueError("The target must be 'scaled_identity' or 'diagonal'.")
    sample = returns.cov().to_numpy(dtype=float)
    try:
        np.linalg.cholesky(sample)
    except np.linalg.LinAlgError:
        volatility = np.sqrt(np.diag(sample))
        sample = get_nearest_correlation(
            sample / np.outer(volatility, volatility)
        ) * np.outer(volatility, volatility)
    if target == "scaled_identity":
        end = np.trace(sample) / len(sample) * np.eye(len(sample))
    else:
        end = np.diag(np.diag(sample))
    if shrinkage in (0, 1):
        covariance = sample if shrinkage == 0 else end
    elif target == "scaled_identity":
        eigenvalues, eigenvectors = np.linalg.eigh(sample)
        eigenvalues = eigenvalues ** (1 - shrinkage) * end[0, 0] ** shrinkage
        covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    else:
        eigenvalues, eigenvectors = linalg.eigh(sample, end)
        basis = end @ eigenvectors
        covariance = (basis * eigenvalues ** (1 - shrinkage)) @ basis.T
    return pd.DataFrame(
        (covariance + covariance.T) / 2, index=returns.columns, columns=returns.columns
    )


def get_regime_adjusted_exponential_covariance(
    returns: pd.DataFrame, half_life: float = 40
) -> tuple[pd.DataFrame, float]:
    """
    The exponentially weighted covariance matrix rescaled by a short-term volatility
    update (STVU): a multiplier φ that compares the risk the covariance predicted with the
    risk that followed, so the estimate catches up when volatility changes faster than the
    exponential average can follow.

    With λ = 2^(-1 / half_life), the returns r_t taken around zero and the recursion
    started at zero, the covariance after t periods is S_t = λ S_(t-1) + (1 - λ) r_t r_t',
    divided by 1 - λ^t to remove the bias of the zero start. From period half_life on,
    every period compares the return of the inverse volatility portfolio w_t (from the
    covariance before that period) with its predicted volatility:

    - z_t = |w_t' r_t| / √(w_t' S_(t-1) w_t) / √(2 / π), which averages 1 when the risk
      is predicted well (E|Z| = √(2 / π) for a standard normal Z).
    - φ is the exponential average of z_t with half-life half_life / 2, started at the
      first z_t and used once there are half_life / 2 of them, clipped to [0.7, 1.6].
    - Σ = φ² S_T.

    The volatility of the portfolio is calibrated rather than every direction of the
    covariance (the Mahalanobis distance), since the directions of the smallest
    eigenvalues, which an inverse volatility portfolio hardly holds, are estimated worst.

    The short-term volatility update follows Paleologo (2025). Referenced against
    skfolio's RegimeAdjustedEWCovariance, whose values it matches with its defaults
    (half_life=40, the PORTFOLIO target with inverse volatility weights, the FIRST_MOMENT
    method, the multiplier clipped to (0.7, 1.6), assume_centered=True, no HAC
    correction).

    See Paleologo, G. A. (2025), The Elements of Quantitative Investing, Wiley Finance,
    and Hawkins, D. M. and Maboudou-Tchao, E. M. (2008), "Multivariate Exponentially
    Weighted Moving Covariance Matrix", Technometrics, 50(2), 155-166.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        half_life (float, optional): The half-life in periods of the exponential weights.
            Defaults to 40.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the multiplier φ.
    """
    if half_life <= 0:
        raise ValueError("The half-life must be positive.")
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    if periods < max(1, int(half_life)):
        raise ValueError(
            f"The regime adjusted covariance needs at least {int(half_life)} periods."
        )
    decay = 2 ** (-1 / half_life)
    regime_decay = 2 ** (-1 / (half_life / 2))
    covariance = np.zeros((assets, assets))
    multiplier, comparisons = 0.0, 0
    for period, row in enumerate(values):
        if period >= max(1, int(half_life)):
            predicted = covariance / (1 - decay**period)
            weights = 1 / np.sqrt(np.maximum(np.diag(predicted), 1e-12))
            weights = weights / weights.sum()
            variance = max(weights @ predicted @ weights, 1e-12)
            surprise = abs(weights @ row) / np.sqrt(variance) / np.sqrt(2 / np.pi)
            multiplier = (
                surprise
                if comparisons == 0
                else regime_decay * multiplier + (1 - regime_decay) * surprise
            )
            comparisons += 1
        covariance = decay * covariance + (1 - decay) * np.outer(row, row)
    covariance = covariance / (1 - decay**periods)
    multiplier = (
        float(np.clip(multiplier, 0.7, 1.6))
        if comparisons >= max(1, int(half_life / 2))
        else 1.0
    )
    return (
        pd.DataFrame(
            multiplier**2 * (covariance + covariance.T) / 2,
            index=returns.columns,
            columns=returns.columns,
        ),
        multiplier,
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

    Referenced against skfolio's DistanceCorrelation and Riskfolio-Lib's dcorr_matrix,
    whose values it matches. It builds a T × T matrix per asset, so it grows with the
    square of the number of periods.

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
      larger of the two for a pair, also the default of skfolio's MutualInformation.
    - "hacine_gharbi": the rule of Hacine-Gharbi et al. (2012) for one series,
      round(z / 6 + 2 / (3 z) + 1 / 3) with z = (8 + 324 T + 12 √(36 T + 729 T²))^(1/3),
      and of Hacine-Gharbi and Ravier (2018) for a pair with correlation ρ,
      round(√(1 + √(1 + 24 T / (1 - ρ²))) / √2), the choice of López de Prado (2020),
      as in Riskfolio-Lib's numBins.

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

    Referenced against skfolio's MutualInformation (codependence and distance) and
    Riskfolio-Lib's mutual_info_matrix and var_info_matrix, whose values it matches with
    the same number of bins.

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

    Referenced against ffn's calc_ftca, whose clusters it matches on the Pearson
    correlation of the returns; any codependence matrix can be passed here.

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
        # On ties the stable sort takes the earliest asset, the order ffn's calc_ftca gives.
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

    - "sample": w_t = 1 / T, referenced against PerformanceAnalytics' M3.MM
      (unbiased=FALSE) and CoSkewnessMatrix.
    - "exponential": w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent
      periods weigh most. PerformanceAnalytics' M3.ewma uses the same weights (its
      default λ = 0.97 is a span of about 65.7) on returns it assumes to have a zero
      mean; here the deviations are taken from the average return, as in the
      exponential covariance, so it matches M3.ewma on the demeaned returns.
    - "semi": the lower semi co-moment, with min(r_ti - μ_i, 0) in place of the
      deviations, so only returns below the average count, as defined by Cajas in
      Riskfolio-Lib.

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

    - "sample": w_t = 1 / T, referenced against PerformanceAnalytics' M4.MM and
      CoKurtosisMatrix and, in its N² × N² arrangement, Riskfolio-Lib's cokurt_matrix
      (method="hist").
    - "exponential": w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent
      periods weigh most; it matches PerformanceAnalytics' M4.ewma on the demeaned
      returns.
    - "semi": the lower semi cokurtosis, with min(r_ti - μ_i, 0) in place of the
      deviations (Cajas, 2022), referenced against Riskfolio-Lib's cokurt_matrix
      (method="semi") in its N² × N² arrangement.

    With standardize, every m_ijkl is divided by σ_i σ_j σ_k σ_l, with σ² = Σ_t w_t
    (r_t - μ)² the weighted second moment of the full deviations, so the diagonal m_iiii
    is the moment kurtosis of each asset (3 for a normal distribution, not the excess).

    See Jondeau, E. and Rockinger, M. (2006), "Optimal Portfolio Allocation under Higher
    Moments", European Financial Management, 12(1), 29-55, and Cajas, D. (2022), "Convex
    Optimization of Portfolio Kurtosis", SSRN 4202967.

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

    The arch package (Sheppard) offers the same schemes, which Riskfolio-Lib's
    bootstrapping draws from.

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
    The averages and covariances are those of the returns per period (ddof=1). The
    parameters match Riskfolio-Lib's bootstrapping (q = 1 - c, n_sim, window) and
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


FACTOR_SELECTIONS = ("ols", "forward", "backward", "pcr")
SELECTION_CRITERIA = ("pvalue", "aic", "bic", "r_squared", "adjusted_r_squared")


def get_stepwise_factors(
    returns: pd.Series,
    factors: pd.DataFrame,
    direction: str = "forward",
    criterion: str = "pvalue",
    threshold: float = 0.05,
) -> list[str]:
    """
    The factors a stepwise regression (Efroymson, 1960) of an asset's returns on factor
    returns (with a constant) selects. Shares similar logic as Riskfolio-Lib's
    forward_regression and backward_regression.

    - direction="forward", criterion="pvalue": starting without factors, add the factor
      with the lowest p-value among those that keep every p-value of the model at or below
      the threshold, until no factor qualifies.
    - direction="backward", criterion="pvalue": starting with every factor, drop the
      factor with the highest p-value until every p-value is at or below the threshold.
      With either direction, when no factor qualifies the single factor with the lowest
      p-value is kept, so the model always holds one factor.
    - criterion="aic", "bic", "r_squared" or "adjusted_r_squared": add (forward) or drop
      (backward) the factor that improves the criterion most (a lower information
      criterion, a higher R squared) for as long as it improves.

    The p-values are those of the t-statistics of the OLS coefficients and the criteria
    are defined as statsmodels defines them (see helpers.get_linear_regression).

    See Efroymson, M. A. (1960), "Multiple Regression Analysis", in Ralston, A. and Wilf,
    H. S. (eds.), Mathematical Methods for Digital Computers, Wiley; Akaike, H. (1974), "A
    New Look at the Statistical Model Identification", IEEE Transactions on Automatic
    Control, 19(6), 716-723; and Schwarz, G. (1978), "Estimating the Dimension of a Model",
    The Annals of Statistics, 6(2), 461-464. Riskfolio-Lib:
    https://github.com/dcajasn/Riskfolio-Lib.

    Args:
        returns (pd.Series): The returns per period of the asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods.
        direction (str, optional): "forward" or "backward". Defaults to "forward".
        criterion (str, optional): "pvalue", "aic", "bic", "r_squared" or
            "adjusted_r_squared". Defaults to "pvalue".
        threshold (float, optional): The highest p-value a selected factor may have.
            Defaults to 0.05.

    Returns:
        list[str]: The selected factors, in the order they were selected (forward) or in
        the order of the factors (backward).
    """
    if direction not in ("forward", "backward"):
        raise ValueError("The direction must be forward or backward.")
    if criterion not in SELECTION_CRITERIA:
        raise ValueError(
            f"The criterion must be one of {', '.join(SELECTION_CRITERIA)}, not "
            f"'{criterion}'."
        )
    target = returns.to_numpy(dtype=float)

    def fit(names: list[str]) -> dict:
        return get_linear_regression(target, factors[names].to_numpy(dtype=float))

    def best_single() -> list[str]:
        pvalues = {name: fit([name])["pvalues"][0] for name in factors.columns}
        return [min(pvalues, key=pvalues.__getitem__)]

    lower_is_better = criterion in ("pvalue", "aic", "bic")
    if criterion == "pvalue":
        if direction == "forward":
            included: list[str] = []
            while True:
                candidates = {}
                for name in factors.columns.difference(included, sort=False):
                    pvalues = fit([*included, name])["pvalues"]
                    if pvalues.max() <= threshold:
                        candidates[name] = pvalues[-1]
                if not candidates:
                    break
                included.append(min(candidates, key=candidates.__getitem__))
            return included or best_single()
        included = list(factors.columns)
        while included:
            pvalues = fit(included)["pvalues"]
            if pvalues.max() <= threshold:
                break
            included.pop(int(np.argmax(pvalues)))
        return included or best_single()

    def score(names: list[str]) -> float:
        value = fit(names)[criterion]
        return value if lower_is_better else -value

    if direction == "forward":
        included, current = [], np.inf
        remaining = list(factors.columns)
        while remaining:
            scores = {name: score([*included, name]) for name in remaining}
            name = min(scores, key=scores.__getitem__)
            if scores[name] >= current:
                break
            included.append(name)
            remaining.remove(name)
            current = scores[name]
        return included
    included = list(factors.columns)
    current = score(included)
    while included:
        scores = {
            name: score([other for other in included if other != name])
            for name in included
        }
        name = min(scores, key=scores.__getitem__)
        if scores[name] >= current:
            break
        included.remove(name)
        current = scores[name]
    return included


def get_principal_components_loadings(
    returns: pd.Series, factors: pd.DataFrame, components: float = 0.95
) -> pd.Series:
    """
    The intercept and factor loadings of a principal components regression (PCR) of an
    asset's returns on factor returns (Jolliffe, 1982). The factors
    are standardized (ddof=1), their first k principal components Z = X_std V_k are the
    regressors of an OLS regression with a constant, and the coefficients γ of the
    components are mapped back to the factors: β = V_k γ / σ_f and the intercept
    α = mean(r) - Σ β_f mean(f). Leaving out the components with little variance removes
    the instability that correlated factors give OLS loadings. Shares similar logic as
    Riskfolio-Lib's PCR (https://github.com/dcajasn/Riskfolio-Lib).

    See Jolliffe, I. T. (1982), "A Note on the Use of Principal Components in
    Regression", Journal of the Royal Statistical Society, Series C, 31(3), 300-303.

    Args:
        returns (pd.Series): The returns per period of the asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods.
        components (float, optional): The number of components to keep (1 or more), or,
            between 0 and 1, the smallest number whose share of the variance of the
            standardized factors exceeds it, as scikit-learn's PCA counts them. Defaults
            to 0.95.

    Returns:
        pd.Series: The Intercept and the loading of every factor.
    """
    if components <= 0:
        raise ValueError("The number of components must be positive.")
    data = factors.to_numpy(dtype=float)
    mean = data.mean(axis=0)
    deviation = data.std(axis=0, ddof=1)
    standardized = (data - mean) / deviation
    _, singular_values, vectors = np.linalg.svd(standardized, full_matrices=False)
    if components < 1:
        explained = singular_values**2 / np.sum(singular_values**2)
        kept = int(np.searchsorted(np.cumsum(explained), components, side="right")) + 1
    else:
        kept = int(components)
    kept = min(kept, len(singular_values))
    loadings = vectors[:kept].T
    regression = get_linear_regression(
        returns.to_numpy(dtype=float), standardized @ loadings
    )
    betas = loadings @ regression["params"][1:] / deviation
    intercept = float(returns.mean()) - float(betas @ mean)
    return pd.Series(
        np.concatenate([[intercept], betas]), index=["Intercept", *factors.columns]
    )


def get_factor_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    selection: str = "ols",
    criterion: str = "pvalue",
    threshold: float = 0.05,
    components: float = 0.95,
) -> pd.DataFrame:
    """
    The loadings matrix of a time series factor model: the intercept and factor loadings
    of the regression of every asset's returns on the factor returns, r_i,t = α_i +
    Σ_f β_i,f F_f,t + ε_i,t (Fama and French, 1993; Connor, 1995). Shares similar logic as
    Riskfolio-Lib's loadings_matrix.

    - "ols": every factor, ordinary least squares.
    - "forward", "backward": the factors a stepwise regression selects (see
      get_stepwise_factors), the others with a loading of zero.
    - "pcr": a principal components regression (see get_principal_components_loadings).

    See Fama, E. F. and French, K. R. (1993), "Common Risk Factors in the Returns on
    Stocks and Bonds", Journal of Financial Economics, 33(1), 3-56, and Connor, G.
    (1995), "The Three Types of Factor Models: A Comparison of Their Explanatory Power",
    Financial Analysts Journal, 51(3), 42-46.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods, without
            missing values.
        selection (str, optional): "ols", "forward", "backward" or "pcr". Defaults to "ols".
        criterion (str, optional): The criterion of the stepwise selection. Defaults to
            "pvalue".
        threshold (float, optional): The highest p-value of the stepwise selection.
            Defaults to 0.05.
        components (float, optional): The components of "pcr". Defaults to 0.95.

    Returns:
        pd.DataFrame: Per asset the Intercept (per period) and the loading of every factor.
    """
    if selection not in FACTOR_SELECTIONS:
        raise ValueError(
            f"The selection must be one of {', '.join(FACTOR_SELECTIONS)}, not "
            f"'{selection}'."
        )
    if not returns.index.equals(factors.index):
        raise ValueError("The returns and the factors must cover the same periods.")
    loadings = pd.DataFrame(
        0.0, index=returns.columns, columns=["Intercept", *factors.columns]
    )
    for asset in returns.columns:
        if selection == "pcr":
            loadings.loc[asset] = get_principal_components_loadings(
                returns[asset], factors, components
            )
            continue
        selected = (
            list(factors.columns)
            if selection == "ols"
            else get_stepwise_factors(
                returns[asset], factors, selection, criterion, threshold
            )
        )
        params = get_linear_regression(
            returns[asset].to_numpy(dtype=float),
            factors[selected].to_numpy(dtype=float),
        )["params"]
        loadings.loc[asset, ["Intercept", *selected]] = params
    return loadings


def get_factor_model(
    returns: pd.DataFrame, factors: pd.DataFrame, loadings: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    The expected returns and covariance matrix a time series factor model implies
    (Connor, 1995), referenced against Riskfolio-Lib's risk_factors and skfolio's
    FactorModel: with B the loadings, α the intercepts, μ_f and Σ_f the mean and sample covariance (ddof=1) of the
    factor returns and D the diagonal matrix of the variances (ddof=1) of the residuals
    ε = r - α - F B',

    - μ = α + B μ_f
    - Σ = B Σ_f B' + D

    The factors explain the co-movement of the assets and the residuals are taken to be
    uncorrelated, which leaves N K + N parameters to estimate instead of N (N + 1) / 2:
    a covariance matrix that is better conditioned when there are many assets. With an
    intercept, the OLS residuals average zero, so μ equals the average return of each asset
    over the same periods; the factor model changes the covariance, not the means.

    See Connor, G. (1995), "The Three Types of Factor Models: A Comparison of Their
    Explanatory Power", Financial Analysts Journal, 51(3), 42-46.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods.
        loadings (pd.DataFrame): The Intercept and factor loadings per asset (see
            get_factor_loadings).

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.Series]: The expected returns, the covariance
        matrix and the residual variance per asset, all per period.
    """
    intercept = loadings["Intercept"].reindex(returns.columns).to_numpy(dtype=float)
    beta = loadings[factors.columns].reindex(returns.columns).to_numpy(dtype=float)
    data = factors.to_numpy(dtype=float)
    residuals = returns.to_numpy(dtype=float) - intercept - data @ beta.T
    residual_variance = residuals.var(axis=0, ddof=1)
    covariance = beta @ np.cov(data, rowvar=False, ddof=1).reshape(
        len(factors.columns), len(factors.columns)
    ) @ beta.T + np.diag(residual_variance)
    return (
        pd.Series(intercept + beta @ data.mean(axis=0), index=returns.columns),
        pd.DataFrame(covariance, index=returns.columns, columns=returns.columns),
        pd.Series(residual_variance, index=returns.columns),
    )


VIEW_OPERATORS = ("==", ">=", "<=")
# The largest miss of a view (on the scaled constraints) that still counts as met.
VIEW_TOLERANCE = 1e-8


def get_entropy_pooling(
    returns: pd.DataFrame,
    mean_views: dict | None = None,
    volatility_views: dict | None = None,
    correlation_views: dict | None = None,
    prior_probabilities: np.ndarray | None = None,
) -> pd.Series:
    """
    The posterior probabilities of the historical scenarios (periods) under views, by
    Entropy Pooling (Meucci, 2008): the probabilities p closest to the prior probabilities
    q (equal, 1 / T, by default) in relative entropy, KL(p ‖ q) = Σ p_t ln(p_t / q_t), that
    satisfy the views. The scenarios stay as they were; only their weights change, so the
    posterior keeps the fat tails and co-movements of the history.

    A view is linear in p: a mean view E_p[r_i] = Σ p_t r_i,t (or of r_i - r_j for a
    relative view "AAPL-MSFT"), a volatility view Σ p_t (r_i,t - m_i)² = σ² with m_i the
    mean after the mean views, and a correlation view Σ p_t r_i,t r_j,t = m_i m_j +
    ρ √(v_i v_j) with the means and variances after the mean and volatility views, which
    are then held at those values: the sequential treatment of views on higher moments of
    Meucci, Ardia and Keel (2011), in the order skfolio's EntropyPooling takes them. A
    view is an equality (a number) or an inequality, (">=", value) or ("<=", value).

    The problem is solved through its dual (Csiszár, 1975; Boyd and Vandenberghe, 2004,
    5.2.3). With the constraints A'p = b and A'p ≤ b (every column of A scaled to unit
    length), the posterior has the exponential-family form p_t ∝ q_t exp(-a_t'λ), and the
    multipliers λ minimize ln Σ q_t exp(-a_t'λ) + λ'b, with λ ≥ 0 for the inequalities
    (SciPy's L-BFGS-B). The multipliers of the moments held fixed are bounded to ±1000, as
    in skfolio, which relaxes them when the views cannot be met exactly. A solution that
    misses a view by more than 1e-8 (on the scaled constraints) raises an error.
    Referenced against skfolio's EntropyPooling (https://github.com/skfolio/skfolio).

    See Meucci, A. (2008), "Fully Flexible Views: Theory and Practice", Risk, 21(10),
    97-102, and Meucci, A., Ardia, D. and Keel, S. (2011), "Fully Flexible Extreme Views",
    Journal of Risk, 14(2), 39-49.

    Args:
        returns (pd.DataFrame): The returns per period (the scenarios), one column per
            asset, without missing values.
        mean_views (dict | None, optional): The mean return per period of an asset
            ("AAPL") or of the difference of two ("AAPL-MSFT"). Defaults to None.
        volatility_views (dict | None, optional): The standard deviation per period of an
            asset. Defaults to None.
        correlation_views (dict | None, optional): The correlation of two assets
            ("AAPL-MSFT"), between -1 and 1. Defaults to None.
        prior_probabilities (np.ndarray | None, optional): The prior probability of every
            scenario. Defaults to None, equal probabilities.

    Returns:
        pd.Series: The posterior probability of every scenario, summing to one.
    """
    data = returns.to_numpy(dtype=float)
    periods, assets = data.shape
    names = returns.columns
    prior = (
        np.full(periods, 1 / periods)
        if prior_probabilities is None
        else np.asarray(prior_probabilities, dtype=float)
    )
    if len(prior) != periods or np.any(prior < 0) or not np.isclose(prior.sum(), 1):
        raise ValueError(
            "The prior probabilities must be one non-negative number per period, summing "
            "to one."
        )
    # Each kind holds the columns of A and the values of b: equalities, inequalities
    # (A'p ≤ b) and the moments held fixed.
    constraints: dict[str, list] = {"equality": [], "inequality": [], "fixed": []}

    def add(kind: str, column: np.ndarray, value: float) -> None:
        scale = np.linalg.norm(column)
        if scale == 0:
            raise ValueError("A view does not vary across the scenarios.")
        constraints[kind].append((column / scale, value / scale))

    def add_view(view_value, column: np.ndarray, transform=None) -> None:
        # The transform turns the value of the view into the value of the constraint,
        # e.g. a volatility into the variance the constraint holds.
        operator, value = (
            ("==", view_value)
            if isinstance(view_value, int | float)
            else tuple(view_value)
        )
        if operator not in VIEW_OPERATORS:
            raise ValueError(
                f"A view is a number or a pair (operator, value) with the operator one of "
                f"{', '.join(VIEW_OPERATORS)}, not {operator!r}."
            )
        target = float(value) if transform is None else float(transform(float(value)))
        if operator == "==":
            add("equality", column, target)
        elif operator == "<=":
            add("inequality", column, target)
        else:
            add("inequality", -column, -target)

    def solve() -> np.ndarray:
        # The posterior has the exponential-family form p_t ∝ q_t exp(-a_t'λ), so only one
        # multiplier per view is searched: λ minimizes the dual ln Σ q_t exp(-a_t'λ) + λ'b,
        # whose gradient is b - A'p. Inequalities (A'p ≤ b) need λ ≥ 0, and the moments
        # held fixed get |λ| ≤ 1000 so that conflicting views relax them rather than fail.
        if not any(constraints.values()):
            return prior
        kinds = {
            "equality": (None, None),
            "inequality": (0, None),
            "fixed": (-1000, 1000),
        }
        rows = [(row, kinds[kind]) for kind in kinds for row in constraints[kind]]
        matrix = np.column_stack([column for (column, _), _ in rows])
        target = np.array([value for (_, value), _ in rows])
        with np.errstate(divide="ignore"):
            log_prior = np.log(prior)

        def posterior(multipliers: np.ndarray) -> np.ndarray:
            exponent = log_prior - matrix @ multipliers
            return np.exp(exponent - logsumexp(exponent))

        def dual(multipliers: np.ndarray) -> tuple[float, np.ndarray]:
            exponent = log_prior - matrix @ multipliers
            return (
                float(logsumexp(exponent) + multipliers @ target),
                target - matrix.T @ posterior(multipliers),
            )

        solution = minimize(
            dual,
            x0=np.zeros(len(target)),
            jac=True,
            bounds=[bound for _, bound in rows],
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
        )
        # The views are met when the gradient vanishes on the equalities and is not
        # negative on the inequalities (the fixed moments may be relaxed); the solver's
        # own flag also fails when the line search stops at the limit of the precision.
        residual = matrix.T @ posterior(solution.x) - target
        equalities = len(constraints["equality"])
        inequalities = len(constraints["inequality"])
        if (
            np.abs(residual[:equalities]).max(initial=0) > VIEW_TOLERANCE
            or residual[equalities : equalities + inequalities].max(initial=0)
            > VIEW_TOLERANCE
        ):
            raise ValueError(
                "The views cannot be met by reweighting the scenarios; they conflict or "
                f"are too extreme ({solution.message})."
            )
        return posterior(solution.x)

    fixed_mean = np.zeros(assets, dtype=bool)
    fixed_variance = np.zeros(assets, dtype=bool)

    def fix(selected: np.ndarray, mean: np.ndarray, variance=None) -> None:
        for index in np.flatnonzero(selected & ~fixed_mean):
            add("fixed", data[:, index], mean[index])
            fixed_mean[index] = True
        if variance is not None:
            for index in np.flatnonzero(selected & ~fixed_variance):
                add("fixed", (data[:, index] - mean[index]) ** 2, variance[index])
                fixed_variance[index] = True

    for view, value in (mean_views or {}).items():
        add_view(value, data @ get_view_vector(view, names))
    probabilities = solve()

    if volatility_views:
        mean = probabilities @ data
        selected = np.zeros(assets, dtype=bool)
        for view, value in volatility_views.items():
            if view not in names:
                raise ValueError(f"The volatility view {view!r} names no asset.")
            index = names.get_loc(view)
            add_view(
                value,
                (data[:, index] - mean[index]) ** 2,
                lambda volatility: volatility**2,
            )
            selected[index] = True
        fix(selected, mean)
        probabilities = solve()

    if correlation_views:
        mean = probabilities @ data
        variance = probabilities @ (data - mean) ** 2
        selected = np.zeros(assets, dtype=bool)
        for view, value in correlation_views.items():
            pick = get_view_vector(view, names)
            if np.count_nonzero(pick) != 2:  # noqa: PLR2004
                raise ValueError(
                    f"The correlation view {view!r} must name two assets, 'NAME-NAME'."
                )
            first, second = np.flatnonzero(pick == 1)[0], np.flatnonzero(pick == -1)[0]
            add_view(
                value,
                data[:, first] * data[:, second],
                lambda correlation, i=first, j=second: mean[i] * mean[j]
                + correlation * np.sqrt(variance[i] * variance[j]),
            )
            selected[[first, second]] = True
        fix(selected, mean, variance)
        probabilities = solve()

    return pd.Series(probabilities, index=returns.index)


POOLING_METHODS = ("linear", "logarithmic")


def get_opinion_pooling(
    probabilities: pd.DataFrame,
    opinion_weights: list[float] | None = None,
    method: str = "linear",
    divergence_penalty: float = 0.0,
) -> pd.Series:
    """
    The consensus of several sets of scenario probabilities, the opinions of different
    experts (e.g. each an Entropy Pooling posterior). The weight π_k of each opinion lies
    between 0 and 1; when they sum to less than one, the rest goes to the prior (equal
    probabilities).

    - "linear": p = Σ π_k p_k, the arithmetic mixture (Stone, 1961), which keeps every
      scenario any opinion gives a probability.
    - "logarithmic": p ∝ Π p_k^π_k, the geometric mixture (Genest and Zidek, 1986), which
      minimizes the weighted relative entropy to the opinions and gives no probability to
      a scenario that any opinion rules out.

    With a divergence penalty α > 0 the weights become π̃_k ∝ π_k exp(-α KL(p_k ‖ c)), with
    c = Σ π_k p_k the linear consensus, which weighs down the opinions far from the others;
    this robust weighting is the one skfolio's OpinionPooling (Delatte and Nicolini)
    defines, and the function is referenced against it.

    See Stone, M. (1961), "The Opinion Pool", The Annals of Mathematical Statistics,
    32(4), 1339-1342; Genest, C. and Zidek, J. V. (1986), "Combining Probability
    Distributions: A Critique and an Annotated Bibliography", Statistical Science, 1(1),
    114-135. skfolio: https://github.com/skfolio/skfolio.

    Args:
        probabilities (pd.DataFrame): The probability of every scenario (rows) per opinion
            (columns), each column summing to one.
        opinion_weights (list[float] | None, optional): The weight of every opinion.
            Defaults to None, equal weights summing to one.
        method (str, optional): "linear" or "logarithmic". Defaults to "linear".
        divergence_penalty (float, optional): The penalty α. Defaults to 0.

    Returns:
        pd.Series: The pooled probability of every scenario, summing to one.
    """
    if method not in POOLING_METHODS:
        raise ValueError("The pooling method must be linear or logarithmic.")
    if divergence_penalty < 0:
        raise ValueError("The divergence penalty must not be negative.")
    opinions = probabilities.to_numpy(dtype=float).T
    count, periods = opinions.shape
    weights = (
        np.full(count, 1 / count)
        if opinion_weights is None
        else np.asarray(opinion_weights, dtype=float)
    )
    if len(weights) != count:
        raise ValueError("There must be one weight per opinion.")
    if np.any(weights < 0) or np.any(weights > 1) or weights.sum() > 1 + 1e-12:
        raise ValueError(
            "The opinion weights must lie between 0 and 1 and sum to at most one."
        )
    # The weight the opinions leave goes to the prior, equal probabilities, as one more
    # opinion.
    if not np.isclose(weights.sum(), 1):
        opinions = np.vstack([opinions, np.full(periods, 1 / periods)])
        weights = np.append(weights, 1 - weights.sum())
    weights = weights / weights.sum()
    if divergence_penalty > 0:
        consensus = weights @ opinions
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(opinions > 0, opinions * np.log(opinions / consensus), 0.0)
        weights = weights * np.exp(-divergence_penalty * terms.sum(axis=1))
        weights = weights / weights.sum()
    if method == "linear":
        pooled = weights @ opinions
    else:
        # The geometric mixture in logarithms; only opinions with weight take part, so a
        # scenario any of them rules out gets no probability.
        weighted = weights > 0
        with np.errstate(divide="ignore"):
            logarithm = weights[weighted] @ np.log(opinions[weighted])
        pooled = np.exp(logarithm - logarithm.max())
        pooled = pooled / pooled.sum()
    return pd.Series(pooled, index=probabilities.index)
