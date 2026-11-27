"""Portfolio Estimation Model"""

__docformat__ = "google"

import itertools

import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.optimize import brentq, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp

from financetoolkit.portfolio.helpers import get_linear_regression, get_view_vector
from financetoolkit.risk.dependence import copula_model
from financetoolkit.risk.volatility import garch_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# pylint: disable=too-many-arguments,too-many-locals

# The smallest eigenvalue a repaired correlation matrix keeps, a floor of 1e-13 as in skfolio's cov_nearest.
MINIMUM_EIGENVALUE = 1e-13

# The lowest codependence that the tail distance -ln(λ) is taken of, a floor of 1e-8 as in
# Riskfolio-Lib, so that two assets that never fall into their tails together get a large but
# finite distance.
MINIMUM_TAIL_DEPENDENCE = 1e-8

GERBER_STATISTICS = (0, 1, 2)
COMOMENT_METHODS = ("sample", "exponential", "semi", "ccc_garch")
BIN_METHODS = ("freedman", "hacine_gharbi")
BOOTSTRAP_METHODS = ("stationary", "circular", "moving", "normal")


def get_james_stein_expected_returns(
    returns: pd.DataFrame,
) -> tuple[pd.Series, float]:
    """
    The James-Stein estimator of the expected returns.

    Every average return pulled towards the grand mean, the average of the average returns,
    by a shrinkage intensity that grows as the averages differ less compared to their
    estimation error. With μ̂ the average returns of the T periods of the N assets, μ̄ their
    average and λ_1, ..., λ_N the eigenvalues of the sample covariance matrix S (ddof=1),
    the estimate is μ = (1 - β) μ̂ + β μ̄ 1 with β = (Σ λ_k - 2 max λ_k) / (T ‖μ̂ - μ̄ 1‖²),
    clipped to [0, 1] (the positive-part James-Stein estimator). This is the form of Meucci
    (2005) for an unknown covariance matrix. With two assets Σ λ_k - 2 max λ_k is negative,
    so the estimator needs at least three assets to shrink.

    See James, W. and Stein, C. (1961), "Estimation with Quadratic Loss", Proceedings of
    the Fourth Berkeley Symposium on Mathematical Statistics and Probability, 1, 361-379,
    and Meucci, A. (2005), Risk and Asset Allocation, Springer, chapter 4.

    Referenced against:
    - skfolio, ShrunkMu (James-Stein): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html

    Shares similar logic:
    - Riskfolio-Lib, mean_vector (James-Stein): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.Series, float]: The expected return per asset per period and the
            shrinkage intensity β between 0 (the averages) and 1 (the grand mean).

    Notes:
    - Unlike skfolio's ShrunkMu and Riskfolio-Lib's mean_vector, β is clipped to [0, 1];
      that only differs when the averages lie so close together that β leaves [0, 1].
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
    The shrinkage estimator of the expected returns of Bodnar, Okhrin and Parolya (2019).

    Μ = α μ̂ + β b, a combination of the average returns μ̂ and a target b with the weights
    α and β that are optimal in the sense of the quadratic loss when the number of assets
    grows with the number of periods. The target is the grand mean, the average of the
    average returns; since α and β scale with b, any multiple of a vector of ones gives the
    same estimate.

    With S the sample covariance matrix (ddof=1), T periods and N assets, u = μ̂'S⁻¹μ̂,
    v = μ̂'S⁻¹b and w = b'S⁻¹b:

    - α = ((u - N / (T - N)) w - v²) / (u w - v²)
    - β = (1 - α) v / w

    See Bodnar, T., Okhrin, O. and Parolya, N. (2019), "Optimal Shrinkage Estimator for
    High-Dimensional Mean Vector", Journal of Multivariate Analysis, 170, 63-79.

    Referenced against:
    - skfolio, ShrunkMu (Bodnar-Okhrin): https://skfolio.org/generated/skfolio.moments.ShrunkMu.html

    Shares similar logic:
    - Riskfolio-Lib, mean_vector (Bodnar-Okhrin-Parolya): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.mean_vector

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
    by the average variance.

    Σ = (1 - δ) S + δ (tr(S) / N) I, with S the sample covariance matrix (ddof=1) of the N
    assets. Unlike Ledoit-Wolf and OAS, the intensity δ is chosen rather than estimated.

    See Ledoit, O. and Wolf, M. (2004), "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Journal of Multivariate Analysis, 88(2), 365-411, for the
    target.

    Referenced against:
    - PyPortfolioOpt, CovarianceShrinkage.shrunk_covariance: https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html#pypfopt.risk_models.CovarianceShrinkage.shrunk_covariance

    Shares similar logic:
    - scikit-learn, ShrunkCovariance: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
    - skfolio, ShrunkCovariance: https://skfolio.org/generated/skfolio.moments.ShrunkCovariance.html
    - Riskfolio-Lib, covar_matrix (fixed shrinkage): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.covar_matrix

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        shrinkage (float, optional): The shrinkage intensity δ between 0 (the sample
            covariance) and 1 (the scaled identity). Defaults to 0.1.

    Returns:
        pd.DataFrame: The covariance matrix per period.

    Notes:
    - PyPortfolioOpt's shrunk_covariance defaults to δ = 0.2. scikit-learn's
      ShrunkCovariance, skfolio's ShrunkCovariance and Riskfolio-Lib's covar_matrix default
      to δ = 0.1, the default here, and shrink the maximum likelihood covariance (divided by
      T rather than T - 1), which is T / (T - 1) times smaller.
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
    The correlation matrix itself when it is positive definite, otherwise the nearest one by
    eigenvalue clipping.

    The eigenvalues below 1e-13 are raised to it and the matrix is rescaled to a unit
    diagonal. See Higham, N. J. (2002), "Computing the Nearest Correlation Matrix - A
    Problem from Finance", IMA Journal of Numerical Analysis, 22(3), 329-343.

    Shares similar logic:
    - skfolio, cov_nearest (eigenvalue clipping): https://skfolio.org/generated/skfolio.utils.stats.cov_nearest.html

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
    The Gerber covariance matrix.

    Σ = D G D, with D the standard deviations (ddof=1) and G the Gerber statistic, a
    correlation that counts only the periods in which returns move beyond a threshold. A
    return r_ti is up when r_ti ≥ c σ_i and down when r_ti ≤ -c σ_i, with c the threshold;
    small moves, which are mostly noise, do not count, and neither does the size of large
    ones, so a few extreme returns do not dominate it.

    Per pair of assets, n_c counts the periods in which both moved beyond the threshold in
    the same direction (concordant), n_d those in opposite directions (discordant) and n_n
    those in which neither did. With U and D the T × N indicators of up and down moves:

    - Statistic 0: g_ij = (n_c - n_d) / (n_c + n_d), the original pairwise statistic, which
      need not give a positive semidefinite matrix.
    - Statistic 1: g_ij = (n_c - n_d) / (T - n_n).
    - Statistic 2: G = H / √(diag(H) diag(H)'), with H = U'U + D'D - U'D - D'U, the matrix
      form that is positive semidefinite by construction.

    A matrix that is not positive definite is replaced by the nearest one by clipping the
    eigenvalues of G (see get_nearest_correlation). The thresholds apply to the returns
    themselves, not to their deviations from the mean, as in the paper.

    See Gerber, S., Markowitz, H. M., Ernst, P. A., Miao, Y., Javid, B. and Sargen, P.
    (2022), "The Gerber Statistic: A Robust Co-Movement Measure for Portfolio
    Optimization", The Journal of Portfolio Management, 48(3), 87-102.

    Referenced against:
    - skfolio, GerberCovariance (statistics 1 and 2): https://skfolio.org/generated/skfolio.moments.GerberCovariance.html
    - Riskfolio-Lib, gerber_cov_stat0 (statistic 0): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat0

    Shares similar logic:
    - Riskfolio-Lib, gerber_cov_stat1 (statistic 1): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat1
    - Riskfolio-Lib, gerber_cov_stat2 (statistic 2): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#GerberStatistic.gerber_cov_stat2

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        threshold (float, optional): The threshold c, in standard deviations, between 0
            and 1. Defaults to 0.5.
        statistic (int, optional): The Gerber statistic. Options are: 0 (the original statistic), 1
            (the positive semidefinite variant), 2 (the matrix form). Defaults to 1.

    Returns:
        pd.DataFrame: The covariance matrix per period.

    Notes:
    - Unlike skfolio and Riskfolio-Lib, which take the standard deviations with ddof=0, the
      standard deviations here take ddof=1, which gives their result with the threshold
      multiplied by √(T / (T - 1)) and the covariance by T / (T - 1). skfolio's
      GerberCovariance gives statistic 1 by default and statistic 2 without its positive
      semidefinite variant.
    """
    if not 0 < threshold < 1:
        raise ValueError("The Gerber threshold must lie between 0 and 1.")
    check_option("statistic", statistic, GERBER_STATISTICS)
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


def get_detoned_covariance(
    returns: pd.DataFrame, components: int = 1, covariance: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    The detoned covariance matrix.

    It is the sample covariance (ddof=1), or another estimate, with the first principal
    components of its correlation matrix, the market, removed. With λ_k and v_k the
    eigenvalues and eigenvectors of the correlation matrix C in descending order, the
    detoned correlation is C - Σ_(k ≤ m) λ_k v_k v_k' for the m components, rescaled to a
    unit diagonal and turned back into a covariance with the standard deviations of the
    estimate. López de Prado (2020, section 2.6) detones the denoised correlation matrix:
    pass the denoised covariance (optimization_model.get_denoised_covariance) as the
    covariance, and the eigenvalues and eigenvectors are those of its correlation matrix.

    The market component makes all assets look alike; without it, the clusters of assets
    that move together for other reasons stand out. The detoned matrix has rank N - m, so
    it is singular: it suits clustering (as the detoned codependence) but cannot be inverted
    by an optimizer.

    This is the detoning of López de Prado (2020), applied to the correlation matrix itself
    rather than to a denoised one; like skfolio, the zero eigenvalues are raised to 1e-13.

    See López de Prado, M. (2020), Machine Learning for Asset Managers, Cambridge
    University Press, chapter 2.

    Referenced against:
    - skfolio, DetoneCovariance (sample covariance): https://skfolio.org/generated/skfolio.moments.DetoneCovariance.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        components (int, optional): The number m of principal components removed.
            Defaults to 1.
        covariance (pd.DataFrame | None, optional): The covariance matrix per period to
            detone, in the order of the returns. Defaults to None, the sample covariance.

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    assets = returns.shape[1]
    if not 1 <= components < assets:
        raise ValueError(
            f"The number of components removed must lie between 1 and {assets - 1}."
        )
    sample = (
        returns.cov()
        if covariance is None
        else covariance.loc[returns.columns, returns.columns]
    ).to_numpy(dtype=float)
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
    The Minimum Covariance Determinant (MCD) covariance matrix.

    It is the covariance of the h periods whose covariance matrix has the smallest
    determinant, so that the periods far from the bulk of the returns (crashes, squeezes,
    data errors) do not inflate it. With T periods and N assets, h = ⌊(T + N + 1) / 2⌋ by
    default, the subset size with the highest breakdown point: the estimate withstands up to
    ⌊(T - N + 1) / 2⌋ outlying periods (Rousseeuw, 1985).

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
    MinCovDet. Without reweighting it is the raw covariance of the best subset, without the
    consistency factor.

    See Rousseeuw, P. J. (1985), "Multivariate Estimation with High Breakdown Point", in
    Grossmann, W., Pflug, G., Vincze, I. and Wertz, W. (eds.), Mathematical Statistics and
    Applications, Vol. B, Reidel, 283-297; Rousseeuw, P. J. and Van Driessen, K. (1999), "A
    Fast Algorithm for the Minimum Covariance Determinant Estimator", Technometrics, 41(3),
    212-223; and Croux, C. and Haesbroeck, G. (1999), "Influence Function and Efficiency of
    the Minimum Covariance Determinant Scatter Matrix Estimator", Journal of Multivariate
    Analysis, 71(2), 161-190.

    Referenced against:
    - scikit-learn, MinCovDet: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html

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

    Notes:
    - Unlike scikit-learn's MinCovDet, which starts from 30 random subsets of h periods, the
      search starts from 500: the random subsets differ, so the two can end in different
      local minima of the determinant, which on small samples moves the estimate by a few
      percent; periods far from the bulk are flagged by both. Without reweighting it is the
      covariance PyPortfolioOpt's min_cov_determinant returns.
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
    (such as similarities between assets).

    It is a planar graph that keeps the strongest links, 3(N - 2) of the N(N - 1) / 2, built
    from 4-cliques (tetrahedra) joined by 3-cliques (triangles, the separators).

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

    Ties in the gain go to the earliest face and then to the lowest vertex index.

    See Massara, G. P., Di Matteo, T. and Aste, T. (2017), "Network Filtering for Big Data:
    Triangulated Maximally Filtered Graph", Journal of Complex Networks, 5(2), 161-178.

    Shares similar logic:
    - Riskfolio-Lib, PMFG_T2s: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#DBHT.PMFG_T2s

    Args:
        weights (np.ndarray): A symmetric N × N matrix of non-negative weights, N ≥ 4.

    Returns:
        tuple[np.ndarray, np.ndarray]: The N - 3 4-cliques (rows of four vertex indices)
        and the N - 4 separators (rows of three vertex indices).

    Notes:
    - Riskfolio-Lib's PMFG_T2s gives the same graph on the test data; it compares the gains
      rounded to six decimals, so two gains within 1e-6 of each other can make it choose
      another face.
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
    The j-LoGo covariance matrix.

    It is the inverse of the sparse inverse covariance (precision matrix) that the
    Local-Global (LoGo) method of Barfuss, Massara, Di Matteo and Aste (2016) builds on the
    TMFG of the correlations. Two assets that the planar graph does not link are
    conditionally independent given the others, so most of the precision matrix is zero and
    far fewer parameters are estimated than in the full matrix.

    The TMFG is a tree of 4-cliques joined by separators, and for such a graph the
    precision matrix of a Gaussian follows from the local covariances alone. With S the
    sample covariance (ddof=1) and the TMFG of the similarities (1 + ρ_ij) / 2 (see
    get_triangulated_maximally_filtered_graph):

    - J = Σ_C [S_CC]⁻¹ - Σ_S [S_SS]⁻¹, each local inverse added to the rows and columns of
      its vertices, over the 4-cliques C and the separators S.
    - Σ = J⁻¹.

    The similarity (1 + ρ_ij) / 2 = 1 - d²_ij, with d_ij = √((1 - ρ_ij) / 2) the correlation
    distance, is the choice of Riskfolio-Lib's j-LoGo covariance. With four or fewer assets
    the graph links every pair and the estimate is the sample covariance.

    See Barfuss, W., Massara, G. P., Di Matteo, T. and Aste, T. (2016), "Parsimonious
    Modeling with Information Filtering Networks", Physical Review E, 94(6), 062306, and
    Massara, G. P., Di Matteo, T. and Aste, T. (2017), "Network Filtering for Big Data:
    Triangulated Maximally Filtered Graph", Journal of Complex Networks, 5(2), 161-178.

    Referenced against:
    - Riskfolio-Lib, covar_matrix (j-LoGo): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.covar_matrix

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
    definite matrices rather than along a straight line.

    With S the sample covariance (ddof=1), T the target and α the shrinkage intensity:

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

    See Bhatia, R. (2007), Positive Definite Matrices, Princeton University Press, and
    Musolas, A., Smith, S. T. and Marzouk, Y. (2021), "Geodesically Parameterized
    Covariance Estimation", SIAM Journal on Matrix Analysis and Applications, 42(2),
    528-556.

    Referenced against:
    - skfolio, GeodesicShrinkageCovariance: https://skfolio.org/generated/skfolio.moments.GeodesicShrinkageCovariance.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        shrinkage (float, optional): The shrinkage intensity α between 0 (the sample
            covariance) and 1 (the target). Defaults to 0.1.
        target (str, optional): The target. Options are: "scaled_identity" (the identity scaled by
            the average variance), "diagonal" (the variances). Defaults to "scaled_identity".

    Returns:
        pd.DataFrame: The covariance matrix per period.
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError("The shrinkage intensity must lie between 0 and 1.")
    check_option("target", target, ("scaled_identity", "diagonal"))
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
    The exponentially weighted covariance matrix rescaled by a short-term volatility update
    (STVU).

    The update is a multiplier φ that compares the risk the covariance predicted with the
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

    The short-term volatility update follows Paleologo (2025).

    See Paleologo, G. A. (2025), The Elements of Quantitative Investing, Wiley Finance,
    and Hawkins, D. M. and Maboudou-Tchao, E. M. (2008), "Multivariate Exponentially
    Weighted Moving Covariance Matrix", Technometrics, 50(2), 155-166.

    Referenced against:
    - skfolio, RegimeAdjustedEWCovariance (its defaults): https://skfolio.org/generated/skfolio.moments.RegimeAdjustedEWCovariance.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        half_life (float, optional): The half-life in periods of the exponential weights.
            Defaults to 40.

    Returns:
        tuple[pd.DataFrame, float]: The covariance matrix per period and the multiplier φ.

    Notes:
    - skfolio's RegimeAdjustedEWCovariance gives the same values with its defaults: a
      half-life of 40, the portfolio target with inverse volatility weights, the first
      moment method, the multiplier clipped to 0.7 and 1.6, returns taken around zero and no
      HAC correction.
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
    The distance correlation of every pair of assets: how much they depend on each other,
    linearly or not.

    It is zero only when their returns are independent, so it also captures non-linear
    co-movement that the Pearson correlation misses. It lies between 0 and 1.

    With a_st = |x_s - x_t| the distances between the returns of one asset in periods s
    and t, A the doubly centred distances (A_st = a_st - ā_s. - ā_.t + ā_..), and B the
    same for the other asset:

    - dCov²(x, y) = (1 / T²) Σ A_st B_st
    - dCor(x, y) = √(dCov²(x, y) / √(dCov²(x, x) dCov²(y, y)))

    It builds a T × T matrix per asset, so it grows with the
    square of the number of periods.

    See Székely, G. J., Rizzo, M. L. and Bakirov, N. K. (2007), "Measuring and Testing
    Dependence by Correlation of Distances", The Annals of Statistics, 35(6), 2769-2794.

    Referenced against:
    - skfolio, DistanceCorrelation: https://skfolio.org/generated/skfolio.distance.DistanceCorrelation.html

    Shares similar logic:
    - Riskfolio-Lib, dcorr_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.dcorr_matrix

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

    - The rule of Freedman and Diaconis (1981), ⌈(max - min) / (2 IQR / T^(1/3))⌉ per series
      (5 for a series without an interquartile range), and the larger of the two for a pair.
    - The rule of Hacine-Gharbi et al. (2012) for one series, round(z / 6 + 2 / (3 z) + 1 /
      3) with z = (8 + 324 T + 12 √(36 T + 729 T²))^(1/3), and of Hacine-Gharbi and Ravier
      (2018) for a pair with correlation ρ, round(√(1 + √(1 + 24 T / (1 - ρ²))) / √2), the
      choice of López de Prado (2020).

    See Freedman, D. and Diaconis, P. (1981), "On the Histogram as a Density Estimator:
    L2 Theory", Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 57,
    453-476, and Hacine-Gharbi, A. and Ravier, P. (2018), "A Binning Formula of
    Bi-histogram for Joint Entropy Estimation Using Mean Square Error Minimization",
    Pattern Recognition Letters, 101, 21-28.

    Shares similar logic:
    - skfolio, MutualInformation (Freedman-Diaconis): https://skfolio.org/generated/skfolio.distance.MutualInformation.html
    - Riskfolio-Lib, numBins (Hacine-Gharbi): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.numBins

    Args:
        first (np.ndarray): The returns of the first asset.
        second (np.ndarray | None, optional): The returns of the second asset, for a
            pair. Defaults to None.
        method (str, optional): The rule. Options are: "freedman" (Freedman and Diaconis),
            "hacine_gharbi" (Hacine-Gharbi). Defaults to "freedman".

    Returns:
        int: The number of bins.
    """
    check_option("method", method, BIN_METHODS)
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
    estimated from histograms of their returns.

    The mutual information I(X, Y) = H(X) + H(Y) - H(X, Y) measures how much knowing one
    return tells about the other, linear or not; the variation of information VI(X, Y) =
    H(X) + H(Y) - 2 I(X, Y) is the information they do not share, a metric distance between
    the assets.

    With p_ab the share of the periods in bin a of X and bin b of Y of the joint
    histogram, and p_a, p_b its margins (natural logarithms):

    - H(X) = -Σ p_a ln p_a, H(Y) likewise
    - I(X, Y) = Σ p_ab ln(p_ab / (p_a p_b))
    - normalized: I(X, Y) / min(H(X), H(Y)) between 0 and 1, and VI(X, Y) / H(X, Y)
      between 0 (identical) and 1 (independent)

    See Meilă, M. (2007), "Comparing Clusterings - An Information Based Distance",
    Journal of Multivariate Analysis, 98(5), 873-895, and López de Prado, M. (2020),
    Machine Learning for Asset Managers, Cambridge University Press, chapter 3.

    Referenced against:
    - skfolio, MutualInformation: https://skfolio.org/generated/skfolio.distance.MutualInformation.html
    - Riskfolio-Lib, mutual_info_matrix (Hacine-Gharbi bins): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.mutual_info_matrix

    Shares similar logic:
    - Riskfolio-Lib, var_info_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.var_info_matrix

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        bins (str | int, optional): The number of bins of every histogram, or the rule that sets it
            per pair (see get_number_of_bins). Options are: "freedman" (Freedman and Diaconis),
            "hacine_gharbi" (Hacine-Gharbi), or an integer. Defaults to "freedman".
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
    returns: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    The lower tail dependence of every pair of assets.

    It is the share of the periods in which one asset fell into its lowest alpha share of
    returns in which the other did too, λ_ij = P(r_j ≤ q_j(α) | r_i ≤ q_i(α)) at α = alpha,
    with q(α) the empirical quantile. It measures how often assets crash together, which the
    correlation, an average over all periods, does not show.

    Each pair comes from the empirical estimator of get_tail_dependence_coefficient
    (quantiles with linear interpolation).

    See Embrechts, P., McNeil, A. and Straumann, D. (2002), "Correlation and Dependence
    in Risk Management: Properties and Pitfalls", in Risk Management: Value at Risk and
    Beyond, Cambridge University Press, 176-223.

    Referenced against:
    - Riskfolio-Lib, ltdi_matrix: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#AuxFunctions.ltdi_matrix

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        alpha (float, optional): The share α of the lowest returns that forms the
            tail, between 0 and 0.5. Defaults to 0.05.

    Returns:
        pd.DataFrame: The lower tail dependence matrix, with ones on the diagonal.

    Notes:
    - Riskfolio-Lib's ltdi_matrix takes the k-th lowest return as the quantile, with k = ⌈T
      α⌉, and divides by k; both count the same periods whenever the k-th lowest return is
      the last one at or below the interpolated quantile.
    """
    if not 0 < alpha < 0.5:  # noqa: PLR2004
        raise InvalidArgumentError(
            f"alpha is the share of the lowest returns that forms the tail, between 0 and 0.5, "
            f"e.g. 0.05, not {alpha!r}. Please pass a value between 0 and 0.5 as alpha."
        )
    columns = returns.columns
    matrix = np.eye(len(columns))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            matrix[i, j] = matrix[j, i] = copula_model.get_tail_dependence_coefficient(
                returns.iloc[:, i], returns.iloc[:, j], q=1 - alpha
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

    Any codependence matrix can be passed, not only the Pearson correlation of the returns.

    See Varadi, D. (2013), "The Fast Threshold Clustering Algorithm (FTCA)", CSS
    Analytics.

    Referenced against:
    - ffn, calc_ftca: https://github.com/pmorissette/ffn/blob/v1.4.1/ffn/core.py#L2297

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
    How much every period counts in a co-moment.

    1 / T each for the sample, semi and CCC-GARCH co-moments (the latter on its scenarios),
    and (1 - α)^age normalized to sum to one, with α = 2 / (span + 1), for the exponential
    one.

    Args:
        periods (int): The number of periods T.
        method (str): The estimator. Options are: "sample" (equal weights), "exponential"
            (exponentially weighted), "semi" (lower semi co-moment), "ccc_garch" (CCC-GARCH
            scenarios).
        span (int): The span in periods of the exponential weights.

    Returns:
        np.ndarray: The weight of every period, oldest first.
    """
    check_option("method", method, COMOMENT_METHODS)
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
    standardized_moments: bool = True,
) -> pd.DataFrame:
    """
    The coskewness matrix M3, the third co-moments of the returns.

    N rows and N² columns with m_ijk = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k), with μ
    the average returns and w_t the weight of period t. The skewness of a portfolio with
    weights w is w'M3(w ⊗ w) / σ_p³, so M3 is what a mean-variance-skewness optimization or
    a modified VaR of a portfolio needs.

    - Sample: w_t = 1 / T.
    - Exponential: w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent periods
      weigh most. The deviations are taken from the average return, as in the exponential
      covariance.
    - Semi: the lower semi co-moment, with min(r_ti - μ_i, 0) in place of the deviations, so
      only returns below the average count, as defined by Cajas (2022).
    - CCC-GARCH: the next-period coskewness of the constant conditional correlation GARCH(1,
      1) model, the sample co-moment of its scenarios, the standardized residuals at the
      next-period volatilities (see get_ccc_garch_covariance).

    With standardized moments, every m_ijk is divided by σ_i σ_j σ_k, with σ² = Σ_t w_t
    (r_t - μ)² the weighted second moment of the full deviations, so the diagonal m_iii is
    the moment skewness of each asset.

    See Jondeau, E. and Rockinger, M. (2006), "Optimal Portfolio Allocation under Higher
    Moments", European Financial Management, 12(1), 29-55, and Martellini, L. and
    Ziemann, V. (2010), "Improved Estimates of Higher-Order Comoments and Implications for
    Portfolio Selection", The Review of Financial Studies, 23(4), 1467-1502.

    Shares similar logic:
    - PerformanceAnalytics (R), M3.MM (sample): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L34
    - PerformanceAnalytics (R), M3.ewma (exponential): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L1486
    - PortfolioAnalytics (R), CCCgarch.MM (CCC-GARCH): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): The estimator of the co-moments. Options are: "sample" (equal
            weights), "exponential" (exponentially weighted), "semi" (lower semi co-moment),
            "ccc_garch" (CCC-GARCH scenarios). Defaults to "sample".
        span (int, optional): The span in periods of the exponential weights. Defaults to 180.
        standardized_moments (bool, optional): Whether to divide by the product of the
            standard deviations. Defaults to True.

    Returns:
        pd.DataFrame: The coskewness matrix, one row per asset i and one column per pair
            of assets (j, k).

    Notes:
    - PerformanceAnalytics' M3.ewma uses the same weights (its default λ = 0.97 is a span of
      about 65.7) on returns it assumes to have a zero mean, so the two agree on demeaned
      returns.
    """
    if method == "ccc_garch":
        returns, method = get_ccc_garch_covariance(returns)[1], "sample"
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    weights = get_comoment_weights(periods, method, span)
    deviations = values - values.mean(axis=0)
    volatility = np.sqrt(weights @ deviations**2)
    if method == "semi":
        deviations = np.minimum(deviations, 0)
    pairs = (deviations[:, :, None] * deviations[:, None, :]).reshape(periods, -1)
    matrix = (deviations * weights[:, None]).T @ pairs
    if standardized_moments:
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
    standardized_moments: bool = True,
) -> pd.DataFrame:
    """
    The cokurtosis matrix M4, the fourth co-moments of the returns.

    N rows and N³ columns with m_ijkl = Σ_t w_t (r_ti - μ_i)(r_tj - μ_j)(r_tk - μ_k)(r_tl -
    μ_l), with μ the average returns and w_t the weight of period t. The kurtosis of a
    portfolio with weights w is w'M4(w ⊗ w ⊗ w) / σ_p⁴.

    - Sample: w_t = 1 / T.
    - Exponential: w_t = λ^(T - t) / Σ λ^(T - s) with λ = 1 - 2 / (span + 1), recent periods
      weigh most.
    - Semi: the lower semi cokurtosis, with min(r_ti - μ_i, 0) in place of the deviations
      (Cajas, 2022).
    - CCC-GARCH: the next-period cokurtosis of the constant conditional correlation GARCH(1,
      1) model, the sample co-moment of its scenarios (see get_ccc_garch_covariance).

    With standardized moments, every m_ijkl is divided by σ_i σ_j σ_k σ_l, with σ² = Σ_t w_t
    (r_t - μ)² the weighted second moment of the full deviations, so the diagonal m_iiii
    is the moment kurtosis of each asset (3 for a normal distribution, not the excess).

    See Jondeau, E. and Rockinger, M. (2006), "Optimal Portfolio Allocation under Higher
    Moments", European Financial Management, 12(1), 29-55, and Cajas, D. (2022), "Convex
    Optimization of Portfolio Kurtosis", SSRN 4202967.

    Shares similar logic:
    - PerformanceAnalytics (R), M4.MM (sample): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L72
    - PerformanceAnalytics (R), M4.ewma (exponential): https://github.com/cran/PerformanceAnalytics/blob/2.1.0/R/MultivariateMoments.R#L1543
    - Riskfolio-Lib, cokurt_matrix (sample and semi, as N² × N²): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.cokurt_matrix
    - PortfolioAnalytics (R), CCCgarch.MM (CCC-GARCH): https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): The estimator of the co-moments. Options are: "sample" (equal
            weights), "exponential" (exponentially weighted), "semi" (lower semi co-moment),
            "ccc_garch" (CCC-GARCH scenarios). Defaults to "sample".
        span (int, optional): The span in periods of the exponential weights. Defaults to 180.
        standardized_moments (bool, optional): Whether to divide by the product of the
            standard deviations. Defaults to True.

    Returns:
        pd.DataFrame: The cokurtosis matrix, one row per asset i and one column per
            triple of assets (j, k, l).
    """
    if method == "ccc_garch":
        returns, method = get_ccc_garch_covariance(returns)[1], "sample"
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
    if standardized_moments:
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
    simulations: int | None = None,
) -> np.ndarray:
    """
    The periods of one bootstrapped history of the same length (or of `simulations` of
    them).

    They are drawn in blocks so that the dependence between consecutive returns (volatility
    clustering) survives.

    - The stationary bootstrap of Politis and Romano (1994), blocks of a random length with
      a geometric distribution of mean block_size: every period continues the block with the
      next period (after the last, the first) with probability 1 - 1 / block_size and starts
      a new block at a random period otherwise.
    - The circular block bootstrap: blocks of block_size periods that start at a random
      period and wrap from the last period to the first (Politis and Romano, 1992).
    - The moving block bootstrap: blocks of block_size periods that start at a random period
      from which a whole block fits (Künsch, 1989).

    The arch package offers the same schemes.

    See Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap", Journal of
    the American Statistical Association, 89(428), 1303-1313, and Künsch, H. R. (1989),
    "The Jackknife and the Bootstrap for General Stationary Observations", The Annals of
    Statistics, 17(3), 1217-1241.

    Args:
        periods (int): The number of periods T of the history.
        method (str, optional): The bootstrap. Options are: "stationary" (stationary bootstrap),
            "circular" (circular block bootstrap), "moving" (moving block bootstrap). Defaults to
            "stationary".
        block_size (int, optional): The (average) number of periods in a block. Defaults
            to 3.
        generator (np.random.Generator | None, optional): The random number generator.
            Defaults to None, a new unseeded one.
        simulations (int | None, optional): The number of histories, each drawn from the generator
            in turn, so that they are the histories of as many calls with one history each. Defaults
            to None, one history.

    Returns:
        np.ndarray: The T periods, numbered from 0, of the bootstrapped history, or a row of
        them per history with `simulations`.
    """
    if not 1 <= block_size < periods:
        raise ValueError(
            f"The block size must lie between 1 and {periods - 1} for {periods} periods."
        )
    generator = generator if generator is not None else np.random.default_rng()
    count = 1 if simulations is None else simulations
    time = np.arange(periods)
    if method == "stationary":
        # The draws of every history in turn: whether each period starts a new block, then
        # the period each block would start at.
        draws = [
            (
                generator.random(periods) < 1 / block_size,
                generator.integers(0, periods, size=periods),
            )  # noqa: E501
            for _ in range(count)
        ]
        starts = np.array([new_block for new_block, _ in draws]).reshape(count, periods)
        random = np.array([first for _, first in draws]).reshape(count, periods)
        starts[:, 0] = True
        # Every period continues the block of the last start before it (or on it), one
        # period further for every period since that start.
        block_start = np.maximum.accumulate(np.where(starts, time, 0), axis=1)
        indices = (
            np.take_along_axis(random, block_start, axis=1) + time - block_start
        ) % periods
        return indices[0] if simulations is None else indices
    blocks = -(-periods // block_size)
    highest = periods if method == "circular" else periods - block_size + 1
    starts = np.array(
        [generator.integers(0, highest, size=blocks) for _ in range(count)]
    ).reshape(count, blocks)
    indices = (starts[:, :, None] + np.arange(block_size)).reshape(count, -1)[
        :, :periods
    ] % periods
    return indices[0] if simulations is None else indices


def get_bootstrap_estimates(
    returns: pd.DataFrame,
    method: str = "stationary",
    n_bootstrap: int = 1000,
    block_size: int = 3,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    How uncertain the average returns and the covariance matrix are.

    Both are estimated again on many simulated histories of the same length, and the spread
    of those estimates gives a confidence interval (box) for every average return and every
    covariance, and a standard error for every average return.

    - With the stationary, circular or moving block bootstrap, the histories are block
      bootstraps of the returns (see get_bootstrap_indices), which keep the dependence
      between consecutive returns.
    - With the normal distribution, the histories are drawn from a multivariate normal
      distribution with the average returns and the sample covariance matrix.

    The bounds are the (1 - c) / 2 and (1 + c) / 2 percentiles of the estimates at
    confidence level c, and the standard error is their standard deviation (ddof=1).
    The averages and covariances are those of the returns per period (ddof=1). The bounds of
    the covariances are per entry and are not adjusted to form positive semidefinite
    matrices.

    See Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap", Journal of
    the American Statistical Association, 89(428), 1303-1313, and Tütüncü, R. H. and
    Koenig, M. (2004), "Robust Asset Allocation", Annals of Operations Research, 132,
    157-187.

    Shares similar logic:
    - Riskfolio-Lib, bootstrapping: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.bootstrapping
    - Riskfolio-Lib, normal_simulation (normal distribution): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.normal_simulation
    - skfolio, BootstrapMuUncertaintySet: https://skfolio.org/generated/skfolio.uncertainty_set.BootstrapMuUncertaintySet.html

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        method (str, optional): How the histories are simulated. Options are: "stationary"
            (stationary bootstrap), "circular" (circular block bootstrap), "moving" (moving block
            bootstrap), "normal" (multivariate normal draws). Defaults to "stationary".
        n_bootstrap (int, optional): The number of simulated histories. Defaults to 1000, as
            skfolio's bootstrap uncertainty sets (Riskfolio-Lib draws 6000).
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

    Notes:
    - The parameters match Riskfolio-Lib's bootstrapping and normal_simulation (q = 1 - c,
      the number of simulations and the block size), whose box and ellipsoid uncertainty
      sets come from these estimates; the draws differ from its arch-based ones, so the
      results agree only in distribution.
    """
    check_option("method", method, BOOTSTRAP_METHODS)
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must lie between 0 and 1.")
    if n_bootstrap < 2:  # noqa: PLR2004
        raise ValueError("At least two bootstrap samples are needed.")
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    generator = np.random.default_rng(seed)
    average = values.mean(axis=0)
    sample = np.cov(values, rowvar=False)

    means = np.empty((n_bootstrap, assets))
    covariances = np.empty((n_bootstrap, assets, assets))
    indices = (
        get_bootstrap_indices(periods, method, block_size, generator, n_bootstrap)
        if method != "normal"
        else None
    )
    for simulation in range(n_bootstrap):
        history = (
            generator.multivariate_normal(average, sample, size=periods)
            if indices is None
            else values[indices[simulation]]
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
    returns (with a constant) selects.

    - Forward selection on the p-value: starting without factors, add the factor with the
      lowest p-value among those that keep every p-value of the model at or below the
      threshold, until no factor qualifies.
    - Backward selection on the p-value: starting with every factor, drop the factor with
      the highest p-value until every p-value is at or below the threshold. With either
      direction, when no factor qualifies the single factor with the lowest p-value is kept,
      so the model always holds one factor.
    - Selection on the Akaike or Schwarz information criterion, the R squared or the
      adjusted R squared: add (forward) or drop (backward) the factor that improves the
      criterion most (a lower information criterion, a higher R squared) for as long as it
      improves.

    The p-values are those of the t-statistics of the OLS coefficients and the criteria
    are defined as statsmodels defines them (see helpers.get_linear_regression).

    See Efroymson, M. A. (1960), "Multiple Regression Analysis", in Ralston, A. and Wilf,
    H. S. (eds.), Mathematical Methods for Digital Computers, Wiley; Akaike, H. (1974), "A
    New Look at the Statistical Model Identification", IEEE Transactions on Automatic
    Control, 19(6), 716-723; and Schwarz, G. (1978), "Estimating the Dimension of a Model",
    The Annals of Statistics, 6(2), 461-464.

    Shares similar logic:
    - Riskfolio-Lib, forward_regression: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.forward_regression
    - Riskfolio-Lib, backward_regression: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.backward_regression

    Args:
        returns (pd.Series): The returns per period of the asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods.
        direction (str, optional): Options are: "forward" (add factors), "backward" (drop factors).
            Defaults to "forward".
        criterion (str, optional): Options are: "pvalue" (p-value), "aic" (Akaike information
            criterion), "bic" (Schwarz information criterion), "r_squared" (R squared),
            "adjusted_r_squared" (adjusted R squared). Defaults to "pvalue".
        threshold (float, optional): The highest p-value a selected factor may have.
            Defaults to 0.05.

    Returns:
        list[str]: The selected factors, in the order they were selected (forward) or in
        the order of the factors (backward).
    """
    check_option("direction", direction, ("forward", "backward"))
    check_option("criterion", criterion, SELECTION_CRITERIA)
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
    asset's returns on factor returns (Jolliffe, 1982).

    The factors are standardized (ddof=1), their first k principal components Z = X_std V_k
    are the regressors of an OLS regression with a constant, and the coefficients γ of the
    components are mapped back to the factors: β = V_k γ / σ_f and the intercept α = mean(r)
    - Σ β_f mean(f). Leaving out the components with little variance removes the instability
      that correlated factors give OLS loadings.

    See Jolliffe, I. T. (1982), "A Note on the Use of Principal Components in
    Regression", Journal of the Royal Statistical Society, Series C, 31(3), 300-303.

    Referenced against:
    - Riskfolio-Lib, PCR: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.PCR

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
    The loadings matrix of a time series factor model.

    These are the intercept and factor loadings of the regression of every asset's returns
    on the factor returns, r_i,t = α_i + Σ_f β_i,f F_f,t + ε_i,t (Fama and French, 1993;
    Connor, 1995).

    - Ordinary least squares on every factor.
    - Forward or backward selection: the factors a stepwise regression selects (see
      get_stepwise_factors), the others with a loading of zero.
    - A principal components regression (see get_principal_components_loadings).

    See Fama, E. F. and French, K. R. (1993), "Common Risk Factors in the Returns on
    Stocks and Bonds", Journal of Financial Economics, 33(1), 3-56, and Connor, G.
    (1995), "The Three Types of Factor Models: A Comparison of Their Explanatory Power",
    Financial Analysts Journal, 51(3), 42-46.

    Referenced against:
    - Riskfolio-Lib, loadings_matrix (backward selection, every factor kept): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.loadings_matrix

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        factors (pd.DataFrame): The factor returns per period, on the same periods, without
            missing values.
        selection (str, optional): The selection of the factors. Options are: "ols" (every factor),
            "forward" (forward selection), "backward" (backward selection), "pcr" (principal
            components regression). Defaults to "ols".
        criterion (str, optional): The criterion of the stepwise selection. Options are: "pvalue"
            (p-value), "aic" (Akaike information criterion), "bic" (Schwarz information criterion),
            "r_squared" (R squared), "adjusted_r_squared" (adjusted R squared). Defaults to
            "pvalue".
        threshold (float, optional): The highest p-value of the stepwise selection.
            Defaults to 0.05.
        components (float, optional): The components of the principal components regression.
            Defaults to 0.95.

    Returns:
        pd.DataFrame: Per asset the Intercept (per period) and the loading of every factor.
    """
    check_option("selection", selection, FACTOR_SELECTIONS)
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
    The expected returns and covariance matrix a time series factor model implies (Connor,
    1995).

    With B the loadings, α the intercepts, μ_f and Σ_f the mean and sample covariance
    (ddof=1) of the factor returns and D the diagonal matrix of the variances (ddof=1) of
    the residuals ε = r - α - F B',

    - μ = α + B μ_f
    - Σ = B Σ_f B' + D

    The factors explain the co-movement of the assets and the residuals are taken to be
    uncorrelated, which leaves N K + N parameters to estimate instead of N (N + 1) / 2:
    a covariance matrix that is better conditioned when there are many assets. With an
    intercept, the OLS residuals average zero, so μ equals the average return of each asset
    over the same periods; the factor model changes the covariance, not the means.

    See Connor, G. (1995), "The Three Types of Factor Models: A Comparison of Their
    Explanatory Power", Financial Analysts Journal, 51(3), 42-46.

    Referenced against:
    - Riskfolio-Lib, risk_factors: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/parameters.html#ParamsEstimation.risk_factors

    Shares similar logic:
    - skfolio, FactorModel: https://skfolio.org/generated/skfolio.prior.FactorModel.html

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
    ranking: list[str] | None = None,
) -> pd.Series:
    """
    The posterior probabilities of the historical scenarios (periods) under views, by
    Entropy Pooling (Meucci, 2008).

    These are the probabilities p closest to the prior probabilities q (equal, 1 / T, by
    default) in relative entropy, KL(p ‖ q) = Σ p_t ln(p_t / q_t), that satisfy the views.
    The scenarios stay as they were; only their weights change, so the posterior keeps the
    fat tails and co-movements of the history.

    A view is linear in p: a mean view E_p[r_i] = Σ p_t r_i,t (or of r_i - r_j for a
    relative view "AAPL-MSFT"), a volatility view Σ p_t (r_i,t - m_i)² = σ² with m_i the
    mean after the mean views, and a correlation view Σ p_t r_i,t r_j,t = m_i m_j +
    ρ √(v_i v_j) with the means and variances after the mean and volatility views, which
    are then held at those values: the sequential treatment of views on higher moments of
    Meucci, Ardia and Keel (2011), in the order skfolio's EntropyPooling takes them. A
    view is an equality (a number) or an inequality, (">=", value) or ("<=", value).

    A ranking is the ordering view of Meucci (2008), its ranking example: with the assets
    listed from the highest expected return to the lowest, the inequalities E_p[r_(k+1)] ≤
    E_p[r_(k)] between every pair of neighbours, Σ p_t (r_(k+1),t - r_(k),t) ≤ 0, which are
    imposed with the mean views. The posterior then agrees with the ordering while staying
    as close to the prior as it can, so assets the history already ranks that way keep their
    means.

    The problem is solved through its dual (Csiszár, 1975; Boyd and Vandenberghe, 2004,
    5.2.3). With the constraints A'p = b and A'p ≤ b (every column of A scaled to unit
    length), the posterior has the exponential-family form p_t ∝ q_t exp(-a_t'λ), and the
    multipliers λ minimize ln Σ q_t exp(-a_t'λ) + λ'b, with λ ≥ 0 for the inequalities
    (SciPy's L-BFGS-B). The multipliers of the moments held fixed are bounded to ±1000, as
    in skfolio, which relaxes them when the views cannot be met exactly. A solution that
    misses a view by more than 1e-8 (on the scaled constraints) raises an error.

    See Meucci, A. (2008), "Fully Flexible Views: Theory and Practice", Risk, 21(10),
    97-102, and Meucci, A., Ardia, D. and Keel, S. (2011), "Fully Flexible Extreme Views",
    Journal of Risk, 14(2), 39-49.

    Referenced against:
    - skfolio, EntropyPooling: https://skfolio.org/generated/skfolio.prior.EntropyPooling.html

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
        ranking (list[str] | None, optional): Assets from the highest expected return to
            the lowest, at least two. Defaults to None.

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
    if ranking is not None:
        unknown = [str(name) for name in ranking if name not in names]
        if unknown or len(set(ranking)) != len(ranking) or len(ranking) <= 1:
            raise ValueError(
                "The ranking lists at least two different assets of the returns, from the "
                "highest expected return to the lowest"
                + (f"; unknown: {', '.join(unknown)}." if unknown else ".")
            )
        for higher, lower in itertools.pairwise(ranking):
            add_view(
                ("<=", 0.0),
                data[:, names.get_loc(lower)] - data[:, names.get_loc(higher)],
            )
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


def get_opinion_entropy_pooling(
    returns: pd.DataFrame, opinion: dict, periods_per_year: float
) -> pd.Series:
    """
    Calculate the entropy pooling probabilities of one opinion: its annualized views on
    expected returns and volatilities scaled to the period of the returns, its correlation
    views and its ranking, applied with get_entropy_pooling.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset.
        opinion (dict): The views, volatility_views, correlation_views and ranking of one
            opinion, each optional.
        periods_per_year (float): The number of return periods in a year.

    Returns:
        pd.Series: The posterior probability of every period.
    """
    unknown = set(opinion) - {
        "views",
        "volatility_views",
        "correlation_views",
        "ranking",
    }
    if unknown:
        raise ValueError(
            "An opinion holds views, volatility_views, correlation_views and "
            f"ranking, not {', '.join(sorted(unknown))}."
        )

    def per_period(views: dict | None, scale) -> dict | None:
        # The views are annualized; the scenarios are returns per period.
        if not views:
            return None
        return {
            name: (
                scale(value)
                if isinstance(value, int | float)
                else (value[0], scale(value[1]))
            )
            for name, value in views.items()
        }

    return get_entropy_pooling(
        returns,
        per_period(opinion.get("views"), lambda value: value / periods_per_year),
        per_period(
            opinion.get("volatility_views"),
            lambda value: value / np.sqrt(periods_per_year),
        ),
        opinion.get("correlation_views") or None,
        ranking=opinion.get("ranking"),
    )


POOLING_METHODS = ("linear", "logarithmic")


def get_opinion_pooling(
    probabilities: pd.DataFrame,
    opinion_weights: list[float] | None = None,
    method: str = "linear",
    divergence_penalty: float = 0.0,
) -> pd.Series:
    """
    The consensus of several sets of scenario probabilities, the opinions of different
    experts (e.g. each an Entropy Pooling posterior).

    The weight π_k of each opinion lies between 0 and 1; when they sum to less than one, the
    rest goes to the prior (equal probabilities).

    - Linear pooling: p = Σ π_k p_k, the arithmetic mixture (Stone, 1961), which keeps every
      scenario any opinion gives a probability.
    - Logarithmic pooling: p ∝ Π p_k^π_k, the geometric mixture (Genest and Zidek, 1986),
      which minimizes the weighted relative entropy to the opinions and gives no probability
      to a scenario that any opinion rules out.

    With a divergence penalty α > 0 the weights become π̃_k ∝ π_k exp(-α KL(p_k ‖ c)), with
    c = Σ π_k p_k the linear consensus, which weighs down the opinions far from the others;
    this robust weighting is the one skfolio's OpinionPooling (Delatte and Nicolini)
    defines.

    See Stone, M. (1961), "The Opinion Pool", The Annals of Mathematical Statistics,
    32(4), 1339-1342; Genest, C. and Zidek, J. V. (1986), "Combining Probability
    Distributions: A Critique and an Annotated Bibliography", Statistical Science, 1(1),
    114-135.

    Referenced against:
    - skfolio, OpinionPooling: https://skfolio.org/generated/skfolio.prior.OpinionPooling.html

    Args:
        probabilities (pd.DataFrame): The probability of every scenario (rows) per opinion
            (columns), each column summing to one.
        opinion_weights (list[float] | None, optional): The weight of every opinion.
            Defaults to None, equal weights summing to one.
        method (str, optional): The pooling. Options are: "linear" (arithmetic mixture),
            "logarithmic" (geometric mixture). Defaults to "linear".
        divergence_penalty (float, optional): The penalty α. Defaults to 0.

    Returns:
        pd.Series: The pooled probability of every scenario, summing to one.
    """
    check_option("method", method, POOLING_METHODS)
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


def get_silhouette_coefficients(
    euclidean: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    The silhouette coefficients of Rousseeuw (1987): how well every asset fits its cluster.

    They are S_i = (b_i - a_i) / max(a_i, b_i), with a_i the average distance of asset i to
    the other assets of its cluster and b_i the smallest average distance to the assets of
    another cluster (S_i = 0 for an asset alone in its cluster). See Rousseeuw, P. J.
    (1987), "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster
    Analysis", Journal of Computational and Applied Mathematics, 20, 53-65.

    Shares similar logic:
    - scikit-learn, silhouette_samples: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html

    Args:
        euclidean (np.ndarray): The N × N distances between the assets.
        labels (np.ndarray): The cluster of every asset.

    Returns:
        np.ndarray: The silhouette coefficient of every asset.
    """
    clusters = np.unique(labels)
    # The average distance of every asset to the assets of every cluster, itself left out.
    members = labels[:, None] == clusters[None, :]
    sizes = members.sum(axis=0) - members
    average = np.divide(
        euclidean @ members,
        sizes,
        out=np.full(members.shape, np.nan),
        where=sizes > 0,
    )
    within = average[members]
    between = np.where(members, np.inf, average).min(axis=1)
    alone = sizes[members] == 0
    with np.errstate(invalid="ignore"):
        return np.where(alone, 0.0, (between - within) / np.maximum(between, within))


# The k-means runs of every number of clusters that ONC tries, as in López de Prado (2020).
ONC_RESTARTS = 10


def get_optimal_number_of_clusters(
    distance: pd.DataFrame,
    maximum_clusters: int | None = None,
    restarts: int = ONC_RESTARTS,
    seed: int | None = 0,
) -> np.ndarray:
    """
    The clusters of the Optimal Number of Clusters (ONC) algorithm of López de Prado and
    Lewis (2019).

    K-means on the rows of the distance matrix, the number of clusters chosen by the quality
    of their silhouettes, and the clusters of below-average quality clustered again.

    - Every asset is a point whose coordinates are its row of the distance matrix (for the
      correlation distance d = √((1 - ρ) / 2)), so two assets are close when they have
      similar distances to all assets.
    - For k = 2 to K clusters, k-means (Lloyd's algorithm from k-means++ starting centres)
      runs `restarts` times; the clustering kept is the one, over every k and run, with the
      highest quality q = E[S] / √V[S] of the silhouette coefficients S of all assets (see
      get_silhouette_coefficients, with the Euclidean distances between the rows).
    - Every cluster gets its own quality, the mean over the standard deviation of the
      silhouettes of its assets. The clusters below the average quality are clustered again
      with the same steps, on the rows and columns of their assets. Their new clusters
      replace them when the average quality of all clusters, with the silhouettes taken
      again on all assets, is above the average quality of the clusters replaced.

    The clustering is that of López de Prado (2020, snippets 4.1 and 4.2): K = N - 1 by
    default, at most ten k-means runs per k, and the standard deviations without degrees of
    freedom correction. A cluster whose silhouettes are all equal (an asset alone, for
    instance) has no quality, so it is left as it is and not part of the averages; in the
    book its quality is undefined and then stops any further split. With fewer than three
    assets there is nothing to choose, so they form one cluster. The k-means starts are
    drawn from numpy's default generator with the seed, so the same distances give the same
    clusters, and the clusters are numbered in the order of their first asset.

    As in the book's snippet 4.2, the clusters are clustered again when at least two are
    below the average quality, and the new clusters are kept when their quality is above
    that of the clusters replaced.

    See López de Prado, M. and Lewis, M. J. (2019), "Detection of False Investment
    Strategies Using Unsupervised Learning Methods", Quantitative Finance, 19(9), 1555-1565,
    and López de Prado, M. (2020), Machine Learning for Asset Managers, Cambridge University
    Press, chapter 4.

    Shares similar logic:
    - RiskLabAI, cluster_k_means_top (López de Prado, Machine Learning for Asset Managers, snippet 4.2): https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/cluster/clustering.py#L194

    Args:
        distance (pd.DataFrame): The N × N distance matrix of the assets.
        maximum_clusters (int | None, optional): The largest number of clusters K tried.
            Defaults to None, N - 1.
        restarts (int, optional): The k-means runs per number of clusters. Defaults to 10.
        seed (int | None, optional): The seed of the k-means starts (numpy's default
            generator). Defaults to 0.

    Returns:
        np.ndarray: The cluster of every asset, numbered from 0 in the order of the assets.

    Notes:
    - Unlike RiskLabAI's cluster_k_means_top, which clusters with scikit-learn's KMeans, the
      k-means runs here are drawn from numpy's default generator with the seed.
    """
    matrix = distance.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:  # noqa: PLR2004
        raise ValueError("The distance matrix must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "The distance matrix must not hold missing or infinite values."
        )
    if restarts < 1:
        raise ValueError("The number of k-means runs must be at least 1.")
    if maximum_clusters is not None and maximum_clusters < 2:  # noqa: PLR2004
        raise ValueError("The largest number of clusters must be at least 2.")
    rng = np.random.default_rng(seed)

    def k_means(points: np.ndarray, count: int) -> np.ndarray:
        # k-means++ starting centres: the first a random point, every next one a point
        # drawn with a probability proportional to its squared distance to the nearest
        # centre so far.
        centres = [points[rng.integers(len(points))]]
        for _ in range(1, count):
            nearest = np.min(
                ((points[:, None, :] - np.array(centres)[None]) ** 2).sum(axis=2),
                axis=1,
            )
            total = nearest.sum()
            if total <= 0:
                # Fewer distinct points than clusters; another point is as good as any.
                centres.append(points[rng.integers(len(points))])
            else:
                centres.append(points[rng.choice(len(points), p=nearest / total)])
        centres = np.array(centres)
        labels = np.full(len(points), -1)
        for _ in range(300):
            squared = ((points[:, None, :] - centres[None]) ** 2).sum(axis=2)
            new_labels = np.argmin(squared, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(count):
                members = labels == cluster
                if members.any():
                    centres[cluster] = points[members].mean(axis=0)
                else:
                    # An empty cluster takes the point farthest from its own centre.
                    farthest = int(np.argmax(squared[np.arange(len(points)), labels]))
                    centres[cluster] = points[farthest]
                    labels[farthest] = cluster
        return labels

    def quality(silhouettes: np.ndarray) -> float:
        spread = silhouettes.std()
        if spread > 0:
            return float(silhouettes.mean() / spread)
        return np.inf if silhouettes.mean() > 0 else -np.inf

    def cluster_qualities(silhouettes: np.ndarray, labels: np.ndarray) -> dict:
        # The quality per cluster; one whose silhouettes are all equal has none.
        return {
            label: float(values.mean() / values.std())
            for label in np.unique(labels)
            if (values := silhouettes[labels == label]).std() > 0
        }

    def base(points: np.ndarray, largest: int) -> tuple[np.ndarray, np.ndarray]:
        euclidean = squareform(pdist(points))
        best, best_labels, best_silhouettes = -np.inf, None, None
        for _ in range(restarts):
            for count in range(2, largest + 1):
                labels = k_means(points, count)
                silhouettes = get_silhouette_coefficients(euclidean, labels)
                score = quality(silhouettes)
                if best_labels is None or score > best:
                    best, best_labels, best_silhouettes = score, labels, silhouettes
        return best_labels, best_silhouettes

    def top(sub: np.ndarray, largest: int) -> np.ndarray:
        count = len(sub)
        largest = min(largest, count - 1)
        if largest < 2:  # noqa: PLR2004
            return np.zeros(count, dtype=int)
        labels, silhouettes = base(sub, largest)
        qualities = cluster_qualities(silhouettes, labels)
        if not qualities:
            return labels
        average = np.mean(list(qualities.values()))
        redo = [label for label, value in qualities.items() if value < average]
        if len(redo) <= 1:
            return labels
        selected = np.isin(labels, redo)
        inner = top(sub[np.ix_(selected, selected)], largest)
        new_labels = labels.copy()
        new_labels[selected] = labels.max() + 1 + inner
        new_qualities = cluster_qualities(
            get_silhouette_coefficients(squareform(pdist(sub)), new_labels), new_labels
        )
        redo_average = np.mean([qualities[label] for label in redo])
        if new_qualities and np.mean(list(new_qualities.values())) > redo_average:
            return new_labels
        return labels

    count = len(matrix)
    labels = top(matrix, count - 1 if maximum_clusters is None else maximum_clusters)
    # Numbered in the order of the first asset of every cluster.
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(np.argsort(first))
    return order[np.searchsorted(np.unique(labels), labels)]


def get_gnpr_distance_matrix(
    returns: pd.DataFrame, theta: float = 0.5, bandwidth: float = 0.01
) -> pd.DataFrame:
    """
    The distance of the Generic Non-Parametric Representation (GNPR) of Marti, Andler,
    Nielsen and Donnat (2016) between every pair of assets.

    It is a mix of how differently two assets move (their copula) and how differently their
    returns are distributed (their marginals), weighted by θ.

    - d_θ² = θ d_1² + (1 - θ) d_0²
    - d_1² = 3 / (T (T² - 1)) Σ_t (R_i,t - R_j,t)², with R the ranks of the returns (ties
      averaged), which is (1 - ρ_S) / 2 without ties for the Spearman correlation ρ_S
    - d_0² = ½ Σ_b (√g_i,b - √g_j,b)², the squared Hellinger distance between the
      histograms g of the two assets' returns, with bins of width `bandwidth` from the lowest
      to the highest return of the pair

    θ = 1 compares only the co-movement, θ = 0 only the distributions; between them two assets
    that move together but have different volatilities or fat tails stay apart. With θ
    between 0 and 1 the distance lies between 0 and 1 and is a metric.

    d_1 is taken on the ranks (the empirical copula), as Marti et al. define it. Marti et al.
    compare the densities of the returns in d_0; here they are estimated with histograms of
    a fixed bin width, `bandwidth`, laid from the lowest to the highest return of the pair.

    See Marti, G., Andler, S., Nielsen, F. and Donnat, P. (2016), "Clustering Financial Time
    Series: How Long Is Enough?", Proceedings of the Twenty-Fifth International Joint
    Conference on Artificial Intelligence (IJCAI-16), 2583-2589, and Marti, G. (2017), "Some
    Contributions to the Clustering of Financial Time Series and Applications to Credit
    Default Swaps", PhD thesis, École Polytechnique.

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        theta (float, optional): The weight θ of the co-movement, between 0 and 1.
            Defaults to 0.5.
        bandwidth (float, optional): The width of the histogram bins, in the units of the
            returns. Defaults to 0.01.

    Returns:
        pd.DataFrame: The distance matrix, with zeros on the diagonal.
    """
    if not 0 <= theta <= 1:
        raise ValueError("The weight theta must lie between 0 and 1.")
    if bandwidth <= 0:
        raise ValueError("The bandwidth must be positive.")
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    if periods < 2:  # noqa: PLR2004
        raise ValueError("The GNPR distance needs at least two periods.")
    ranks = returns.rank().to_numpy(dtype=float)
    copula = (
        3
        / (periods * (periods**2 - 1))
        * ((ranks[:, :, None] - ranks[:, None, :]) ** 2).sum(axis=0)
    )
    marginal = np.zeros((assets, assets))
    for first, second in itertools.combinations(range(assets), 2):
        pair = values[:, [first, second]]
        edges = np.arange(pair.min(), pair.max() + bandwidth, bandwidth)
        if len(edges) < 2:  # noqa: PLR2004
            edges = np.array([pair.min(), pair.min() + bandwidth])
        histograms = [np.histogram(column, edges)[0] / periods for column in pair.T]
        marginal[first, second] = marginal[second, first] = 0.5 * np.sum(
            (np.sqrt(histograms[0]) - np.sqrt(histograms[1])) ** 2
        )
    distance = np.sqrt(np.clip(theta * copula + (1 - theta) * marginal, 0, None))
    np.fill_diagonal(distance, 0)
    return pd.DataFrame(distance, index=returns.columns, columns=returns.columns)


# The number of random splits NERCOME averages over, Lam's (2016) M = 50.
NERCOME_SPLITS = 50


def get_nercome_covariance(
    returns: pd.DataFrame, splits: int = NERCOME_SPLITS, seed: int | None = 0
) -> pd.DataFrame:
    """
    The Nonparametric Eigenvalue-Regularized COvariance Matrix Estimator (NERCOME) of Lam
    (2016), which builds on the split-sample estimator of Abadir, Distaso and Žikeš (2014).

    The eigenvectors come from one part of the periods and the eigenvalues from the other,
    so the eigenvalues are not fitted to the noise in the eigenvectors that the sample
    covariance overfits.

    - The T periods are put in a random order and split into the first m and the other
      T - m, with sample covariances S_1 and S_2 (ddof=1).
    - Σ_m = P_1 diag(P_1' S_2 P_1) P_1', with P_1 the eigenvectors of S_1: the eigenvectors of
      the first part with the variances the second part shows along them.
    - The estimate is the average of Σ_m over M random splits.
    - The split m is chosen from 2√T, 0.2T, 0.4T, 0.6T, 0.8T, T - 2.5√T and T - 1.5√T
      (rounded) as the one with the smallest ‖(1 / M) Σ_s (Σ_m,s - S_2,s)‖² (Frobenius), the
      practical choice of Lam (2016, section 4); splits of N periods or fewer, whose S_1 is
      singular, are left out.

    The estimate is positive semi-definite, and positive definite when the first part has more
    periods than there are assets. The random orders are drawn from numpy's default
    generator with the seed, the same orders for every candidate split, so the same returns
    give the same estimate.

    See Abadir, K. M., Distaso, W. and Žikeš, F. (2014), "Design-Free Estimation of Variance
    Matrices", Journal of Econometrics, 181(2), 165-180, and Lam, C. (2016), "Nonparametric
    Eigenvalue-Regularized Precision or Covariance Matrix Estimator", The Annals of
    Statistics, 44(3), 928-953.

    Shares similar logic:
    - RiskLabAI, nercome_denoised_covariance: https://github.com/RiskLabAI/RiskLabAI.py/blob/v3.2.0/src/RiskLabAI/data/denoise/nercome.py#L54

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        splits (int, optional): The number M of random splits averaged. Defaults to 50.
        seed (int | None, optional): The seed of the random orders (numpy's default
            generator). Defaults to 0.

    Returns:
        pd.DataFrame: The covariance matrix per period.

    Notes:
    - Unlike RiskLabAI's nercome_denoised_covariance, which splits at 2T / 3 and estimates
      the correlation matrix of the standardized returns, scaled back with the sample
      standard deviations, the covariance itself is estimated, as in Lam (2016), so the
      variances are regularized as well.
    """
    if splits < 1:
        raise ValueError("The number of splits must be at least 1.")
    values = returns.to_numpy(dtype=float)
    periods, assets = values.shape
    candidates = sorted(
        {
            int(round(size))
            for size in (
                2 * np.sqrt(periods),
                0.2 * periods,
                0.4 * periods,
                0.6 * periods,
                0.8 * periods,
                periods - 2.5 * np.sqrt(periods),
                periods - 1.5 * np.sqrt(periods),
            )
            if assets < round(size) <= periods - 2
        }
    )
    if not candidates:
        raise ValueError(
            "NERCOME needs a first part with more periods than assets and a second part "
            f"of at least two periods; {periods} periods are too few for {assets} assets."
        )
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(periods) for _ in range(splits)]
    results = []
    for size in candidates:
        estimates, differences = np.zeros((assets, assets)), np.zeros((assets, assets))
        for order in orders:
            first = np.cov(values[order[:size]], rowvar=False, ddof=1)
            second = np.cov(values[order[size:]], rowvar=False, ddof=1)
            vectors = np.linalg.eigh(first)[1]
            estimate = (
                vectors * np.einsum("ki,kl,li->i", vectors, second, vectors)
            ) @ vectors.T
            estimates += estimate / splits
            differences += (estimate - second) / splits
        results.append((float(np.sum(differences**2)), estimates))
    # The first split size with the smallest error, as the sizes are tried in order.
    best = min(results, key=lambda result: result[0])[1]
    best = (best + best.T) / 2
    return pd.DataFrame(best, index=returns.columns, columns=returns.columns)


def get_ccc_garch_covariance(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The next-period covariance matrix of the Constant Conditional Correlation (CCC) GARCH
    model of Bollerslev (1990).

    Every asset has its own GARCH(1, 1) volatility and the correlations between the
    standardized residuals are constant.

    - ε_t = r_t - r̄, the returns around their average
    - σ²_i,t = ω_i + α_i ε²_i,t-1 + β_i σ²_i,t-1 per asset, fitted by Gaussian (quasi-)
      maximum likelihood from the backcast (see risk.volatility.garch_model.get_garch_weights)
    - R = the correlation matrix of the standardized residuals z_t = ε_t / σ_t
    - Σ_T+1 = D_T+1 R D_T+1 with D_T+1 = diag(σ_i,T+1), σ²_i,T+1 = ω_i + α_i ε²_i,T + β_i σ²_i,T

    With constant correlations the likelihood splits into a part per asset and the
    correlations, so the two-step estimate is the maximum likelihood one for R given the
    volatilities. The covariance follows the current volatility of every asset, which a
    sample covariance averages away. The scenarios returned are the standardized residuals
    rescaled to the next-period volatilities, z_i,t σ_i,T+1 / s(z_i) with s the sample
    standard deviation (ddof=1), whose sample covariance is Σ_T+1 and whose co-moments are
    the next-period coskewness and cokurtosis of the model.

    The GARCH(1, 1) legs follow the same model as the arch package's zero-mean GARCH on the
    demeaned returns.

    See Bollerslev, T. (1990), "Modelling the Coherence in Short-Run Nominal Exchange Rates:
    A Multivariate Generalized ARCH Model", The Review of Economics and Statistics, 72(3),
    498-505, and, for the scenarios rescaled to the next-period volatilities, Boudt, K.,
    Carl, P. and Peterson, B. (2013), "Asset Allocation with Conditional Value-at-Risk
    Budgets", Journal of Risk, 15(3), 39-68.

    Shares similar logic:
    - PortfolioAnalytics (R), CCCgarch.MM: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/moment.functions.R#L22

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The next-period covariance matrix and the
            scenarios, the rescaled standardized residuals per period.

    Notes:
    - PortfolioAnalytics' CCCgarch.MM (Boudt, Peterson and Carl) builds the same moments
      from fGarch's GARCH(1, 1) fits and falls back on a constant volatility when a fitted
      α_i is below 0.01; here the fitted volatility is kept.
    """
    if len(returns) < 100:  # noqa: PLR2004
        raise ValueError("The CCC-GARCH model needs at least 100 periods.")
    residuals = (returns - returns.mean()).to_numpy(dtype=float)
    standardized = np.empty_like(residuals)
    volatility = np.empty(residuals.shape[1])
    for column, name in enumerate(returns.columns):
        omega, alpha, beta = garch_model.get_garch_weights(residuals[:, column])
        if np.isnan([omega, alpha, beta]).any():
            raise ValueError(f"The GARCH model of {name} did not converge.")
        variance = garch_model.get_garch(residuals[:, column], [omega, alpha, beta])
        standardized[:, column] = residuals[:, column] / np.sqrt(variance)
        volatility[column] = np.sqrt(
            omega + alpha * residuals[-1, column] ** 2 + beta * variance[-1]
        )
    correlation = np.corrcoef(standardized, rowvar=False)
    covariance = correlation * np.outer(volatility, volatility)
    scenarios = standardized * volatility / standardized.std(axis=0, ddof=1)
    return (
        pd.DataFrame(covariance, index=returns.columns, columns=returns.columns),
        pd.DataFrame(scenarios, index=returns.index, columns=returns.columns),
    )


ROBUST_COVARIANCE_METHODS = ("mm", "rocke")

# Rocke's translated biweight spans the squared distances within γ = min(χ²_N(1 - α) / N - 1,
# 1) of their scale, with α = 0.005 as in Maronna, Martin, Yohai and Salibián-Barrera (2019).
ROCKE_ALPHA = 0.005

# The iterations stop once no location or shape entry moves by more than this, relative to
# the scale, or after the maximum number of iterations.
ROBUST_TOLERANCE = 1e-12
ROBUST_ITERATIONS = 1000


def get_bisquare_constants(
    assets: int, delta: float, efficiency: float
) -> tuple[float, float]:
    """
    The tuning constants of Tukey's bisquare for the S-estimator and the MM-estimator of the
    location and scatter of N assets at the normal distribution.

    The bisquare is ρ(d) = 1 - (1 - (d / c)²)³ for d ≤ c and 1 above.

    - The S-estimator's c_0 makes it consistent at the normal distribution: E[ρ(√X)] = δ for
      X ~ χ²_N, which in closed form is P(X > c²) + E[3 X / c² - 3 X² / c⁴ + X³ / c⁶; X ≤ c²]
      with E[X^k; X ≤ a] = N (N + 2) ⋯ (N + 2k - 2) F_(N+2k)(a).
    - The MM-estimator's c_1 gives the shape the asymptotic efficiency `efficiency` relative
      to the sample covariance, 1 / σ_1 with σ_1 = N (N + 2) E[ψ(d)² d²] / E[ψ'(d) d² +
      (N + 1) ψ(d) d]² for d ~ χ_N and ψ = ρ' (Lopuhaä, 1989, equation 5.5;
      Salibián-Barrera, Van Aelst and Willems, 2006), integrated by Gauss-Legendre quadrature.

    See Lopuhaä, H. P. (1989), "On the Relation between S-Estimators and M-Estimators of
    Multivariate Location and Covariance", The Annals of Statistics, 17(4), 1662-1683, and
    Salibián-Barrera, M., Van Aelst, S. and Willems, G. (2006), "Principal Components
    Analysis Based on Multivariate MM Estimators with Fast and Robust Bootstrap", Journal of
    the American Statistical Association, 101(475), 1198-1211.

    Referenced against:
    - statsmodels, tukeybiweight_mvmean_eff (shape efficiency): https://github.com/statsmodels/statsmodels/blob/v0.15.0/statsmodels/robust/tools.py#L586

    Shares similar logic:
    - statsmodels, eff_mvshape: https://github.com/statsmodels/statsmodels/blob/v0.15.0/statsmodels/robust/tools.py#L414

    Args:
        assets (int): The number of assets N.
        delta (float): The average ρ δ of the S-estimator, between 0 and 1.
        efficiency (float): The shape efficiency of the MM-estimator, between 0 and 1.

    Returns:
        tuple[float, float]: The constants c_0 and c_1.
    """

    def average_rho(c: float) -> float:
        square = c**2
        moments = [
            np.prod([assets + 2 * j for j in range(k)])
            * stats.chi2.cdf(square, assets + 2 * k)
            for k in (1, 2, 3)
        ]
        return float(
            stats.chi2.sf(square, assets)
            + 3 * moments[0] / square
            - 3 * moments[1] / square**2
            + moments[2] / square**3
        )

    # Gauss-Legendre nodes on [0, c]: the integrands are polynomials in d times the χ_N
    # density, smooth on the interval, so 200 nodes integrate them to machine precision.
    nodes, node_weights = np.polynomial.legendre.leggauss(200)

    def shape_efficiency(c: float) -> float:
        d = c * (nodes + 1) / 2
        weights = node_weights * c / 2 * stats.chi.pdf(d, assets)
        u = (d / c) ** 2
        psi = d * (1 - u) ** 2
        derivative = (1 - u) ** 2 - 4 * u * (1 - u)
        numerator = weights @ (psi**2 * d**2)
        denominator = weights @ (derivative * d**2 + (assets + 1) * psi * d)
        return float(denominator**2 / (assets * (assets + 2) * numerator))

    upper = np.sqrt(stats.chi2.ppf(1 - 1e-12, assets)) * 10
    first = brentq(lambda c: average_rho(c) - delta, 1e-3, upper, xtol=1e-14)
    # The S-estimator's own efficiency rises with the number of assets; when it already
    # reaches the efficiency asked for, the MM step keeps its constant.
    if shape_efficiency(first) >= efficiency:
        return float(first), float(first)
    second = brentq(
        lambda c: shape_efficiency(c) - efficiency, first, upper, xtol=1e-12
    )
    return float(first), float(second)


def get_robust_covariance(
    returns: pd.DataFrame,
    method: str = "mm",
    efficiency: float = 0.95,
    seed: int | None = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    The robust location and covariance matrix of the returns by an MM-estimator or Rocke's
    S-estimator, which outliers barely move.

    Both are iteratively reweighted means and covariances in which periods far from the bulk
    of the returns (in Mahalanobis distance) get a weight that falls to zero, so crashes,
    squeezes and data errors barely move them. An S-estimator (Davies, 1987) is the location
    μ and scatter Σ = s V with det(V) = 1 that give the smallest scale s solving (1 / T) Σ_t
    ρ(d²_t / s) = δ, with d²_t = (r_t - μ)' V⁻¹ (r_t - μ) and δ = (1 - N / T) / 2, the
    largest breakdown point: the estimate withstands almost half the periods being outliers
    (Lopuhaä and Rousseeuw, 1991). It is found by iterating μ = Σ w_t r_t / Σ w_t and V ∝ Σ
    w_t (r_t - μ)(r_t - μ)' with the weights w_t = ρ'(d²_t / s), halving the step whenever
    the scale would rise.

    - The MM-estimator (Tatsuoka and Tyler, 2000; Salibián-Barrera, Van Aelst and Willems,
      2006). First an S-estimator with Tukey's bisquare, ρ(d) = 1 - (1 - (d / c_0)²)³ for d
      ≤ c_0, consistent at the normal distribution. Then, with its scale s held fixed, the
      M-estimator with a bisquare of a larger c_1 that minimizes Σ_t ρ(d_t / √s) over μ and
      V, by the same reweighting: Σ = s V. The second step brings the efficiency of the
      shape at the normal distribution to `efficiency` (see get_bisquare_constants) and
      keeps the breakdown point of the first.
    - Rocke's S-estimator, with the translated biweight of Rocke (1996) on the squared
      distances, ρ(x) = 0 for x < 1 - γ, ((x - 1) / (4γ)) (3 - ((x - 1) / γ)²) + ½ for |x -
      1| ≤ γ and 1 for x > 1 + γ, with γ = min(χ²_N(1 - α) / N - 1, 1) and α = 0.005
      (Maronna, Martin, Yohai and Salibián-Barrera, 2019). Its weights vanish at both small
      and large distances, which keeps its efficiency and robustness as the number of assets
      grows, where the bisquare S-estimator becomes ever less robust. The scatter s V is
      made consistent at the normal distribution by matching the median of the squared
      distances to the median of χ²_N.

    The S-iteration starts from the minimum covariance determinant estimate (see
    get_minimum_covariance_determinant, with the seed) with the coordinatewise median as the
    location, and from the sample mean and covariance; the one that ends at the smaller scale
    is kept. The iterations stop when no entry of the location and shape moves by more than
    1e-12 (relative to the scale), so the estimate is the fixed point rather than wherever a
    looser stop leaves it, and the same returns with the same seed give the same estimate.

    The estimates follow the papers, with the bisquare constants of statsmodels, and are
    checked to recover the covariance of normal returns with outliers added.

    See Davies, P. L. (1987), "Asymptotic Behaviour of S-Estimates of Multivariate Location
    Parameters and Dispersion Matrices", The Annals of Statistics, 15(3), 1269-1292; Lopuhaä,
    H. P. and Rousseeuw, P. J. (1991), "Breakdown Points of Affine Equivariant Estimators of
    Multivariate Location and Covariance Matrices", The Annals of Statistics, 19(1), 229-248;
    Rocke, D. M. (1996), "Robustness Properties of S-Estimators of Multivariate Location and
    Shape in High Dimension", The Annals of Statistics, 24(3), 1327-1345; Tatsuoka, K. S. and
    Tyler, D. E. (2000), "On the Uniqueness of S-Functionals and M-Functionals under
    Nonelliptical Distributions", The Annals of Statistics, 28(4), 1219-1243; and Maronna, R.
    A., Martin, R. D., Yohai, V. J. and Salibián-Barrera, M. (2019), Robust Statistics:
    Theory and Methods (with R), 2nd edition, Wiley, chapter 6.

    Shares similar logic:
    - PortfolioAnalytics (R), custom.covRob.MM: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/custom.covRob.R#L29
    - PortfolioAnalytics (R), custom.covRob.Rocke: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/custom.covRob.R#L57

    Args:
        returns (pd.DataFrame): The returns per period, at least two assets, without missing
            values.
        method (str, optional): The estimator. Options are: "mm" (MM-estimator), "rocke" (Rocke's
            S-estimator). Defaults to "mm".
        efficiency (float, optional): The shape efficiency of the MM-estimator at the normal
            distribution, between 0.5 and 0.99. Defaults to 0.95.
        seed (int | None, optional): The seed of the minimum covariance determinant start.
            Defaults to 0.

    Returns:
        tuple[pd.Series, pd.DataFrame]: The location and the covariance matrix per period.

    Notes:
    - PortfolioAnalytics' custom.covRob.MM and custom.covRob.Rocke call RobStatTM's covRobMM
      and covRobRocke, which start from the kurtosis-based estimate of Peña and Prieto
      (2007); from another start, the iterations can end in another local minimum of the
      scale on small samples.
    """
    check_option("method", method, ROBUST_COVARIANCE_METHODS)
    if not 0.5 <= efficiency <= 0.99:  # noqa: PLR2004
        raise ValueError("The efficiency must lie between 0.5 and 0.99.")
    data = returns.to_numpy(dtype=float)
    periods, assets = data.shape
    if assets < 2:  # noqa: PLR2004
        raise ValueError("The robust covariance needs at least two assets.")
    if periods <= 2 * assets:
        raise ValueError(
            "The robust covariance needs more than twice as many periods as assets."
        )
    delta = 0.5 * (1 - assets / periods)
    if method == "mm":
        tuning, mm_tuning = get_bisquare_constants(assets, delta, efficiency)

        def rho(x: np.ndarray) -> np.ndarray:
            u = np.minimum(x / tuning**2, 1)
            return 1 - (1 - u) ** 3

        def weight(x: np.ndarray) -> np.ndarray:
            return np.where(x < tuning**2, (1 - x / tuning**2) ** 2, 0.0)

    else:
        gamma = min(stats.chi2.ppf(1 - ROCKE_ALPHA, assets) / assets - 1, 1.0)

        def rho(x: np.ndarray) -> np.ndarray:
            z = np.clip((x - 1) / gamma, -1, 1)
            return 0.5 + 0.25 * z * (3 - z**2)

        def weight(x: np.ndarray) -> np.ndarray:
            z = (x - 1) / gamma
            return np.where(np.abs(z) < 1, 1 - z**2, 0.0)

    def squared_distances(location: np.ndarray, shape: np.ndarray) -> np.ndarray:
        deviations = data - location
        return np.einsum("ti,ij,tj->t", deviations, np.linalg.inv(shape), deviations)

    def scale_of(distances: np.ndarray) -> float:
        # The scale s solving mean ρ(d² / s) = δ; the mean falls as s grows, from the share
        # of positive distances (s near 0) to 0, so the root is bracketed in logarithms.
        positive = distances[distances > 0]
        if len(positive) == 0:
            raise ValueError("Every period lies at the location; there is no scatter.")

        def excess(logarithm: float) -> float:
            return float(rho(distances / np.exp(logarithm)).mean() - delta)

        low = np.log(positive.min()) - 10
        high = np.log(positive.max()) + 10
        return float(np.exp(brentq(excess, low, high, xtol=1e-15)))

    def unit(scatter: np.ndarray) -> np.ndarray:
        scatter = (scatter + scatter.T) / 2
        sign, logarithm = np.linalg.slogdet(scatter)
        if sign <= 0:
            raise ValueError("The weighted covariance of the returns is singular.")
        return scatter / np.exp(logarithm / assets)

    def reweighted(location, shape, weights) -> tuple[np.ndarray, np.ndarray]:
        if weights.sum() <= 0:
            raise ValueError("Every period got a zero weight.")
        new_location = weights @ data / weights.sum()
        deviations = data - new_location
        return new_location, unit((deviations * weights[:, None]).T @ deviations)

    def iterate(location, shape, objective, weights_of, fixed_scale=None):
        # Reweighting until the reweighted location and shape no longer move, with the step
        # halved (up to ten times) whenever it would raise the objective by more than its
        # numerical precision (1e-12 relative: near the fixed point the objective changes
        # by the square of the step, below the precision of the scale's root, so a stricter
        # test would stop short of it); when no halved step lowers it, the point is kept.
        # The S objective is the scale itself; the MM step holds the scale fixed.
        value = objective(location, shape)
        for _ in range(ROBUST_ITERATIONS):
            target = reweighted(location, shape, weights_of(location, shape))
            # The move of the full step: the location in Mahalanobis units and the shape,
            # which has a unit determinant, entry by entry.
            step = target[0] - location
            change = max(
                np.sqrt(
                    step
                    @ np.linalg.solve(shape, step)
                    / (value if fixed_scale is None else fixed_scale)
                ),
                np.abs(target[1] - shape).max(),
            )
            for _ in range(10):
                new_value = objective(*target)
                if new_value <= value * (1 + 1e-12):
                    break
                target = (
                    (location + target[0]) / 2,
                    unit((shape + target[1]) / 2),
                )
            else:
                break
            location, shape, value = target[0], target[1], new_value
            if change <= ROBUST_TOLERANCE:
                break
        return location, shape, value

    def s_objective(location, shape) -> float:
        return scale_of(squared_distances(location, shape))

    def s_weights(location, shape) -> np.ndarray:
        distances = squared_distances(location, shape)
        return weight(distances / scale_of(distances))

    robust_start = get_minimum_covariance_determinant(returns, seed=seed).to_numpy()
    starts = [
        (np.median(data, axis=0), unit(robust_start)),
        (data.mean(axis=0), unit(np.cov(data, rowvar=False))),
    ]
    location, shape, scale = min(
        (iterate(start, scatter, s_objective, s_weights) for start, scatter in starts),
        key=lambda fit: fit[2],
    )
    if method == "mm":

        def mm_objective(location, shape) -> float:
            u = np.minimum(
                squared_distances(location, shape) / (scale * mm_tuning**2), 1
            )
            return float(np.sum(1 - (1 - u) ** 3))

        def mm_weights(location, shape) -> np.ndarray:
            x = squared_distances(location, shape) / (scale * mm_tuning**2)
            return np.where(x < 1, (1 - x) ** 2, 0.0)

        location, shape, _ = iterate(
            location, shape, mm_objective, mm_weights, fixed_scale=scale
        )
        covariance = scale * shape
    else:
        covariance = scale * shape
        covariance = covariance * (
            np.median(squared_distances(location, covariance))
            / stats.chi2.ppf(0.5, assets)
        )
    return (
        pd.Series(location, index=returns.columns),
        pd.DataFrame(covariance, index=returns.columns, columns=returns.columns),
    )


# The centroid of the expected normal order statistics of Almgren and Chriss (2006):
# α = A - B n^(-β).
CENTROID_CONSTANTS = (0.4424, 0.1185, 0.21)


def get_centroid_expected_returns(
    returns: pd.DataFrame,
    ranking: list | dict | pd.Series,
    periods_per_year: int,
) -> pd.Series:
    """
    The expected returns implied by an ordering of the assets alone, by the centroid of
    Almgren and Chriss (2006).

    It is the expected value of the risk-adjusted returns given only that they fall in the
    stated order, which is the centre of mass of the cone of vectors in that order.

    - For a complete ordering of n assets, the centroid is the vector of expected order
      statistics of n standard normal draws, approximated by
      c_j = Φ⁻¹((n + 1 - j - α) / (n - 2α + 1)) with α = 0.4424 - 0.1185 n^(-0.21), for the
      asset ranked j-th from the top.
    - Assets ranked together (a bucket, or equal scores) share the average c of the
      positions they occupy, the centroid when nothing orders them within the bucket.
    - μ_i = c_i σ_i, with σ_i the annualized volatility (ddof=1): the centroid applies to
      the risk-adjusted returns, so every asset's Sharpe ratio is its centroid score.

    Only the ordering is used, never the size of the past returns, so the estimate carries
    no estimation error of the means; the maximum Sharpe portfolio depends only on the
    direction of μ, the other optimizers also on its scale.

    The centroid values are checked against the Monte Carlo mean of sorted normal draws.

    See Almgren, R. and Chriss, N. (2006), "Optimal Portfolios from Ordering Information",
    Journal of Risk, 9(1), 1-47.

    Shares similar logic:
    - PortfolioAnalytics (R), ac.ranking: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/ac_ranking.R#L31
    - PortfolioAnalytics (R), centroid.complete.mc: https://github.com/cran/PortfolioAnalytics/blob/2.1.2/R/ac_ranking.R#L97

    Args:
        returns (pd.DataFrame): The returns per period, one column per asset, without
            missing values.
        ranking (list | dict | pd.Series): The ordering, from the highest expected return to
            the lowest: a list of assets, a list whose entries can be lists of assets ranked
            together, or scores per asset (a dict or Series, higher is better, equal scores
            ranked together). Every asset must be ranked.
        periods_per_year (int): The number of periods in a year, e.g. 252 for daily data.

    Returns:
        pd.Series: The annualized expected return per asset.

    Notes:
    - PortfolioAnalytics takes the order from the lowest expected return to the highest and
      rescales the centroid to lie between -0.05 and 0.05; its centroid.buckets gives every
      bucket the centroid of as many draws as there are buckets, where here a bucket
      averages the positions it holds.
    """
    if isinstance(ranking, dict | pd.Series):
        scores = pd.Series(ranking, dtype=float)
        if scores.isna().any():
            raise ValueError("Every score of the ranking must be a number.")
        groups = [
            list(scores.index[scores == value])
            for value in sorted(scores.unique(), reverse=True)
        ]
    else:
        groups = [
            list(entry) if isinstance(entry, list | tuple) else [entry]
            for entry in ranking
        ]
    ranked = [asset for group in groups for asset in group]
    if sorted(map(str, ranked)) != sorted(map(str, returns.columns)) or len(
        set(ranked)
    ) != len(ranked):
        raise ValueError(
            "The ranking must hold every asset once: "
            f"{', '.join(map(str, returns.columns))}."
        )
    count = len(ranked)
    a, b, beta = CENTROID_CONSTANTS
    alpha = a - b * count ** (-beta)
    centroid = stats.norm.ppf(
        (count + 1 - np.arange(1, count + 1) - alpha) / (count - 2 * alpha + 1)
    )
    scores, position = {}, 0
    for group in groups:
        value = centroid[position : position + len(group)].mean()
        scores.update(dict.fromkeys(group, value))
        position += len(group)
    volatility = returns.std() * np.sqrt(periods_per_year)
    return pd.Series(scores).reindex(returns.columns) * volatility
