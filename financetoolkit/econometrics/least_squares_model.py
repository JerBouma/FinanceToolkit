"""Least Squares Model"""

__docformat__ = "google"

import numpy as np
from scipy import linalg, stats

from financetoolkit.utilities.error_model import check_option

# The relative cutoff below which a singular value of the design counts as zero in the
# pseudo-inverse, the LAPACK-style default for a numerically exact least squares solution.
PSEUDO_INVERSE_CUTOFF = 1e-15

# Above this value the alternating series of the Kolmogorov distribution converges within a
# dozen terms; below it the Jacobi theta form does.
KOLMOGOROV_SERIES_SWITCH = 1.0
KOLMOGOROV_TERMS = 30

COVARIANCE_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC")


def has_constant(x: np.ndarray) -> bool:
    """
    Whether a design matrix holds a constant, explicitly or implicitly.

    The rule decides between the centered and the uncentered R-squared and how many
    regressors the model F-statistic and adjusted R-squared count:

    - Explicit: a column whose values are all equal and not zero
    - Implicit: otherwise, rank([1, X]) = rank(X), so a combination of the columns
      (such as a full set of dummies) is a column of ones

    Referenced against:
    - statsmodels, OLS (its k_constant): https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html

    Args:
        x (np.ndarray): The design matrix, shape `(n, k)`.

    Returns:
        bool: Whether the design has a constant.
    """
    column_maximum = x.max(axis=0)
    constant_columns = column_maximum == x.min(axis=0)

    if np.any(constant_columns & (column_maximum != 0)):
        return True

    augmented = np.column_stack([np.ones(x.shape[0]), x])
    return bool(np.linalg.matrix_rank(augmented) == np.linalg.matrix_rank(x))


def get_wald_statistic(
    coefficients: np.ndarray,
    covariance: np.ndarray,
    restriction_matrix: np.ndarray,
    restriction_values: np.ndarray | None = None,
) -> tuple[float, int]:
    """
    Calculate the Wald statistic of linear restrictions R·β = r: how far the estimated
    coefficients are from satisfying them, given their covariance (Wald, 1943).

    The formula is as follows:

    - W = (Rβ̂ - r)' (R·Cov(β̂)·R')⁺ (Rβ̂ - r)
    - q = rank(R·Cov(β̂)·R'), the number of independent restrictions

    W is chi-squared(q) distributed asymptotically and W / q is the F(q, n - k) version.
    The Moore-Penrose inverse ⁺ and the rank keep the statistic defined when two
    restrictions are linearly dependent; q then counts them once (the convention of
    statsmodels' wald_test).

    Referenced against:
    - statsmodels, RegressionResults.wald_test: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.wald_test.html

    Args:
        coefficients (np.ndarray): The estimated coefficients β̂, shape `(k,)`.
        covariance (np.ndarray): Their covariance matrix, shape `(k, k)`.
        restriction_matrix (np.ndarray): The restriction matrix R, shape `(q, k)`.
        restriction_values (np.ndarray | None, optional): The hypothesized values r,
            shape `(q,)`. Defaults to None, meaning zeros.

    Returns:
        tuple[float, int]: The Wald statistic W and the number of restrictions q.

    Notes:
        Reference: Wald, A. (1943). "Tests of Statistical Hypotheses Concerning Several
        Parameters When the Number of Observations is Large." Transactions of the
        American Mathematical Society, 54(3), 426-482.
    """
    restriction_matrix = np.atleast_2d(restriction_matrix)
    if restriction_values is None:
        restriction_values = np.zeros(restriction_matrix.shape[0])

    distance = restriction_matrix @ coefficients - restriction_values
    restriction_covariance = restriction_matrix @ covariance @ restriction_matrix.T

    if np.isnan(restriction_covariance).any():
        raise ValueError("The covariance of the restrictions contains NaN values.")

    statistic = float(distance @ np.linalg.pinv(restriction_covariance) @ distance)
    restrictions = min(
        restriction_matrix.shape[0], int(np.linalg.matrix_rank(restriction_covariance))
    )

    return statistic, restrictions


def _get_coefficient_covariance(
    whitened_x: np.ndarray,
    whitened_residuals: np.ndarray,
    pseudo_inverse: np.ndarray,
    residual_degrees_of_freedom: int,
    cov_type: str,
    clusters: np.ndarray | None,
    lags: int | None,
) -> np.ndarray:
    """
    The covariance matrix of least squares coefficients, see `fit_least_squares`.
    """
    observations = whitened_x.shape[0]
    # (X'X)⁻¹, as the product of the pseudo-inverse with itself.
    bread = pseudo_inverse @ pseudo_inverse.T

    if cov_type == "nonrobust":
        residual_variance = (
            whitened_residuals @ whitened_residuals / residual_degrees_of_freedom
        )
        return residual_variance * bread

    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        squared_residuals = whitened_residuals**2

        if cov_type == "HC1":
            squared_residuals = (
                observations / residual_degrees_of_freedom * squared_residuals
            )
        elif cov_type in ("HC2", "HC3"):
            # hₜ = x̃ₜ'Bx̃ₜ, the diagonal of the hat matrix.
            leverage = np.einsum("ij,ik,kj->i", whitened_x, whitened_x, bread)
            squared_residuals = squared_residuals / (
                (1 - leverage) if cov_type == "HC2" else (1 - leverage) ** 2
            )

        return pseudo_inverse @ (squared_residuals[:, None] * pseudo_inverse.T)

    scores = whitened_x * whitened_residuals[:, None]

    if cov_type == "HAC":
        lags = int(lags) if lags is not None else 0
        meat = scores.T @ scores

        for lag in range(1, lags + 1):
            # Bartlett weight of Newey-West (1987), which keeps the estimate positive semi-definite.  # noqa: E501
            autocovariance = scores[lag:].T @ scores[:-lag]
            meat += (1 - lag / (lags + 1)) * (autocovariance + autocovariance.T)

        return (bread @ meat) @ bread.T

    # Cluster-robust: the scores are summed within each cluster before the outer product.
    _, cluster_index = np.unique(clusters, return_inverse=True)
    number_of_clusters = int(cluster_index.max()) + 1
    cluster_scores = np.zeros((number_of_clusters, scores.shape[1]))
    np.add.at(cluster_scores, cluster_index.ravel(), scores)

    parameters = whitened_x.shape[1]
    correction = (
        number_of_clusters
        / (number_of_clusters - 1)
        * (observations - 1)
        / (observations - parameters)
    )

    return correction * ((bread @ (cluster_scores.T @ cluster_scores)) @ bread.T)


def get_pseudo_inverse(x: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Calculate the Moore-Penrose pseudo-inverse X⁺ of a design matrix and its rank, from the
    singular value decomposition X = USV': X⁺ = VS⁺U', with S⁺ the reciprocals of the
    singular values above PSEUDO_INVERSE_CUTOFF times the largest one (and zero for the
    rest). X⁺y is the least squares solution of y on X; X⁺Y solves every column of Y at once,
    such as the regressions of many assets on the same factors.

    Args:
        x (np.ndarray): The design matrix, shape `(n, k)`.

    Returns:
        tuple[np.ndarray, int]: The pseudo-inverse, shape `(k, n)`, and the rank of `x`.
    """
    left, singular_values, right = np.linalg.svd(x, full_matrices=False)
    keep = singular_values > PSEUDO_INVERSE_CUTOFF * singular_values.max()
    inverse_singular_values = np.zeros_like(singular_values)
    inverse_singular_values[keep] = 1 / singular_values[keep]
    pseudo_inverse = right.T @ (inverse_singular_values[:, None] * left.T)

    return pseudo_inverse, int(np.linalg.matrix_rank(np.diag(singular_values)))


def fit_least_squares(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    cov_type: str = "nonrobust",
    clusters: np.ndarray | None = None,
    lags: int | None = None,
) -> dict:
    """
    Fit a least squares regression: how a series moves with a set of regressors, by Ordinary,
    Weighted or Generalized Least Squares.

    Weighted Least Squares takes a weight per observation and Generalized Least Squares an
    error covariance structure Ω. Each one is OLS on whitened data, ỹ = Py and X̃ = PX, with
    P = I (OLS), P = diag(√w) (WLS) or P = L⁻¹ with LL' = Ω the Cholesky factor (GLS,
    Aitken, 1935).

    The formula is as follows:

    - β̂ = X̃⁺ỹ, with X̃⁺ the Moore-Penrose pseudo-inverse, which is (X̃'X̃)⁻¹X̃' when X̃ has
      full column rank
    - ẽ = ỹ - X̃β̂ (the whitened residuals) and e = y - Xβ̂ (the residuals)
    - SSR = ẽ'ẽ, σ̂² = SSR / (n - k), with k = rank(X̃)
    - R² = 1 - SSR / TSS, with TSS = ‖P(y - m)‖², m = (Pι)'(Py) / ‖Pι‖² when the design has a
      constant and TSS = ‖Py‖² when it has none (see `has_constant`)
    - Adjusted R² = 1 - (n - c) / (n - k) · (1 - R²), with c = 1 when there is a constant
    - ln L = -n/2 · (ln(2π) + ln(SSR / n) + 1) + ½ Σ ln wₜ (WLS) - ½ ln|Ω| (GLS)

    The covariance of β̂, with B = (X̃'X̃)⁻¹ and gₜ = x̃ₜẽₜ:

    - Non-robust: σ̂²·B
    - HC0: B (Σ ẽₜ² x̃ₜx̃ₜ') B, White (1980)
    - HC1: HC0 · n / (n - k), HC2: ẽₜ² / (1 - hₜ), HC3: ẽₜ² / (1 - hₜ)², with the leverage
      hₜ = x̃ₜ'Bx̃ₜ, MacKinnon and White (1985)
    - HAC: B (Σₜ gₜgₜ' + Σⱼ₌₁ᴸ (1 - j / (L + 1)) Σₜ (gₜgₜ₋ⱼ' + gₜ₋ⱼgₜ')) B with L = `lags`,
      Newey and West (1987), without a small-sample correction
    - Cluster-robust: G / (G - 1) · (n - 1) / (n - k) · B (Σ_g (Σ_{t∈g} gₜ)(Σ_{t∈g} gₜ)') B, with G
      the number of clusters

    The t-statistics are β̂ⱼ / SE(β̂ⱼ). Their p-values come from the Student-t distribution
    with n - k degrees of freedom for the non-robust covariance and from the standard normal
    distribution for the robust estimators, which are asymptotic results. These are
    statsmodels' defaults; the sandwich estimators use the whitened data, so a robust WLS
    covariance is robust on the weighted scale.

    Referenced against:
    - statsmodels, OLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
    - statsmodels, WLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html
    - statsmodels, GLS: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html

    Args:
        y (np.ndarray): The dependent variable, shape `(n,)`.
        x (np.ndarray): The design matrix, shape `(n, k)`, with a constant column if the
            model has one.
        weights (np.ndarray | None, optional): The WLS weights, shape `(n,)`. Defaults to
            None.
        omega (np.ndarray | None, optional): The GLS error covariance structure, shape
            `(n, n)`. Defaults to None.
        cov_type (str, optional): The covariance estimator, one of "nonrobust", "HC0",
            "HC1", "HC2", "HC3", "cluster" and "HAC". Defaults to "nonrobust".
        clusters (np.ndarray | None, optional): The cluster label of each observation,
            used by "cluster". Defaults to None.
        lags (int | None, optional): The truncation lag L of "HAC". Defaults to None,
            meaning 0.

    Returns:
        dict: The keys `coefficients`, `standard_errors`, `t_statistics`, `p_values`,
        `residuals`, `fitted_values`, `covariance_matrix`, `r_squared`,
        `adjusted_r_squared`, `residual_variance`, `sum_of_squared_residuals` (whitened),
        `log_likelihood`, `degrees_of_freedom` (n - k), `inference_degrees_of_freedom`
        (G - 1 for "cluster", otherwise n - k, the denominator of Wald F-tests),
        `n_observations` and `rank`. R² is NaN when TSS is zero and the adjusted R² also
        when n - k is not positive.

    Notes:
        References: Aitken, A.C. (1935). "On Least Squares and Linear Combinations of
        Observations." Proceedings of the Royal Society of Edinburgh, 55, 42-48. White, H.
        (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct
        Test for Heteroskedasticity." Econometrica, 48(4), 817-838. MacKinnon, J.G. and
        White, H. (1985). "Some Heteroskedasticity-Consistent Covariance Matrix Estimators
        with Improved Finite Sample Properties." Journal of Econometrics, 29(3), 305-325.
        Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
        Econometrica, 55(3), 703-708. Liang, K.-Y. and Zeger, S.L. (1986). "Longitudinal
        Data Analysis Using Generalized Linear Models." Biometrika, 73(1), 13-22.
    """
    check_option("cov_type", cov_type, COVARIANCE_TYPES)

    observations = len(y)
    log_likelihood_adjustment = 0.0

    if omega is not None:
        cholesky_factor = linalg.cholesky(omega, lower=True)
        whitened_y = linalg.solve_triangular(cholesky_factor, y, lower=True)
        whitened_x = linalg.solve_triangular(cholesky_factor, x, lower=True)
        whitened_ones = linalg.solve_triangular(
            cholesky_factor, np.ones(observations), lower=True
        )
        log_likelihood_adjustment = -np.sum(np.log(np.diag(cholesky_factor)))
    elif weights is not None:
        root_weights = np.sqrt(weights)
        whitened_y = root_weights * y
        whitened_x = root_weights[:, None] * x
        whitened_ones = root_weights
        log_likelihood_adjustment = 0.5 * np.sum(np.log(weights))
    else:
        whitened_y, whitened_x, whitened_ones = y, x, np.ones(observations)

    pseudo_inverse, rank = get_pseudo_inverse(whitened_x)

    coefficients = pseudo_inverse @ whitened_y
    fitted_values = x @ coefficients
    residuals = y - fitted_values
    whitened_residuals = whitened_y - whitened_x @ coefficients
    sum_of_squared_residuals = float(whitened_residuals @ whitened_residuals)
    degrees_of_freedom = observations - rank

    constant = has_constant(x)
    if constant:
        weighted_mean = (whitened_ones @ whitened_y) / (whitened_ones @ whitened_ones)
        if omega is not None:
            centered = linalg.solve_triangular(
                cholesky_factor, y - weighted_mean, lower=True
            )
        else:
            centered = whitened_y - whitened_ones * weighted_mean
        total_sum_of_squares = float(centered @ centered)
    else:
        total_sum_of_squares = float(whitened_y @ whitened_y)

    # R² has no meaning when the outcome has no variation to explain (0 / 0).
    r_squared = adjusted_r_squared = float("nan")
    if np.isfinite(total_sum_of_squares) and total_sum_of_squares != 0:
        r_squared = 1 - sum_of_squared_residuals / total_sum_of_squares
        if degrees_of_freedom > 0:
            adjusted_r_squared = 1 - (observations - int(constant)) / (
                degrees_of_freedom
            ) * (1 - r_squared)

    covariance = _get_coefficient_covariance(
        whitened_x,
        whitened_residuals,
        pseudo_inverse,
        degrees_of_freedom,
        cov_type,
        clusters,
        lags,
    )
    standard_errors = np.sqrt(np.diag(covariance))
    # A perfect fit has zero standard errors, which leaves the t-statistics undefined (NaN).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = coefficients / standard_errors

    if cov_type == "nonrobust":
        p_values = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    else:
        p_values = 2 * stats.norm.sf(np.abs(t_statistics))

    inference_degrees_of_freedom = degrees_of_freedom
    if cov_type == "cluster":
        inference_degrees_of_freedom = len(np.unique(clusters)) - 1

    half_observations = observations / 2
    # A perfect fit (SSR = 0) has an unbounded likelihood, +inf.
    with np.errstate(divide="ignore"):
        log_likelihood = (
            -half_observations
            * (np.log(2 * np.pi) + np.log(sum_of_squared_residuals / observations) + 1)
            + log_likelihood_adjustment
        )

    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "t_statistics": t_statistics,
        "p_values": p_values,
        "residuals": residuals,
        "fitted_values": fitted_values,
        "covariance_matrix": covariance,
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "residual_variance": (
            sum_of_squared_residuals / degrees_of_freedom
            if degrees_of_freedom > 0
            else float("nan")
        ),
        "sum_of_squared_residuals": sum_of_squared_residuals,
        "log_likelihood": float(log_likelihood),
        "degrees_of_freedom": degrees_of_freedom,
        "inference_degrees_of_freedom": inference_degrees_of_freedom,
        "n_observations": observations,
        "rank": rank,
    }


def get_lagged_values(values: np.ndarray, lags: int) -> np.ndarray:
    """
    Calculate the lagged values of a series side by side, on the observations that have all
    of them.

    Row t - `lags` holds (x(t-1), ..., x(t-lags)) for t = lags, ..., n - 1.

    Args:
        values (np.ndarray): The series, shape `(n,)`.
        lags (int): The number of lags.

    Returns:
        np.ndarray: The lagged values, shape `(n - lags, lags)`.
    """
    return np.column_stack(
        [values[lags - lag : len(values) - lag] for lag in range(1, lags + 1)]
    )


def get_autocorrelations(values: np.ndarray, lags: int) -> np.ndarray:
    """
    Calculate the sample autocorrelations of a series: how strongly it moves with its own
    past values, from lag 1 up to a number of lags.

    Every autocovariance of the demeaned series is divided by n (the biased estimator, which
    keeps the autocorrelation matrix positive semi-definite):

    - ρₖ = Σₜ₌ₖ₊₁ⁿ (xₜ - x̄)(xₜ₋ₖ - x̄) / Σₜ₌₁ⁿ (xₜ - x̄)²

    Referenced against:
    - statsmodels, acf: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html

    Args:
        values (np.ndarray): The series, shape `(n,)`.
        lags (int): The highest lag.

    Returns:
        np.ndarray: The autocorrelations ρ₁ to ρ_lags.
    """
    demeaned = values - values.mean()
    variance = demeaned @ demeaned

    return np.array(
        [demeaned[lag:] @ demeaned[:-lag] / variance for lag in range(1, lags + 1)]
    )


def get_kolmogorov_survival(statistic: float) -> float:
    """
    Calculate the survival function of the Kolmogorov distribution: the probability that the
    supremum of the absolute value of a Brownian bridge exceeds a statistic.

    It is the asymptotic distribution of the OLS-CUSUM statistic of Ploberger and Krämer
    (1992):

    - P(sup|B(t)| > x) = 2 Σₖ₌₁^∞ (-1)ᵏ⁻¹ exp(-2k²x²), for x ≥ 1
    - P(sup|B(t)| > x) = 1 - √(2π) / x · Σₖ₌₁^∞ exp(-(2k - 1)²π² / (8x²)), for x < 1

    Both series are exact; each converges within a few terms on its side of 1.

    Referenced against:
    - scipy, kstwobign: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstwobign.html

    Args:
        statistic (float): The value x.

    Returns:
        float: The survival probability.

    Notes:
        References: Kolmogorov, A. (1933). "Sulla determinazione empirica di una legge di
        distribuzione." Giornale dell'Istituto Italiano degli Attuari, 4, 83-91. Ploberger,
        W. and Krämer, W. (1992). "The CUSUM Test with OLS Residuals." Econometrica, 60(2),
        271-285.
        statsmodels' breaks_cusumolsresid uses the survival function of scipy's kstwobign.
    """
    if statistic <= 0:
        return 1.0

    terms = np.arange(1, KOLMOGOROV_TERMS + 1)

    if statistic >= KOLMOGOROV_SERIES_SWITCH:
        signs = np.where(terms % 2 == 1, 1.0, -1.0)
        return float(min(2 * np.sum(signs * np.exp(-2 * terms**2 * statistic**2)), 1.0))

    cumulative = (
        np.sqrt(2 * np.pi)
        / statistic
        * np.sum(np.exp(-((2 * terms - 1) ** 2) * np.pi**2 / (8 * statistic**2)))
    )
    return float(1 - cumulative)
