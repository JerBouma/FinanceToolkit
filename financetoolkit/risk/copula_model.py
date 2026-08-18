"""Copula Model"""

__docformat__ = "google"

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

MINIMUM_TAIL_OBSERVATIONS = 2
MEDIAN_QUANTILE = 0.5
MINIMUM_OBSERVATIONS_FOR_COPULA_FIT = 10

# Below ~2 the second moment is undefined; above ~60 it matches the gaussian.
STUDENT_T_DOF_BOUNDS = (2.05, 60.0)


def get_tail_dependence_coefficient(
    series_a: pd.Series,
    series_b: pd.Series,
    q: float = 0.95,
    method: str = "empirical",
    dof: float = 4.0,
) -> pd.Series:
    """
    Calculate the Upper and Lower Tail Dependence Coefficients between two series.

    Correlation (see e.g. `risk_controller.Risk.get_value_at_risk`'s use of the
    gaussian distribution) only captures the *average* co-movement between two
    series -- it says nothing about whether they are more likely to crash together
    than an equivalent gaussian relationship would imply. The Tail Dependence
    Coefficient answers that specific question: the probability that one series is
    in extreme distress, given that the other one already is.

    - Upper Tail Dependence: lambda_U = lim_(u -> 1) P(F_B(B) > u | F_A(A) > u)
    - Lower Tail Dependence: lambda_L = lim_(u -> 0) P(F_B(B) <= u | F_A(A) <= u)

    Where `F_A` and `F_B` are the marginal (cumulative) distributions of `series_a`
    and `series_b`. `lambda_L` is generally the more relevant figure for risk
    management purposes, since it measures joint crash risk (both assets suffering
    extreme losses together) rather than joint boom risk.

    Three ways of estimating the coefficient are supported via `method`:

    - "empirical" (default): a nonparametric plug-in estimate, replacing the limit at
    `u -> 0` / `u -> 1` with a finite threshold quantile `q`:
    `lambda_U(q) = P(B > quantile(B, q) | A > quantile(A, q))`, and symmetrically
    for `lambda_L(1 - q)`. This makes no distributional assumption but is
    sensitive to the choice of `q` and needs enough observations in the tail to be
    reliable.
    - "gaussian": the tail dependence implied by a bivariate gaussian copula with the
    same linear (Pearson) correlation as `series_a` and `series_b`. This is
    included mainly as a cautionary baseline: a gaussian copula has *zero*
    asymptotic tail dependence for any correlation below 1 (Embrechts, McNeil &
    Straumann, 1999), so `lambda_U = lambda_L = 0` is returned regardless of the
    observed correlation -- demonstrating why gaussian-based dependence models
    (e.g. the gaussian VaR in `var_model.get_var_gaussian`) can understate joint
    crash risk relative to the empirical estimate above.
    - "student-t": the tail dependence implied by a bivariate Student-T copula with
    `dof` degrees of freedom and the same linear correlation `rho` as `series_a`
    and `series_b`, which (unlike the gaussian copula) has nonzero, symmetric tail
    dependence given by the closed-form expression:
    `lambda = 2 * t_sf(SQRT((dof + 1) * (1 - rho) / (1 + rho)), dof + 1)`
    where `t_sf` is the Student-T survival function.

    For more information about the method, see the following papers:

    - Embrechts, P., McNeil, A., & Straumann, D. (1999). "Correlation: Pitfalls and
    Alternatives." RISK Magazine, 12, 69-71.
    - Poon, S.H., Rockinger, M., & Tawn, J. (2004). "Extreme Value Dependence in
    Financial Markets: Diagnostics, Models, and Financial Implications." Review of
    Financial Studies, 17(2), 581-610.

    Also known as: tail dependence, extremal dependence coefficient.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).
        q (float, optional): The threshold quantile used for the "empirical" method,
        in (0.5, 1). Defaults to 0.95 (the top/bottom 5% of each series).
        method (str, optional): The estimation method, one of "empirical", "gaussian"
        or "student-t". Defaults to "empirical".
        dof (float, optional): The degrees of freedom of the Student-T copula, only
        used when `method="student-t"`. Lower values imply heavier tails and
        therefore higher tail dependence for the same correlation. Defaults to 4.0.

    Returns:
        pd.Series: The Lower and Upper Tail Dependence Coefficients, the linear
        (Pearson) correlation used by the "gaussian" and "student-t" methods, and the
        number of (paired, non-missing) observations used.

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If `method` is not one of "empirical", "gaussian" or "student-t",
        or if `q` is not in (0.5, 1).
    """
    if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")
    if method not in ("empirical", "gaussian", "student-t"):
        raise ValueError("method must be 'empirical', 'gaussian' or 'student-t'.")
    if q <= MEDIAN_QUANTILE or q >= 1:
        raise ValueError("q must be in the range (0.5, 1).")

    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    n = len(aligned)

    if n <= MINIMUM_TAIL_OBSERVATIONS:
        return pd.Series(
            {
                "Lower Tail Dependence": np.nan,
                "Upper Tail Dependence": np.nan,
                "Correlation": np.nan,
                "Observations": n,
            }
        )

    a = aligned.iloc[:, 0].to_numpy()
    b = aligned.iloc[:, 1].to_numpy()
    correlation = np.corrcoef(a, b)[0, 1]

    if method == "empirical":
        upper_a = a > np.quantile(a, q)
        upper_b = b > np.quantile(b, q)
        lower_a = a <= np.quantile(a, 1 - q)
        lower_b = b <= np.quantile(b, 1 - q)

        upper_tail_dependence = (
            np.sum(upper_a & upper_b) / np.sum(upper_a)
            if np.sum(upper_a) > 0
            else np.nan
        )
        lower_tail_dependence = (
            np.sum(lower_a & lower_b) / np.sum(lower_a)
            if np.sum(lower_a) > 0
            else np.nan
        )
    elif method == "gaussian":
        # A gaussian copula has zero tail dependence for any correlation below 1.
        upper_tail_dependence = 0.0 if correlation < 1 else 1.0
        lower_tail_dependence = upper_tail_dependence
    else:
        tail_dependence = 2 * stats.t.sf(
            np.sqrt((dof + 1) * (1 - correlation) / (1 + correlation)), dof + 1
        )
        upper_tail_dependence = tail_dependence
        lower_tail_dependence = tail_dependence

    return pd.Series(
        {
            "Lower Tail Dependence": lower_tail_dependence,
            "Upper Tail Dependence": upper_tail_dependence,
            "Correlation": correlation,
            "Observations": n,
        }
    )


def get_pseudo_observations(series: pd.Series) -> pd.Series:
    """
    Transform a series to pseudo-observations on the unit interval, via its
    empirical (rank-based) cumulative distribution function.

    Copulas describe the dependence structure between series independently of
    their marginal distributions (Sklar's theorem). Rather than assuming a
    parametric marginal distribution for each series, the standard "canonical
    maximum likelihood" approach used throughout this module instead works
    directly with each series' own empirical ranks -- every copula function in
    this module expects its `u`/`v` (or `series_a`/`series_b`) inputs to already
    be on this pseudo-observation scale.

    The formula is as follows:

    - u_i = rank(x_i) / (n + 1)

    Where rank(x_i) is the ascending rank of observation i among the n
    observations in the series. Dividing by (n + 1) rather than n keeps every
    pseudo-observation strictly inside (0, 1), which every copula function in
    this module requires (several are undefined at exactly 0 or 1).

    Also known as: empirical CDF transform, copula data transform, rank transform.

    Args:
        series (pd.Series): The series to transform.

    Returns:
        pd.Series: The pseudo-observations, in (0, 1), same index as `series`.
    """
    return series.rank() / (len(series) + 1)


def _align_pseudo_observations(
    series_a: pd.Series, series_b: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two series on their shared, non-missing index and transform both to
    pseudo-observations (see `get_pseudo_observations`). Shared by every
    `get_<family>_copula_parameters` fitting function in this module.

    Args:
        series_a (pd.Series): The first series.
        series_b (pd.Series): The second series.

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than `MINIMUM_OBSERVATIONS_FOR_COPULA_FIT` paired,
            non-missing observations are available.

    Returns:
        tuple[np.ndarray, np.ndarray]: The pseudo-observations of `series_a` and
            `series_b`.
    """
    if not isinstance(series_a, pd.Series) or not isinstance(series_b, pd.Series):
        raise TypeError("Expects pd.Series, no other value.")

    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()

    if len(aligned) < MINIMUM_OBSERVATIONS_FOR_COPULA_FIT:
        raise ValueError(
            f"At least {MINIMUM_OBSERVATIONS_FOR_COPULA_FIT} paired, non-missing "
            "observations are required to fit a copula."
        )

    u = get_pseudo_observations(aligned.iloc[:, 0]).to_numpy()
    v = get_pseudo_observations(aligned.iloc[:, 1]).to_numpy()

    return u, v


def _get_akaike_information_criterion(
    log_likelihood: float, number_of_parameters: int
) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for a fitted copula, used to
    compare which copula family best fits a given pair of series -- the lower the
    AIC, the better the fit relative to its number of parameters.

    The formula is as follows:

    - AIC = 2 * k - 2 * log_likelihood

    Where k is the number of free parameters in the model.

    Args:
        log_likelihood (float): The maximized log-likelihood of the fitted copula.
        number_of_parameters (int): The number of free parameters fitted.

    Returns:
        float: The Akaike Information Criterion.
    """
    return 2 * number_of_parameters - 2 * log_likelihood


def get_gaussian_copula_density(
    u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, rho: float
) -> np.ndarray:
    """
    Evaluate the bivariate gaussian copula density at pseudo-observations `u`
    and `v` (see `get_pseudo_observations`).

    The formula is as follows:

    - c(u, v; rho) = (1 / sqrt(1 - rho^2)) * exp((2 * rho * z_u * z_v - rho^2 *
    (z_u^2 + z_v^2)) / (2 * (1 - rho^2)))

    Where z_u = Phi^-1(u) and z_v = Phi^-1(v) are the standard normal quantiles
    of the pseudo-observations, and rho is the (linear) correlation parameter.
    Note that a gaussian copula has zero tail dependence for any rho strictly
    below 1 -- see `get_tail_dependence_coefficient`.

    Args:
        u (pd.Series | np.ndarray): The first series' pseudo-observations, in
            (0, 1).
        v (pd.Series | np.ndarray): The second series' pseudo-observations, in
            (0, 1).
        rho (float): The correlation parameter, in (-1, 1).

    Returns:
        np.ndarray: The copula density at each (u, v) pair.
    """
    z_u, z_v = stats.norm.ppf(u), stats.norm.ppf(v)

    return (1 / np.sqrt(1 - rho**2)) * np.exp(
        (2 * rho * z_u * z_v - rho**2 * (z_u**2 + z_v**2)) / (2 * (1 - rho**2))
    )


def get_gaussian_copula_parameters(
    series_a: pd.Series, series_b: pd.Series
) -> dict[str, float]:
    """
    Calibrate a bivariate gaussian copula to two series: both series are
    transformed to pseudo-observations (see `get_pseudo_observations`), then to
    standard normal quantiles, whose sample (Pearson) correlation is taken as the
    copula's rho parameter. This "normal scores" plug-in estimator is the standard
    calibration for the gaussian copula and is consistent and asymptotically
    equivalent to the canonical maximum likelihood estimate (which has no closed
    form, since the normal scores' variances are fixed at one rather than
    estimated), typically agreeing with it to within a few thousandths.

    Also known as: gaussian copula calibration, normal copula fit.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than 10 paired, non-missing observations are
            available.

    Returns:
        dict[str, float]: The fitted "Rho", the "Lower Tail Dependence" and
        "Upper Tail Dependence" implied by it (always 0.0 for a gaussian
        copula), the "Log-Likelihood", the "AIC" and the number of
        "Observations" used.
    """
    u, v = _align_pseudo_observations(series_a, series_b)

    z_u, z_v = stats.norm.ppf(u), stats.norm.ppf(v)
    rho = float(np.corrcoef(z_u, z_v)[0, 1])

    log_likelihood = float(np.sum(np.log(get_gaussian_copula_density(u, v, rho))))

    return {
        "Rho": rho,
        "Lower Tail Dependence": 0.0,
        "Upper Tail Dependence": 0.0,
        "Log-Likelihood": log_likelihood,
        "AIC": _get_akaike_information_criterion(log_likelihood, 1),
        "Observations": len(u),
    }


def get_gaussian_copula_simulation(
    rho: float, n_simulations: int = 10_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a bivariate gaussian copula.

    The result is on the pseudo-observation (0, 1) scale -- to obtain simulated
    returns, apply each column's own marginal quantile function (e.g. the
    empirical quantile function of the original return series) to it.

    Args:
        rho (float): The correlation parameter, in (-1, 1), see
            `get_gaussian_copula_parameters`.
        n_simulations (int, optional): The number of joint draws to simulate.
            Defaults to 10,000.
        random_state (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    rng = np.random.default_rng(random_state)

    z = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n_simulations
    )

    return pd.DataFrame({"U": stats.norm.cdf(z[:, 0]), "V": stats.norm.cdf(z[:, 1])})


def get_student_t_copula_density(
    u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, rho: float, dof: float
) -> np.ndarray:
    """
    Evaluate the bivariate Student-T copula density at pseudo-observations `u`
    and `v` (see `get_pseudo_observations`).

    The formula is as follows:

    - c(u, v; rho, dof) = [Gamma((dof + 2) / 2) * Gamma(dof / 2) / Gamma((dof +
    1) / 2)^2] * (1 / sqrt(1 - rho^2)) * (1 + (x^2 + y^2 - 2 * rho * x * y) /
    (dof * (1 - rho^2)))^(-(dof + 2) / 2) / [(1 + x^2 / dof)^(-(dof + 1) / 2) *
    (1 + y^2 / dof)^(-(dof + 1) / 2)]

    Where x = t_dof^-1(u) and y = t_dof^-1(v) are the Student-T quantiles of the
    pseudo-observations, rho is the correlation parameter and dof the degrees of
    freedom. Unlike the gaussian copula, the Student-T copula has symmetric,
    nonzero tail dependence -- see `get_tail_dependence_coefficient`.

    Args:
        u (pd.Series | np.ndarray): The first series' pseudo-observations, in
            (0, 1).
        v (pd.Series | np.ndarray): The second series' pseudo-observations, in
            (0, 1).
        rho (float): The correlation parameter, in (-1, 1).
        dof (float): The degrees of freedom. Lower values imply heavier tails.

    Returns:
        np.ndarray: The copula density at each (u, v) pair.
    """
    x, y = stats.t.ppf(u, dof), stats.t.ppf(v, dof)

    log_normalizing_constant = (
        gammaln((dof + 2) / 2) + gammaln(dof / 2) - 2 * gammaln((dof + 1) / 2)
    )

    log_density = (
        log_normalizing_constant
        - 0.5 * np.log(1 - rho**2)
        - (dof + 2)
        / 2
        * np.log(1 + (x**2 + y**2 - 2 * rho * x * y) / (dof * (1 - rho**2)))
        + (dof + 1) / 2 * np.log(1 + x**2 / dof)
        + (dof + 1) / 2 * np.log(1 + y**2 / dof)
    )

    return np.exp(log_density)


def get_student_t_copula_parameters(
    series_a: pd.Series,
    series_b: pd.Series,
    dof_bounds: tuple[float, float] = STUDENT_T_DOF_BOUNDS,
) -> dict[str, float]:
    """
    Calibrate a bivariate Student-T copula to two series, via profile maximum
    likelihood (Demarta & McNeil, 2005): for each candidate degrees of freedom,
    the correlation is estimated as the sample correlation of the pseudo-
    observations' Student-T quantiles at that degrees of freedom (the same plug-in
    estimator as in `get_gaussian_copula_parameters`), and the degrees of freedom
    that maximizes the resulting log-likelihood is found via bounded 1-D
    optimization.

    See: Demarta, S., & McNeil, A.J. (2005), "The T Copula and Related Copulas",
    International Statistical Review, 73(1), 111-129.

    Also known as: Student-T copula calibration, T copula fit.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).
        dof_bounds (tuple[float, float], optional): The bounds to search the
            degrees of freedom over. Defaults to (2.05, 60.0).

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than 10 paired, non-missing observations are
            available.

    Returns:
        dict[str, float]: The fitted "Rho" and "Degrees of Freedom", the "Lower
        Tail Dependence" and "Upper Tail Dependence" implied by them, the
        "Log-Likelihood", the "AIC" and the number of "Observations" used.
    """
    u, v = _align_pseudo_observations(series_a, series_b)

    def negative_log_likelihood(dof: float) -> float:
        x, y = stats.t.ppf(u, dof), stats.t.ppf(v, dof)
        rho = np.corrcoef(x, y)[0, 1]

        return -float(np.sum(np.log(get_student_t_copula_density(u, v, rho, dof))))

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=dof_bounds, method="bounded"
    )

    dof = float(result.x)
    x, y = stats.t.ppf(u, dof), stats.t.ppf(v, dof)
    rho = float(np.corrcoef(x, y)[0, 1])
    log_likelihood = -float(result.fun)

    tail_dependence = 2 * stats.t.sf(
        np.sqrt((dof + 1) * (1 - rho) / (1 + rho)), dof + 1
    )

    return {
        "Rho": rho,
        "Degrees of Freedom": dof,
        "Lower Tail Dependence": tail_dependence,
        "Upper Tail Dependence": tail_dependence,
        "Log-Likelihood": log_likelihood,
        "AIC": _get_akaike_information_criterion(log_likelihood, 2),
        "Observations": len(u),
    }


def get_student_t_copula_simulation(
    rho: float, dof: float, n_simulations: int = 10_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a bivariate Student-T copula, via the
    standard normal-mixture construction of the multivariate T distribution
    (correlated gaussian variates divided by an independent chi-squared shock,
    shared across both series -- the mechanism that produces the T copula's
    tail dependence, unlike the gaussian copula).

    The result is on the pseudo-observation (0, 1) scale -- to obtain simulated
    returns, apply each column's own marginal quantile function (e.g. the
    empirical quantile function of the original return series) to it.

    Args:
        rho (float): The correlation parameter, in (-1, 1), see
            `get_student_t_copula_parameters`.
        dof (float): The degrees of freedom, see
            `get_student_t_copula_parameters`.
        n_simulations (int, optional): The number of joint draws to simulate.
            Defaults to 10,000.
        random_state (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    rng = np.random.default_rng(random_state)

    z = rng.multivariate_normal(
        mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n_simulations
    )
    chi_squared_shock = rng.chisquare(dof, size=n_simulations)
    t = z * np.sqrt(dof / chi_squared_shock)[:, None]

    return pd.DataFrame(
        {"U": stats.t.cdf(t[:, 0], dof), "V": stats.t.cdf(t[:, 1], dof)}
    )


def _simulate_archimedean_copula(
    conditional_cdf: Callable[[float, float], float],
    n_simulations: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from any Archimedean copula, via the standard
    conditional distribution method: draw U ~ Uniform(0, 1) and an independent
    T ~ Uniform(0, 1), then solve for V such that the conditional distribution
    of V given U, h(v | u) = dC(u, v) / du, equals T (h is monotonically
    increasing in v, so the equation has a unique root). Shared by every
    `get_<archimedean family>_copula_simulation` function in this module.

    Args:
        conditional_cdf (Callable[[float, float], float]): A function
            `h(u, v)` returning the family's h-function (dC(u, v) / du) at a
            given (u, v) pair.
        n_simulations (int): The number of joint draws to simulate.
        random_state (int): The seed for the random number generator.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    rng = np.random.default_rng(random_state)

    u = rng.uniform(size=n_simulations)
    t = rng.uniform(size=n_simulations)

    v = np.array(
        [
            optimize.brentq(
                lambda candidate_v, ui=ui, ti=ti: conditional_cdf(ui, candidate_v) - ti,
                1e-10,
                1 - 1e-10,
            )
            for ui, ti in zip(u, t)
        ]
    )

    return pd.DataFrame({"U": u, "V": v})


def get_clayton_copula_density(
    u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, theta: float
) -> np.ndarray:
    """
    Evaluate the bivariate Clayton copula density at pseudo-observations `u`
    and `v` (see `get_pseudo_observations`).

    The formula is as follows:

    - c(u, v; theta) = (1 + theta) * (u * v)^(-1 - theta) * (u^-theta + v^-theta
    - 1)^(-1 / theta - 2)

    Where theta > 0 controls the strength of dependence (theta -> 0 is
    independence, theta -> infinity is perfect dependence). The Clayton copula
    has nonzero *lower* tail dependence only (see
    `get_clayton_copula_parameters`), making it well suited to modeling joint
    crash risk without assuming symmetric joint boom risk.

    Also known as: Clayton dependence, Archimedean copula (Clayton family).

    Args:
        u (pd.Series | np.ndarray): The first series' pseudo-observations, in
            (0, 1).
        v (pd.Series | np.ndarray): The second series' pseudo-observations, in
            (0, 1).
        theta (float): The dependence parameter. Must be > 0.

    Returns:
        np.ndarray: The copula density at each (u, v) pair.
    """
    return (
        (1 + theta)
        * (u * v) ** (-1 - theta)
        * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 2)
    )


def _get_clayton_conditional_cdf(u: float, v: float, theta: float) -> float:
    """
    The Clayton copula's h-function, h(v | u) = dC(u, v) / du. Used by
    `get_clayton_copula_simulation` to simulate via conditional inversion (see
    `_simulate_archimedean_copula`).
    """
    return u ** (-theta - 1) * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 1)


def get_clayton_copula_parameters(
    series_a: pd.Series, series_b: pd.Series
) -> dict[str, float]:
    """
    Calibrate a bivariate Clayton copula to two series, via maximum likelihood:
    both series are transformed to pseudo-observations (see
    `get_pseudo_observations`), and theta is found by numerically maximizing the
    resulting Clayton copula log-likelihood.

    Also known as: Clayton copula calibration, Clayton copula fit.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than 10 paired, non-missing observations are
            available.

    Returns:
        dict[str, float]: The fitted "Theta", the "Lower Tail Dependence" and
        "Upper Tail Dependence" implied by it (the latter always 0.0 for a
        Clayton copula), the "Log-Likelihood", the "AIC" and the number of
        "Observations" used.
    """
    u, v = _align_pseudo_observations(series_a, series_b)

    def negative_log_likelihood(theta: float) -> float:
        return -float(np.sum(np.log(get_clayton_copula_density(u, v, theta))))

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=(1e-4, 100.0), method="bounded"
    )

    theta = float(result.x)
    log_likelihood = -float(result.fun)

    return {
        "Theta": theta,
        "Lower Tail Dependence": 2 ** (-1 / theta),
        "Upper Tail Dependence": 0.0,
        "Log-Likelihood": log_likelihood,
        "AIC": _get_akaike_information_criterion(log_likelihood, 1),
        "Observations": len(u),
    }


def get_clayton_copula_simulation(
    theta: float, n_simulations: int = 10_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a bivariate Clayton copula, via
    conditional inversion (see `_simulate_archimedean_copula`).

    The result is on the pseudo-observation (0, 1) scale -- to obtain simulated
    returns, apply each column's own marginal quantile function (e.g. the
    empirical quantile function of the original return series) to it.

    Args:
        theta (float): The dependence parameter, > 0, see
            `get_clayton_copula_parameters`.
        n_simulations (int, optional): The number of joint draws to simulate.
            Defaults to 10,000.
        random_state (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    return _simulate_archimedean_copula(
        lambda u, v: _get_clayton_conditional_cdf(u, v, theta),
        n_simulations,
        random_state,
    )


def get_gumbel_copula_density(
    u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, theta: float
) -> np.ndarray:
    """
    Evaluate the bivariate Gumbel copula density at pseudo-observations `u` and
    `v` (see `get_pseudo_observations`).

    The formula is as follows:

    - Let a = (-ln u)^theta + (-ln v)^theta, s = a^(1 / theta), C = exp(-s)
    - c(u, v; theta) = C * (u * v)^-1 * a^(1 / theta - 2) * (ln(u) * ln(v))^
    (theta - 1) * (s + theta - 1)

    Where theta >= 1 controls the strength of dependence (theta = 1 is
    independence, theta -> infinity is perfect dependence). The Gumbel copula
    has nonzero *upper* tail dependence only (see
    `get_gumbel_copula_parameters`), making it well suited to modeling joint
    boom/rally risk without assuming symmetric joint crash risk.

    Also known as: Gumbel dependence, Gumbel-Hougaard copula, Archimedean copula
    (Gumbel family).

    Args:
        u (pd.Series | np.ndarray): The first series' pseudo-observations, in
            (0, 1).
        v (pd.Series | np.ndarray): The second series' pseudo-observations, in
            (0, 1).
        theta (float): The dependence parameter. Must be >= 1.

    Returns:
        np.ndarray: The copula density at each (u, v) pair.
    """
    log_u, log_v = -np.log(u), -np.log(v)
    a = log_u**theta + log_v**theta
    copula_cdf = np.exp(-(a ** (1 / theta)))

    return (
        copula_cdf
        * (u * v) ** (-1)
        * a ** (1 / theta - 2)
        * (log_u * log_v) ** (theta - 1)
        * (a ** (1 / theta) + theta - 1)
    )


def _get_gumbel_conditional_cdf(u: float, v: float, theta: float) -> float:
    """
    The Gumbel copula's h-function, h(v | u) = dC(u, v) / du. Used by
    `get_gumbel_copula_simulation` to simulate via conditional inversion (see
    `_simulate_archimedean_copula`).
    """
    log_u, log_v = -np.log(u), -np.log(v)
    a = log_u**theta + log_v**theta
    copula_cdf = np.exp(-(a ** (1 / theta)))

    return copula_cdf * (1 / u) * log_u ** (theta - 1) * a ** (1 / theta - 1)


def get_gumbel_copula_parameters(
    series_a: pd.Series, series_b: pd.Series
) -> dict[str, float]:
    """
    Calibrate a bivariate Gumbel copula to two series, via maximum likelihood:
    both series are transformed to pseudo-observations (see
    `get_pseudo_observations`), and theta is found by numerically maximizing the
    resulting Gumbel copula log-likelihood.

    Also known as: Gumbel copula calibration, Gumbel-Hougaard copula fit.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than 10 paired, non-missing observations are
            available.

    Returns:
        dict[str, float]: The fitted "Theta", the "Lower Tail Dependence"
        (always 0.0 for a Gumbel copula) and "Upper Tail Dependence" implied by
        it, the "Log-Likelihood", the "AIC" and the number of "Observations"
        used.
    """
    u, v = _align_pseudo_observations(series_a, series_b)

    def negative_log_likelihood(theta: float) -> float:
        return -float(np.sum(np.log(get_gumbel_copula_density(u, v, theta))))

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=(1.0, 50.0), method="bounded"
    )

    theta = float(result.x)
    log_likelihood = -float(result.fun)

    return {
        "Theta": theta,
        "Lower Tail Dependence": 0.0,
        "Upper Tail Dependence": 2 - 2 ** (1 / theta),
        "Log-Likelihood": log_likelihood,
        "AIC": _get_akaike_information_criterion(log_likelihood, 1),
        "Observations": len(u),
    }


def get_gumbel_copula_simulation(
    theta: float, n_simulations: int = 10_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a bivariate Gumbel copula, via conditional
    inversion (see `_simulate_archimedean_copula`).

    The result is on the pseudo-observation (0, 1) scale -- to obtain simulated
    returns, apply each column's own marginal quantile function (e.g. the
    empirical quantile function of the original return series) to it.

    Args:
        theta (float): The dependence parameter, >= 1, see
            `get_gumbel_copula_parameters`.
        n_simulations (int, optional): The number of joint draws to simulate.
            Defaults to 10,000.
        random_state (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    return _simulate_archimedean_copula(
        lambda u, v: _get_gumbel_conditional_cdf(u, v, theta),
        n_simulations,
        random_state,
    )


def get_frank_copula_density(
    u: pd.Series | np.ndarray, v: pd.Series | np.ndarray, theta: float
) -> np.ndarray:
    """
    Evaluate the bivariate Frank copula density at pseudo-observations `u` and
    `v` (see `get_pseudo_observations`).

    The formula is as follows:

    - c(u, v; theta) = [theta * (1 - e^-theta) * e^(-theta * (u + v))] / [(1 -
    e^-theta) - (1 - e^(-theta * u)) * (1 - e^(-theta * v))]^2

    Where theta (any nonzero real number) controls both the strength and sign
    of dependence -- unlike Clayton and Gumbel, Frank can represent negative
    dependence (theta < 0). The Frank copula has *zero* tail dependence in
    both tails, making it a flexible baseline for dependence that is
    concentrated in the middle of the distribution rather than the extremes.

    Also known as: Frank dependence, Archimedean copula (Frank family).

    Args:
        u (pd.Series | np.ndarray): The first series' pseudo-observations, in
            (0, 1).
        v (pd.Series | np.ndarray): The second series' pseudo-observations, in
            (0, 1).
        theta (float): The dependence parameter. Must be nonzero.

    Returns:
        np.ndarray: The copula density at each (u, v) pair.
    """
    numerator = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
    denominator = (
        (1 - np.exp(-theta)) - (1 - np.exp(-theta * u)) * (1 - np.exp(-theta * v))
    ) ** 2

    return numerator / denominator


def _get_frank_conditional_cdf(u: float, v: float, theta: float) -> float:
    """
    The Frank copula's h-function, h(v | u) = dC(u, v) / du. Used by
    `get_frank_copula_simulation` to simulate via conditional inversion (see
    `_simulate_archimedean_copula`).
    """
    d = np.exp(-theta) - 1
    n = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)

    return np.exp(-theta * u) * (np.exp(-theta * v) - 1) / (d + n)


def get_frank_copula_parameters(
    series_a: pd.Series, series_b: pd.Series
) -> dict[str, float]:
    """
    Calibrate a bivariate Frank copula to two series, via maximum likelihood:
    both series are transformed to pseudo-observations (see
    `get_pseudo_observations`), and theta is found by numerically maximizing the
    resulting Frank copula log-likelihood.

    Also known as: Frank copula calibration, Frank copula fit.

    Args:
        series_a (pd.Series): The first series (e.g. returns of one asset).
        series_b (pd.Series): The second series (e.g. returns of another asset).

    Raises:
        TypeError: If `series_a` or `series_b` is not a pd.Series.
        ValueError: If fewer than 10 paired, non-missing observations are
            available.

    Returns:
        dict[str, float]: The fitted "Theta", the "Lower Tail Dependence" and
        "Upper Tail Dependence" (both always 0.0 for a Frank copula), the
        "Log-Likelihood", the "AIC" and the number of "Observations" used.
    """
    u, v = _align_pseudo_observations(series_a, series_b)

    def negative_log_likelihood(theta: float) -> float:
        return -float(np.sum(np.log(get_frank_copula_density(u, v, theta))))

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=(-50.0, 50.0), method="bounded"
    )

    theta = float(result.x)
    log_likelihood = -float(result.fun)

    return {
        "Theta": theta,
        "Lower Tail Dependence": 0.0,
        "Upper Tail Dependence": 0.0,
        "Log-Likelihood": log_likelihood,
        "AIC": _get_akaike_information_criterion(log_likelihood, 1),
        "Observations": len(u),
    }


def get_frank_copula_simulation(
    theta: float, n_simulations: int = 10_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a bivariate Frank copula, via conditional
    inversion (see `_simulate_archimedean_copula`).

    The result is on the pseudo-observation (0, 1) scale -- to obtain simulated
    returns, apply each column's own marginal quantile function (e.g. the
    empirical quantile function of the original return series) to it.

    Args:
        theta (float): The dependence parameter, nonzero, see
            `get_frank_copula_parameters`.
        n_simulations (int, optional): The number of joint draws to simulate.
            Defaults to 10,000.
        random_state (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        pd.DataFrame: `n_simulations` simulated ("U", "V") pseudo-observation
        pairs.
    """
    return _simulate_archimedean_copula(
        lambda u, v: _get_frank_conditional_cdf(u, v, theta),
        n_simulations,
        random_state,
    )
