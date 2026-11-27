"""Dispersion Model: moments, order statistics and the OWA (ordered weighted average) measures"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import binom

from financetoolkit.risk import risk_model
from financetoolkit.risk.value_at_risk import evar_model
from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# The smallest tail probability of the Tail Gini's discretization, the same as Riskfolio-Lib's (Cajas, 2021).
TAIL_GINI_SMALLEST_ALPHA = 0.0001

RANGE_MEASURES = ["range", "var", "cvar", "tail_gini", "evar"]
L_MOMENT_RISK_METHODS = ["msd", "mss", "crra"]

# The order of the L-scale, the denominator of the L-moment ratios of order 3 and higher.
L_SCALE_ORDER = 2


def get_owa_gmd_weights(observations: int) -> np.ndarray:
    """
    Calculate the ordered weighted average (OWA) weights of the Gini Mean Difference: the
    weights that turn returns sorted from low to high into how far apart two returns lie on average.

    Applied to the sorted returns they give the mean absolute difference between every two
    distinct returns (Yitzhaki 1982, Cajas 2021).

    The formula is as follows:

        Weight_i = 2 * (2i - 1 - N) / (N * (N - 1)) for i = 1, ..., N

    Shares similar logic:
    - Riskfolio-Lib, owa_gmd: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_gmd

    Args:
        observations (int): The number of returns N, 2 or more.

    Returns:
        np.ndarray: The N weights.
    """
    rank = np.arange(1, observations + 1)

    return 2 * (2 * rank - 1 - observations) / (observations * (observations - 1))


def get_owa_cvar_weights(observations: int, alpha: float) -> np.ndarray:
    """
    Calculate the ordered weighted average (OWA) weights of the Conditional Value at Risk: the
    weights that turn returns sorted from low to high into the average loss in the worst tail.

    Applied to the sorted returns they give the average of the worst alpha share of the
    returns, the observation at the boundary counting with its fraction (Rockafellar and
    Uryasev 2002, Cajas 2021). The result is a return, negative for a loss.

    The formula is as follows:

        Weight_i = 1 / (N * alpha) for i < k, Weight_k = 1 - (k - 1) / (N * alpha), else 0

    Where k = ceil(N * alpha).

    Shares similar logic:
    - Riskfolio-Lib, owa_cvar: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_cvar

    Args:
        observations (int): The number of returns N.
        alpha (float): The tail probability, between 0 and 1.

    Returns:
        np.ndarray: The N weights, summing to 1.
    """
    boundary = int(np.ceil(observations * alpha)) - 1
    weights = np.zeros(observations)
    weights[:boundary] = 1 / (observations * alpha)
    weights[boundary] = 1 - weights[:boundary].sum()

    return weights


def get_owa_tail_gini_weights(
    observations: int, alpha: float = 0.05, points: int = 100
) -> np.ndarray:
    """
    Calculate the ordered weighted average (OWA) weights of the Tail Gini: the weights that
    turn returns sorted from low to high into a tail loss that weighs the worst returns most.

    The weights are a weighted sum of the Conditional Value at Risk at `points` tail
    probabilities from 0.0001 up to alpha, which approximates the Tail Gini of Furman, Wang and
    Zitikis (2017) as in Cajas (2021). Applied to the returns sorted from low to high they give
    a return, negative for a loss.

    The formula is as follows:

        Weights = Σ c_j * CVaR Weights(alpha_j)
        c_1 = alpha_2 * alpha_1 / alpha^2
        c_j = (alpha_(j+1) - alpha_(j-1)) * alpha_j / alpha^2 for 1 < j < m
        c_m = (alpha_m - alpha_(m-1)) / alpha

    Where alpha_1 < ... < alpha_m = alpha are evenly spaced from 0.0001 to alpha.

    Shares similar logic:
    - Riskfolio-Lib, owa_tg: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_tg

    Args:
        observations (int): The number of returns N.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        points (int, optional): The number of tail probabilities m. Defaults to 100.

    Returns:
        np.ndarray: The N weights.
    """
    if not TAIL_GINI_SMALLEST_ALPHA < alpha < 1:
        raise InvalidArgumentError(
            f"Alpha must be between {TAIL_GINI_SMALLEST_ALPHA} and 1 for the Tail Gini."
        )

    alphas = np.linspace(alpha, TAIL_GINI_SMALLEST_ALPHA, int(points))[::-1]
    factors = np.empty(len(alphas))
    factors[0] = alphas[1] * alphas[0] / alphas[-1] ** 2
    factors[1:-1] = (alphas[2:] - alphas[:-2]) * alphas[1:-1] / alphas[-1] ** 2
    factors[-1] = (alphas[-1] - alphas[-2]) / alphas[-1]

    return sum(
        factor * get_owa_cvar_weights(observations, tail_alpha)
        for factor, tail_alpha in zip(factors, alphas, strict=True)
    )


def get_owa_l_moment_weights(observations: int, order: int) -> np.ndarray:
    """
    Calculate the ordered weighted average (OWA) weights of the sample L-moment of an order:
    the weights that turn returns sorted from low to high into Hosking's (1990) unbiased
    estimate of the L-moment, which describes the shape of the distribution.

    The formula is as follows:

        Weight_i = Σ_j (-1)^j * C(k - 1, j) * C(i - 1, k - 1 - j) * C(N - i, j) / (k * C(N, k))

    Where C is the binomial coefficient, k the order and j runs from 0 to k - 1.

    Shares similar logic:
    - Riskfolio-Lib, owa_l_moment: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_l_moment

    Args:
        observations (int): The number of returns N, at least the order.
        order (int): The order k, 1 or higher (1 is the mean).

    Returns:
        np.ndarray: The N weights.
    """
    rank = np.arange(1, observations + 1)
    weights = sum(
        (-1) ** j
        * binom(order - 1, j)
        * binom(rank - 1, order - 1 - j)
        * binom(observations - rank, j)
        for j in range(order)
    )

    return weights / (order * binom(observations, order))


@lru_cache(maxsize=256)
def get_owa_l_moment_crm_weights(
    observations: int,
    order: int = 4,
    method: str = "msd",
    risk_aversion: float = 0.5,
    max_weight: float = 0.5,
) -> np.ndarray:
    """
    Calculate the ordered weighted average (OWA) weights of the L-moment convex risk measure:
    the weights that turn returns sorted from low to high into a risk measure of the spread
    and the shape of the distribution, which weighs the worst returns most.

    The measure is based on the L-moments of order 2 up to `order` (Cajas 2023): a combination
    Σ φ_k * (-1)^k * λ_k whose weights φ are non-negative, sum to 1 and decrease with the order,
    chosen so that the OWA weights increase from the best to the worst return (which makes the
    measure convex). The odd L-moments enter with a minus sign, so a negative L-skewness adds
    risk.

    The φ follow from one of three methods:

    - Minimum squared differences: the smallest sum of squared differences between
      neighbouring OWA weights.
    - Minimum sum of squares: the smallest sum of squared OWA weights.
    - Constant relative risk aversion: φ_k proportional to Π (risk_aversion + i - 1) / k!
      (i = 1 .. k - 1), with the OWA weights made non-decreasing by a running maximum.

    The first two are solved for φ with at most `max_weight` per L-moment, as convex
    programs over the k - 1 weights (Riskfolio-Lib solves them with cvxpy).

    Shares similar logic:
    - Riskfolio-Lib, owa_l_moment_crm: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/auxiliary.html#OwaWeights.owa_l_moment_crm

    Args:
        observations (int): The number of returns N, at least the order.
        order (int, optional): The highest L-moment order, 2 or higher. Defaults to 4.
        method (str, optional): How the weights of the L-moments are chosen. Options are: "msd"
            (minimum squared differences), "mss" (minimum sum of squares), "crra" (constant
            relative risk aversion). Defaults to "msd".
        risk_aversion (float, optional): The risk aversion of the constant relative risk
            aversion method, between 0 and 1. Defaults to 0.5.
        max_weight (float, optional): The largest weight of one L-moment for the minimum
            squared differences and minimum sum of squares methods, between 0 and 1.
            Defaults to 0.5.

    Returns:
        np.ndarray: The N weights.

    Notes:
    - Riskfolio-Lib's maximum entropy method is left out: it maximizes the entropy of a bound
    above the absolute OWA weights, which is the same for every feasible φ, so its weights
    depend on the solver rather than on the problem.
    """
    check_option("method", method, L_MOMENT_RISK_METHODS)
    if order < 2 or int(order) != order:  # noqa: PLR2004
        raise ValueError("The order must be an integer of 2 or higher.")
    if not 0 < risk_aversion < 1 or not 0 < max_weight < 1:
        raise ValueError(
            "The risk aversion and the maximum weight lie between 0 and 1."
        )

    moment_weights = np.column_stack(
        [
            (-1) ** k * get_owa_l_moment_weights(observations, k)
            for k in range(2, order + 1)
        ]
    )
    moments = moment_weights.shape[1]

    if method == "crra":
        factors = np.cumprod(risk_aversion + np.arange(moments)) / np.cumprod(
            np.arange(2, moments + 2)
        )
        weights = moment_weights @ (factors / factors.sum())

        return np.maximum.accumulate(weights)

    if max_weight * moments < 1:
        raise ValueError(
            "The maximum weight times the number of L-moments must be at least 1."
        )

    # The OWA weights must not decrease from the worst to the best return.
    differences = np.diff(moment_weights, axis=0)

    # Scaled so that both objectives are of the order of 1 whatever the number of returns.
    def objective(phi: np.ndarray) -> float:
        if method == "msd":
            return float(np.sum((differences @ phi) ** 2)) * observations**3

        return float(np.sum((moment_weights @ phi) ** 2)) * observations

    solution = minimize(
        objective,
        np.full(moments, 1 / moments),
        method="SLSQP",
        bounds=[(0, max_weight)] * moments,
        constraints=[
            {"type": "eq", "fun": lambda phi: phi.sum() - 1},
            {"type": "ineq", "fun": lambda phi: phi[:-1] - phi[1:]},
            {"type": "ineq", "fun": lambda phi: (differences @ phi) * observations**2},
        ],
        options={"ftol": 1e-14, "maxiter": 1000},
    )

    phi = np.clip(solution.x, 0, None)

    # SLSQP can stop on a line search message at the optimum, so the solution is judged on its constraints instead of its status.  # noqa: E501
    tolerance = 1e-8
    if (
        abs(phi.sum() - 1) > tolerance
        or (phi[1:] - phi[:-1]).max() > tolerance
        or phi.max() > max_weight + tolerance
        or (differences @ phi).min() * observations**2 < -tolerance
    ):
        raise ValueError(
            "No L-moment weights meet the constraints: use a higher max_weight or order, "
            "or method='crra'."
        )

    return moment_weights @ (phi / phi.sum())


def get_central_moment(
    returns: pd.Series | pd.DataFrame, order: int = 3, window_size: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the central moment of returns of an order: the average of the deviations from
    the mean raised to that power, the spread (order 2) or the shape (order 3 or 4) of the returns.

    The formula is as follows:

        Central Moment = (1 / N) * Σ (Return - Mean)^order

    Referenced against:
    - skfolio, third_central_moment: https://skfolio.org/generated/skfolio.measures.third_central_moment.html
    - skfolio, fourth_central_moment: https://skfolio.org/generated/skfolio.measures.fourth_central_moment.html
    - Riskfolio-Lib, Kurtosis: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Kurtosis

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        order (int, optional): The order, 2 or higher. Defaults to 3.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The central moment, see
        `risk_model.apply_to_observations`.

    Notes:
    - Riskfolio-Lib's Kurtosis is the square root of the fourth central moment.
    """
    if order < 2 or int(order) != order:  # noqa: PLR2004
        raise ValueError("The order must be an integer of 2 or higher.")

    return risk_model.apply_to_observations(
        returns,
        lambda samples: ((samples - samples.mean(axis=1, keepdims=True)) ** order).mean(
            axis=1
        ),
        window_size,
    )


def get_worst_realization(
    returns: pd.Series | pd.DataFrame, window_size: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Worst Realization of returns: the lowest return, the worst loss in the
    sample (Young 1998).

    Referenced against:
    - Riskfolio-Lib, WR: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.WR
    - skfolio, worst_realization: https://skfolio.org/generated/skfolio.measures.worst_realization.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The lowest return, negative for a loss.

    Notes:
    - Unlike Riskfolio-Lib and skfolio, which report the loss as a positive number, the
    lowest return keeps its sign.
    """
    return risk_model.apply_to_observations(
        returns, lambda samples: samples.min(axis=1), window_size, 1
    )


def get_gini_mean_difference(
    returns: pd.Series | pd.DataFrame, window_size: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Gini Mean Difference of returns: how far apart two returns lie on average,
    the mean absolute difference between every two distinct returns (Yitzhaki 1982).

    The formula is as follows:

        Gini Mean Difference = Σ_i Σ_j |Return_i - Return_j| / (N * (N - 1))

    It is calculated as the ordered weighted average of the sorted returns with the weights
    of `get_owa_gmd_weights`, and equals twice the second L-moment.

    Referenced against:
    - Riskfolio-Lib, GMD: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.GMD
    - skfolio, gini_mean_difference: https://skfolio.org/generated/skfolio.measures.gini_mean_difference.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Gini Mean Difference, a positive decimal.
    """
    return risk_model.apply_to_observations(
        returns,
        lambda samples: np.sort(samples, axis=1)
        @ get_owa_gmd_weights(samples.shape[1]),
        window_size,
    )


def get_tail_gini(
    returns: pd.Series | pd.DataFrame,
    alpha: float = 0.05,
    points: int = 100,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Tail Gini of returns: the loss in the worst tail of the returns, weighing
    the worst returns within that tail more than the Conditional Value at Risk does.

    It is a weighted average of the Conditional Value at Risk at tail probabilities from
    0.0001 up to alpha (see `get_owa_tail_gini_weights`).

    Referenced against:
    - Riskfolio-Lib, TG: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.TG

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        points (int, optional): The number of tail probabilities. Defaults to 100.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The Tail Gini as a return, negative for a loss.

    Notes:
    - Unlike Riskfolio-Lib, which reports the loss as a positive number, the Tail Gini is a
    return, negative for a loss.
    """
    get_owa_tail_gini_weights(2, alpha, points)  # Validates alpha up front.

    return risk_model.apply_to_observations(
        returns,
        lambda samples: np.sort(samples, axis=1)
        @ get_owa_tail_gini_weights(samples.shape[1], alpha, points),
        window_size,
    )


def get_l_moment(
    returns: pd.Series | pd.DataFrame,
    order: int = 2,
    ratio: bool = False,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the sample L-moment of returns of an order (Hosking 1990), or its L-moment
    ratio: the location, spread, asymmetry or tail weight of the returns, robust to outliers.

    The formula is as follows:

        L-moment λ_k = Σ_i Weight_i * Return_(i)
        L-moment ratio τ_k = λ_k / λ_2 for k >= 3, τ_2 = λ_2 / λ_1 (the L-CV)

    Where Return_(i) are the returns sorted from low to high and the weights are those of
    `get_owa_l_moment_weights`.

    Referenced against:
    - Riskfolio-Lib, L_Moment: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.L_Moment
    - scipy, lmoment: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lmoment.html

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        order (int, optional): The order k, 1 or higher (2 or higher for a ratio). Defaults to 2.
        ratio (bool, optional): Whether to return the L-moment ratio. Defaults to False.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The L-moment or L-moment ratio.
    """
    if order < 1 or int(order) != order or (ratio and order < 2):  # noqa: PLR2004
        raise ValueError(
            "The order must be an integer of 1 or higher, and 2 or higher for a ratio."
        )

    def l_moment(samples: np.ndarray) -> np.ndarray:
        observations = samples.shape[1]
        ordered = np.sort(samples, axis=1)
        moment = ordered @ get_owa_l_moment_weights(observations, order)

        if not ratio:
            return moment

        # The L-scale (order 2) is divided by the mean, every higher order by the L-scale.
        scale = ordered @ get_owa_l_moment_weights(
            observations, 1 if order == L_SCALE_ORDER else L_SCALE_ORDER
        )

        return np.divide(
            moment, scale, out=np.full(len(samples), np.nan), where=scale != 0
        )

    return risk_model.apply_to_observations(
        returns, l_moment, window_size, max(order, 2)
    )


def get_l_moment_risk_measure(
    returns: pd.Series | pd.DataFrame,
    order: int = 4,
    method: str = "msd",
    risk_aversion: float = 0.5,
    max_weight: float = 0.5,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the L-moment convex risk measure of returns: a risk measure of the spread and
    the shape of the returns that weighs the worst returns most (Cajas 2023).

    It is based on the L-moments of order 2 up to `order`, the ordered weighted average of the
    sorted returns with the weights of `get_owa_l_moment_crm_weights`.

    Referenced against:
    - Riskfolio-Lib, L_Moment_CRM (constant relative risk aversion): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.L_Moment_CRM

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        order (int, optional): The highest L-moment order, 2 or higher. Defaults to 4.
        method (str, optional): How the weights of the L-moments are chosen. Options are: "msd"
            (minimum squared differences), "mss" (minimum sum of squares), "crra" (constant
            relative risk aversion). Defaults to "msd".
        risk_aversion (float, optional): The risk aversion of the constant relative risk
            aversion method. Defaults to 0.5.
        max_weight (float, optional): The largest weight of one L-moment. Defaults to 0.5.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The risk measure, a positive decimal.

    Notes:
    - The minimum squared differences and minimum sum of squares methods are solved with tight
    tolerances; the default solver tolerances of Riskfolio-Lib can give slightly different values.
    """
    # The parameters are validated up front, as a period fails on them only when it is calculated.
    check_option("method", method, L_MOMENT_RISK_METHODS)
    if order < 2 or int(order) != order:  # noqa: PLR2004
        raise ValueError("The order must be an integer of 2 or higher.")
    if method != "crra" and max_weight * (order - 1) < 1:
        raise ValueError(
            "The maximum weight times the number of L-moments must be at least 1."
        )

    return risk_model.apply_to_observations(
        returns,
        lambda samples: np.sort(samples, axis=1)
        @ get_owa_l_moment_crm_weights(
            samples.shape[1], order, method, risk_aversion, max_weight
        ),
        window_size,
        max(order, 2),
    )


def get_risk_range(
    returns: pd.Series | pd.DataFrame,
    measure: str = "range",
    alpha: float = 0.05,
    beta: float | None = None,
    points: int = 100,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate a range measure of returns (Cajas 2021): how far the gains (the upper tail) lie
    from the losses (the lower tail), each measured the same way.

    The formula is as follows:

        Range = max(Return) - min(Return)
        VaR Range = Quantile(Return, 1 - beta) - Quantile(Return, alpha)
        CVaR Range = mean(Return >= Quantile(Return, 1 - beta)) - mean(Return <= Quantile(Return, alpha))
        Tail Gini Range = -Tail Gini(-Return, beta) - Tail Gini(Return, alpha)
        EVaR Range = -EVaR(-Return, beta) - EVaR(Return, alpha)

    The quantiles are interpolated linearly and the CVaR is the average of the returns
    beyond the quantile, as in the historic Value at Risk and Conditional Value at Risk of
    the Toolkit; the Tail Gini is that of `get_tail_gini` and the EVaR the historic Entropic
    Value at Risk of `evar_model.get_evar_historic`.

    Referenced against:
    - Riskfolio-Lib, RG (Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.RG
    - Riskfolio-Lib, TGRG (Tail Gini Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.TGRG
    - Riskfolio-Lib, EVRG (EVaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EVRG

    Shares similar logic:
    - Riskfolio-Lib, VRG (VaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.VRG
    - Riskfolio-Lib, CVRG (CVaR Range): https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.CVRG

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        measure (str, optional): The measure of the tails. Options are: "range" (Range),
            "var" (VaR Range), "cvar" (CVaR Range), "tail_gini" (Tail Gini Range), "evar"
            (EVaR Range). Defaults to "range".
        alpha (float, optional): The tail probability of the losses. Defaults to 0.05.
        beta (float | None, optional): The tail probability of the gains. Defaults to None,
            which uses alpha.
        points (int, optional): The number of tail probabilities of the Tail Gini.
            Defaults to 100.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The range, a positive decimal.
    """
    check_option("measure", measure, RANGE_MEASURES)

    beta = alpha if beta is None else beta

    if measure == "tail_gini":
        get_owa_tail_gini_weights(2, alpha, points)
        get_owa_tail_gini_weights(2, beta, points)
    elif measure != "range" and not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("Alpha and beta lie between 0 and 1.")

    def risk_range(samples: np.ndarray) -> np.ndarray:
        if measure == "range":
            return samples.max(axis=1) - samples.min(axis=1)

        if measure == "evar":
            return -evar_model.get_evar_samples(
                -samples, beta
            ) - evar_model.get_evar_samples(samples, alpha)

        if measure == "tail_gini":
            ordered = np.sort(samples, axis=1)
            observations = samples.shape[1]
            losses = ordered @ get_owa_tail_gini_weights(observations, alpha, points)
            gains = ordered[:, ::-1] @ get_owa_tail_gini_weights(
                observations, beta, points
            )

            return gains - losses

        lower = np.quantile(samples, alpha, axis=1, keepdims=True)
        upper = np.quantile(samples, 1 - beta, axis=1, keepdims=True)

        if measure == "var":
            return (upper - lower)[:, 0]

        return np.nanmean(
            np.where(samples >= upper, samples, np.nan), axis=1
        ) - np.nanmean(np.where(samples <= lower, samples, np.nan), axis=1)

    return risk_model.apply_to_observations(returns, risk_range, window_size)


def get_skewness_kurtosis_ratio(
    returns: pd.Series | pd.DataFrame, window_size: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Skewness-Kurtosis Ratio of returns (Bacon 2008): how lopsided the returns
    are relative to how heavy their tails are.

    It is the skewness over the (Pearson, not excess) kurtosis, both the bias-adjusted sample
    estimates of `risk_model.get_skewness` and `risk_model.get_kurtosis`.

    The formula is as follows:

        Skewness-Kurtosis Ratio = Skewness / Kurtosis

    Shares similar logic:
    - PerformanceAnalytics (R), SkewnessKurtosisRatio: https://github.com/braverock/PerformanceAnalytics/blob/v2.0.4/R/SkewnessKurtosisRatio.R#L31

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The ratio, NaN where the kurtosis is undefined.
    """
    if window_size:
        return risk_model.get_skewness(
            returns, window=window_size
        ) / risk_model.get_kurtosis(returns, fisher=False, window=window_size)

    return risk_model.get_skewness(returns) / risk_model.get_kurtosis(
        returns, fisher=False
    )
