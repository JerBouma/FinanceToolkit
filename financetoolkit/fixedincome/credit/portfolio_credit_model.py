"""Portfolio Credit Model

The default of many names at once, in the one-factor copula model of Vasicek (1987) and Li
(2000): name i defaults before the horizon when its latent variable falls below the threshold
that matches its default probability p_i,

- X_i = √ρ_i Z + √(1 - ρ_i) ε_i, default when X_i ≤ F⁻¹(p_i)

with Z the common (systematic) factor and ε_i the name's own risk, independent standard normals
in the Gaussian copula. The asset correlation of two names is √(ρ_i ρ_j). Given the factor the
names default independently, with the conditional default probability

- p_i(z) = Φ((Φ⁻¹(p_i) - √ρ_i z) / √(1 - ρ_i))

so the loss distribution of the portfolio is a mixture over the factor of the loss
distributions of independent names. In the large homogeneous portfolio limit of Vasicek (2002)
the conditional loss is certain and the mixture has a closed form; for a finite portfolio the
conditional distribution is built name by name with the recursion of Andersen, Sidenius and
Basu (2003) and integrated over the factor with Gauss-Hermite quadrature.

The Student-t copula scales all latent variables by one common √(ν / W), W ~ χ²_ν, which makes
joint defaults more likely in the tails (tail dependence) at the same correlation; it is
integrated over Z and W (Mashal and Zeevi, 2002; O'Kane, 2008, ch. 21).

Default probabilities are cumulative to the horizon (1 - Q(T) of a survival curve, see
`credit_curve_model.get_survival_probability`); losses are shares of the portfolio notional.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gamma, ndtr, roots_genlaguerre, roots_hermitenorm
from scipy.stats import (
    norm,
    t as student_t,
)

from financetoolkit.fixedincome.bonds.bond_model import get_cash_flow_periods
from financetoolkit.fixedincome.credit.cds_model import get_credit_default_swap
from financetoolkit.fixedincome.credit.credit_curve_model import (
    DEFAULT_RECOVERY_RATE,
    get_survival_probability,
)
from financetoolkit.utilities.error_model import check_option

COPULAS = ("gaussian", "student_t")
# The finest default loss unit is a hundredth of the smallest loss given default: losses that
# are not multiples of it are rounded to it, an error of at most half a percent of that loss.
MAXIMUM_UNIT_DIVISOR = 100
# How close to a whole multiple of the unit a loss must be: loose enough that tiny
# perturbations of the inputs keep the same unit.
UNIT_TOLERANCE = 1e-6


def _check_vasicek(
    default_probability: float, correlation: float, recovery_rate: float
) -> None:
    """Check the inputs of the large homogeneous portfolio formulas."""
    if not 0 < default_probability < 1:
        raise ValueError("The default probability must lie strictly between 0 and 1.")
    if not 0 < correlation < 1:
        raise ValueError("The correlation must lie strictly between 0 and 1.")
    if not 0 <= recovery_rate < 1:
        raise ValueError("The recovery rate must lie between 0 and 1.")


def get_vasicek_loss_distribution(
    loss: float | np.ndarray,
    default_probability: float,
    correlation: float,
    recovery_rate: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the probability that the loss of a large homogeneous portfolio stays at or
    below a level: Vasicek's (1987, 2002) limit of the one-factor Gaussian copula.

    In a portfolio of many small loans with the same default probability p and asset
    correlation ρ the share of loans that default given the factor is no longer random, it
    is p(Z), so the loss distribution is that of p(Z) over the factor. It is the
    distribution behind the Basel II and III internal ratings-based capital formula, which
    takes its 99.9% quantile.

    The formula is as follows:

    - P(L ≤ x) = Φ((√(1 - ρ) Φ⁻¹(x / (1 - R)) - Φ⁻¹(p)) / √ρ) for 0 < x < 1 - R
    - P(L ≤ x) = 1 for x ≥ 1 - R, 0 for x < 0

    where L is the portfolio loss as a share of its notional, R the recovery rate (so a share
    θ of defaults loses (1 - R) θ), Φ the standard normal distribution and Φ⁻¹ its inverse.
    The mean loss is (1 - R) p whatever the correlation; the correlation spreads it out.

    For more information, see the following sources:

    - Vasicek, O. (1987). "Probability of Loss on Loan Portfolio." KMV Corporation.
    - Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." Risk, 15(12), 160-162.
    - Basel Committee on Banking Supervision (2005). "An Explanatory Note on the Basel II IRB
      Risk Weight Functions."

    Referenced against:
    - FinancePy, portfolio_cdf_lhp: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_lhp.py#L60

    Args:
        loss (float | np.ndarray): The loss level(s) x, as a share of the notional.
        default_probability (float): The default probability p of every name to the horizon.
        correlation (float): The asset correlation ρ, strictly between 0 and 1.
        recovery_rate (float, optional): The recovery rate R. Defaults to 0, which makes the
            loss the share of names that default.

    Returns:
        float | np.ndarray: The probability that the loss is at most x.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    portfolio_credit_model.get_vasicek_loss_distribution(
        loss=0.05, default_probability=0.02, correlation=0.15
    )
    ```

    Which returns:

    ```
    0.9173129708733823
    ```
    """
    _check_vasicek(default_probability, correlation, recovery_rate)
    share = np.asarray(loss, dtype=float) / (1 - recovery_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = norm.cdf(
            (
                np.sqrt(1 - correlation) * norm.ppf(np.clip(share, 0.0, 1.0))
                - norm.ppf(default_probability)
            )
            / np.sqrt(correlation)
        )
    value = np.where(share >= 1, 1.0, np.where(share < 0, 0.0, inside))
    return float(value) if np.ndim(value) == 0 else value


def get_vasicek_loss_density(
    loss: float | np.ndarray,
    default_probability: float,
    correlation: float,
    recovery_rate: float = 0.0,
) -> float | np.ndarray:
    """
    Calculate the probability density of the loss of a large homogeneous portfolio: how
    likely each loss level is, the derivative of Vasicek's loss distribution
    (`get_vasicek_loss_distribution`).

    For a correlation below one half it is zero at both ends and peaks near the mean loss;
    above one half it piles up at no loss and at the full loss.

    The formula is as follows:

    - f(x) = 1 / (1 - R) · √((1 - ρ) / ρ) · exp(½ Φ⁻¹(θ)² - (Φ⁻¹(p) - √(1 - ρ) Φ⁻¹(θ))² / (2ρ))
    - θ = x / (1 - R), for 0 < x < 1 - R, and 0 elsewhere

    with p the default probability, ρ the asset correlation and R the recovery rate.

    For more information, see the following sources:

    - Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." Risk, 15(12), 160-162.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      section 16.3.

    Referenced against:
    - FinancePy, lhp_analytical_density: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_lhp.py#L219

    Shares similar logic:
    - FinancePy, lhp_density: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_lhp.py#L137

    Args:
        loss (float | np.ndarray): The loss level(s) x, as a share of the notional.
        default_probability (float): The default probability p of every name to the horizon.
        correlation (float): The asset correlation ρ, strictly between 0 and 1.
        recovery_rate (float, optional): The recovery rate R. Defaults to 0.

    Returns:
        float | np.ndarray: The density of the loss at x.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    portfolio_credit_model.get_vasicek_loss_density(
        loss=0.05, default_probability=0.02, correlation=0.15
    )
    ```

    Which returns:

    ```
    3.5179545316827525
    ```
    """
    _check_vasicek(default_probability, correlation, recovery_rate)
    share = np.asarray(loss, dtype=float) / (1 - recovery_rate)
    inside = (share > 0) & (share < 1)
    quantile = norm.ppf(np.where(inside, share, 0.5))
    density = (
        np.sqrt((1 - correlation) / correlation)
        * np.exp(
            0.5 * quantile**2
            - (norm.ppf(default_probability) - np.sqrt(1 - correlation) * quantile) ** 2
            / (2 * correlation)
        )
        / (1 - recovery_rate)
    )
    value = np.where(inside, density, 0.0)
    return float(value) if np.ndim(value) == 0 else value


def get_vasicek_tranche_loss(
    attachment: float,
    detachment: float,
    default_probability: float,
    correlation: float,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
) -> float:
    """
    Calculate the expected loss of a tranche of a large homogeneous portfolio, as a share of
    the tranche notional: how much of the tranche the portfolio's defaults are expected to
    wipe out.

    A tranche from the attachment K_1 to the detachment K_2 absorbs the portfolio losses
    between them; the share of it that survives, 1 - the expected loss, is the tranche
    survival probability with which its premium leg is valued like a CDS (O'Kane, 2008, ch.
    12). Given the factor the loss is certain, so the expectation is a single integral,
    split at the factor value at which the loss reaches the strike.

    The formula is as follows:

    - Tranche Loss = (E[min(L, K_2)] - E[min(L, K_1)]) / (K_2 - K_1)
    - E[min(L, K)] = K Φ(A) + (1 - R) ∫_A^∞ Φ((Φ⁻¹(p) - √ρ z) / √(1 - ρ)) φ(z) dz
    - A = (Φ⁻¹(p) - √(1 - ρ) Φ⁻¹(K / (1 - R))) / √ρ, for K < 1 - R
    - E[min(L, K)] = (1 - R) p, for K ≥ 1 - R

    where L = (1 - R) p(Z) is the conditional loss, which exceeds K when the factor Z falls
    below A, φ and Φ the standard normal density and distribution, p the default
    probability, ρ the asset correlation and R the recovery rate. The integral is evaluated
    with adaptive quadrature. Over the whole capital structure (K_1 = 0, K_2 = 1) the
    tranche losses add up to the portfolio's expected loss (1 - R) p.

    For more information, see the following sources:

    - Vasicek, O. (2002). "The Distribution of Loan Portfolio Value." Risk, 15(12), 160-162.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      sections 16.3 and 16.4.

    Referenced against:
    - FinancePy, tr_surv_prob_lhp: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_lhp.py#L17

    Shares similar logic:
    - FinancePy, exp_min_lk: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_lhp.py#L257

    Args:
        attachment (float): The attachment K_1, as a share of the portfolio notional.
        detachment (float): The detachment K_2 > K_1, as a share of the portfolio notional.
        default_probability (float): The default probability p of every name to the horizon.
        correlation (float): The asset correlation ρ, strictly between 0 and 1.
        recovery_rate (float, optional): The recovery rate R. Defaults to 0.4, the ISDA
            standard for senior unsecured debt.

    Returns:
        float: The expected tranche loss, as a share of the tranche notional.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    portfolio_credit_model.get_vasicek_tranche_loss(
        attachment=0.03, detachment=0.07, default_probability=0.05, correlation=0.2
    )
    ```

    Which returns:

    ```
    0.19468467607231035
    ```
    """
    _check_vasicek(default_probability, correlation, recovery_rate)
    if not 0 <= attachment < detachment <= 1:
        raise ValueError("The tranche needs 0 ≤ attachment < detachment ≤ 1.")
    threshold = norm.ppf(default_probability)
    severity = 1 - recovery_rate

    def expected_minimum(strike: float) -> float:
        if strike <= 0:
            return 0.0
        if strike >= severity:
            return severity * default_probability
        boundary = (
            threshold - np.sqrt(1 - correlation) * norm.ppf(strike / severity)
        ) / np.sqrt(correlation)
        tail, _ = quad(
            lambda z: ndtr(
                (threshold - np.sqrt(correlation) * z) / np.sqrt(1 - correlation)
            )
            * np.exp(-0.5 * z**2)
            / np.sqrt(2 * np.pi),
            boundary,
            np.inf,
            epsabs=1e-15,
            epsrel=1e-12,
            limit=200,
        )
        return strike * norm.cdf(boundary) + severity * tail

    return float(expected_minimum(detachment) - expected_minimum(attachment)) / (
        detachment - attachment
    )


def _factor_nodes(
    copula: str, degrees_of_freedom: float | None, integration_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The quadrature nodes and weights of the copula's common variables: the factor Z on
    Gauss-Hermite nodes and, for the Student-t copula, the scale √(W / ν) of a chi-squared W
    with ν degrees of freedom on generalised Gauss-Laguerre nodes (W = 2u, u ~ Γ(ν / 2)).
    Returns the factor values, the scales (1 for the Gaussian copula) and the weights.
    """
    check_option("copula", copula, COPULAS)
    if integration_points <= 1:
        raise ValueError("Use at least 2 integration points.")
    factor, factor_weights = roots_hermitenorm(integration_points)
    factor_weights = factor_weights / np.sqrt(2 * np.pi)
    if copula == "gaussian":
        return factor, np.ones_like(factor), factor_weights
    if degrees_of_freedom is None or degrees_of_freedom <= 0:
        raise ValueError("The Student-t copula needs positive degrees of freedom.")
    gamma_nodes, gamma_weights = roots_genlaguerre(
        integration_points, degrees_of_freedom / 2 - 1
    )
    gamma_weights = gamma_weights / gamma(degrees_of_freedom / 2)
    scales = np.sqrt(2 * gamma_nodes / degrees_of_freedom)
    return (
        np.repeat(factor, integration_points),
        np.tile(scales, integration_points),
        np.outer(factor_weights, gamma_weights).ravel(),
    )


def _conditional_loss_recursion(
    default_probabilities: np.ndarray,
    correlations: np.ndarray,
    units: np.ndarray,
    copula: str,
    degrees_of_freedom: float | None,
    integration_points: int,
) -> np.ndarray:
    """
    The distribution of the portfolio loss in loss units, by the recursion of Andersen,
    Sidenius and Basu (2003) given the factor, integrated over the copula's common variables.
    """
    factor, scales, weights = _factor_nodes(
        copula, degrees_of_freedom, integration_points
    )
    if copula == "gaussian":
        thresholds = norm.ppf(default_probabilities)
    else:
        thresholds = student_t.ppf(default_probabilities, degrees_of_freedom)
    # p_i(z, w) = Φ((F⁻¹(p_i) √(W / ν) - √ρ_i z) / √(1 - ρ_i)), nodes by names.
    conditional = norm.cdf(
        (
            thresholds[None, :] * scales[:, None]
            - np.sqrt(correlations)[None, :] * factor[:, None]
        )
        / np.sqrt(1 - correlations)[None, :]
    )
    distribution = np.zeros((len(factor), int(units.sum()) + 1))
    distribution[:, 0] = 1.0
    for name, unit in enumerate(units):
        probability = conditional[:, [name]]
        shifted = np.zeros_like(distribution)
        shifted[:, unit:] = distribution[:, : distribution.shape[1] - unit]
        distribution = distribution * (1 - probability) + shifted * probability
    return weights @ distribution


def _check_portfolio(
    default_probabilities: np.ndarray | list | float,
    correlation: np.ndarray | list | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the default probabilities and correlations and give one of each per name."""
    probabilities = np.atleast_1d(np.asarray(default_probabilities, dtype=float))
    if (
        probabilities.ndim != 1
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Pass one default probability per name, between 0 and 1.")
    correlations = np.broadcast_to(
        np.asarray(correlation, dtype=float), probabilities.shape
    ).copy()
    if np.any(correlations < 0) or np.any(correlations >= 1):
        raise ValueError("The correlations must lie between 0 and 1 (excluded).")
    return probabilities, correlations


def get_portfolio_loss_distribution(
    default_probabilities: np.ndarray | list,
    correlation: np.ndarray | list | float,
    recovery_rates: np.ndarray | list | float = DEFAULT_RECOVERY_RATE,
    notionals: np.ndarray | list | float = 1.0,
    copula: str = "gaussian",
    degrees_of_freedom: float | None = None,
    loss_unit: float | None = None,
    integration_points: int = 64,
) -> pd.Series:
    """
    Calculate the loss distribution of a portfolio of named credits in the one-factor
    Gaussian or Student-t copula: the probability of every portfolio loss, with the
    recursion of Andersen, Sidenius and Basu (2003).

    Given the common factor the names default independently, so the distribution of the loss
    is built by adding one name at a time: each moves its loss up by its loss given default
    with its conditional default probability. The conditional distributions are then
    averaged over the factor (and, for the Student-t copula, the common scale) with Gaussian
    quadrature. It holds names with different default probabilities, correlations,
    recoveries and notionals; as the number of names grows the distribution of a homogeneous
    portfolio tends to Vasicek's large portfolio limit (`get_vasicek_loss_distribution`).

    The formula is as follows:

    - Gaussian: p_i(z) = Φ((Φ⁻¹(p_i) - √ρ_i z) / √(1 - ρ_i))
    - Student-t: p_i(z, w) = Φ((t_ν⁻¹(p_i) √(w / ν) - √ρ_i z) / √(1 - ρ_i)), w ~ χ²_ν
    - P^(k)(l | z) = P^(k-1)(l | z) (1 - p_k(z)) + P^(k-1)(l - u_k | z) p_k(z),
      P^(0)(l | z) = 1 for l = 0
    - P(L = l) = ∫ P^(N)(l | z) φ(z) dz (and over w for the Student-t copula)

    where p_i is the default probability of name i, ρ_i its correlation with the factor,
    t_ν⁻¹ the inverse Student-t distribution with ν degrees of freedom and u_k the loss of
    name k in loss units: its loss given default N_k (1 - R_k) divided by the loss unit and
    rounded to a whole number. The integrals use Gauss-Hermite nodes for the factor and
    generalised Gauss-Laguerre nodes for the chi-squared scale.

    For more information, see the following sources:

    - Vasicek, O. (1987). "Probability of Loss on Loan Portfolio." KMV Corporation.
    - Li, D.X. (2000). "On Default Correlation: A Copula Function Approach." The Journal of
      Fixed Income, 9(4), 43-54.
    - Andersen, L., Sidenius, J. and Basu, S. (2003). "All Your Hedges in One Basket."
      Risk, 16(11), 67-72.
    - Mashal, R. and Zeevi, A. (2002). "Beyond Correlation: Extreme Co-movements Between
      Financial Assets." Columbia University working paper.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 13 and 18.

    The Student-t copula is checked against a Monte Carlo simulation of the copula.

    Referenced against:
    - FinancePy, loss_dbn_recursion_gcd: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_onefactor.py#L29

    Shares similar logic:
    - FinancePy, homog_basket_loss_dbn: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_onefactor.py#L95

    Args:
        default_probabilities (np.ndarray | list): The default probability p_i of every name
            to the horizon.
        correlation (np.ndarray | list | float): The correlation ρ_i of every name with the
            factor (between 0 and 1, excluded), or one for all; two names have an asset
            correlation of √(ρ_i ρ_j).
        recovery_rates (np.ndarray | list | float, optional): The recovery rate R_i of every
            name, or one for all. Defaults to 0.4.
        notionals (np.ndarray | list | float, optional): The notional N_i of every name, or
            one for all. Defaults to 1.
        copula (str, optional): The copula. Options are: "gaussian" (the one-factor Gaussian
            copula), "student_t" (the one-factor Student-t copula). Defaults to "gaussian".
        degrees_of_freedom (float | None, optional): The degrees of freedom ν of the
            Student-t copula. Defaults to None.
        loss_unit (float | None, optional): The loss unit, in the currency of the notionals;
            None takes the largest unit of which every loss given default is a whole
            multiple, the smallest loss divided by 1, 2, ..., 100, and else a hundredth of
            the smallest loss. Defaults to None.
        integration_points (int, optional): The quadrature nodes for the factor (and as
            many for the scale). Defaults to 64.

    Returns:
        pd.Series: The probability of every loss, indexed by the loss as a share of the
        portfolio notional.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    portfolio_credit_model.get_portfolio_loss_distribution(
        default_probabilities=[0.01, 0.02, 0.03, 0.05, 0.08], correlation=0.3
    )
    ```

    Which returns:

    ```
    Loss
    0.00    0.842908
    0.12    0.129548
    0.24    0.022884
    0.36    0.004009
    0.48    0.000595
    0.60    0.000055
    ```
    """
    probabilities, correlations = _check_portfolio(default_probabilities, correlation)
    recoveries = np.broadcast_to(
        np.asarray(recovery_rates, dtype=float), probabilities.shape
    )
    notionals = np.broadcast_to(np.asarray(notionals, dtype=float), probabilities.shape)
    if np.any(recoveries < 0) or np.any(recoveries > 1) or np.any(notionals <= 0):
        raise ValueError(
            "The recovery rates must lie between 0 and 1 and the notionals be positive."
        )
    losses = notionals * (1 - recoveries)
    if loss_unit is None:
        positive = losses[losses > 0]
        smallest = float(positive.min()) if len(positive) else 1.0
        # The largest unit, the smallest loss over 1, 2, ..., 100, of which every loss is a
        # whole multiple.
        loss_unit = smallest / MAXIMUM_UNIT_DIVISOR
        for divisor in range(1, MAXIMUM_UNIT_DIVISOR + 1):
            ratios = losses * divisor / smallest
            if np.all(np.abs(ratios - np.rint(ratios)) <= UNIT_TOLERANCE):
                loss_unit = smallest / divisor
                break
    if loss_unit <= 0:
        raise ValueError("The loss unit must be positive.")
    units = np.rint(losses / loss_unit).astype(int)

    distribution = _conditional_loss_recursion(
        probabilities,
        correlations,
        units,
        copula,
        degrees_of_freedom,
        integration_points,
    )
    return pd.Series(
        distribution,
        index=pd.Index(
            np.arange(len(distribution)) * loss_unit / notionals.sum(), name="Loss"
        ),
        name="Probability",
    )


def get_tranche_expected_loss(
    loss_distribution: pd.Series, attachment: float, detachment: float
) -> float:
    """
    Calculate the expected loss of a tranche as a share of its notional, from a portfolio
    loss distribution (`get_portfolio_loss_distribution`): how much of the tranche the
    portfolio's defaults are expected to wipe out.

    The tranche from the attachment K_1 to the detachment K_2 absorbs the portfolio losses
    between them; 1 - the expected loss is the tranche survival probability, with which the
    premium and protection legs of a tranche are valued like those of a CDS (O'Kane, 2008,
    ch. 12). The tranche losses of the whole capital structure, weighted by their widths,
    add up to the expected portfolio loss.

    The formula is as follows:

    - Tranche Loss = SUM_l P(L = l) min(max(l - K_1, 0), K_2 - K_1) / (K_2 - K_1)
    - Tranche Survival Probability = 1 - Tranche Loss

    where l are the portfolio losses as shares of the portfolio notional.

    For more information, see the following sources:

    - Andersen, L., Sidenius, J. and Basu, S. (2003). "All Your Hedges in One Basket."
      Risk, 16(11), 67-72.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 12 and 18.

    Referenced against:
    - FinancePy, tranche_surv_prob_recursion: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gauss_copula_onefactor.py#L150

    Args:
        loss_distribution (pd.Series): The probability of every loss, indexed by the loss as
            a share of the portfolio notional.
        attachment (float): The attachment K_1, as a share of the portfolio notional.
        detachment (float): The detachment K_2 > K_1, as a share of the portfolio notional.

    Returns:
        float: The expected tranche loss, as a share of the tranche notional.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    distribution = portfolio_credit_model.get_portfolio_loss_distribution(
        default_probabilities=[0.05] * 125, correlation=0.2
    )

    portfolio_credit_model.get_tranche_expected_loss(
        distribution, attachment=0.03, detachment=0.07
    )
    ```

    Which returns:

    ```
    0.2035087075102297
    ```
    """
    if not 0 <= attachment < detachment <= 1:
        raise ValueError("The tranche needs 0 ≤ attachment < detachment ≤ 1.")
    losses = loss_distribution.index.to_numpy(dtype=float)
    tranche = np.clip(losses - attachment, 0.0, detachment - attachment)
    return float(
        loss_distribution.to_numpy(dtype=float) @ tranche / (detachment - attachment)
    )


def get_nth_to_default_probability(
    default_probabilities: np.ndarray | list,
    correlation: np.ndarray | list | float,
    n: int = 1,
    copula: str = "gaussian",
    degrees_of_freedom: float | None = None,
    integration_points: int = 64,
) -> float:
    """
    Calculate the probability that at least n names of a basket default before the horizon:
    the probability that an nth-to-default basket pays out.

    The number of defaults is the loss of a portfolio in which every name loses one unit, so
    its distribution follows from the recursion of `get_portfolio_loss_distribution`. A
    first-to-default basket pays when any name defaults, so it is worth more the less the
    names are correlated; for the last names to default it is the other way around.

    The formula is as follows:

    - P(N ≥ n) = 1 - SUM_(k=0)^(n-1) P(N = k)

    with N the number of defaults to the horizon in the one-factor copula. Without
    correlation the first-to-default probability is 1 - PROD(1 - p_i).

    For more information, see the following sources:

    - Li, D.X. (2000). "On Default Correlation: A Copula Function Approach." The Journal of
      Fixed Income, 9(4), 43-54.
    - Andersen, L., Sidenius, J. and Basu, S. (2003). "All Your Hedges in One Basket."
      Risk, 16(11), 67-72.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 14.

    Shares similar logic:
    - FinancePy, CDSBasket: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/credit/cds_basket.py#L38

    Args:
        default_probabilities (np.ndarray | list): The default probability p_i of every name
            to the horizon.
        correlation (np.ndarray | list | float): The correlation ρ_i of every name with the
            factor (between 0 and 1, excluded), or one for all.
        n (int, optional): The default that triggers the basket, between 1 and the number of
            names. Defaults to 1.
        copula (str, optional): The copula. Options are: "gaussian" (the one-factor Gaussian
            copula), "student_t" (the one-factor Student-t copula). Defaults to "gaussian".
        degrees_of_freedom (float | None, optional): The degrees of freedom ν of the
            Student-t copula. Defaults to None.
        integration_points (int, optional): The quadrature nodes for the factor (and as
            many for the scale). Defaults to 64.

    Returns:
        float: The probability of at least n defaults.

    As an example:

    ```python
    from financetoolkit.fixedincome.credit import portfolio_credit_model

    portfolio_credit_model.get_nth_to_default_probability(
        default_probabilities=[0.01, 0.02, 0.03, 0.05, 0.08], correlation=0.3, n=2
    )
    ```

    Which returns:

    ```
    0.027543651684869497
    ```
    """
    probabilities, correlations = _check_portfolio(default_probabilities, correlation)
    if not 1 <= n <= len(probabilities):
        raise ValueError(f"n must lie between 1 and the {len(probabilities)} names.")
    counts = _conditional_loss_recursion(
        probabilities,
        correlations,
        np.ones(len(probabilities), dtype=int),
        copula,
        degrees_of_freedom,
        integration_points,
    )
    return float(1 - counts[:n].sum())


def get_nth_to_default_swap(
    n: int,
    maturity: float,
    zero_curve: pd.Series,
    hazard_rates: np.ndarray | list,
    hazard_times: np.ndarray | list,
    correlation: np.ndarray | list | float,
    coupon: float = 0.01,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    notional: float = 1.0,
    frequency: int = 4,
    copula: str = "gaussian",
    degrees_of_freedom: float | None = None,
    integration_points: int = 64,
) -> pd.Series:
    """
    Calculate the par spread and value of an nth-to-default basket swap: what protection
    that pays (1 - R) of the notional at the nth default among the names of the basket costs
    and is worth, a CDS whose premium is paid until that default.

    The time of the nth default has a survival curve of its own, the probability of fewer
    than n defaults at each date, which the one-factor copula gives at every premium date
    (`get_nth_to_default_probability`, with the default probabilities of each name's
    survival curve at that date and one factor for all dates). Between the dates its hazard
    rate is held flat, and the swap is then valued as a CDS on that curve with the ISDA CDS
    Standard Model (`cds_model.get_credit_default_swap`).

    The formula is as follows:

    - Q_n(t) = 1 - P(N(t) ≥ n), p_i(t) = 1 - Q_i(t)
    - λ_n,k = -ln(Q_n(t_k) / Q_n(t_(k-1))) / (t_k - t_(k-1)) on the premium dates t_k
    - Par Spread, RPV01, legs and upfront as for a single-name CDS on λ_n

    where Q_i is the survival curve of name i (piecewise flat hazard rates, see
    `credit_curve_model.get_survival_probability`) and N(t) the number of defaults by t. All
    names share one recovery rate and the same notional, so the payout does not depend on
    which name defaults nth. The zero curve is annually compounded; the hazard rates are
    continuous and per year.

    For more information, see the following sources:

    - Li, D.X. (2000). "On Default Correlation: A Copula Function Approach." The Journal of
      Fixed Income, 9(4), 43-54.
    - Laurent, J.-P. and Gregory, J. (2005). "Basket Default Swaps, CDOs and Factor
      Copulas." Journal of Risk, 7(4), 103-122.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 14.

    Referenced against:
    - FinancePy, CDSBasket.value_1f_gaussian_homo: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/credit/cds_basket.py#L354

    Args:
        n (int): The default that triggers the protection, between 1 and the number of
            names.
        maturity (float): The years until the protection ends.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        hazard_rates (np.ndarray | list): The piecewise flat hazard rates per year, names by
            hazard times.
        hazard_times (np.ndarray | list): The maturities up to which each hazard rate holds,
            shared by the names.
        correlation (np.ndarray | list | float): The correlation ρ_i of every name with the
            factor (between 0 and 1, excluded), or one for all.
        coupon (float, optional): The running coupon per year, as a decimal. Defaults to 0.01.
        recovery_rate (float, optional): The recovery rate of every name. Defaults to 0.4.
        notional (float, optional): The notional. Defaults to 1.
        frequency (int, optional): The number of premium payments per year. Defaults to 4.
        copula (str, optional): The copula. Options are: "gaussian" (the one-factor Gaussian
            copula), "student_t" (the one-factor Student-t copula). Defaults to "gaussian".
        degrees_of_freedom (float | None, optional): The degrees of freedom ν of the
            Student-t copula. Defaults to None.
        integration_points (int, optional): The quadrature nodes for the factor (and as
            many for the scale). Defaults to 64.

    Returns:
        pd.Series: The Par Spread, RPV01, Full RPV01, Protection Leg, Premium Leg, Accrued
        Premium, Upfront and Mark-to-Market of `cds_model.get_credit_default_swap`.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.credit import portfolio_credit_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    portfolio_credit_model.get_nth_to_default_swap(
        n=1, maturity=5, zero_curve=zero_curve,
        hazard_rates=[[0.01], [0.015], [0.02], [0.025], [0.03]], hazard_times=[5],
        correlation=0.3
    )
    ```

    Which returns:

    ```
    Par Spread         0.048218
    RPV01              3.831740
    Full RPV01         3.831740
    Protection Leg     0.184759
    Premium Leg        0.038317
    Accrued Premium    0.000000
    Upfront            0.146441
    Mark-to-Market     0.146441
    ```
    """
    hazard_rates = np.atleast_2d(np.asarray(hazard_rates, dtype=float))
    if maturity <= 0:
        raise ValueError("The maturity must be positive.")
    dates = get_cash_flow_periods(maturity, frequency) / frequency
    dates = dates[dates > 0]
    survival = np.array(
        [
            1
            - get_nth_to_default_probability(
                1
                - np.array(
                    [
                        get_survival_probability(rates, hazard_times, date)
                        for rates in hazard_rates
                    ]
                ).ravel(),
                correlation,
                n,
                copula,
                degrees_of_freedom,
                integration_points,
            )
            for date in dates
        ]
    )
    hazard = -np.diff(np.log(np.concatenate([[1.0], survival]))) / np.diff(
        np.concatenate([[0.0], dates])
    )
    return get_credit_default_swap(
        maturity,
        zero_curve,
        hazard,
        dates,
        coupon=coupon,
        recovery_rate=recovery_rate,
        notional=notional,
        frequency=frequency,
    )
