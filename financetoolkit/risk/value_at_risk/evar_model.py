"""Entropic Value at Risk Model"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp

from financetoolkit.risk import risk_model
from financetoolkit.utilities.error_model import validate_alpha

ALPHA_CONSTRAINT = 0.5

# The largest log of the inverse risk aversion the historic EVaR searches: beyond it the bound
# is within 1e-20 of the largest loss of a sample scaled to [-1, 0], see get_evar_samples.
MAXIMUM_LOG_INVERSE_RISK_AVERSION = 50.0

# Two levels when a 'within period' index nests days inside a period (2020Q1).
MULTI_PERIOD_INDEX_LEVELS = 2


def get_evar_gaussian(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int | None = None
) -> pd.Series | pd.DataFrame:
    """
    Calculate the gaussian Entropic Value at Risk (EVaR) of returns: the tightest bound on the
    tail loss that the moment generating function of a normal distribution guarantees.

    The EVaR is the tightest upper bound on the loss that follows from the Chernoff
    inequality; for gaussian returns it has the closed form:

    - EVaR = mean - std * SQRT(-2 * ln(alpha))

    Where `std` is the sample standard deviation (with n - 1 in the denominator). It bounds the gaussian VaR and
    CVaR at the same `alpha`.

    For more information see:

    - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure."
    Journal of Optimization Theory and Applications, 155(3), 1105-1123.
    - https://en.wikipedia.org/wiki/Entropic_value_at_risk

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int | None, optional): The size of the rolling window. Defaults to None,
            which gives one value per column (or per period and column).

    Returns:
        pd.Series | pd.DataFrame: EVaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if window_size:
        return risk_model.apply_to_observations(
            returns,
            lambda samples: samples.mean(axis=1)
            - samples.std(axis=1, ddof=1) * np.sqrt(-2 * np.log(alpha)),
            window_size,
        )

    if (
        isinstance(returns, pd.DataFrame)
        and returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS
    ):
        periods = returns.index.get_level_values(0).unique()
        period_data_list = []

        for sub_period in periods:
            period_data = get_evar_gaussian(returns.loc[sub_period], alpha)
            period_data.name = sub_period

            if not period_data.empty:
                period_data_list.append(period_data)

        value_at_risk = pd.concat(period_data_list, axis=1)

        return value_at_risk.T

    # The entropic bound is expressed on losses; subtracting it keeps the result a negative return, consistent with the VaR and CVaR functions in this module.  # noqa: E501
    return returns.mean() - returns.std(ddof=1) * np.sqrt(-2 * np.log(alpha))


def get_evar_samples(samples: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate the historical Entropic Value at Risk (EVaR) of every row of an array of
    returns without missing values: the tightest entropic bound on their tail loss, as
    `get_evar_historic` reports it.

    The EVaR of the losses L = -Return is the smallest entropic bound over the risk
    aversion z > 0 (Ahmadi-Javid 2012):

    - EVaR = inf over z > 0 of z * ln(mean(exp(L / z)) / alpha)

    The losses of each row are scaled to [-1, 0] (EVaR is translation equivariant and
    positively homogeneous), so the exponentials cannot overflow, and with t = 1 / z the
    minimum is where the derivative of (ln(Σ exp(t * y)) - ln(n * alpha)) / t changes sign,
    which brentq finds on log t. When the largest loss holds a share of the sample of at
    least alpha the infimum is that largest loss (z goes to 0).

    Referenced against:
    - skfolio, evar: https://skfolio.org/generated/skfolio.measures.evar.html
    - Riskfolio-Lib, EVaR_Hist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EVaR_Hist

    Args:
        samples (np.ndarray): The returns, one sample per row.
        alpha (float): The tail probability, between 0 and 1.

    Returns:
        np.ndarray: The EVaR of every row as a return, negative for a loss.
    """
    losses = -samples
    largest = losses.max(axis=1)
    spread = largest - losses.min(axis=1)
    result = largest.astype(float)
    log_tail = np.log(samples.shape[1] * alpha)

    for row in np.flatnonzero(spread > 0):
        scaled = (losses[row] - largest[row]) / spread[row]

        # The largest loss covers the tail on its own: the bound falls to it as z goes to 0.
        if np.count_nonzero(scaled == 0) >= samples.shape[1] * alpha:
            continue

        def gradient_sign(log_t: float, scaled=scaled) -> float:
            t = np.exp(log_t)
            log_total = logsumexp(t * scaled)
            weights = np.exp(t * scaled - log_total)

            return t * (weights @ scaled) - log_total + log_tail

        # The scaled losses have a variance of at most 1/4 under any weights, so the
        # derivative is negative at t = sqrt(-2 ln(alpha)), and it is positive for a large t.
        lower = 0.5 * np.log(-2 * np.log(alpha))
        upper = lower + 1

        while gradient_sign(upper) <= 0 and upper < MAXIMUM_LOG_INVERSE_RISK_AVERSION:
            upper = min(upper + max(abs(upper), 1.0), MAXIMUM_LOG_INVERSE_RISK_AVERSION)

        if gradient_sign(upper) <= 0:
            continue

        t = np.exp(brentq(gradient_sign, lower, upper, xtol=1e-14))
        result[row] += spread[row] * (logsumexp(t * scaled) - log_tail) / t

    return -result


def get_evar_historic(
    returns: pd.Series | pd.DataFrame, alpha: float, window_size: int | None = None
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the historical Entropic Value at Risk (EVaR) of returns: the tightest bound on
    the tail loss that the observed returns guarantee, more severe than the CVaR.

    The bound follows from the Chernoff inequality, using the empirical moment generating
    function of the returns rather than a distribution:

    - EVaR = -inf over z > 0 of z * ln(mean(exp(-Return / z)) / alpha)

    It lies between the historical Conditional Value at Risk and the worst return.

    For more information see:

    - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure."
    Journal of Optimization Theory and Applications, 155(3), 1105-1123.

    Referenced against:
    - skfolio, evar: https://skfolio.org/generated/skfolio.measures.evar.html
    - Riskfolio-Lib, EVaR_Hist: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.EVaR_Hist

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        alpha (float): The tail probability (e.g., 0.05 for the 95% confidence level).
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The EVaR as a return, negative for a loss.
    """
    validate_alpha(alpha)

    return risk_model.apply_to_observations(
        returns, lambda samples: get_evar_samples(samples, alpha), window_size
    )


def get_entropic_risk_measure(
    returns: pd.Series | pd.DataFrame,
    risk_aversion: float = 1.0,
    alpha: float = 0.05,
    window_size: int | None = None,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the Entropic Risk Measure (ERM) of returns at a given risk aversion z: the
    entropic bound on the tail loss for that risk aversion, which the Entropic Value at Risk
    minimizes over z.

    The formula is as follows:

    - ERM = -z * ln(mean(exp(-Return / z)) / alpha)

    For every z it is at least as severe as the historical EVaR at the same alpha, and equal
    to it at the z that minimizes the bound.

    For more information see:

    - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure."
    Journal of Optimization Theory and Applications, 155(3), 1105-1123.

    Referenced against:
    - skfolio, entropic_risk_measure: https://skfolio.org/generated/skfolio.measures.entropic_risk_measure.html
    - Riskfolio-Lib, Entropic_RM: https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/risk.html#RiskFunctions.Entropic_RM

    Args:
        returns (pd.Series | pd.DataFrame): A Series or DataFrame of returns, optionally with
            a (period, date) index.
        risk_aversion (float, optional): The risk aversion z, larger than 0. Defaults to 1.
        alpha (float, optional): The tail probability. Defaults to 0.05.
        window_size (int | None, optional): The size of the rolling window. Defaults to None.

    Returns:
        pd.Series | pd.DataFrame | float: The ERM as a return, negative for a loss.
    """
    validate_alpha(alpha)

    if risk_aversion <= 0:
        raise ValueError("The risk aversion must be larger than 0.")

    return risk_model.apply_to_observations(
        returns,
        lambda samples: -risk_aversion
        * (
            logsumexp(-samples / risk_aversion, axis=1)
            - np.log(samples.shape[1] * alpha)
        ),
        window_size,
    )
