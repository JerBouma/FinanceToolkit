"""VaR Backtesting Model"""

__docformat__ = "google"

import pandas as pd
from scipy import special, stats

MINIMUM_OBSERVATIONS_FOR_TRANSITIONS = 2


def _align(returns: pd.Series, var_estimates: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([returns, var_estimates], axis=1, join="inner").dropna()
    aligned.columns = ["returns", "var"]

    return aligned


def get_kupiec_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    alpha: float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Kupiec's Proportion of Failures (POF) test for a Value at Risk model.

    The test compares the observed breach rate (the fraction of periods in which the
    realized return was worse than the estimated VaR) against the breach rate implied
    by `alpha`, via a likelihood-ratio statistic that is asymptotically chi-squared
    distributed with 1 degree of freedom under the null hypothesis that the VaR model
    is correctly calibrated:

    - LR_POF = -2 * ln[(1 - alpha)^(n - x) * alpha^x] + 2 * ln[(1 - x/n)^(n - x) * (x/n)^x]

    Where `x` is the number of breaches and `n` is the number of observations.

    A significant result (low p-value) indicates that the VaR model's breach rate is
    inconsistent with its stated confidence level, i.e. it is mis-calibrated.

    For more information about the method, see the following paper:

    - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement
    Models." The Journal of Derivatives, 3(2), 73-84.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at
        Risk estimates, aligned to the same index as `returns`.
        alpha (float): The confidence level the VaR estimates were built with
        (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: The Kupiec statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_kupiec_test(returns[column], var_estimates[column], alpha)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        aligned = _align(returns, var_estimates)
        breaches = int((aligned["returns"] < aligned["var"]).sum())
        n = len(aligned)

        if n == 0:
            return pd.Series(
                {"Kupiec Statistic": float("nan"), "P-Value": float("nan")}
            )

        observed_rate = breaches / n

        log_likelihood_null = special.xlogy(n - breaches, 1 - alpha) + special.xlogy(
            breaches, alpha
        )
        log_likelihood_observed = special.xlogy(
            n - breaches, 1 - observed_rate
        ) + special.xlogy(breaches, observed_rate)

        lr_statistic = -2 * (log_likelihood_null - log_likelihood_observed)
        p_value = stats.chi2.sf(lr_statistic, 1)

        return pd.Series({"Kupiec Statistic": lr_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_christoffersen_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Christoffersen's independence test for a Value at Risk model.

    Kupiec's test checks whether the overall breach rate matches the expected rate,
    but says nothing about whether breaches cluster together in time (e.g. during a
    volatility spike) rather than occurring independently. This test builds a
    first-order Markov chain of hit/no-hit outcomes and tests, via a likelihood-ratio
    statistic asymptotically chi-squared distributed with 1 degree of freedom, whether
    the probability of a breach depends on whether the previous period also breached.

    A significant result (low p-value) indicates that breaches are not independent
    over time, i.e. the VaR model under-reacts to changing volatility.

    For more information about the method, see the following paper:

    - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International
    Economic Review, 39(4), 841-862.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at
        Risk estimates, aligned to the same index as `returns`.

    Returns:
        pd.Series | pd.DataFrame: The Christoffersen statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_christoffersen_test(returns[column], var_estimates[column])
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        aligned = _align(returns, var_estimates)
        breach = (aligned["returns"] < aligned["var"]).astype(int).to_numpy()

        if len(breach) < MINIMUM_OBSERVATIONS_FOR_TRANSITIONS:
            return pd.Series(
                {"Christoffersen Statistic": float("nan"), "P-Value": float("nan")}
            )

        previous, current = breach[:-1], breach[1:]

        n00 = int(((previous == 0) & (current == 0)).sum())
        n01 = int(((previous == 0) & (current == 1)).sum())
        n10 = int(((previous == 1) & (current == 0)).sum())
        n11 = int(((previous == 1) & (current == 1)).sum())

        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        log_likelihood_restricted = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(
            n01 + n11, pi
        )
        log_likelihood_unrestricted = (
            special.xlogy(n00, 1 - pi01)
            + special.xlogy(n01, pi01)
            + special.xlogy(n10, 1 - pi11)
            + special.xlogy(n11, pi11)
        )

        lr_statistic = -2 * (log_likelihood_restricted - log_likelihood_unrestricted)
        p_value = stats.chi2.sf(lr_statistic, 1)

        return pd.Series({"Christoffersen Statistic": lr_statistic, "P-Value": p_value})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
