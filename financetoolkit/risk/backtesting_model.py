"""VaR Backtesting Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy import special, stats

MINIMUM_OBSERVATIONS_FOR_TRANSITIONS = 2
MINIMUM_OBSERVATIONS_FOR_ES_BACKTEST = 5


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


def get_acerbi_szekely_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    cvar_estimates: pd.Series | pd.DataFrame,
    alpha: float,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Acerbi-Szekely (2014) "Z2" Expected Shortfall backtest.

    Kupiec's and Christoffersen's tests above only backtest the VaR estimate itself
    -- they check how often (and how independently) the VaR threshold is breached,
    but say nothing about the *severity* of the losses on those breach days, which is
    exactly the extra information a Conditional VaR (Expected Shortfall) estimate is
    supposed to add over VaR. The Z2 statistic fills that gap by comparing the actual
    loss on each breach day to the CVaR estimate that was supposed to describe the
    average loss on such days:

    - Z2 = (1 / (n * alpha)) * SUM_t[ r_t * 1{r_t <= VaR_t} / CVaR_t ] - 1

    Since `CVaR_t = E[r_t | r_t <= VaR_t]` by construction and breaches occur with
    probability `alpha`, `Z2 = 0` in expectation under a correctly calibrated model.
    `Z2 > 0` indicates the model *underestimates* tail risk (realized losses on breach
    days are, on average, worse than the CVaR predicted -- i.e. `abs(r_t) > abs(CVaR_t)`
    on breach days), while `Z2 < 0` indicates the model is conservative.

    Note that the original paper writes the statistic as `SUM[...] / (n * alpha) + 1`
    with the Expected Shortfall quoted as a positive loss; this module keeps CVaR on
    the same negative-return scale as every other measure here, which flips the sign
    of the constant to `- 1` and, with it, the sign of the statistic relative to the
    paper. The magnitude and the p-value are unaffected.

    The Z2 statistic has no closed-form null distribution (the indicator function
    makes it non-pivotal), so -- following the Monte Carlo approach recommended in the
    original paper -- the p-value here is obtained via a nonparametric (i.i.d.,
    paired) bootstrap: `n_bootstrap` resamples of the `(return, VaR, CVaR)` triples are
    drawn with replacement, Z2 is recomputed on each, and the resulting bootstrap
    Standard Error is used to studentize the observed Z2 into an (asymptotically
    normal) z-score.

    For more information about the method, see the following paper:

    - Acerbi, C., & Szekely, B. (2014). "Back-Testing Expected Shortfall." RISK
    Magazine, 27(11), 76-81.

    Also known as: Acerbi-Szekely test, ES backtest, Z2 test.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at
        Risk estimates, aligned to the same index as `returns`.
        cvar_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Conditional
        Value at Risk (Expected Shortfall) estimates, aligned to the same index as
        `returns`.
        alpha (float): The confidence level the VaR/CVaR estimates were built with
        (e.g., 0.05 for 95% confidence).
        n_bootstrap (int, optional): The number of bootstrap resamples used to estimate
        the Standard Error of Z2. Defaults to 1000.
        random_state (int, optional): The seed for the bootstrap random number
        generator, for reproducibility. Defaults to 42.

    Returns:
        pd.Series | pd.DataFrame: The Z2 statistic, its bootstrap Standard Error, its
        (studentized, two-sided) p-value, and the number of breaches observed.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_acerbi_szekely_test(
                    returns[column],
                    var_estimates[column],
                    cvar_estimates[column],
                    alpha,
                    n_bootstrap,
                    random_state,
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        aligned = pd.concat(
            [returns, var_estimates, cvar_estimates], axis=1, join="inner"
        ).dropna()
        aligned.columns = ["returns", "var", "cvar"]
        n = len(aligned)

        if n <= MINIMUM_OBSERVATIONS_FOR_ES_BACKTEST:
            return pd.Series(
                {
                    "Acerbi-Szekely Statistic": np.nan,
                    "Standard Error": np.nan,
                    "P-Value": np.nan,
                    "Breaches": np.nan,
                }
            )

        breach = (aligned["returns"] <= aligned["var"]).to_numpy()
        r = aligned["returns"].to_numpy()
        cvar = aligned["cvar"].to_numpy()

        def _z2(
            breach_mask: np.ndarray, r_values: np.ndarray, cvar_values: np.ndarray
        ) -> float:
            return float(np.sum(r_values * breach_mask / cvar_values) / (n * alpha) - 1)

        z2_statistic = _z2(breach, r, cvar)

        rng = np.random.default_rng(random_state)
        bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))
        bootstrap_z2 = np.array(
            [_z2(breach[idx], r[idx], cvar[idx]) for idx in bootstrap_indices]
        )
        standard_error = float(np.std(bootstrap_z2, ddof=1))

        if standard_error == 0:
            p_value = np.nan if z2_statistic == 0 else 0.0
        else:
            z_score = z2_statistic / standard_error
            p_value = 2 * stats.norm.sf(abs(z_score))

        return pd.Series(
            {
                "Acerbi-Szekely Statistic": z2_statistic,
                "Standard Error": standard_error,
                "P-Value": p_value,
                "Breaches": int(breach.sum()),
            }
        )
