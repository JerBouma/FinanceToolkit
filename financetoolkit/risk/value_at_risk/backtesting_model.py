"""VaR Backtesting Model"""

__docformat__ = "google"


import numpy as np
import pandas as pd
from scipy import special, stats

MINIMUM_OBSERVATIONS_FOR_TRANSITIONS = 2
MINIMUM_OBSERVATIONS_FOR_ES_BACKTEST = 5

# The Weibull likelihood of the duration test needs two durations, one of which is not censored.
MINIMUM_DURATIONS = 2

# The t-statistic of the exceedance residuals needs their standard deviation (ddof=1).
MINIMUM_BREACHES_FOR_RESIDUAL_TEST = 2

# The search interval of the Weibull shape, as in rugarch's VaRDurTest.
WEIBULL_SHAPE_BOUNDS = (0.001, 10.0)

# The random orderings of the breaches behind the Monte Carlo p-value of the duration test:
# 999 make (R + 1) * 5% a whole number, so the test at 5% has exactly its size (Dufour, 2006).
DURATION_SIMULATIONS = 999
GOLDEN_SECTION_STEPS = 40


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
    Calculate Kupiec's Proportion of Failures (POF) test: whether a Value at Risk model is
    breached as often as its tail probability says it should be.

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

    Shares similar logic:
    - rugarch (R), VaRTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L418

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at
        Risk estimates, aligned to the same index as `returns`.
        alpha (float): The tail probability the VaR estimates were built with
        (e.g., 0.05 for the 95% confidence level).

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
        if not isinstance(var_estimates, pd.Series):
            raise TypeError("A Series of returns needs a Series of VaR estimates.")
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
    Calculate Christoffersen's independence test: whether the breaches of a Value at Risk
    model cluster in time rather than arriving independently.

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

    Shares similar logic:
    - rugarch (R), VaRTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L418

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
        if not isinstance(var_estimates, pd.Series):
            raise TypeError("A Series of returns needs a Series of VaR estimates.")
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
    seed: int = 0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the Acerbi-Szekely (2014) "Z2" Expected Shortfall backtest: whether the losses
    on the days a Value at Risk is breached are as severe as the Conditional Value at Risk
    predicted.

    Kupiec's and Christoffersen's tests above only backtest the VaR estimate itself
    -- they check how often (and how independently) the VaR threshold is breached,
    but say nothing about the *severity* of the losses on those breach days, which is
    exactly the extra information a Conditional VaR (Expected Shortfall) estimate is
    supposed to add over VaR. The Z2 statistic fills that gap by comparing the actual
    loss on each breach day to the CVaR estimate that was supposed to describe the
    average loss on such days:

    - Z2 = (1 / (n * alpha)) * SUM_t[ r_t * 1{r_t < VaR_t} / CVaR_t ] - 1

    Since `CVaR_t = E[r_t | r_t < VaR_t]` by construction and breaches occur with
    probability `alpha`, `Z2 = 0` in expectation under a correctly calibrated model.
    A breach is a return strictly below the VaR, as in the paper (X_t + VaR_t < 0) and
    as in `get_kupiec_test` and `get_christoffersen_test`.
    `Z2 > 0` indicates the model *underestimates* tail risk (realized losses on breach
    days are, on average, worse than the CVaR predicted -- i.e. `abs(r_t) > abs(CVaR_t)`
    on breach days), while `Z2 < 0` indicates the model is conservative.

    Note that the original paper writes the statistic as `SUM[...] / (n * alpha) + 1`
    with the Expected Shortfall quoted as a positive loss; this module keeps CVaR on
    the same negative-return scale as every other measure here, which flips the sign
    of the constant to `- 1` and, with it, the sign of the statistic relative to the
    paper. The magnitude and the p-value are unaffected.

    The Z2 statistic has no closed-form null distribution. The paper obtains its
    p-value by simulating returns from the model's own predictive distributions and
    recomputing Z2 on each simulated path. That needs the full forecast distribution of
    every day, which a rolling VaR/CVaR path does not carry, so this function uses an
    approximation instead: `n_bootstrap` paired resamples of the observed
    `(return, VaR, CVaR)` triples are drawn with replacement (i.i.d. bootstrap), Z2 is
    recomputed on each, and the bootstrap Standard Error studentizes the observed Z2
    into a z-score that is compared with the standard normal distribution.

    The test is one-sided, as in the paper: only an underestimation of tail risk
    rejects the model. On this module's sign convention that is a positive Z2, so the
    p-value is P(Z > z-score); a conservative model (Z2 < 0) gives a p-value above 0.5.
    A CVaR of 0 on any day (no loss to scale the breach by) gives a NaN statistic,
    Standard Error and p-value rather than an infinite one.

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
        alpha (float): The tail probability the VaR/CVaR estimates were built with
        (e.g., 0.05 for the 95% confidence level).
        n_bootstrap (int, optional): The number of bootstrap resamples used to estimate
        the Standard Error of Z2. Defaults to 1000.
        seed (int, optional): The seed for the bootstrap random number
        generator, for reproducibility. Defaults to 0.

    Returns:
        pd.Series | pd.DataFrame: The Z2 statistic, its bootstrap Standard Error, its
        (studentized, one-sided) p-value, and the number of breaches observed.
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
                    seed,
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

        breach = (aligned["returns"] < aligned["var"]).to_numpy()
        r = aligned["returns"].to_numpy()
        # A CVaR of 0 (an asset whose price did not move, or a short estimation window whose
        # tail holds only flat days) scales no loss, so it gives NaN rather than an infinite
        # statistic.
        cvar = aligned["cvar"].replace(0, np.nan).to_numpy()

        def _z2(
            breach_mask: np.ndarray, r_values: np.ndarray, cvar_values: np.ndarray
        ) -> float:
            return float(np.sum(r_values * breach_mask / cvar_values) / (n * alpha) - 1)

        rng = np.random.default_rng(seed)
        bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))

        # Without a CVaR there is no statistic; numpy would warn about it once for every
        # resample.
        with np.errstate(divide="ignore", invalid="ignore"):
            z2_statistic = _z2(breach, r, cvar)
            bootstrap_z2 = np.array(
                [_z2(breach[idx], r[idx], cvar[idx]) for idx in bootstrap_indices]
            )
            standard_error = float(np.std(bootstrap_z2, ddof=1))

        if not np.isfinite(z2_statistic) or not np.isfinite(standard_error):
            z2_statistic, standard_error, p_value = np.nan, np.nan, np.nan
        elif standard_error == 0:
            p_value = np.nan if z2_statistic == 0 else float(z2_statistic < 0)
        else:
            # One-sided: only Z2 > 0 (tail risk underestimated) counts against the model.
            z_score = z2_statistic / standard_error
            p_value = stats.norm.sf(z_score)

        return pd.Series(
            {
                "Acerbi-Szekely Statistic": z2_statistic,
                "Standard Error": standard_error,
                "P-Value": p_value,
                "Breaches": int(breach.sum()),
            }
        )


def get_conditional_coverage_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    alpha: float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Christoffersen's conditional coverage test: whether a Value at Risk model is
    breached as often as it should be and with breaches that arrive independently.

    The test joins Kupiec's unconditional coverage test (is the breach rate alpha?) and
    Christoffersen's independence test (does a breach make the next one more likely?) into
    one likelihood ratio, which is asymptotically chi-squared distributed with 2 degrees of
    freedom under the null hypothesis that breaches occur independently with probability
    alpha:

    - LR_cc = LR_uc + LR_ind

    Where LR_uc is the statistic of `get_kupiec_test` on all n observations and LR_ind that
    of `get_christoffersen_test` on the n - 1 transitions of the breach sequence. A breach is
    a return strictly below the VaR.

    For more information about the method, see the following paper:

    - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts." International Economic
    Review, 39(4), 841-862.

    Shares similar logic:
    - rugarch (R), VaRTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L418

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at Risk
            estimates, aligned to the same index as `returns`.
        alpha (float): The tail probability the VaR estimates were built with (e.g., 0.05
            for the 95% confidence level).

    Returns:
        pd.Series | pd.DataFrame: The conditional coverage statistic and its p-value.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_conditional_coverage_test(
                    returns[column], var_estimates[column], alpha
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        statistic = (
            get_kupiec_test(returns, var_estimates, alpha)["Kupiec Statistic"]
            + get_christoffersen_test(returns, var_estimates)[
                "Christoffersen Statistic"
            ]
        )

        return pd.Series(
            {
                "Conditional Coverage Statistic": statistic,
                "P-Value": stats.chi2.sf(statistic, 2),
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def _duration_statistics(breaches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The likelihood ratio of the duration test and the fitted Weibull Shape for every row of
    breaches (True) and non-breaches, NaN for a row with fewer than two durations or none
    uncensored. The Shape is found by a golden-section search on all rows at once.
    """
    rows = []
    for breach in breaches:
        breach_times = np.flatnonzero(breach) + 1
        durations = np.diff(breach_times).astype(float)
        censored = np.zeros(len(durations))
        if len(breach_times) and not breach[0]:
            durations = np.concatenate(([breach_times[0]], durations))
            censored = np.concatenate(([1.0], censored))
        if len(breach_times) and not breach[-1]:
            durations = np.concatenate((durations, [len(breach) - breach_times[-1]]))
            censored = np.concatenate((censored, [1.0]))
        rows.append((durations, censored))

    count = max(1, *(len(durations) for durations, _ in rows))
    # Padded durations of 1 with a weight of 0 leave the sums unchanged.
    d = np.ones((len(rows), count))
    c = np.zeros((len(rows), count))
    weight = np.zeros((len(rows), count))
    for row, (durations, censored) in enumerate(rows):
        d[row, : len(durations)] = durations
        c[row, : len(censored)] = censored
        weight[row, : len(durations)] = 1.0

    uncensored = np.sum(weight * (1 - c), axis=1)
    valid = (np.sum(weight, axis=1) >= MINIMUM_DURATIONS) & (uncensored > 0)
    uncensored = np.where(valid, uncensored, 1.0)
    log_d = np.log(d)

    def log_likelihood(shape: np.ndarray) -> np.ndarray:
        # The scale that maximizes the likelihood for this shape (see `get_duration_test`)
        # gives ln (a * D)^b = b ln D + ln(uncensored / Σ D^b).
        shape = shape[:, None]
        log_powered = shape * log_d
        log_hazard = (
            log_powered
            + (
                np.log(uncensored)
                - np.log(
                    np.maximum(np.sum(weight * np.exp(log_powered), axis=1), 1e-300)
                )
            )[:, None]
        )
        # Uncensored durations add ln f(D) = ln b - ln D + ln (a * D)^b - (a * D)^b and censored
        # ones their log survival, -(a * D)^b.
        return np.sum(
            weight
            * ((1 - c) * (np.log(shape) - log_d + log_hazard) - np.exp(log_hazard)),
            axis=1,
        )

    # Golden-section search for the largest likelihood: the interval shrinks by a factor of
    # 0.618 per step, to below 1e-7 of the Shape after 40 steps, where the likelihood ratio is
    # flat to about 1e-14.
    ratio = (np.sqrt(5) - 1) / 2
    lower = np.full(len(rows), WEIBULL_SHAPE_BOUNDS[0])
    upper = np.full(len(rows), WEIBULL_SHAPE_BOUNDS[1])
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value, right_value = log_likelihood(left), log_likelihood(right)
    for _ in range(GOLDEN_SECTION_STEPS):
        move_up = left_value < right_value
        lower = np.where(move_up, left, lower)
        upper = np.where(move_up, upper, right)
        new_left = np.where(move_up, right, upper - ratio * (upper - lower))
        new_right = np.where(move_up, lower + ratio * (upper - lower), left)
        new_value = log_likelihood(np.where(move_up, new_right, new_left))
        left_value, right_value = (
            np.where(move_up, right_value, new_value),
            np.where(move_up, new_value, left_value),
        )
        left, right = new_left, new_right

    shape = (lower + upper) / 2
    statistic = np.maximum(
        2 * (log_likelihood(shape) - log_likelihood(np.ones(len(rows)))), 0.0
    )

    return np.where(valid, statistic, np.nan), np.where(valid, shape, np.nan)


def get_duration_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    simulations: int = DURATION_SIMULATIONS,
    seed: int = 0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate Christoffersen and Pelletier's duration test: whether the time between the
    breaches of a Value at Risk model is memoryless, or breaches come in clusters.

    Under a correct model, breaches arrive without memory: the number of periods between two
    breaches (the duration) is exponentially distributed, with a constant probability of a
    breach whatever the time since the last one. The test fits a Weibull distribution to the
    durations, which nests the exponential distribution at a Shape b of 1, and compares the
    two likelihoods; a Shape below 1 means breaches cluster (a breach is most likely soon
    after another one):

    - f(D) = a^b * b * D^(b - 1) * exp(-(a * D)^b) and S(D) = exp(-(a * D)^b)
    - ln L(a, b) = C_1 * ln S(D_1) + (1 - C_1) * ln f(D_1) + Σ_(i=2)^(N-1) ln f(D_i) + C_N * ln S(D_N)
      + (1 - C_N) * ln f(D_N)
    - LR = 2 * (ln L(a_hat, b_hat) - ln L(a_tilde, 1))
    - P-Value = (1 + #{LR_r >= LR}) / (R + 1), with LR_r the statistic of the r-th of R = `simulations`
      random orderings of the same breaches

    Where D_i = t_i - t_(i-1) are the periods between consecutive breaches at t_1 < ... < t_N.
    A sample that does not start with a breach adds the periods up to the first breach as a
    censored duration (C_1 = 1), and one that does not end with a breach the periods after
    the last breach (C_N = 1): a censored duration only says that the next breach came later,
    so it enters with its survival probability. For a given b, the likelihood is largest at
    a = (number of uncensored durations / Σ D_i^b)^(1 / b), so b is found by maximizing over
    a single parameter.

    The p-value is a Monte Carlo p-value (Dufour, 2006), as Christoffersen and Pelletier
    (2004) use, because the χ² distribution with 1 degree of freedom of the LR is reached
    slowly: with a correct 5% VaR over 1,000 periods it rejects at 5% about 9% of the time.
    Under the null the breaches are independent with a constant probability, so given their
    number every ordering of them is equally likely: the random orderings give the exact
    distribution of the LR under the null, whatever the breach probability.

    For more information about the method, see the following papers:

    - Christoffersen, P.F. and Pelletier, D. (2004). "Backtesting Value-at-Risk: A
    Duration-Based Approach." Journal of Financial Econometrics, 2(1), 84-108.
    - Dufour, J.-M. (2006). "Monte Carlo Tests with Nuisance Parameters: A General Approach
    to Finite-Sample Inference and Nonstandard Asymptotics." Journal of Econometrics,
    133(2), 443-477.

    Shares similar logic:
    - rugarch (R), VaRDurTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L469

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): A Series or Dataframe of Value at Risk
            estimates, aligned to the same index as `returns`.
        simulations (int, optional): The number of random orderings of the breaches.
            Defaults to 999.
        seed (int, optional): The seed of the random orderings. Defaults to 0.

    Returns:
        pd.Series | pd.DataFrame: The duration statistic, its p-value and the fitted Weibull
        Shape b; NaN with fewer than two durations.

    Notes:
    - rugarch's VaRDurTest defines the durations and their censoring in the same way and
      compares the LR with the χ² distribution instead of the Monte Carlo p-value.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_duration_test(
                    returns[column], var_estimates[column], simulations, seed
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        if not isinstance(var_estimates, pd.Series):
            raise TypeError("A Series of returns needs a Series of VaR estimates.")
        aligned = _align(returns, var_estimates)
        breach = (aligned["returns"] < aligned["var"]).to_numpy()
        generator = np.random.default_rng(seed)
        orderings = np.array(
            [generator.permutation(breach) for _ in range(simulations)]
        )
        statistics, shapes = _duration_statistics(
            np.vstack([breach[None, :], orderings.reshape(simulations, len(breach))])
        )
        lr_statistic, shape = float(statistics[0]), float(shapes[0])

        # An ordering with fewer than two durations (all breaches in a row at an end) has no
        # statistic and counts as one that is not as large.
        p_value = (
            (1 + np.sum(statistics[1:] >= lr_statistic)) / (simulations + 1)
            if np.isfinite(lr_statistic)
            else np.nan
        )

        return pd.Series(
            {
                "Duration Statistic": lr_statistic,
                "P-Value": p_value,
                "Weibull Shape": shape,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_mcneil_frey_test(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    cvar_estimates: pd.Series | pd.DataFrame,
    volatility_estimates: pd.Series | pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.Series | pd.DataFrame:
    """
    Calculate McNeil and Frey's exceedance residual test: whether an Expected Shortfall (CVaR)
    model understates the losses beyond its Value at Risk.

    On every breach day the gap between the return and the Expected Shortfall forecast,
    in units of the forecast volatility, is an exceedance residual. If the model describes
    the tail correctly, the residuals have a mean of zero; when it understates the losses
    beyond the VaR, they are positive:

    - e_t = (CVaR_t - Return_t) / Volatility_t, for every t with Return_t < VaR_t
    - Statistic = mean(e) / (std(e) / √k)
    - P-Value = (1 / B) * Σ_b 1{Statistic*_b >= Statistic}

    Where k is the number of breaches, std the sample standard deviation (with k - 1 in the
    denominator) and
    Statistic*_b the statistic of the b-th of B resamples (with replacement) of the residuals
    less their mean, which have a mean of zero as under the null hypothesis. The test is
    one-sided: only an understatement of the tail losses rejects the model. Returns, VaR and
    CVaR are negative for a loss, so a positive residual is a loss beyond the CVaR.

    For more information about the method, see the following papers:

    - McNeil, A.J. and Frey, R. (2000). "Estimation of Tail-Related Risk Measures for
    Heteroscedastic Financial Time Series: An Extreme Value Approach." Journal of Empirical
    Finance, 7(3-4), 271-300.
    - Efron, B. and Tibshirani, R.J. (1993). "An Introduction to the Bootstrap." Chapman and
    Hall, section 16.4.

    Shares similar logic:
    - rugarch (R), ESTest: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-tests.R#L440

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): The Value at Risk estimates, aligned to the
            same index as `returns`.
        cvar_estimates (pd.Series | pd.DataFrame): The Conditional Value at Risk (Expected
            Shortfall) estimates, aligned to the same index.
        volatility_estimates (pd.Series | pd.DataFrame): The volatility forecasts, aligned to
            the same index.
        n_bootstrap (int, optional): The number of bootstrap resamples. Defaults to 1000.
        seed (int, optional): The seed of the bootstrap. Defaults to 0.

    Returns:
        pd.Series | pd.DataFrame: The McNeil-Frey statistic, its bootstrap p-value and the
        number of breaches; NaN with fewer than two breaches.

    Notes:
    - rugarch's ESTest tests the unscaled gap CVaR_t - Return_t with a normal p-value and, in
      its bootstrap version, reports the average of the p-values of the resamples instead.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_mcneil_frey_test(
                    returns[column],
                    var_estimates[column],
                    cvar_estimates[column],
                    volatility_estimates[column],
                    n_bootstrap,
                    seed,
                )
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        aligned = pd.concat(
            [returns, var_estimates, cvar_estimates, volatility_estimates],
            axis=1,
            join="inner",
        ).dropna()
        aligned.columns = ["returns", "var", "cvar", "volatility"]
        breaches = aligned[aligned["returns"] < aligned["var"]]
        residuals = (
            (breaches["cvar"] - breaches["returns"]) / breaches["volatility"]
        ).to_numpy()
        k = len(residuals)

        def t_statistic(values: np.ndarray) -> np.ndarray:
            return (
                values.mean(axis=-1)
                / values.std(axis=-1, ddof=1)
                * np.sqrt(values.shape[-1])
            )

        if k < MINIMUM_BREACHES_FOR_RESIDUAL_TEST or np.ptp(residuals) == 0:
            return pd.Series(
                {"McNeil-Frey Statistic": np.nan, "P-Value": np.nan, "Breaches": k}
            )

        statistic = float(t_statistic(residuals))
        rng = np.random.default_rng(seed)
        centered = residuals - residuals.mean()
        resamples = centered[rng.integers(0, k, size=(n_bootstrap, k))]

        with np.errstate(divide="ignore", invalid="ignore"):
            bootstrap_statistics = t_statistic(resamples)

        # A resample of identical residuals has no statistic; it is left out of the p-value.
        bootstrap_statistics = bootstrap_statistics[np.isfinite(bootstrap_statistics)]

        return pd.Series(
            {
                "McNeil-Frey Statistic": statistic,
                "P-Value": float(np.mean(bootstrap_statistics >= statistic)),
                "Breaches": k,
            }
        )

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


def get_tick_loss(
    returns: pd.Series | pd.DataFrame,
    var_estimates: pd.Series | pd.DataFrame,
    alpha: float,
) -> pd.Series | pd.DataFrame:
    """
    Calculate the tick (quantile) loss of a Value at Risk model: how well its VaR estimates
    work as alpha quantiles of the returns, lower being better.

    The VaR at alpha is the value that minimizes the expected tick loss, so of two VaR
    models on the same returns the one with the lower loss is the better quantile forecast.
    Unlike the coverage tests it weighs how far each return falls from the VaR, so it ranks
    models rather than accepting or rejecting one:

    - Tick Loss = (1 / n) * Σ (alpha - 1{Return_t < VaR_t}) * (Return_t - VaR_t)

    Every term is non-negative: a breach costs (1 - alpha) times its size and every other
    period alpha times the distance to the VaR.

    For more information about the method, see the following papers:

    - Koenker, R. and Bassett, G. (1978). "Regression Quantiles." Econometrica, 46(1), 33-50.
    - González-Rivera, G., Lee, T.H. and Mishra, S. (2004). "Forecasting Volatility: A Reality
    Check Based on Option Pricing, Utility Function, Value-at-Risk, and Predictive
    Likelihood." International Journal of Forecasting, 20(4), 629-645.

    Shares similar logic:
    - rugarch (R), VaRloss: https://github.com/cran/rugarch/blob/1.5-6/R/rugarch-methods.R#L3839

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of realized returns.
        var_estimates (pd.Series | pd.DataFrame): The Value at Risk estimates, aligned to the
            same index as `returns`.
        alpha (float): The tail probability the VaR estimates were built with.

    Returns:
        pd.Series | pd.DataFrame: The tick loss.

    Notes:
    - rugarch's VaRloss smooths the breach indicator with a steep logistic function and reports
      the loss of every period in percentage points; the loss here is the exact one, in
      decimals, averaged.
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                column: get_tick_loss(returns[column], var_estimates[column], alpha)
                for column in returns.columns
            }
        )
    if isinstance(returns, pd.Series):
        if not isinstance(var_estimates, pd.Series):
            raise TypeError("A Series of returns needs a Series of VaR estimates.")
        aligned = _align(returns, var_estimates)
        gap = aligned["returns"] - aligned["var"]
        loss = (alpha - (gap < 0)) * gap

        return pd.Series({"Tick Loss": loss.mean() if len(loss) else np.nan})

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")
