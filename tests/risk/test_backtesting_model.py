"""VaR Backtesting Model Tests"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import (
    optimize,
    stats as spstats,
)

from financetoolkit.risk.value_at_risk import backtesting_model

SIGNIFICANCE_LEVEL = 0.05


def test_get_kupiec_test(recorder):
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(500) * 0.02)
    var_estimates = pd.Series(np.percentile(returns, 5) * np.ones(500))
    recorder.capture(
        backtesting_model.get_kupiec_test(returns, var_estimates, alpha=0.05).round(4)
    )


def test_get_kupiec_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_estimates_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_kupiec_test(
            returns_df, var_estimates_df, alpha=0.05
        ).round(4)
    )


def test_get_kupiec_test_never_breached(recorder):
    returns = pd.Series(np.random.default_rng(2).standard_normal(300) * 0.01)
    var_estimates = pd.Series(-1.0 * np.ones(300))
    recorder.capture(
        backtesting_model.get_kupiec_test(returns, var_estimates, alpha=0.05).round(4)
    )


def test_get_christoffersen_test(recorder):
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_normal(500) * 0.02)
    var_estimates = pd.Series(np.percentile(returns, 5) * np.ones(500))
    recorder.capture(
        backtesting_model.get_christoffersen_test(returns, var_estimates).round(4)
    )


def test_get_christoffersen_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_estimates_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_christoffersen_test(returns_df, var_estimates_df).round(4)
    )


def test_get_acerbi_szekely_test_well_calibrated(recorder):
    rng = np.random.default_rng(9)
    sigma = 0.02
    alpha = 0.05
    returns = pd.Series(rng.standard_normal(1000) * sigma)
    za = spstats.norm.ppf(alpha)
    true_var = za * sigma
    true_cvar = -sigma * spstats.norm.pdf(za) / alpha
    result = backtesting_model.get_acerbi_szekely_test(
        returns,
        pd.Series(np.full(1000, true_var)),
        pd.Series(np.full(1000, true_cvar)),
        alpha=alpha,
        n_bootstrap=200,
        seed=1,
    )
    assert abs(result["Acerbi-Szekely Statistic"]) < 1
    assert result["P-Value"] > SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_acerbi_szekely_test_underestimated_risk(recorder):
    rng = np.random.default_rng(9)
    sigma = 0.02
    alpha = 0.05
    returns = pd.Series(rng.standard_normal(1000) * sigma)
    za = spstats.norm.ppf(alpha)
    true_var = za * sigma
    true_cvar = -sigma * spstats.norm.pdf(za) / alpha
    understated_cvar = true_cvar * 0.3
    result = backtesting_model.get_acerbi_szekely_test(
        returns,
        pd.Series(np.full(1000, true_var)),
        pd.Series(np.full(1000, understated_cvar)),
        alpha=alpha,
        n_bootstrap=200,
        seed=1,
    )
    assert result["Acerbi-Szekely Statistic"] > 0
    assert result["P-Value"] < SIGNIFICANCE_LEVEL
    recorder.capture(result.round(4))


def test_get_acerbi_szekely_test_dataframe(recorder):
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(500) * 0.02,
            "MSFT": rng.standard_normal(500) * 0.02,
        }
    )
    var_df = pd.DataFrame(
        {
            "AAPL": np.percentile(returns_df["AAPL"], 5) * np.ones(500),
            "MSFT": np.percentile(returns_df["MSFT"], 5) * np.ones(500),
        }
    )
    cvar_df = pd.DataFrame(
        {
            "AAPL": returns_df["AAPL"][returns_df["AAPL"] <= var_df["AAPL"]].mean()
            * np.ones(500),
            "MSFT": returns_df["MSFT"][returns_df["MSFT"] <= var_df["MSFT"]].mean()
            * np.ones(500),
        }
    )
    recorder.capture(
        backtesting_model.get_acerbi_szekely_test(
            returns_df, var_df, cvar_df, alpha=0.05, n_bootstrap=100, seed=1
        ).round(4)
    )


def test_get_acerbi_szekely_test_too_few_observations(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    var_estimates = pd.Series([-0.01, -0.01, -0.01])
    cvar_estimates = pd.Series([-0.02, -0.02, -0.02])
    recorder.capture(
        backtesting_model.get_acerbi_szekely_test(
            returns, var_estimates, cvar_estimates, alpha=0.05
        )
    )


def test_get_acerbi_szekely_test_breach_is_strictly_below_the_var():
    returns = pd.Series([-0.05, -0.02, 0.01, 0.02, -0.01, 0.03, 0.0, -0.03])
    var_estimates = pd.Series([-0.05] * 8)
    cvar_estimates = pd.Series([-0.06] * 8)

    result = backtesting_model.get_acerbi_szekely_test(
        returns, var_estimates, cvar_estimates, alpha=0.05, n_bootstrap=50
    )

    # A return equal to the VaR is no breach, as in the Kupiec and Christoffersen tests.
    assert result["Breaches"] == 0
    kupiec = backtesting_model.get_kupiec_test(returns, var_estimates, 0.05)
    assert (
        kupiec["Kupiec Statistic"]
        == backtesting_model.get_kupiec_test(returns, var_estimates - 1e-9, 0.05)[
            "Kupiec Statistic"
        ]
    )


def test_get_acerbi_szekely_test_is_one_sided():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    # A model that overstates the tail losses: breaches are rare and mild.
    var_estimates = pd.Series(-0.03, index=returns.index)
    cvar_estimates = pd.Series(-0.04, index=returns.index)

    result = backtesting_model.get_acerbi_szekely_test(
        returns, var_estimates, cvar_estimates, alpha=0.05
    )

    assert result["Acerbi-Szekely Statistic"] < 0
    assert result["P-Value"] > 0.5  # noqa: PLR2004


def _breach_sample(n=1000, seed=3):
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.standard_t(4, n) * 0.01)
    var_estimates = pd.Series(-0.016 * np.ones(n))
    return returns, var_estimates


def test_get_conditional_coverage_test_adds_kupiec_and_christoffersen(recorder):
    returns, var_estimates = _breach_sample()
    result = backtesting_model.get_conditional_coverage_test(
        returns, var_estimates, alpha=0.05
    )

    expected = (
        backtesting_model.get_kupiec_test(returns, var_estimates, 0.05).iloc[0]
        + backtesting_model.get_christoffersen_test(returns, var_estimates).iloc[0]
    )
    assert result["Conditional Coverage Statistic"] == pytest.approx(expected)
    assert result["P-Value"] == pytest.approx(spstats.chi2.sf(expected, 2))
    recorder.capture(result.round(4))


def test_get_duration_test_matches_a_two_parameter_weibull_fit(recorder):
    returns, var_estimates = _breach_sample()
    result = backtesting_model.get_duration_test(returns, var_estimates)

    # The durations and their censoring written out, and both Weibull parameters fitted together.
    hits = np.flatnonzero((returns < var_estimates).to_numpy()) + 1
    durations = [hits[0], *np.diff(hits), len(returns) - hits[-1]]
    censored = np.array([1] + [0] * (len(durations) - 2) + [1])
    durations = np.array(durations, dtype=float)

    def negative_log_likelihood(log_parameters):
        scale, shape = np.exp(log_parameters)
        cumulative_hazard = (scale * durations) ** shape
        log_density = (
            shape * np.log(scale)
            + np.log(shape)
            + (shape - 1) * np.log(durations)
            - cumulative_hazard
        )
        return -np.sum(np.where(censored == 1, -cumulative_hazard, log_density))

    weibull = optimize.minimize(
        negative_log_likelihood,
        [np.log(0.05), 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 20000},
    )
    exponential = optimize.minimize_scalar(
        lambda log_scale: negative_log_likelihood([log_scale, 0.0]),
        bounds=(-10, 0),
        method="bounded",
        options={"xatol": 1e-12},
    )

    assert result["Duration Statistic"] == pytest.approx(
        2 * (exponential.fun - weibull.fun), rel=1e-6
    )
    assert result["Weibull Shape"] == pytest.approx(np.exp(weibull.x[1]), rel=1e-5)
    recorder.capture(result.round(4))


def test_get_duration_test_detects_clustered_breaches():
    # Breaches in bursts of five: short durations within a burst, long ones between them.
    returns = pd.Series(np.zeros(1000))
    for start in range(50, 1000, 200):
        returns.iloc[start : start + 5] = -1.0
    var_estimates = pd.Series(-0.5 * np.ones(1000))

    result = backtesting_model.get_duration_test(returns, var_estimates)

    assert result["Weibull Shape"] < 1
    assert result["P-Value"] < SIGNIFICANCE_LEVEL


def test_get_duration_test_needs_two_durations():
    returns = pd.Series(np.zeros(100))
    returns.iloc[50] = -1.0
    result = backtesting_model.get_duration_test(
        returns, pd.Series(-0.5 * np.ones(100))
    )
    # One breach gives two censored durations and none that ends in a breach on both sides.
    assert result.isna().all()


def test_get_mcneil_frey_test_statistic_and_direction(recorder):
    returns, var_estimates = _breach_sample()
    volatility = pd.Series(0.01 * np.ones(len(returns)))

    understated = backtesting_model.get_mcneil_frey_test(
        returns, var_estimates, pd.Series(-0.02 * np.ones(len(returns))), volatility
    )
    overstated = backtesting_model.get_mcneil_frey_test(
        returns, var_estimates, pd.Series(-0.04 * np.ones(len(returns))), volatility
    )

    breaches = returns[returns < var_estimates]
    residuals = (-0.02 - breaches) / 0.01
    assert understated["McNeil-Frey Statistic"] == pytest.approx(
        residuals.mean() / residuals.std(ddof=1) * np.sqrt(len(residuals))
    )
    assert understated["Breaches"] == len(breaches)
    # The test is one-sided: a CVaR that understates the losses is rejected, one that
    # overstates them is not.
    assert understated["P-Value"] < SIGNIFICANCE_LEVEL
    assert overstated["P-Value"] > 1 - SIGNIFICANCE_LEVEL
    recorder.capture(pd.concat([understated, overstated], axis=1).round(4))


def test_get_mcneil_frey_test_too_few_breaches():
    returns = pd.Series(np.zeros(100))
    returns.iloc[10] = -1.0
    ones = pd.Series(np.ones(100))
    result = backtesting_model.get_mcneil_frey_test(
        returns, -0.5 * ones, -0.8 * ones, ones
    )
    assert np.isnan(result["McNeil-Frey Statistic"])
    assert result["Breaches"] == 1


def test_get_tick_loss_matches_the_check_function():
    returns, var_estimates = _breach_sample()
    loss = backtesting_model.get_tick_loss(returns, var_estimates, alpha=0.05)

    expected = np.mean(
        [
            (0.05 - (value < var)) * (value - var)
            for value, var in zip(returns, var_estimates, strict=True)
        ]
    )
    assert loss["Tick Loss"] == pytest.approx(expected)

    # The true 5% quantile of the returns has a lower loss than a VaR far from it.
    true_quantile = pd.Series(np.quantile(returns, 0.05) * np.ones(len(returns)))
    far = pd.Series(-0.05 * np.ones(len(returns)))
    assert (
        backtesting_model.get_tick_loss(returns, true_quantile, 0.05)["Tick Loss"]
        < backtesting_model.get_tick_loss(returns, far, 0.05)["Tick Loss"]
    )


def test_new_backtests_dataframe(recorder):
    returns, var_estimates = _breach_sample()
    frame = pd.DataFrame({"A": returns, "B": returns.iloc[::-1].to_numpy()})
    var_frame = pd.DataFrame({"A": var_estimates, "B": var_estimates})
    recorder.capture(
        pd.concat(
            [
                backtesting_model.get_conditional_coverage_test(frame, var_frame, 0.05),
                backtesting_model.get_duration_test(frame, var_frame),
                backtesting_model.get_tick_loss(frame, var_frame, 0.05),
            ]
        ).round(4)
    )


def test_get_duration_test_has_a_monte_carlo_p_value():
    returns, var_estimates = _breach_sample()
    result = backtesting_model.get_duration_test(
        returns, var_estimates, simulations=199, seed=1
    )

    # The p-value counts the random orderings of the same breaches with a statistic at least
    # as large, so it is a multiple of 1 / (R + 1) and the same for the same seed.
    assert result["P-Value"] * 200 == pytest.approx(round(result["P-Value"] * 200))
    pd.testing.assert_series_equal(
        result,
        backtesting_model.get_duration_test(
            returns, var_estimates, simulations=199, seed=1
        ),
    )
    ordered = backtesting_model._duration_statistics(
        (returns < var_estimates).to_numpy()[None, :]
    )
    assert ordered[0][0] == pytest.approx(result["Duration Statistic"])


def test_get_duration_test_keeps_its_size_under_the_null():
    rng = np.random.default_rng(12)
    var_estimates = pd.Series(np.full(500, spstats.norm.ppf(0.05)))
    p_values = [
        backtesting_model.get_duration_test(
            pd.Series(rng.standard_normal(500)), var_estimates, simulations=99
        )["P-Value"]
        for _ in range(120)
    ]

    # The χ² distribution rejects about 9% of the time at 5%; the orderings give the exact
    # null distribution given the number of breaches (5% up to 2 pp of simulation error).
    assert np.mean(np.array(p_values) <= SIGNIFICANCE_LEVEL) < 0.1


def test_acerbi_szekely_test_without_a_cvar_does_not_warn_per_resample():
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300))
    zero = pd.Series(0.0, index=returns.index)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = backtesting_model.get_acerbi_szekely_test(
            returns, zero, zero, alpha=0.05, n_bootstrap=50
        )

    assert np.isnan(result["Standard Error"]) or not np.isfinite(
        result["Acerbi-Szekely Statistic"]
    )


def test_acerbi_szekely_test_with_a_cvar_of_zero_is_nan_rather_than_infinite():
    returns = pd.Series(np.random.default_rng(4).normal(0, 0.01, 300))
    var = pd.Series(-0.015, index=returns.index)
    cvar = pd.Series(-0.02, index=returns.index)
    # A breach on a day whose CVaR is 0 would divide a loss by 0.
    returns.iloc[10], cvar.iloc[10] = -0.05, 0.0

    result = backtesting_model.get_acerbi_szekely_test(
        returns, var, cvar, alpha=0.05, n_bootstrap=50
    )

    assert np.isnan(result["Acerbi-Szekely Statistic"])
    assert np.isnan(result["Standard Error"])
    assert np.isnan(result["P-Value"])
