"""Conditional Value at Risk Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from financetoolkit.risk.value_at_risk import cvar_model, var_model


def test_get_cvar_historic(recorder):
    recorder.capture(
        cvar_model.get_cvar_historic(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_gaussian(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_gaussian(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_studentt(recorder):
    recorder.capture(
        cvar_model.get_cvar_studentt(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_laplace(recorder):
    recorder.capture(
        cvar_model.get_cvar_laplace(
            returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
        )
    )


def test_get_cvar_historic_rolling(recorder):
    returns = pd.Series([0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.15, -0.05, 0.08, 0.12])
    recorder.capture(
        cvar_model.get_cvar_historic(returns=returns, alpha=0.05, window=3).round(4)
    )


def test_get_cvar_historic_rolling_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [0.3, 0.2, 0.1, 0, 0.06, -0.1, -0.05],
            "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04, -0.08, -0.03],
        }
    )
    recorder.capture(
        cvar_model.get_cvar_historic(returns=returns_df, alpha=0.05, window=3).round(4)
    )


def test_get_cvar_logistic(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_logistic(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_cornish_fisher(recorder):
    recorder.capture(
        round(
            cvar_model.get_cvar_cornish_fisher(
                returns=pd.Series([0.3, 0.2, 0.1, 0, 0.06]), alpha=0.05
            ),
            4,
        )
    )


def test_get_cvar_cornish_fisher_dataframe(recorder):
    returns_df = pd.DataFrame(
        {"AAPL": [0.3, 0.2, 0.1, 0, 0.06], "MSFT": [0.25, 0.15, 0.08, -0.02, 0.04]}
    )
    recorder.capture(
        cvar_model.get_cvar_cornish_fisher(returns=returns_df, alpha=0.05).round(4)
    )


def test_get_cvar_cornish_fisher_more_extreme_than_var(recorder):
    # CVaR should always be at least as extreme (more negative) than VaR at the same alpha.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(2000) * 0.01)

    var_cf = var_model.get_var_cornish_fisher(returns=returns, alpha=0.05)
    cvar_cf = cvar_model.get_cvar_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(bool(cvar_cf <= var_cf))


def test_get_cvar_cornish_fisher_more_extreme_for_skewed_fat_tailed_data(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(
        np.concatenate(
            [
                rng.standard_normal(1900) * 0.005,
                -np.abs(rng.standard_normal(100)) * 0.05,
            ]
        )
    )

    gaussian = cvar_model.get_cvar_gaussian(returns=returns, alpha=0.05)
    cornish_fisher = cvar_model.get_cvar_cornish_fisher(returns=returns, alpha=0.05)

    recorder.capture(bool(cornish_fisher < gaussian))


def test_get_cvar_cornish_fisher_multiindex(recorder):
    periods = ["2023Q1", "2023Q2"]
    dates = pd.date_range("2023-01-01", periods=10, freq="D")

    multi_index = pd.MultiIndex.from_product(
        [periods[:1], dates[:5]], names=["Period", "Date"]
    )
    returns_multi = pd.DataFrame(
        {"AAPL": np.random.default_rng(1).normal(0.001, 0.02, 5)}, index=multi_index
    )

    result = cvar_model.get_cvar_cornish_fisher(returns=returns_multi, alpha=0.05)
    recorder.capture(result.shape)


def test_get_cvar_evt(recorder):
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.15, 0.01, -0.02, 0.04, -0.2, 0.02, -0.01, -0.18, 0.03]
    )
    recorder.capture(
        round(
            cvar_model.get_cvar_evt(
                returns=returns, alpha=0.05, threshold_percentile=0.7
            ),
            4,
        )
    )


def test_get_cvar_evt_dataframe(recorder):
    returns_df = pd.DataFrame(
        {
            "AAPL": [
                0.02,
                -0.01,
                0.03,
                -0.15,
                0.01,
                -0.02,
                0.04,
                -0.2,
                0.02,
                -0.01,
                -0.18,
                0.03,
            ],
            "MSFT": [
                0.01,
                -0.02,
                0.02,
                -0.12,
                0.02,
                -0.01,
                0.03,
                -0.17,
                0.01,
                -0.02,
                -0.14,
                0.02,
            ],
        }
    )
    recorder.capture(
        cvar_model.get_cvar_evt(
            returns=returns_df, alpha=0.05, threshold_percentile=0.7
        ).round(4)
    )


def test_get_cvar_evt_at_least_as_extreme_as_var_evt(recorder):
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_t(3, 2000) * 0.01)

    var_evt = var_model.get_var_evt(
        returns=returns, alpha=0.05, threshold_percentile=0.9
    )
    cvar_evt = cvar_model.get_cvar_evt(
        returns=returns, alpha=0.05, threshold_percentile=0.9
    )

    recorder.capture(bool((cvar_evt <= var_evt).all()))


def test_get_cvar_evt_insufficient_exceedances(recorder):
    # Too few exceedances to fit a GPD, so this returns NaN rather than raising.
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
    recorder.capture(
        cvar_model.get_cvar_evt(returns=returns, alpha=0.05, threshold_percentile=0.95)
    )


def test_get_cvar_gaussian_uses_sample_standard_deviation():
    returns = pd.Series([0.03, -0.02, 0.01, -0.04, 0.02, 0.005, -0.01])
    z = stats.norm.ppf(0.05)
    expected = returns.mean() - returns.std(ddof=1) * stats.norm.pdf(z) / 0.05

    assert np.isclose(cvar_model.get_cvar_gaussian(returns, 0.05), expected)


def test_get_cvar_studentt_matches_the_fitted_distribution():
    rng = np.random.default_rng(1)
    returns = pd.Series(stats.t.rvs(4, size=3000, random_state=rng) * 0.01 + 0.0005)
    dof, location, scale = stats.t.fit(returns)

    # The Expected Shortfall is the mean of the fitted quantile function over the tail.
    expected = (
        integrate.quad(lambda p: location + scale * stats.t.ppf(p, dof), 0, 0.05)[0]
        / 0.05
    )

    assert np.isclose(
        cvar_model.get_cvar_studentt(returns, 0.05).iloc[0], expected, rtol=1e-6
    )


def test_get_cvar_studentt_without_finite_mean_is_nan():
    returns = pd.Series(stats.t.rvs(0.7, size=2000, random_state=3) * 0.01)

    assert stats.t.fit(returns)[0] <= 1
    assert np.isnan(cvar_model.get_cvar_studentt(returns, 0.05).iloc[0])


def test_get_cvar_cornish_fisher_matches_the_integral():
    rng = np.random.default_rng(1)
    returns = pd.Series(stats.t.rvs(6, size=3000, random_state=rng) * 0.01)
    mean, std = returns.mean(), returns.std(ddof=1)
    skewness, kurtosis = returns.skew(), returns.kurt()

    def quantile(p):
        z = stats.norm.ppf(p)
        z_cf = (
            z
            + (z**2 - 1) * skewness / 6
            + (z**3 - 3 * z) * kurtosis / 24
            - (2 * z**3 - 5 * z) * skewness**2 / 36
        )
        return mean + z_cf * std

    expected = integrate.quad(quantile, 0, 0.05, limit=200)[0] / 0.05

    assert np.isclose(
        cvar_model.get_cvar_cornish_fisher(returns, 0.05), expected, rtol=1e-3
    )


def test_get_cvar_evt_alpha_outside_fitted_tail_raises():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500))

    with pytest.raises(ValueError, match="outside the tail"):
        cvar_model.get_cvar_evt(returns, alpha=0.2, threshold_percentile=0.9)


def test_rolling_cvar_historic_matches_the_cvar_of_every_window():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.standard_t(5, size=(300, 3)) * 0.01, columns=["A", "B", "C"]
    ).round(4)
    returns.iloc[:40, 1] = np.nan
    returns.iloc[150, 2] = np.nan

    def window_cvar(window):
        value_at_risk = np.percentile(window, 5)
        return window[window <= value_at_risk].mean()

    # A missing return is a day the ticker didn't trade: each window holds its own last
    # 30 returns, as for the ticker on its own.
    expected = pd.DataFrame(
        {
            column: returns[column].dropna().rolling(30).apply(window_cvar, raw=True)
            for column in returns.columns
        }
    ).reindex(returns.index)
    rolling_cvar = cvar_model.get_cvar_historic(returns, 0.05, window=30)

    pd.testing.assert_frame_equal(rolling_cvar, expected, rtol=1e-12, atol=1e-15)
    pd.testing.assert_series_equal(
        cvar_model.get_cvar_historic(returns["A"], 0.05, window=30),
        expected["A"],
        rtol=1e-12,
        atol=1e-15,
    )
    assert (
        cvar_model.get_cvar_historic(returns.iloc[:10], 0.05, window=30)
        .isna()
        .all()
        .all()
    )


def test_out_of_sample_gaussian_estimates_match_a_fit_per_window():
    from financetoolkit.risk.value_at_risk import value_at_risk_controller

    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(200, 2)), columns=["A", "B"])
    returns.iloc[:30, 1] = np.nan

    for measure, estimator in [
        ("var", var_model.get_var_gaussian),
        ("cvar", cvar_model.get_cvar_gaussian),
    ]:
        estimates = value_at_risk_controller._get_out_of_sample_estimates(
            returns, measure, "gaussian", 0.05, 50
        )
        expected = pd.DataFrame(
            {
                column: returns[column]
                .dropna()
                .rolling(50)
                .apply(lambda window: estimator(window, 0.05), raw=False)
                .shift(1)
                .reindex(returns.index)
                for column in returns
            }
        )

        pd.testing.assert_frame_equal(estimates, expected, rtol=1e-12)


def test_get_cvar_historic_of_a_period_with_a_single_return_is_nan():
    index = pd.period_range("2021-12-31", periods=6, freq="D")
    returns = pd.DataFrame({"A": [-0.02, 0.01, -0.03, 0.02, 0.0, -0.01]}, index=index)
    within = returns.set_axis(pd.MultiIndex.from_arrays([index.asfreq("Y"), index]))

    result = cvar_model.get_cvar_historic(within, 0.2)

    assert np.isnan(result.loc[pd.Period("2021", "Y"), "A"])
    assert result.loc[pd.Period("2022", "Y"), "A"] == pytest.approx(-0.03)


# Forty returns of a Student-t(4) draw; the kernel reference is the integral of the return
# times scipy's gaussian_kde(bw_method="silverman") density below its 5% quantile, over 0.05.
KERNEL_REFERENCE_RETURNS = pd.Series(
    [0.0004, -0.0137, 0.0009, -0.0064, 0.0025, 0.0177, -0.0455, -0.0052, 0.0026, -0.0068,
     -0.0241, -0.007, 0.0001, -0.001, -0.0152, -0.0147, -0.0041, -0.0149, -0.0015, 0.0059,
     0.0036, -0.0195, 0.0084, -0.0079, -0.0059, 0.0107, -0.0016, -0.0031, -0.0039, -0.0035,
     0.0002, 0.0143, -0.0145, -0.0299, 0.0015, -0.0072, -0.0021, 0.0113, 0.0011, -0.0077]
)  # fmt: skip


def test_get_cvar_kernel_reference_value():
    assert np.isclose(
        cvar_model.get_cvar_kernel(KERNEL_REFERENCE_RETURNS, 0.05),
        -0.039120756350325525,
        atol=1e-12,
    )
    assert cvar_model.get_cvar_kernel(
        KERNEL_REFERENCE_RETURNS, 0.05
    ) < var_model.get_var_kernel(KERNEL_REFERENCE_RETURNS, 0.05)


def test_get_weighted_cvar():
    frame = pd.DataFrame({"A": KERNEL_REFERENCE_RETURNS})
    first = cvar_model.get_cvar_historic(frame, 0.05)
    second = cvar_model.get_cvar_historic(frame, 0.2)
    weighted = cvar_model.get_weighted_cvar([first, second], [0.25, 0.75])
    assert np.isclose(weighted["A"], 0.25 * first["A"] + 0.75 * second["A"])

    with pytest.raises(ValueError):
        cvar_model.get_weighted_cvar([first, second], [0.5, 0.6])
    with pytest.raises(ValueError):
        cvar_model.get_weighted_cvar([first, second], [1.0])


def _higher_moment_objective(losses, order, alpha):
    return (
        lambda threshold: threshold
        + np.mean(np.maximum(losses - threshold, 0) ** order) ** (1 / order) / alpha
    )


@pytest.mark.parametrize(
    ("size", "order", "alpha"),
    [(1000, 2, 0.05), (250, 2, 0.05), (300, 3, 0.1), (500, 1.5, 0.01), (15, 2, 0.05)],
)
def test_higher_moment_coherent_risk_is_the_minimum_of_krokhmal(size, order, alpha):
    # Krokhmal (2007): min over ζ of ζ + (1 / alpha) * mean(max(L - ζ, 0)^p)^(1/p), minimized
    # independently with scipy's bounded Brent search and checked on a dense grid.
    from scipy.optimize import minimize_scalar

    returns = np.random.default_rng(1).standard_t(4, size=size) * 0.01
    losses = -returns
    objective = _higher_moment_objective(losses, order, alpha)
    reference = minimize_scalar(
        objective,
        bounds=(losses.min() - 1, losses.max()),
        method="bounded",
        options={"xatol": 1e-14},
    ).fun
    grid = min(
        objective(z) for z in np.linspace(losses.min() - 0.2, losses.max(), 4001)
    )

    result = -cvar_model.get_higher_moment_coherent_risk(
        pd.Series(returns), order, alpha
    )

    assert result <= min(reference, grid) + 1e-15
    assert result == pytest.approx(reference, abs=1e-8)


def test_higher_moment_coherent_risk_of_order_one_is_the_exact_cvar():
    # Rockafellar and Uryasev (2002): the average of the n * alpha largest losses, the one at
    # the boundary counted with its fraction (1000 * 0.05 = 50 exactly, 37 * 0.05 = 1.85).
    rng = np.random.default_rng(2)

    for size in (1000, 37):
        returns = rng.normal(0, 0.01, size)
        losses = np.sort(-returns)[::-1]
        tail = size * 0.05
        whole = int(tail)
        expected = (losses[:whole].sum() + (tail - whole) * losses[whole]) / tail

        assert -cvar_model.get_higher_moment_coherent_risk(
            pd.Series(returns), 1, 0.05
        ) == pytest.approx(expected, abs=1e-14)


def test_higher_moment_coherent_risk_rises_with_the_order_and_not_with_the_sample():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_t(4, 2000) * 0.01)
    by_order = [
        cvar_model.get_higher_moment_coherent_risk(returns, order, 0.05)
        for order in (1, 1.5, 2)
    ]

    assert by_order[0] > by_order[1] > by_order[2] > returns.min()
    # With n^(-1 / p) >= alpha the minimum is at the worst loss: 2000^(-1/3) = 0.079 for p = 3.
    assert cvar_model.get_higher_moment_coherent_risk(
        returns, 3, 0.05
    ) == pytest.approx(returns.min(), abs=1e-15)
    assert cvar_model.get_higher_moment_coherent_risk(returns, 3, 0.1) > returns.min()
    # The moment is a mean, so the same returns repeated give the same measure (with a norm
    # without 1 / √n it would grow towards the worst loss).
    assert cvar_model.get_higher_moment_coherent_risk(
        pd.concat([returns] * 3), 2, 0.05
    ) == pytest.approx(by_order[2], abs=1e-12)


def test_higher_moment_coherent_risk_per_period_and_rolling():
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (60, 2)),
        columns=["A", "B"],
        index=pd.date_range("2024-01-01", periods=60, freq="D"),
    )
    rolling = cvar_model.get_higher_moment_coherent_risk(returns, window_size=20)

    assert rolling.iloc[:19].isna().all().all()
    assert rolling.iloc[-1]["A"] == pytest.approx(
        cvar_model.get_higher_moment_coherent_risk(returns["A"].iloc[-20:]), abs=1e-14
    )

    with pytest.raises(ValueError, match="order"):
        cvar_model.get_higher_moment_coherent_risk(returns, order=0.5)


def test_expected_quadratic_shortfall_is_the_cvar_of_the_squared_losses():
    # The Rockafellar and Uryasev form minimized by scipy's linprog: min ζ + Σ u / (n alpha)
    # with u >= S - ζ and u >= 0, S the squared losses.
    from scipy.optimize import linprog

    rng = np.random.default_rng(5)

    for size, alpha in [(250, 0.05), (37, 0.05), (500, 0.01)]:
        returns = rng.standard_t(4, size) * 0.01
        squared = np.maximum(-returns, 0) ** 2
        cost = np.concatenate([[1.0], np.full(size, 1 / (size * alpha))])
        constraints = np.hstack([-np.ones((size, 1)), -np.eye(size)])
        reference = linprog(
            cost,
            A_ub=constraints,
            b_ub=-squared,
            bounds=[(None, None)] + [(0, None)] * size,
            method="highs",
        ).fun

        assert cvar_model.get_expected_quadratic_shortfall(
            pd.Series(returns), alpha
        ) == pytest.approx(reference, rel=1e-9)


def test_expected_quadratic_shortfall_ignores_gains():
    losses = pd.Series([-0.03, -0.01, 0.0, 0.02, 0.05] * 20)

    # alpha * n = 20: the twenty losses of 3% exactly.
    assert cvar_model.get_expected_quadratic_shortfall(losses, 0.2) == pytest.approx(
        0.0009, abs=1e-15
    )
    assert cvar_model.get_expected_quadratic_shortfall(
        pd.Series([0.01, 0.02, 0.03] * 10), 0.05
    ) == pytest.approx(0)
