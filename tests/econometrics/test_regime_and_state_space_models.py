"""Regime Switching, Structural Break, Time-Varying Beta, DCC-GARCH and Pairs Trading Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics import (
    pairs_trading_model,
    regime_model,
    state_space_model,
)

# pylint: disable=missing-function-docstring

INDEX = pd.period_range("2015-01-01", periods=1500, freq="D")


def test_markov_regime_switching_separates_calm_and_turbulent_periods():
    rng = np.random.default_rng(1)
    state = np.zeros(len(INDEX), dtype=int)
    for t in range(1, len(INDEX)):
        stay = 0.98 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    returns = np.where(
        state == 0,
        rng.normal(0.001, 0.008, len(INDEX)),
        rng.normal(-0.002, 0.025, len(INDEX)),
    )

    summary, probabilities = regime_model.get_markov_regime_switching(
        pd.Series(returns, INDEX)
    )

    assert summary["Regime 1 Volatility"] == pytest.approx(0.008, rel=0.15)
    assert summary["Regime 2 Volatility"] == pytest.approx(0.025, rel=0.15)
    assert summary["Regime 1 Expected Duration"] > summary["Regime 2 Expected Duration"]
    accuracy = ((probabilities["Regime 2"] > 0.5).to_numpy() == (state == 1)).mean()
    assert accuracy > 0.9  # noqa: PLR2004


def test_structural_breaks_find_the_mean_shifts():
    rng = np.random.default_rng(2)
    series = pd.Series(
        np.r_[
            rng.normal(0.02, 0.01, 500),
            rng.normal(0.05, 0.01, 600),
            rng.normal(0.03, 0.01, 400),
        ],
        INDEX,
    )

    segments = regime_model.get_structural_breaks(series)

    assert list(segments["Observations"]) == pytest.approx([500, 600, 400], abs=5)
    assert list(segments["Mean"]) == pytest.approx([0.02, 0.05, 0.03], abs=0.003)
    assert segments.loc[1, "Start"] == "2015-01-01"

    one_break = regime_model.get_structural_breaks(series, breaks=1)
    assert len(one_break) == 2  # noqa: PLR2004
    with pytest.raises(ValueError, match="at most"):
        regime_model.get_structural_breaks(series, breaks=10)


def test_time_varying_beta_follows_a_drifting_beta():
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, len(INDEX))
    beta = np.linspace(0.5, 1.5, len(INDEX))
    stock = 0.0002 + beta * market + rng.normal(0, 0.005, len(INDEX))

    result, variances = state_space_model.get_time_varying_beta(
        pd.Series(stock, INDEX), pd.Series(market, INDEX)
    )

    assert list(result.columns) == ["Alpha", "Beta", "Beta Standard Error"]
    # The filtered beta lags a steady drift a little; on average it follows it.
    assert result["Beta"].iloc[200:500].mean() == pytest.approx(
        beta[200:500].mean(), abs=0.1
    )
    assert result["Beta"].iloc[-300:].mean() == pytest.approx(
        beta[-300:].mean(), abs=0.1
    )
    assert variances["beta variance"] > 0


def test_time_varying_beta_stays_flat_for_a_constant_beta():
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, len(INDEX))
    stock = 0.0002 + 1.0 * market + rng.normal(0, 0.005, len(INDEX))

    result, variances = state_space_model.get_time_varying_beta(
        pd.Series(stock, INDEX), pd.Series(market, INDEX)
    )

    # The constant beta is the nested model with both coefficient variances at zero: the
    # maximum likelihood cannot be below it. Fitted on the raw daily scale, the optimizer
    # stops at its starting beta variance (1e-3), 19 below it, and the beta wanders.
    model = state_space_model._TimeVaryingRegression(stock, market)
    residuals = stock - np.polyval(np.polyfit(market, stock, 1), market)
    constant = model.loglike(np.array([residuals.var(ddof=2), 0.0, 0.0]))
    estimated = model.loglike(np.array(list(variances.values())))
    assert estimated >= constant - 0.01  # noqa: PLR2004
    assert variances["beta variance"] < 1e-6  # noqa: PLR2004

    beta = result["Beta"].iloc[100:]
    assert beta.std() < 0.03  # noqa: PLR2004
    assert beta.mean() == pytest.approx(1.0, abs=0.05)
    assert beta.iloc[-1] == pytest.approx(np.polyfit(market, stock, 1)[0], abs=0.01)


def test_time_varying_beta_is_unidentified_on_the_first_date():
    rng = np.random.default_rng(6)
    market = rng.normal(0, 0.01, 200)
    market[:3] = 0.0
    stock = 0.5 * market + rng.normal(0, 0.005, 200)
    index = INDEX[:200]

    result, _ = state_space_model.get_time_varying_beta(
        pd.Series(stock, index), pd.Series(market, index)
    )

    # A market return of zero says nothing about the beta, so the diffuse start lasts
    # until two returns with information have been seen (dates 0 to 3).
    assert result.iloc[:3].isna().all().all()
    assert result.iloc[3:].notna().all().all()
    with pytest.raises(ValueError, match="vary"):
        state_space_model.get_time_varying_beta(
            pd.Series(stock, index), pd.Series(np.zeros(200), index)
        )


def test_dcc_garch_picks_up_a_rise_in_correlation():
    rng = np.random.default_rng(4)
    shocks = rng.standard_normal((len(INDEX), 2))
    correlation = np.where(np.arange(len(INDEX)) < 750, 0.2, 0.8)
    returns = pd.DataFrame(
        {
            "A": shocks[:, 0] * 0.01,
            "B": (
                correlation * shocks[:, 0] + np.sqrt(1 - correlation**2) * shocks[:, 1]
            )
            * 0.01,
        },
        index=INDEX,
    )

    correlations, parameters = state_space_model.get_dcc_garch(returns)

    assert list(correlations.columns) == ["A / B"]
    assert correlations["A / B"].iloc[300:700].mean() == pytest.approx(0.2, abs=0.1)
    assert correlations["A / B"].iloc[-400:].mean() == pytest.approx(0.8, abs=0.1)
    assert (
        parameters["DCC Alpha"] >= 0
        and parameters["DCC Alpha"] + parameters["DCC Beta"] < 1
    )


def test_pairs_trading_of_a_cointegrated_pair():
    rng = np.random.default_rng(5)
    second = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(INDEX))))
    spread = np.zeros(len(INDEX))
    for t in range(1, len(INDEX)):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.01)
    first = np.exp(0.5 + 0.8 * np.log(second) + spread)

    series, summary = pairs_trading_model.get_pairs_trading(
        pd.Series(first, INDEX), pd.Series(second, INDEX)
    )

    assert summary["Hedge Ratio"] == pytest.approx(0.8, abs=0.02)
    assert summary["Half-Life"] == pytest.approx(-np.log(2) / np.log(0.95), rel=0.3)
    assert summary["Cointegration P-Value"] < 0.01  # noqa: PLR2004
    assert set(series["Signal"].unique()) <= {-1, 0, 1}
    assert summary["Number of Trades"] > 0


def test_signals_follow_the_z_score_rules():
    z_score = pd.Series([0.0, 2.5, 1.0, 0.3, -2.2, -1.0, 0.1])
    assert list(pairs_trading_model.get_signals(z_score)) == [0, -1, -1, 0, 1, 1, 0]
    assert pairs_trading_model.get_half_life(pd.Series(np.arange(50.0))) == np.inf


def test_signals_exit_when_the_z_score_jumps_across_the_band():
    # From 0.8 to -0.7 the z-score never lies within 0.5 of zero, but it crossed the exit
    # level, so the short closes; at -2.6 a long opens.
    jump = pd.Series([2.5, 0.8, -0.7, -2.6, -0.4])
    assert list(pairs_trading_model.get_signals(jump)) == [-1, -1, 0, 1, 0]

    # Straight from beyond one entry threshold to beyond the other, the position reverses.
    reversal = pd.Series([0.0, 2.5, -2.6, 0.2, np.nan, 3.0])
    assert list(pairs_trading_model.get_signals(reversal)) == [0, -1, 1, 0, 0, -1]

    # With an exit threshold of 0, positions close when the spread crosses its mean.
    mean_crossing = pd.Series([-2.5, -0.3, 0.1])
    assert list(pairs_trading_model.get_signals(mean_crossing, exit_=0.0)) == [1, 1, 0]


def test_structural_breaks_trim_with_the_integer_part():
    rng = np.random.default_rng(8)
    series = pd.Series(rng.normal(0, 1, 99), INDEX[:99])

    # [0.15 * 99] = 14 leaves room for 99 // 14 - 1 = 6 breaks; rounding up to 15 would
    # leave room for 5.
    segments = regime_model.get_structural_breaks(series, breaks=6)
    assert len(segments) == 7  # noqa: PLR2004
    with pytest.raises(ValueError, match="at least 14"):
        regime_model.get_structural_breaks(series, breaks=7)


def test_the_exit_threshold_lies_below_the_entry_threshold():
    with pytest.raises(ValueError, match="below the entry threshold"):
        pairs_trading_model.get_signals(pd.Series([0.0, 2.5]), entry=1.0, exit_=1.5)
