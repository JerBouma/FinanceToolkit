"""Regime, Break, Change Point, Time-Varying Beta, DCC-GARCH, Pairs Trading and Trading Rule Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.econometrics.dynamics import (
    pairs_trading_model,
    regime_model,
    state_space_model,
)

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
    model = state_space_model._time_varying_regression(stock, market)
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


def test_markov_regime_switching_is_reproducible_and_keeps_the_random_state():
    rng = np.random.default_rng(7)
    returns = pd.Series(
        np.concatenate([rng.normal(0.001, 0.005, 150), rng.normal(-0.002, 0.02, 150)])
    )

    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    first, _ = regime_model.get_markov_regime_switching(returns)
    # The caller's random numbers are unaffected by the fit's search.
    assert np.random.random() == expected_next

    second, _ = regime_model.get_markov_regime_switching(returns)
    assert first == second


def test_dcc_garch_refuses_perfectly_correlated_series():
    series = np.random.default_rng(1).normal(0, 0.01, 300)
    returns = pd.DataFrame({"A": series, "B": series})

    with pytest.raises(ValueError, match="perfectly correlated"):
        state_space_model.get_dcc_garch(returns)


def _optimal_partitioning(data, cost, penalty, minimum_segment):
    """The exact segmentation without pruning (Jackson et al., 2005), same costs."""
    n = len(data)
    if cost == "mean":
        changes = np.diff(data)
        scale = (
            np.median(np.abs(changes - np.median(changes)))
            / regime_model.NORMAL_MEDIAN_ABSOLUTE_DEVIATION
            / np.sqrt(2)
        )
        data = data / scale
    elif cost == "variance":
        data = data - data.mean()
    floor = regime_model.VARIANCE_FLOOR * np.var(data)

    def segment_cost(start, end):
        segment = data[start:end]
        if cost == "mean":
            return np.sum((segment - segment.mean()) ** 2)
        if cost == "variance":
            return (end - start) * np.log(max(np.mean(segment**2), floor))
        return (end - start) * np.log(max(np.var(segment), floor))

    optimum = np.full(n + 1, np.inf)
    optimum[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    for end in range(minimum_segment, n + 1):
        for start in [0, *range(minimum_segment, end - minimum_segment + 1)]:
            value = optimum[start] + segment_cost(start, end) + penalty
            if value < optimum[end]:
                optimum[end], last[end] = value, start
    boundaries = [n]
    while boundaries[-1] > 0:
        boundaries.append(int(last[boundaries[-1]]))
    return sorted(boundaries[1:-1])


@pytest.mark.parametrize(
    ("cost", "minimum_segment"),
    [("mean", 1), ("mean", 5), ("variance", 2), ("variance", 5), ("mean_variance", 5)],
)
def test_change_points_equal_the_exact_segmentation(cost, minimum_segment):
    rng = np.random.default_rng(4)
    data = np.concatenate(
        [rng.normal(0, 1, 40), rng.normal(1.5, 0.4, 30), rng.normal(-0.5, 2.5, 50)]
    )
    result = regime_model.get_change_points(
        pd.Series(data), cost=cost, minimum_segment=minimum_segment
    )
    ends = result["Observations"].cumsum().to_numpy()[:-1].tolist()

    assert ends == _optimal_partitioning(
        data, cost, result["Penalty"].iloc[0], minimum_segment
    )


def test_change_points_date_a_change_in_volatility(recorder):
    rng = np.random.default_rng(8)
    returns = pd.Series(
        np.concatenate([rng.normal(0, 0.01, 300), rng.normal(0, 0.03, 200)]),
        index=INDEX[:500],
    )
    result = regime_model.get_change_points(returns, cost="variance")

    assert result["Observations"].tolist() == [300, 200]
    # The BIC of the variance cost: two parameters per change.
    assert result["Penalty"].iloc[0] == pytest.approx(2 * np.log(500))
    both = regime_model.get_change_points(returns)
    assert both["Observations"].iloc[0] == 300
    assert both["Penalty"].iloc[0] == pytest.approx(3 * np.log(500))
    recorder.capture(result.round(4))
    recorder.capture(both.round(4))


def test_change_points_date_a_shift_in_the_mean(recorder):
    rng = np.random.default_rng(9)
    series = pd.Series(
        np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)]),
        index=INDEX[:200],
    )
    result = regime_model.get_change_points(series, cost="mean")
    assert len(result) == 2
    assert result["Observations"].iloc[0] == pytest.approx(100, abs=3)
    # A larger penalty finds fewer change points.
    assert len(regime_model.get_change_points(series, cost="mean", penalty=1e6)) == 1
    recorder.capture(result.round(4))


def test_change_points_need_a_valid_cost_and_segment():
    series = pd.Series(np.arange(20.0))
    with pytest.raises(ValueError, match="cost"):
        regime_model.get_change_points(series, cost="median")
    with pytest.raises(ValueError, match="variance"):
        regime_model.get_change_points(series, cost="variance", minimum_segment=1)
    with pytest.raises(ValueError, match="constant"):
        regime_model.get_change_points(pd.Series(np.ones(20)))


def test_ornstein_uhlenbeck_fit_recovers_the_process():
    rng = np.random.default_rng(12)
    levels = np.empty(5000)
    levels[0] = 1.0
    for t in range(1, 5000):
        levels[t] = 0.1 + 0.95 * levels[t - 1] + 0.02 * rng.standard_normal()
    fit = pairs_trading_model.get_ornstein_uhlenbeck_fit(pd.Series(levels))

    assert fit["Phi"] == pytest.approx(0.95, abs=0.01)
    assert fit["Sigma"] == pytest.approx(0.02, rel=0.03)
    assert fit["Long-Run Mean"] == pytest.approx(2.0, abs=0.05)
    assert fit["Half-Life"] == pytest.approx(np.log(2) / -np.log(fit["Phi"]), rel=1e-6)


def test_trading_rule_mesh_of_independent_gains_has_a_closed_form():
    # With phi = 0 every step is an independent N(drift, 1) draw: a path exits above (below)
    # the band with probability p_u (p_d) per step, at the truncated normal mean beyond it,
    # or at the maximum holding period with an unconditional draw.
    drift, horizon, paths = 0.3, 20, 100000
    mesh = pairs_trading_model.get_trading_rule_mesh(
        0.0,
        drift,
        maximum_threshold=2.0,
        grid_size=2,
        maximum_holding_period=horizon,
        simulations=paths,
        seed=3,
    )
    for (profit_taking, stop_loss), row in mesh.iterrows():
        up = stats.norm.sf(profit_taking - drift)
        down = stats.norm.cdf(-stop_loss - drift)
        stay = 1 - up - down
        upper = stats.truncnorm(profit_taking - drift, np.inf, loc=drift)
        lower = stats.truncnorm(-np.inf, -stop_loss - drift, loc=drift)
        early = 1 - stay ** (horizon - 1)
        mean = (up * upper.mean() + down * lower.mean()) / (up + down) * early + (
            1 - early
        ) * drift
        second = (up * upper.moment(2) + down * lower.moment(2)) / (
            up + down
        ) * early + (1 - early) * (1 + drift**2)
        deviation = np.sqrt(second - mean**2)
        steps = np.arange(1, horizon)
        holding = np.sum(steps * (1 - stay) * stay ** (steps - 1)) + horizon * stay ** (
            horizon - 1
        )

        assert row["Mean"] == pytest.approx(mean, abs=4 * deviation / np.sqrt(paths))
        assert row["Standard Deviation"] == pytest.approx(deviation, rel=0.01)
        assert row["Average Holding Period"] == pytest.approx(holding, rel=0.01)


def test_trading_rule_mesh_is_deterministic_and_uses_common_paths():
    first = pairs_trading_model.get_trading_rule_mesh(
        0.97, 0.05, grid_size=4, simulations=2000
    )
    second = pairs_trading_model.get_trading_rule_mesh(
        0.97, 0.05, grid_size=4, simulations=2000
    )
    pd.testing.assert_frame_equal(first, second)
    # A wider stop-loss keeps a position at least as long on the same paths.
    holding = first["Average Holding Period"].unstack()
    assert (holding.diff(axis=1).dropna(axis=1) >= 0).all().all()


def test_optimal_trading_rule_of_a_mean_reverting_price(recorder):
    rng = np.random.default_rng(6)
    levels = np.empty(1000)
    levels[0] = 4.0
    for t in range(1, 1000):
        levels[t] = 0.2 + 0.95 * levels[t - 1] + 0.02 * rng.standard_normal()
    # Start the last price well below its long-run mean of 4: a long position is expected to
    # gain, so the optimal rule lets the profit run and has a positive Sharpe ratio.
    levels[-1] = 3.85
    summary, mesh = pairs_trading_model.get_optimal_trading_rule(
        pd.Series(levels, index=INDEX[:1000]),
        grid_size=8,
        maximum_holding_period=50,
        simulations=4000,
    )

    assert summary["Forecast"] > 5
    assert summary["Sharpe Ratio"] > 0
    assert summary["Sharpe Ratio"] == mesh["Sharpe Ratio"].max()
    assert len(mesh) == 64
    # Monte Carlo results are recorded to two decimals: the Sharpe ratio of a rule has a
    # simulation error of about 0.02 at 4,000 paths, and a path that lands on a threshold
    # after a last-bit difference on another platform moves a mean by at most
    # (profit-taking + stop-loss) / paths.
    recorder.capture(pd.Series(summary).round(2))


def test_ornstein_uhlenbeck_fit_of_a_constant_series_raises():
    # The residuals of a constant series are zero or rounding noise depending on the linear
    # algebra library, so the series itself is checked.
    with pytest.raises(ValueError, match="constant"):
        pairs_trading_model.get_ornstein_uhlenbeck_fit(pd.Series(np.full(400, 50.0)))
