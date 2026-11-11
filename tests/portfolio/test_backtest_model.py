"""Portfolio Backtest Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio import backtest_model, optimization_model

# pylint: disable=missing-function-docstring


def _returns(days: int = 400, assets: int = 3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0004, 0.012, (days, assets)),
        index=pd.PeriodIndex(pd.bdate_range("2021-01-01", periods=days), freq="D"),
        columns=[f"A{i}" for i in range(assets)],
    )


def _equal(window: pd.DataFrame) -> pd.Series:
    return pd.Series(1 / window.shape[1], index=window.columns)


def _minimum_variance(window: pd.DataFrame) -> pd.Series:
    return optimization_model.get_minimum_variance_weights(window.cov())


def test_rebalancing_dates_are_month_ends_after_the_lookback():
    returns = _returns()
    dates = backtest_model.get_rebalancing_dates(returns.index, "monthly", 100)

    month_ends = returns.groupby(returns.index.asfreq("M")).tail(1).index
    assert set(dates) <= set(month_ends)
    assert all(returns.index.get_loc(date) >= 99 for date in dates)  # noqa: PLR2004
    assert dates[-1] < returns.index[-1]
    assert len(backtest_model.get_rebalancing_dates(returns.index, "yearly", 100)) == 1
    with pytest.raises(ValueError, match="rebalancing must be one of"):
        backtest_model.get_rebalancing_dates(returns.index, "daily", 100)


def test_weights_only_use_returns_up_to_the_rebalancing_date():
    returns = _returns()
    dates = backtest_model.get_rebalancing_dates(returns.index, "monthly", 100)
    _, weights, _ = backtest_model.get_walk_forward_backtest(
        returns, {"MV": _minimum_variance}, dates, 100
    )

    # Changing every return after a rebalancing date leaves the weights set on it.
    changed = returns.copy()
    cut = returns.index.get_loc(dates[2])
    changed.iloc[cut + 1 :] *= -3
    _, changed_weights, _ = backtest_model.get_walk_forward_backtest(
        changed, {"MV": _minimum_variance}, dates, 100
    )
    assert np.allclose(weights.iloc[:3], changed_weights.iloc[:3])
    assert not np.allclose(weights.iloc[3:], changed_weights.iloc[3:])


def test_weights_drift_and_costs_are_taken_on_rebalancing():
    returns = _returns()
    dates = backtest_model.get_rebalancing_dates(returns.index, "monthly", 100)
    net, _, turnover = backtest_model.get_walk_forward_backtest(
        returns, {"1/N": _equal}, dates, 100, transaction_cost=0.002
    )

    # The same portfolio from the value held in each asset.
    positions = [returns.index.get_loc(date) for date in dates] + [len(returns) - 1]
    wealth, held, path = 1.0, None, []
    for start, end in zip(positions[:-1], positions[1:], strict=True):
        target = np.full(3, 1 / 3)
        if held is not None:
            wealth *= 1 - 0.002 * np.abs(target - held).sum()
        values = wealth * target
        for day in range(start + 1, end + 1):
            values = values * (1 + returns.iloc[day].to_numpy())
            path.append(values.sum())
        held, wealth = values / values.sum(), values.sum()

    assert np.allclose((1 + net["1/N"]).cumprod(), path)
    assert net.index[0] == returns.index[positions[0] + 1]
    # The first investment is free; the drift makes every later rebalancing trade.
    assert np.isnan(turnover["1/N"].iloc[0])
    assert (turnover["1/N"].iloc[1:] > 0).all()


def test_a_maximum_turnover_moves_part_of_the_way_to_the_new_weights():
    returns = _returns()
    dates = backtest_model.get_rebalancing_dates(returns.index, "monthly", 100)
    _, free_weights, free = backtest_model.get_walk_forward_backtest(
        returns, {"MV": _minimum_variance}, dates, 100
    )
    _, capped_weights, capped = backtest_model.get_walk_forward_backtest(
        returns, {"MV": _minimum_variance}, dates, 100, maximum_turnover=0.01
    )

    assert free["MV"].max() > 0.01  # noqa: PLR2004
    assert capped["MV"].max() <= 0.01 + 1e-12  # noqa: PLR2004
    assert np.allclose(capped_weights.sum(axis=1), 1)
    assert np.allclose(free_weights.iloc[0], capped_weights.iloc[0])


def test_a_strategy_without_weights_keeps_its_drifted_weights(caplog):
    returns = _returns()
    dates = backtest_model.get_rebalancing_dates(returns.index, "monthly", 100)

    def failing_after_the_first(window: pd.DataFrame) -> pd.Series:
        if window.index[-1] > dates[0]:
            raise ValueError("No weights.")
        return _equal(window)

    _, weights, turnover = backtest_model.get_walk_forward_backtest(
        returns, {"Failing": failing_after_the_first}, dates, 100
    )
    assert np.allclose(turnover["Failing"].iloc[1:], 0)
    assert not np.allclose(weights.iloc[1], weights.iloc[0])
    assert "found no weights on" in caplog.text


def test_backtest_statistics():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0]})
    turnover = pd.DataFrame({"A": [np.nan, 0.1, 0.3]})
    statistics = backtest_model.get_backtest_statistics(
        returns, turnover, 4, risk_free_rate=0.01, transaction_cost=0.001
    )
    row = statistics.loc["A"]

    assert row["Annualized Return"] == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    assert row["Annualized Volatility"] == pytest.approx(returns["A"].std() * 2)
    assert row["Sharpe Ratio"] == pytest.approx(
        (returns["A"].mean() * 4 - 0.01) / (returns["A"].std() * 2)
    )
    assert row["Maximum Drawdown"] == pytest.approx(-0.02)
    assert row["Average Turnover"] == pytest.approx(0.2)
    assert row["Annualized Costs"] == pytest.approx(0.0004)


def test_a_backtest_needs_a_rebalancing_date():
    with pytest.raises(ValueError, match="No rebalancing date"):
        backtest_model.get_walk_forward_backtest(_returns(), {"1/N": _equal}, [], 100)


def test_each_strategy_is_measured_over_its_own_periods():
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.0], "Benchmark": [0.02, np.nan, 0.01, np.nan]}
    )
    statistics = backtest_model.get_backtest_statistics(
        returns, pd.DataFrame({"A": [np.nan]}), 4
    )

    # A gap in the benchmark leaves the four periods of A as they are.
    assert statistics.loc["A", "Annualized Return"] == pytest.approx(
        1.01 * 0.98 * 1.03 - 1
    )
    assert statistics.loc["Benchmark", "Annualized Return"] == pytest.approx(
        (1.02 * 1.01) ** 2 - 1
    )


def test_holdings_earn_the_returns_of_days_left_out_of_the_estimates():
    holding = _returns(days=300)
    # A0's market is closed every tenth day: 0% that day, the next day covers both.
    closed = holding.index[5::10]
    holding.loc[closed, "A0"] = 0.0
    estimation = holding.drop(index=closed)
    dates = backtest_model.get_rebalancing_dates(estimation.index, "monthly", 100)

    net, _, _ = backtest_model.get_walk_forward_backtest(
        estimation, {"1/N": _equal}, dates, 100, holding_returns=holding
    )

    assert net.index.equals(holding.index[holding.index.get_loc(dates[0]) + 1 :])
    # Value per asset, rebalanced to 1/N on every date, as an independent check.
    value = np.full(3, 1 / 3)
    path = []
    for date, row in holding.loc[net.index].iterrows():
        before = value.sum()
        value = value * (1 + row.to_numpy())
        path.append(value.sum() / before - 1)
        if date in dates:
            value = np.full(3, value.sum() / 3)
    np.testing.assert_allclose(net["1/N"], path, atol=1e-14)


def test_in_sample_expectations_average_what_each_rebalance_expected():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    weights = pd.concat(
        {
            "Maximum Sharpe": pd.DataFrame(
                [[1.0, 0.0], [0.5, 0.5]], index=dates, columns=["A", "B"]
            )
        },
        axis=1,
    )
    mu = pd.Series({"A": 0.20, "B": 0.10})
    sigma = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"]
    )
    estimates = {dates[0]: (0.0, mu, sigma), dates[1]: (0.02, mu, sigma)}

    expectations = backtest_model.get_in_sample_expectations(weights, estimates)

    assert expectations.loc["Maximum Sharpe", "In-Sample Return"] == pytest.approx(
        (0.20 + 0.15) / 2
    )
    second_sharpe = (0.15 - 0.02) / np.sqrt(0.25 * 0.04 + 0.25 * 0.01)
    assert expectations.loc[
        "Maximum Sharpe", "In-Sample Sharpe Ratio"
    ] == pytest.approx((0.20 / 0.2 + second_sharpe) / 2)
