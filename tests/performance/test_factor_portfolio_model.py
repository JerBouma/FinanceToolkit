"""Factor Portfolio Model Tests"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from arch.bootstrap import StationaryBootstrap
from scipy import stats

from financetoolkit.performance import factor_portfolio_model
from financetoolkit.utilities.error_model import InvalidArgumentError

PERIODS = pd.period_range("2015-01", periods=60, freq="M")
TICKERS = [f"T{number}" for number in range(20)]


def make_panel(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns, a characteristic and market caps of twenty tickers over sixty months, with gaps."""
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(
        rng.normal(0.01, 0.06, (60, 20)), index=PERIODS, columns=TICKERS
    )
    characteristic = pd.DataFrame(
        rng.normal(0, 1, (60, 20)), index=PERIODS, columns=TICKERS
    )
    market_cap = pd.DataFrame(
        rng.lognormal(10, 1, (60, 20)), index=PERIODS, columns=TICKERS
    )
    # A ticker listed later, one delisted early and scattered missing values.
    returns.iloc[:12, 0] = np.nan
    characteristic.iloc[:12, 0] = np.nan
    returns.iloc[40:, 1] = np.nan
    characteristic.iloc[40:, 1] = np.nan
    characteristic.iloc[5, 3] = np.nan
    returns.iloc[7, 4] = np.nan

    return returns, characteristic, market_cap


def loop_quantile_returns(
    characteristic, returns, quantiles, market_cap=None, holding_period=1, lag=1
):
    """A period-by-period loop over the sorts, with pandas' qcut for the quantiles."""
    result = pd.DataFrame(np.nan, index=returns.index, columns=range(1, quantiles + 1))
    first = None
    for position in range(lag, len(returns)):
        if characteristic.iloc[position - lag].count() >= quantiles:
            first = position
            break
    for position in range(first, len(returns)):
        rebalance = first + ((position - first) // holding_period) * holding_period
        sort = characteristic.iloc[rebalance - lag].dropna()
        if len(sort) < quantiles:
            continue
        labels = pd.qcut(sort, quantiles, labels=False) + 1
        for quantile in range(1, quantiles + 1):
            members = labels.index[labels == quantile]
            period_returns = returns.iloc[position][members].dropna()
            if market_cap is None:
                result.iloc[position, quantile - 1] = period_returns.mean()
            else:
                caps = market_cap.iloc[position - 1][period_returns.index]
                result.iloc[position, quantile - 1] = (
                    period_returns * caps
                ).sum() / caps.sum()
    return result


def test_assign_breakpoint_portfolios_matches_qcut():
    _, characteristic, _ = make_panel()
    assignment = factor_portfolio_model.assign_breakpoint_portfolios(
        characteristic, [0.2, 0.4, 0.6, 0.8]
    )

    for date, values in characteristic.iterrows():
        expected = pd.qcut(values.dropna(), 5, labels=False) + 1
        pd.testing.assert_series_equal(
            assignment.loc[date, expected.index],
            expected.astype(float),
            check_names=False,
        )


def test_assign_breakpoint_portfolios_ties_and_too_few_tickers():
    characteristic = pd.DataFrame(
        {
            "A": [1.0, 1.0],
            "B": [1.0, 2.0],
            "C": [1.0, np.nan],
            "D": [3.0, np.nan],
            "E": [4.0, np.nan],
        },
        index=PERIODS[:2],
    )
    assignment = factor_portfolio_model.assign_breakpoint_portfolios(
        characteristic, [0.5]
    )

    # Equal values share a portfolio; two tickers still fill two portfolios.
    assert assignment.iloc[0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert assignment.iloc[1, :2].tolist() == [1.0, 2.0]

    too_few = factor_portfolio_model.assign_breakpoint_portfolios(
        characteristic, [0.25, 0.5, 0.75]
    )
    assert too_few.iloc[1].isna().all()

    with pytest.raises(InvalidArgumentError):
        factor_portfolio_model.assign_breakpoint_portfolios(characteristic, [0.7, 0.3])


@pytest.mark.parametrize("value_weighted", [False, True])
@pytest.mark.parametrize("holding_period", [1, 3])
def test_get_quantile_portfolios_matches_loop(value_weighted, holding_period):
    returns, characteristic, market_cap = make_panel()
    market_cap = market_cap if value_weighted else None

    result = factor_portfolio_model.get_quantile_portfolios(
        characteristic,
        returns,
        quantiles=5,
        market_cap=market_cap,
        holding_period=holding_period,
    )
    expected = loop_quantile_returns(
        characteristic, returns, 5, market_cap, holding_period
    )

    np.testing.assert_allclose(
        result.iloc[:, :5].to_numpy(), expected.to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        result["Top Minus Bottom"].to_numpy(),
        (expected[5] - expected[1]).to_numpy(),
        atol=1e-12,
    )
    # The first period has no earlier characteristic, so no portfolios (no look-ahead).
    assert result.iloc[0].isna().all()


def test_get_quantile_portfolios_uses_no_future_characteristic():
    returns, characteristic, _ = make_panel()
    base = factor_portfolio_model.get_quantile_portfolios(
        characteristic, returns, quantiles=3
    )

    changed = characteristic.copy()
    changed.iloc[30:] = -changed.iloc[30:]
    altered = factor_portfolio_model.get_quantile_portfolios(
        changed, returns, quantiles=3
    )

    # The returns up to and including period 30 use characteristics up to period 29 only.
    pd.testing.assert_frame_equal(base.iloc[:31], altered.iloc[:31])
    assert not base.iloc[31:].equals(altered.iloc[31:])


def test_get_quantile_portfolios_holds_between_rebalancing_dates():
    returns, characteristic, _ = make_panel()
    assignment = factor_portfolio_model.assign_breakpoint_portfolios(
        characteristic, [0.5]
    )
    weights = factor_portfolio_model.get_sorted_portfolio_weights(
        assignment, returns, holding_period=12
    )

    # The first sort is of period 0, applied from period 1; every period of a holding period holds the tickers of
    # the sort before its rebalancing date that have a return in that period.
    for position in range(1, 60):
        rebalancing = 1 + ((position - 1) // 12) * 12
        expected = (assignment.iloc[rebalancing - 1] == 1.0) & returns.iloc[
            position
        ].notna()
        pd.testing.assert_series_equal(
            weights[1.0].iloc[position].notna(), expected, check_names=False
        )


def test_get_double_sort_portfolios_matches_hand_calculation():
    returns, characteristic, market_cap = make_panel(3)
    result = factor_portfolio_model.get_double_sort_portfolios(
        characteristic, market_cap, returns
    )

    position = 20
    size = (
        market_cap.iloc[position - 1]
        .where(characteristic.iloc[position - 1].notna())
        .dropna()
    )
    value = characteristic.iloc[position - 1].dropna()
    size_group = np.where(size > size.quantile(0.5), "Big", "Small")
    size_group = pd.Series(size_group, index=size.index)
    low, high = value.quantile(0.3), value.quantile(0.7)
    value_group = pd.Series(
        np.where(value <= low, "Low", np.where(value <= high, "Neutral", "High")),
        value.index,
    )
    period_returns = returns.iloc[position]
    caps = market_cap.iloc[position - 1]

    portfolio = {}
    for size_name in ("Small", "Big"):
        for value_name in ("Low", "Neutral", "High"):
            members = [
                ticker
                for ticker in value.index
                if size_group.get(ticker) == size_name
                and value_group[ticker] == value_name
                and not np.isnan(period_returns[ticker])
            ]
            portfolio[f"{size_name} {value_name}"] = (
                period_returns[members] * caps[members]
            ).sum() / caps[members].sum()
            assert result.iloc[position][f"{size_name} {value_name}"] == pytest.approx(
                portfolio[f"{size_name} {value_name}"], abs=1e-12
            )

    smb = (
        portfolio["Small Low"] + portfolio["Small Neutral"] + portfolio["Small High"]
    ) / 3 - (
        portfolio["Big Low"] + portfolio["Big Neutral"] + portfolio["Big High"]
    ) / 3
    hml = (portfolio["Small High"] + portfolio["Big High"]) / 2 - (
        portfolio["Small Low"] + portfolio["Big Low"]
    ) / 2
    assert result.iloc[position]["Size Factor"] == pytest.approx(smb, abs=1e-12)
    assert result.iloc[position]["Characteristic Factor"] == pytest.approx(
        hml, abs=1e-12
    )


def test_get_portfolio_turnover():
    returns, characteristic, market_cap = make_panel()
    assignment = factor_portfolio_model.assign_breakpoint_portfolios(
        characteristic, [0.5]
    )
    weights = factor_portfolio_model.get_sorted_portfolio_weights(
        assignment, returns, holding_period=1
    )
    turnover = factor_portfolio_model.get_portfolio_turnover(weights, returns)

    position = 25
    current = weights[1.0].iloc[position].fillna(0)
    previous = weights[1.0].iloc[position - 1]
    grown = previous * (1 + returns.iloc[position - 1])
    drifted = (grown / grown.sum()).fillna(0)
    assert turnover.iloc[position][1.0] == pytest.approx(
        0.5 * (current - drifted).abs().sum(), abs=1e-12
    )
    assert np.isnan(turnover.iloc[1][1.0])

    # A value-weighted portfolio that keeps its tickers and whose market caps grow with the returns trades nothing.
    held = pd.DataFrame(1.0, index=PERIODS, columns=TICKERS[2:6])
    steady_returns = returns[TICKERS[2:6]].fillna(0.01)
    caps = (1 + steady_returns).cumprod() * 100
    value_weights = factor_portfolio_model.get_sorted_portfolio_weights(
        held, steady_returns, caps
    )
    value_turnover = factor_portfolio_model.get_portfolio_turnover(
        value_weights, steady_returns
    )
    np.testing.assert_allclose(value_turnover[1.0].iloc[3:].to_numpy(), 0, atol=1e-12)


def test_get_factor_statistics_matches_statsmodels_hac():
    rng = np.random.default_rng(4)
    # An autocorrelated series, where the Newey-West standard error differs from the plain one.
    noise = rng.normal(0.004, 0.03, 121)
    spread = pd.Series(
        noise[1:] + 0.5 * noise[:-1],
        index=pd.period_range("2010-01", periods=120, freq="M"),
    )

    statistics = factor_portfolio_model.get_factor_statistics(
        spread.rename("Spread"), periods_per_year=12, lags=4
    )
    fit = sm.OLS(spread.to_numpy(), np.ones(120)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
    )

    assert statistics.loc["Spread", "Newey-West Standard Error"] == pytest.approx(
        fit.bse[0], rel=1e-10
    )
    assert statistics.loc["Spread", "t-Statistic"] == pytest.approx(
        fit.tvalues[0], rel=1e-10
    )
    assert statistics.loc["Spread", "P-Value"] == pytest.approx(
        fit.pvalues[0], rel=1e-8
    )
    assert statistics.loc["Spread", "Sharpe Ratio"] == pytest.approx(
        spread.mean() / spread.std(ddof=1) * np.sqrt(12), rel=1e-12
    )
    # The default lags follow the Newey and West (1994) rule: floor(4 × 1.2^(2/9)) = 4.
    assert factor_portfolio_model.get_factor_statistics(spread).iloc[0]["Lags"] == 4


def test_get_monotonicity_test_matches_arch_bootstrap():
    rng = np.random.default_rng(0)
    periods = pd.period_range("2010-01", periods=120, freq="M")
    portfolio_returns = pd.DataFrame(
        rng.normal([0.0, 0.005, 0.01, 0.015, 0.02], 0.03, (120, 5)),
        index=periods,
        columns=[f"Quantile {number}" for number in range(1, 6)],
    )
    portfolio_returns["Top Minus Bottom"] = (
        portfolio_returns["Quantile 5"] - portfolio_returns["Quantile 1"]
    )

    result = factor_portfolio_model.get_monotonicity_test(
        portfolio_returns, bootstraps=10000
    )

    # The same statistic with the stationary bootstrap of arch: the p-values agree within the simulation error.
    differences = np.diff(portfolio_returns.iloc[:, :5].to_numpy(), axis=1)
    sample = differences.mean(axis=0)
    bootstrap = StationaryBootstrap(
        120 ** (1 / 3), differences, seed=np.random.default_rng(5)
    )
    centred = np.array(
        [data[0][0].mean(axis=0) - sample for data in bootstrap.bootstrap(10000)]
    )

    assert result.loc["Increasing", "Statistic"] == pytest.approx(
        sample.min(), abs=1e-15
    )
    assert result.loc["Increasing", "P-Value"] == pytest.approx(
        np.mean(centred.min(axis=1) > sample.min()), abs=0.02
    )
    assert result.loc["Decreasing", "P-Value"] == pytest.approx(
        np.mean((-centred).min(axis=1) > (-sample).min()), abs=0.02
    )
    spearman = stats.spearmanr(range(5), portfolio_returns.iloc[:, :5].mean())
    assert result.loc["Spearman Rank Correlation", "Statistic"] == pytest.approx(
        spearman.statistic
    )
    assert result.loc["Spearman Rank Correlation", "P-Value"] == pytest.approx(
        spearman.pvalue
    )


def test_get_monotonicity_test_rejects_clear_monotonic_and_keeps_flat():
    rng = np.random.default_rng(1)
    periods = pd.period_range("2010-01", periods=240, freq="M")
    rising = pd.DataFrame(
        rng.normal([0.0, 0.01, 0.02, 0.03], 0.02, (240, 4)), index=periods
    )
    flat = pd.DataFrame(rng.normal(0.01, 0.02, (240, 4)), index=periods)

    assert (
        factor_portfolio_model.get_monotonicity_test(rising).loc[
            "Increasing", "P-Value"
        ]
        < 0.01
    )
    assert (
        factor_portfolio_model.get_monotonicity_test(flat).loc["Increasing", "P-Value"]
        > 0.05
    )
