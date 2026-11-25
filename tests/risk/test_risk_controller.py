"""Risk Controller Tests""" ""

import copy

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model
from financetoolkit.risk.distribution import fractal_model
from financetoolkit.risk.helpers import (
    align_annualized_rate,
    compound_into_shared_dates,
)
from financetoolkit.risk.value_at_risk import var_model
from financetoolkit.utilities.error_model import InvalidArgumentError

# AAPL's EGARCH fit sits on a near-flat ridge a different libm/BLAS can tip either way; see garch_model.py.
EGARCH_PLATFORM_DRIFT = pytest.mark.xfail(
    reason="AAPL's EGARCH fit sits on a near-flat likelihood ridge; macOS vs Linux "
    "picks a different near-tied optimum",
    strict=False,
)


def test_collect_all_metrics(recorder, risk_module):
    recorder.capture(risk_module.collect_all_metrics())
    recorder.capture(risk_module.collect_all_metrics(growth=True))
    recorder.capture(risk_module.collect_all_metrics(growth=True, lag=[1, 2, 3]))


def test_get_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_value_at_risk())
    recorder.capture(risk_module.get_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_value_at_risk(period="monthly"))
    recorder.capture(
        risk_module.get_value_at_risk(period="yearly", within_period=True, growth=True)
    )
    recorder.capture(
        risk_module.get_value_at_risk(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(
            period="monthly", within_period=False, method="evt"
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(
            period="monthly", within_period=False, method="cornish-fisher"
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(period="monthly", within_period=False, rolling=6)
    )


def test_get_conditional_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_value_at_risk())
    recorder.capture(risk_module.get_conditional_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_conditional_value_at_risk(period="monthly"))
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, rolling=6
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, method="cornish-fisher"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, method="evt"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, method="studentt"
        )
    )


def test_get_conditional_value_at_risk_studentt_differs_from_var(recorder, risk_module):
    # Regression test: studentt used to dispatch to VaR, so CVaR must be as extreme.
    var = risk_module.get_value_at_risk(
        period="monthly", within_period=False, method="studentt"
    )
    cvar = risk_module.get_conditional_value_at_risk(
        period="monthly", within_period=False, method="studentt"
    )
    assert (cvar <= var).all().all()


def test_get_conditional_drawdown_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_drawdown_at_risk())
    recorder.capture(risk_module.get_conditional_drawdown_at_risk(within_period=False))
    recorder.capture(risk_module.get_conditional_drawdown_at_risk(period="monthly"))
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(period="monthly", rolling=6)
    )


def test_get_tail_ratio(recorder, risk_module):
    recorder.capture(risk_module.get_tail_ratio())
    recorder.capture(risk_module.get_tail_ratio(within_period=False))
    recorder.capture(risk_module.get_tail_ratio(period="monthly"))
    recorder.capture(
        risk_module.get_tail_ratio(period="yearly", within_period=True, growth=True)
    )
    recorder.capture(
        risk_module.get_tail_ratio(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(risk_module.get_tail_ratio(period="monthly", rolling=6))


def test_get_entropic_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_entropic_value_at_risk())
    recorder.capture(risk_module.get_entropic_value_at_risk(within_period=False))
    recorder.capture(risk_module.get_entropic_value_at_risk(period="monthly"))
    recorder.capture(
        risk_module.get_entropic_value_at_risk(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_entropic_value_at_risk(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )


def test_get_garch(recorder, risk_module):
    recorder.capture(risk_module.get_garch())
    recorder.capture(risk_module.get_garch(period="monthly"))
    recorder.capture(risk_module.get_garch(period="yearly", growth=True))
    recorder.capture(risk_module.get_garch(period="yearly", growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_garch(method="gjr"))
    recorder.capture(risk_module.get_garch(method="gjr", period="monthly"))
    # Two yearly returns barely pin down a GJR fit: the optimum lies on a flat ridge whose
    # third decimal moves with the platform's floating point, so it is recorded to two.
    recorder.capture(
        risk_module.get_garch(method="gjr", period="yearly", growth=True).round(2)
    )
    with pytest.raises(InvalidArgumentError, match="method must be one of 'garch'"):
        risk_module.get_garch(method="harch")


def test_get_garch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_garch_forecast())
    recorder.capture(risk_module.get_garch_forecast(period="monthly"))
    recorder.capture(risk_module.get_garch_forecast(period="yearly", growth=True))
    recorder.capture(
        risk_module.get_garch_forecast(period="yearly", growth=True, lag=[1, 2, 3])
    )
    recorder.capture(risk_module.get_garch_forecast(method="gjr"))

    # A GJR-GARCH fit on three years of months lands differently per platform's libraries (in
    # the third digit), so the monthly forecast is checked rather than recorded.
    monthly = risk_module.get_garch_forecast(method="gjr", period="monthly")
    assert monthly.shape == (10, 3)
    assert (monthly > 0).all().all()


def test_get_garch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_garch_parameters())
    recorder.capture(risk_module.get_garch_parameters(period="monthly"))
    recorder.capture(risk_module.get_garch_parameters(method="gjr"))
    recorder.capture(risk_module.get_garch_parameters(method="gjr", period="monthly"))


def test_get_garch_forecast_gjr_monthly_matches_the_arch_package(
    risk_module, monkeypatch
):
    arch = pytest.importorskip("arch")
    from tests.risk.test_garch_fit import simulate_gjr_garch

    # Twenty years of months simulated well inside the stationarity region (persistence
    # Alpha + Gamma / 2 + Beta of 0.78 and 0.92 once fitted), so that the fit and its
    # forecast are well defined on every platform, unlike the fixture's 36 months.
    months = pd.period_range("2005-01", periods=240, freq="M")
    returns = pd.DataFrame(
        {
            ticker: simulate_gjr_garch(2e-4, 0.05, 0.15, 0.75, n=240, seed=seed)
            for ticker, seed in (("A", 4), ("B", 10))
        },
        index=months,
    )
    # The memoized monthly returns are replaced for this test only.
    monkeypatch.setitem(risk_module._data_cache, ("monthly", False), returns)

    forecast = risk_module.get_garch_forecast(
        method="gjr", period="monthly", rounding=12
    )

    assert forecast.index.tolist() == list(
        pd.period_range("2025-01", periods=10, freq="M")
    )
    for ticker in returns:
        reference = (
            arch.arch_model(
                returns[ticker] * 100,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                rescale=False,
            )
            .fit(disp="off")
            .forecast(horizon=10)
            .variance.iloc[-1]
            .to_numpy()
            / 1e4
        )
        np.testing.assert_allclose(forecast[ticker], reference, rtol=1e-3)


# The EGARCH variants have tests of their own, as only they carry the platform drift mark.
# The first one fits EGARCH three times (daily, monthly, yearly), which takes minutes on a
# loaded machine; without the longer timeout the xfail mark would hide that timeout as a pass.
@pytest.mark.timeout(600)
@EGARCH_PLATFORM_DRIFT
def test_get_garch_egarch(recorder, risk_module):
    recorder.capture(risk_module.get_garch(method="egarch"))
    recorder.capture(risk_module.get_garch(method="egarch", period="monthly"))
    recorder.capture(
        risk_module.get_garch(method="egarch", period="yearly", growth=True)
    )


@EGARCH_PLATFORM_DRIFT
def test_get_garch_forecast_egarch(recorder, risk_module):
    recorder.capture(risk_module.get_garch_forecast(method="egarch"))
    recorder.capture(risk_module.get_garch_forecast(method="egarch", period="monthly"))


@EGARCH_PLATFORM_DRIFT
def test_get_garch_parameters_egarch(recorder, risk_module):
    recorder.capture(risk_module.get_garch_parameters(method="egarch"))
    recorder.capture(
        risk_module.get_garch_parameters(method="egarch", period="monthly")
    )


def test_get_var_backtest(recorder, risk_module):
    recorder.capture(risk_module.get_var_backtest(window=100))
    recorder.capture(risk_module.get_var_backtest(window=100, method="gaussian"))
    recorder.capture(risk_module.get_var_backtest(window=100, test="kupiec"))
    recorder.capture(risk_module.get_var_backtest(window=100, test="christoffersen"))


def test_get_maximum_drawdown(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown())
    recorder.capture(risk_module.get_maximum_drawdown(within_period=False))
    recorder.capture(risk_module.get_maximum_drawdown(period="monthly"))
    recorder.capture(
        risk_module.get_maximum_drawdown(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_maximum_drawdown(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )


def test_get_maximum_drawdown_duration(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown_duration())
    recorder.capture(risk_module.get_maximum_drawdown_duration(within_period=False))
    recorder.capture(risk_module.get_maximum_drawdown_duration(period="monthly"))
    recorder.capture(
        risk_module.get_maximum_drawdown_duration(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_maximum_drawdown_duration(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )


def test_the_deepest_drawdown_in_the_table_is_the_maximum_drawdown(risk_module):
    table = risk_module.get_drawdown_table(top=3, rounding=10)
    deepest = table.xs(1, level="Rank")

    assert np.allclose(
        deepest["Drawdown"],
        risk_module.get_maximum_drawdown(rounding=10)[deepest.index],
    )
    assert np.allclose(
        deepest["To Trough"],
        risk_module.get_maximum_drawdown_duration()[deepest.index],
    )
    assert (table.groupby(level="Ticker").size() <= 3).all()
    assert np.allclose(
        risk_module.get_drawdown_series(rounding=10).min()[deepest.index],
        deepest["Drawdown"],
    )


def test_get_maximum_drawdown_recovery_time(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown_recovery_time())
    recorder.capture(
        risk_module.get_maximum_drawdown_recovery_time(within_period=False)
    )
    recorder.capture(risk_module.get_maximum_drawdown_recovery_time(period="monthly"))
    recorder.capture(
        risk_module.get_maximum_drawdown_recovery_time(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_maximum_drawdown_recovery_time(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )


def test_get_ulcer_index(recorder, risk_module):
    recorder.capture(risk_module.get_ulcer_index())
    recorder.capture(risk_module.get_ulcer_index(window=5))
    recorder.capture(risk_module.get_ulcer_index(period="monthly"))
    recorder.capture(risk_module.get_ulcer_index(growth=True))
    recorder.capture(risk_module.get_ulcer_index(growth=True, lag=[1, 2, 3]))


def test_get_pain_index(recorder, risk_module):
    recorder.capture(risk_module.get_pain_index())
    recorder.capture(risk_module.get_pain_index(period="quarterly"))
    recorder.capture(risk_module.get_pain_index(period="monthly"))
    recorder.capture(risk_module.get_pain_index(growth=True, lag=[1, 2]))


def test_get_pain_index_is_the_pain_index_of_the_pain_ratio(
    risk_module, performance_module
):
    from financetoolkit.performance import performance_controller  # noqa: PLC0415

    pain_index = risk_module.get_pain_index(rounding=10)
    # The Pain Ratio divides by the same function on the same daily returns within each year.  # noqa: E501
    assert performance_controller.get_pain_index is risk_model.get_pain_index
    returns = performance_module._get_column("yearly", "Return", within_period=True)
    expected = risk_model.get_pain_index(returns).loc[
        performance_module._start_date : performance_module._end_date
    ]

    # The Performance module keeps the Benchmark apart, the Risk module shows it as a column.  # noqa: E501
    assert list(pain_index.columns) == [*expected.columns, "Benchmark"]
    pd.testing.assert_frame_equal(
        pain_index[expected.columns],
        expected.round(10),
        check_names=False,
        check_freq=False,
    )
    benchmark = performance_module._get_column(
        "yearly", "Return", within_period=True, benchmark=True
    )
    assert pain_index["Benchmark"].to_numpy() == pytest.approx(
        risk_model.get_pain_index(benchmark.to_frame())
        .iloc[:, 0]
        .loc[performance_module._start_date : performance_module._end_date],
        abs=1e-10,
    )
    assert (
        (pain_index <= risk_module.get_ulcer_index(window=None, rounding=10))
        .all()
        .all()
    )


def test_get_skewness(recorder, risk_module):
    recorder.capture(risk_module.get_skewness())
    recorder.capture(risk_module.get_skewness(within_period=False))
    recorder.capture(risk_module.get_skewness(period="monthly"))
    recorder.capture(
        risk_module.get_skewness(period="yearly", within_period=True, growth=True)
    )
    recorder.capture(
        risk_module.get_skewness(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(risk_module.get_skewness(period="monthly", rolling=6))


def test_get_kurtosis(recorder, risk_module):
    recorder.capture(risk_module.get_kurtosis())
    recorder.capture(round(risk_module.get_kurtosis(within_period=False), 4))
    recorder.capture(risk_module.get_kurtosis(period="monthly"))
    recorder.capture(
        risk_module.get_kurtosis(period="yearly", within_period=True, growth=True)
    )
    recorder.capture(
        risk_module.get_kurtosis(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(risk_module.get_kurtosis(period="monthly", rolling=6))


def test_get_downside_deviation(recorder, risk_module):
    recorder.capture(risk_module.get_downside_deviation())
    recorder.capture(risk_module.get_downside_deviation(within_period=False))
    recorder.capture(risk_module.get_downside_deviation(period="monthly"))
    recorder.capture(risk_module.get_downside_deviation(minimum_acceptable_return=0.01))
    recorder.capture(
        risk_module.get_downside_deviation(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_downside_deviation(
            period="yearly", within_period=True, growth=True, lag=[1, 2, 3]
        )
    )
    recorder.capture(risk_module.get_downside_deviation(period="monthly", rolling=6))


def test_get_variance(recorder, risk_module):
    recorder.capture(risk_module.get_variance())
    recorder.capture(risk_module.get_variance(period="monthly"))
    recorder.capture(risk_module.get_variance(growth=True))
    recorder.capture(risk_module.get_variance(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_variance(period="monthly", rolling=6))


def test_get_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_volatility())
    recorder.capture(risk_module.get_volatility(period="monthly"))
    recorder.capture(risk_module.get_volatility(growth=True))
    recorder.capture(risk_module.get_volatility(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_volatility(period="monthly", rolling=6))


def test_get_volatility_parkinson(recorder, risk_module):
    recorder.capture(risk_module.get_volatility(method="parkinson"))
    recorder.capture(risk_module.get_volatility(method="parkinson", period="monthly"))
    recorder.capture(risk_module.get_volatility(method="parkinson", growth=True))
    recorder.capture(
        risk_module.get_volatility(method="parkinson", growth=True, lag=[1, 2, 3])
    )


def test_get_volatility_garman_klass(recorder, risk_module):
    recorder.capture(risk_module.get_volatility(method="garman_klass"))
    recorder.capture(
        risk_module.get_volatility(method="garman_klass", period="monthly")
    )
    recorder.capture(risk_module.get_volatility(method="garman_klass", growth=True))


def test_get_volatility_rogers_satchell(recorder, risk_module):
    recorder.capture(risk_module.get_volatility(method="rogers_satchell"))
    recorder.capture(
        risk_module.get_volatility(method="rogers_satchell", period="monthly")
    )
    recorder.capture(risk_module.get_volatility(method="rogers_satchell", growth=True))


def test_get_volatility_yang_zhang(recorder, risk_module):
    recorder.capture(risk_module.get_volatility(method="yang_zhang"))
    recorder.capture(risk_module.get_volatility(method="yang_zhang", period="monthly"))
    recorder.capture(risk_module.get_volatility(method="yang_zhang", growth=True))


def test_get_excess_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_excess_volatility())
    recorder.capture(risk_module.get_excess_volatility(period="monthly"))
    recorder.capture(risk_module.get_excess_volatility(growth=True))
    recorder.capture(risk_module.get_excess_volatility(growth=True, lag=[1, 2, 3]))
    recorder.capture(risk_module.get_excess_volatility(period="monthly", rolling=6))


def test_get_hill_estimator(recorder, risk_module):
    recorder.capture(risk_module.get_hill_estimator(period="yearly"))
    recorder.capture(
        risk_module.get_hill_estimator(period="yearly", tail="right", k=0.15)
    )


def test_get_amihud_illiquidity(recorder, risk_module):
    recorder.capture(risk_module.get_amihud_illiquidity(period="quarterly", scale=1e12))
    recorder.capture(risk_module.get_amihud_illiquidity(period="monthly", scale=1e12))
    recorder.capture(
        risk_module.get_amihud_illiquidity(
            period="yearly", within_period=True, growth=True, scale=1e12
        )
    )


def test_get_roll_spread(recorder, risk_module):
    recorder.capture(risk_module.get_roll_spread(period="quarterly"))
    recorder.capture(
        risk_module.get_roll_spread(period="quarterly", within_period=False)
    )


def test_get_har_rv_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_har_rv_forecast().tail())
    recorder.capture(risk_module.get_har_rv_forecast(estimator="parkinson").tail())
    recorder.capture(risk_module.get_har_rv_forecast(estimator="garman_klass").tail())
    recorder.capture(
        risk_module.get_har_rv_forecast(estimator="rogers_satchell").tail()
    )


def test_get_har_rv_forecast_invalid_estimator(risk_module):
    # An unknown option raises, naming the valid ones, rather than returning empty.
    with pytest.raises(InvalidArgumentError, match="estimator must be one of"):
        risk_module.get_har_rv_forecast(estimator="bad")


def test_get_tail_dependence_coefficient(recorder, risk_module):
    recorder.capture(
        risk_module.get_tail_dependence_coefficient("AAPL", "MSFT", period="yearly")
    )
    recorder.capture(
        risk_module.get_tail_dependence_coefficient(
            "AAPL", "MSFT", period="yearly", method="gaussian"
        )
    )


def test_get_covar(recorder, risk_module):
    recorder.capture(risk_module.get_covar("AAPL", "MSFT", period="yearly"))


def test_get_marginal_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_marginal_value_at_risk(period="yearly"))
    recorder.capture(
        risk_module.get_marginal_value_at_risk(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="yearly"
        )
    )
    recorder.capture(
        risk_module.get_marginal_value_at_risk(period="yearly", method="gaussian")
    )


def test_get_component_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_component_value_at_risk(period="yearly"))
    recorder.capture(
        risk_module.get_component_value_at_risk(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="yearly"
        )
    )
    recorder.capture(
        risk_module.get_component_value_at_risk(period="yearly", method="gaussian")
    )


def test_get_acerbi_szekely_test(recorder, risk_module):
    recorder.capture(risk_module.get_acerbi_szekely_test(window=100))
    recorder.capture(risk_module.get_acerbi_szekely_test(window=100, method="gaussian"))


def _risk_with_late_listing(risk_module, listing_date="2021-01-01"):
    """A copy of the Risk module in which MSFT has no data before `listing_date`."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

    historical_data = {}
    # Read per period: the Toolkit's module loads a period when it is first asked for.
    for period in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        data = risk_module._historical_data[period].copy()
        if not data.empty and "MSFT" in data.columns.get_level_values(1):
            before_listing = data.index.to_timestamp() < listing_date
            for column in data.columns.get_level_values(0).unique():
                data.loc[before_listing, (column, "MSFT")] = np.nan
        historical_data[period] = data

    return Risk(
        tickers=risk_module._tickers,
        historical_data=historical_data,
        start_date=risk_module._start_date,
        end_date=risk_module._end_date,
    )


def test_shorter_history_does_not_shorten_other_tickers(risk_module):
    late_listing = _risk_with_late_listing(risk_module)

    for method in ("get_value_at_risk", "get_maximum_drawdown"):
        full = getattr(risk_module, method)(rounding=10)
        shortened = getattr(late_listing, method)(rounding=10)

        # AAPL and the Benchmark keep their full history.
        assert full["AAPL"] == shortened["AAPL"]
        assert full["Benchmark"] == shortened["Benchmark"]


def test_within_period_without_data_is_nan(risk_module):
    late_listing = _risk_with_late_listing(risk_module)

    value_at_risk = late_listing.get_value_at_risk(period="yearly", within_period=True)
    maximum_drawdown = late_listing.get_maximum_drawdown(
        period="yearly", within_period=True
    )
    skewness = late_listing.get_skewness(period="yearly", within_period=True)

    for result in (value_at_risk, maximum_drawdown, skewness):
        assert result.loc["2020", "MSFT"] != result.loc["2020", "MSFT"]  # NaN
        assert result.loc["2020", "AAPL"] == result.loc["2020", "AAPL"]
        assert result.loc["2021", "MSFT"] == result.loc["2021", "MSFT"]


def test_var_backtest_rolls_each_ticker_over_its_own_history(risk_module):
    late_listing = _risk_with_late_listing(risk_module)

    full = risk_module.get_var_backtest(window=100, rounding=10)
    shortened = late_listing.get_var_backtest(window=100, rounding=10)

    assert full["AAPL"].equals(shortened["AAPL"])
    assert shortened["MSFT"].notna().all()


def test_get_ulcer_index_weekly_has_values(risk_module):
    ulcer_index = risk_module.get_ulcer_index(period="weekly")

    assert ulcer_index.iloc[1:].notna().all().all()


def test_get_value_at_risk_cf_is_not_a_distribution(risk_module):
    # "cornish-fisher" is the one name of the Cornish-Fisher VaR.
    with pytest.raises(InvalidArgumentError, match="'cornish-fisher'"):
        risk_module.get_value_at_risk(method="cf")


def test_get_hurst_exponent_uses_log_prices(risk_module):
    hurst_exponent = risk_module.get_hurst_exponent(rounding=10)
    log_prices = np.log(risk_module._historical_data["daily"]["Adj Close"])

    for ticker in hurst_exponent.index:
        assert hurst_exponent[ticker] == pytest.approx(
            fractal_model.get_generalized_hurst_exponent(log_prices[ticker]), abs=1e-9
        )


def test_garch_parameters_fit_each_ticker_on_its_own_history(risk_module):
    risk = copy.copy(risk_module)
    risk._data_cache = {}
    risk._historical_data = dict(risk_module._historical_data)
    daily = risk_module._historical_data["daily"].copy()
    tickers = daily["Return"].columns
    # The first ticker lists 100 days later than the others.
    daily.loc[daily.index[:100], ("Return", tickers[0])] = np.nan
    risk._historical_data["daily"] = daily

    full = risk_module.get_garch_parameters(rounding=10)
    late = risk.get_garch_parameters(rounding=10)

    # The other tickers keep their full history, and so their fit.
    pd.testing.assert_series_equal(full[tickers[1]], late[tickers[1]])
    assert not np.allclose(full[tickers[0]], late[tickers[0]])

    garch = risk.get_garch(rounding=10)
    assert garch[tickers[0]].iloc[:99].isna().all()
    assert garch[tickers[1]].notna().all()


def test_the_recovery_needed_is_the_gain_back_to_the_running_peak(risk_module):
    drawdowns = risk_module.get_drawdown_series(rounding=10)
    recovery = risk_module.get_drawdown_series(recovery=True, rounding=10)

    pd.testing.assert_frame_equal(recovery, -drawdowns / (1 + drawdowns), atol=1e-8)

    table = risk_module.get_drawdown_table(top=3, rounding=10)
    expected = -table["Drawdown"] / (1 + table["Drawdown"])
    pd.testing.assert_series_equal(
        table["Recovery Needed"], expected, check_names=False, atol=1e-8
    )


def test_a_single_return_over_the_whole_range_is_reported(risk_module, caplog):
    module = copy.copy(risk_module)
    yearly = risk_module._historical_data["yearly"]["Return"].dropna(how="all")
    module._data_cache = {("yearly", False): yearly.iloc[-1:]}

    module._get_returns("yearly", False)

    assert "Fewer than two yearly returns" in caplog.text


def test_pairwise_measures_default_to_the_benchmark(risk_module):
    tail_dependence = risk_module.get_tail_dependence_coefficient()
    covar = risk_module.get_covar()

    # Every ticker is paired with the benchmark, which is not paired with itself.
    assert list(tail_dependence.index) == ["AAPL", "MSFT"]
    assert list(covar.index) == ["AAPL", "MSFT"]
    pd.testing.assert_series_equal(
        tail_dependence.loc["AAPL"],
        risk_module.get_tail_dependence_coefficient("AAPL", "Benchmark"),
        check_names=False,
    )
    # CoVaR conditions the benchmark (the system) on each ticker's distress.
    pd.testing.assert_series_equal(
        covar.loc["MSFT"],
        risk_module.get_covar("Benchmark", "MSFT"),
        check_names=False,
    )
    # One side given pairs it with the benchmark.
    assert list(risk_module.get_tail_dependence_coefficient("MSFT").index) == ["MSFT"]


def test_pairwise_measures_keep_the_role_of_a_given_ticker(risk_module):
    # A given system is measured against every other ticker, the benchmark included.
    exposure = risk_module.get_covar(ticker="AAPL")
    assert list(exposure.index) == ["MSFT", "Benchmark"]
    pd.testing.assert_series_equal(
        exposure.loc["MSFT"], risk_module.get_covar("AAPL", "MSFT"), check_names=False
    )

    # A given conditioning ticker is paired with the benchmark as the system.
    contribution = risk_module.get_covar(conditioning_ticker="AAPL")
    assert list(contribution.index) == ["AAPL"]
    pd.testing.assert_series_equal(
        contribution.loc["AAPL"],
        risk_module.get_covar("Benchmark", "AAPL"),
        check_names=False,
    )

    # The same for tail dependence: a given ticker_b is paired with every other ticker.
    tail_dependence = risk_module.get_tail_dependence_coefficient(ticker_b="AAPL")
    assert list(tail_dependence.index) == ["MSFT", "Benchmark"]


def _toolkit_with_historical(historical):
    from financetoolkit import Toolkit

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )

    return toolkit


def test_a_constant_price_has_no_skewness_or_kurtosis(risk_module):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        historical[(column, "MSFT")] = 100.0
    historical[("Return", "MSFT")] = 0.0
    constant = _toolkit_with_historical(historical).risk

    skewness = constant.get_skewness()
    kurtosis = constant.get_kurtosis()

    assert np.isnan(skewness["MSFT"]) and np.isnan(kurtosis["MSFT"])
    assert skewness["AAPL"] == risk_module.get_skewness()["AAPL"]


@pytest.mark.parametrize(
    "method", ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
)
def test_a_bad_bar_only_leaves_out_that_day(risk_module, method):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    historical.loc["2021-03-01":"2021-03-04", ("Open", "AAPL")] = 0.0
    historical.loc["2021-03-01":"2021-03-04", ("Low", "AAPL")] = 0.0
    bad_bars = _toolkit_with_historical(historical).risk

    volatility = bad_bars.get_volatility(method=method)
    reference = risk_module.get_volatility(method=method)

    assert np.isfinite(volatility.loc["2021", "AAPL"])
    # Four of 252 days left out move the yearly estimate by little.
    assert volatility.loc["2021", "AAPL"] == pytest.approx(
        reference.loc["2021", "AAPL"], rel=0.05
    )
    pd.testing.assert_series_equal(volatility["MSFT"], reference["MSFT"])

    if method != "yang_zhang":
        forecast = bad_bars.get_har_rv_forecast(estimator=method)["AAPL"]
        assert forecast.notna().any() and np.isfinite(forecast.dropna()).all()


def test_get_mean_absolute_deviation_per_month(risk_module):
    result = risk_module.get_mean_absolute_deviation(period="monthly", rounding=None)
    daily = risk_module._historical_data["daily"]["Return"]["AAPL"]
    june = daily[(daily.index.year == 2021) & (daily.index.month == 6)]

    assert result.loc["2021-06", "AAPL"] == pytest.approx(
        (june - june.mean()).abs().mean(), abs=5e-5
    )


def test_get_mean_absolute_deviation_refuses_a_daily_period(risk_module):
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        risk_module.get_mean_absolute_deviation(period="daily")


def test_get_coefficient_of_variation_per_year(risk_module):
    result = risk_module.get_coefficient_of_variation(period="yearly", rounding=None)
    daily = risk_module._historical_data["daily"]["Return"]["MSFT"]
    year = daily[daily.index.year == 2021]

    assert result.loc["2021", "MSFT"] == pytest.approx(
        year.std() / year.mean(), rel=1e-3
    )


def test_get_autocorrelation_per_lag(risk_module):
    result = risk_module.get_autocorrelation(lags=3, rounding=None)
    values = risk_module._historical_data["daily"]["Return"]["AAPL"].dropna()
    deviations = (values - values.mean()).to_numpy()

    assert list(result.index) == [1, 2, 3]
    assert result.index.name == "Lag"
    assert result.loc[1, "AAPL"] == pytest.approx(
        np.sum(deviations[:-1] * deviations[1:]) / np.sum(deviations**2), abs=5e-5
    )
    # Daily stock returns are close to uncorrelated over time.
    assert (result.abs() < 0.3).all().all()


def test_get_copula_parameters_of_one_pair(risk_module):
    result = risk_module.get_copula_parameters(
        ticker_a="AAPL", ticker_b="MSFT", rounding=None
    )

    # Apple and Microsoft move together: a clearly positive Gaussian correlation.
    assert 0.5 < result["Rho"] < 1
    assert result["AIC"] < 0

    clayton = risk_module.get_copula_parameters(
        ticker_a="AAPL", ticker_b="MSFT", copula="clayton", rounding=None
    )
    assert clayton["Theta"] > 0


def test_get_copula_parameters_refuses_an_unknown_family(risk_module):
    with pytest.raises(InvalidArgumentError, match="copula must be one of"):
        risk_module.get_copula_parameters(
            ticker_a="AAPL", ticker_b="MSFT", copula="normal"
        )


def test_get_best_fitting_copula_ranks_the_families_by_aic(risk_module):
    result = risk_module.get_best_fitting_copula(
        ticker_a="AAPL", ticker_b="MSFT", rounding=None
    )

    assert sorted(result.index) == [
        "Clayton",
        "Frank",
        "Gaussian",
        "Gumbel",
        "Student-T",
    ]
    assert result["AIC"].is_monotonic_increasing

    gaussian = risk_module.get_copula_parameters(
        ticker_a="AAPL", ticker_b="MSFT", rounding=None
    )
    assert result.loc["Gaussian", "AIC"] == pytest.approx(gaussian["AIC"])


def test_get_bid_ask_spread(recorder, risk_module):
    recorder.capture(risk_module.get_bid_ask_spread())
    recorder.capture(risk_module.get_bid_ask_spread(period="quarterly"))
    recorder.capture(risk_module.get_bid_ask_spread(period="quarterly", rolling=2))
    for method in ("corwin_schultz", "abdi_ranaldo"):
        recorder.capture(risk_module.get_bid_ask_spread(method=method))
        recorder.capture(
            risk_module.get_bid_ask_spread(method=method, period="quarterly")
        )
        recorder.capture(
            risk_module.get_bid_ask_spread(method=method, period="quarterly", rolling=2)
        )
    pd.testing.assert_frame_equal(
        risk_module.get_bid_ask_spread(), risk_module.get_bid_ask_spread(method="edge")
    )
    with pytest.raises(InvalidArgumentError, match="'corwin_schultz'"):
        risk_module.get_bid_ask_spread(method="roll")


def test_get_share_turnover(recorder, risk_module):
    recorder.capture(risk_module.get_share_turnover())
    recorder.capture(risk_module.get_share_turnover(period="quarterly", rolling=2))

    # The turnover of a month is its Volume over the share count of the statements.
    volume = risk_module._historical_data["daily"]["Volume"]["AAPL"]
    january = (volume.index >= pd.Period("2021-01-01", "D")) & (
        volume.index <= pd.Period("2021-01-31", "D")
    )
    shares = risk_module._shares_outstanding_function()[0].loc["AAPL", "2020"]
    assert np.isclose(
        risk_module.get_share_turnover(rounding=10).loc["2021-01", "AAPL"],
        volume[january].sum() / shares,
    )


def test_get_idiosyncratic_volatility(recorder, risk_module, monkeypatch):
    # Fixed factors in place of the Ken French download, on the dates of the fixture.
    dates = risk_module._historical_data["daily"].index
    rng = np.random.default_rng(11)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 4)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RF"],
    )
    factors["RF"] = 0.0001
    monkeypatch.setattr(
        "financetoolkit.performance.performance_model."
        "obtain_fama_and_french_three_factor_dataset",
        lambda region="United States": factors,
    )

    recorder.capture(risk_module.get_idiosyncratic_volatility())
    recorder.capture(risk_module.get_idiosyncratic_volatility(model="capm"))
    recorder.capture(
        risk_module.get_idiosyncratic_volatility(
            period="quarterly", rolling=4, annualize=True
        )
    )


def test_var_backtest_window_defaults_to_a_year_of_each_period(risk_module):
    # Daily keeps 252 returns, the other periods a year of them with a minimum of 20.
    pd.testing.assert_frame_equal(
        risk_module.get_var_backtest(rounding=10),
        risk_module.get_var_backtest(window=252, rounding=10),
    )
    pd.testing.assert_frame_equal(
        risk_module.get_var_backtest(period="weekly", rounding=10),
        risk_module.get_var_backtest(period="weekly", window=52, rounding=10),
    )
    assert risk_module.get_var_backtest(period="weekly").notna().all().all()
    assert (
        risk_module.get_acerbi_szekely_test(period="weekly", n_bootstrap=50)
        .notna()
        .all()
        .all()
    )


def test_var_backtest_with_evt_and_a_short_window_is_nan(risk_module, caplog):
    with caplog.at_level("WARNING"):
        result = risk_module.get_var_backtest(
            period="weekly", method="evt", test="kupiec"
        )

    # 52 weekly returns hold 3 losses above the 95th percentile, too few to fit.
    assert result.isna().all().all()
    assert sum("Extreme Value Theory" in m for m in caplog.messages) == 1


def test_marginal_value_at_risk_compounds_the_dates_one_ticker_missed(risk_module):
    risk = copy.copy(risk_module)
    risk._historical_data = dict(risk_module._historical_data)
    daily = risk_module._historical_data["daily"].copy()
    tickers = list(daily["Return"].columns[:2])
    # The first ticker misses every fifth date, as on its own exchange's holidays.
    missed = daily.index[1::5]
    daily.loc[missed, ("Return", tickers[0])] = np.nan
    risk._historical_data["daily"] = daily

    returns = daily["Return"][tickers].dropna(how="all")
    shared = compound_into_shared_dates(returns)
    weights = pd.Series(0.5, index=tickers)

    assert len(shared) == returns.notna().all(axis=1).sum()
    pd.testing.assert_series_equal(
        risk.get_marginal_value_at_risk(
            weights=dict.fromkeys(tickers, 0.5), rounding=10
        ),
        var_model.get_marginal_var(shared, weights, 0.05)
        .round(10)
        .rename("Marginal VaR"),
        check_names=False,
    )


def test_risk_free_rate_of_a_partial_period_covers_its_dates(risk_module):
    daily_returns = risk_module._historical_data["daily"]["Return"]
    per_observation = risk_module._get_risk_free_rate("daily")
    yearly = risk_module._get_risk_free_rate("yearly")

    # Stock-only data earns a trading day of the annualized rate per return.
    annualized = align_annualized_rate(
        risk_module._annualized_risk_free_rate_data["daily"], daily_returns.index
    )
    expected = ((1 + annualized) ** (1 / 252) - 1).to_numpy()[:, None]
    observed = daily_returns.notna().to_numpy()
    assert np.allclose(
        per_observation.to_numpy()[observed],
        np.broadcast_to(expected, observed.shape)[observed],
    )
    assert per_observation.isna().to_numpy()[~observed].all()

    # Each year compounds the annualized rate over the calendar days since the date before
    # each date of the ticker in it that holds a return, so a full year earns the
    # annualized rate.
    for ticker in daily_returns.columns:
        dates = daily_returns[ticker].dropna().index
        first = daily_returns.index.get_loc(dates[0])
        previous = (
            daily_returns.index[first - 1].ordinal if first else dates[0].ordinal - 1
        )
        days = np.diff(np.append(previous, dates.asi8))
        growth = pd.Series(
            (1 + annualized.reindex(dates).to_numpy()) ** (days / 365), index=dates
        )
        for year in yearly.index:
            in_year = dates.asfreq("Y") == year
            assert yearly.loc[year, ticker] == pytest.approx(growth[in_year].prod() - 1)


def _weekend_risk(rate=0.04, tickers=("BTC", "SPY")):
    """A stock (trading days) and a cryptocurrency (every calendar day) at a flat rate."""
    from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

    rng = np.random.default_rng(2)
    days = pd.period_range("2021-01-01", "2022-06-30", freq="D")
    weekday = days.dayofweek < 5  # noqa: PLR2004
    returns = pd.DataFrame(
        {
            "BTC": rng.normal(0.001, 0.03, len(days)),
            "SPY": np.where(weekday, rng.normal(0.0005, 0.01, len(days)), np.nan),
            "Benchmark": np.where(weekday, rng.normal(0.0005, 0.01, len(days)), np.nan),
        },
        index=days,
    )
    returns.iloc[0] = np.nan
    # Without the cryptocurrency the data only holds the trading days.
    returns = returns[[*tickers, "Benchmark"]]
    returns = returns[returns.notna().any(axis=1) | (returns.index == days[0])]
    prices = (1 + returns.fillna(0)).cumprod() * 100
    daily = pd.concat({"Return": returns, "Adj Close": prices, "Close": prices}, axis=1)
    historical_data = {"daily": daily, "intraday": pd.DataFrame()}
    # The return of a period runs from the last price of the period before to its last price.
    for period, frequency in (
        ("weekly", "W"),
        ("monthly", "M"),
        ("quarterly", "Q"),
        ("yearly", "Y"),
    ):
        period_prices = prices.groupby(returns.index.asfreq(frequency)).last()
        historical_data[period] = pd.concat(
            {
                "Return": period_prices / period_prices.shift() - 1,
                "Adj Close": period_prices,
                "Close": period_prices,
            },
            axis=1,
        )
    return Risk(
        tickers=list(tickers),
        historical_data=historical_data,
        risk_free_rate_data={"daily": pd.Series(rate, index=days)},
        rounding=None,
    )


@pytest.mark.parametrize("period", ["weekly", "monthly", "quarterly", "yearly"])
def test_the_excess_volatility_of_a_stock_ignores_a_cryptocurrency_next_to_it(period):
    mixed = _weekend_risk()
    alone = _weekend_risk(tickers=("SPY",))

    # The weekends the cryptocurrency adds neither change the rate a stock earns per
    # return nor move a weekend at the turn of a period out of the period of the stock's
    # next trading day.
    for column in ("SPY", "Benchmark"):
        pd.testing.assert_series_equal(
            mixed._risk_free_rate_data[period][column].dropna(),
            alone._risk_free_rate_data[period][column].dropna(),
        )
    for rolling in (None, 4):
        pd.testing.assert_series_equal(
            mixed.get_excess_volatility(period=period, rolling=rolling)["SPY"].dropna(),
            alone.get_excess_volatility(period=period, rolling=rolling)["SPY"].dropna(),
            rtol=1e-12,
        )

    # July 2021 ends on a Saturday, so the stock's July runs from Wednesday 2021-06-30 to
    # Friday 2021-07-30 and the weekend accrues in its August, up to Tuesday 2021-08-31.
    monthly = mixed._risk_free_rate_data["monthly"]
    assert monthly.loc["2021-07", "SPY"] == pytest.approx(1.04 ** (30 / 365) - 1)
    assert monthly.loc["2021-08", "SPY"] == pytest.approx(1.04 ** (32 / 365) - 1)
    assert monthly.loc["2021-07", "BTC"] == pytest.approx(1.04 ** (31 / 365) - 1)
    assert monthly.loc["2021-08", "BTC"] == pytest.approx(1.04 ** (31 / 365) - 1)


def _weekend_ohlc_risk(tickers=("BTC", "SPY")):
    """
    Daily Open, High, Low and Close of a cryptocurrency (every calendar day) and a stock
    (trading days), with the stock's weekends as the flat bars without a Return that the
    Toolkit fills in when another ticker trades on them.
    """
    from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

    rng = np.random.default_rng(3)
    days = pd.period_range("2021-01-01", "2021-12-31", freq="D")
    weekday = days.dayofweek < 5  # noqa: PLR2004
    fields = {field: {} for field in ("Open", "High", "Low", "Close", "Return")}
    for ticker in ("BTC", "SPY", "Benchmark"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        open_ = close * np.exp(rng.normal(0, 0.003, len(days)))
        high = np.maximum(open_, close) * np.exp(rng.uniform(0, 0.01, len(days)))
        low = np.minimum(open_, close) * np.exp(-rng.uniform(0, 0.01, len(days)))
        traded = np.ones(len(days), dtype=bool) if ticker == "BTC" else weekday
        for field, values in zip(
            ("Open", "High", "Low", "Close"), (open_, high, low, close), strict=True
        ):
            fields[field][ticker] = pd.Series(values, index=days)[traded]
        fields["Return"][ticker] = fields["Close"][ticker].pct_change()
    selected = [*tickers, "Benchmark"]
    frames = {field: pd.DataFrame(values)[selected] for field, values in fields.items()}
    # A flat bar at the last Close on the dates only another ticker traded, without a Return.
    last_close = frames["Close"].ffill()
    for field in ("Open", "High", "Low", "Close"):
        frames[field] = frames[field].fillna(last_close)
    frames["Adj Close"] = frames["Close"]
    data = pd.concat(frames, axis=1)
    return Risk(
        tickers=list(tickers),
        historical_data={"daily": data, "intraday": pd.DataFrame()},
        risk_free_rate_data={"daily": pd.Series(0.04, index=days)},
        rounding=None,
    )


@pytest.mark.parametrize("method", ["corwin_schultz", "abdi_ranaldo", "edge"])
def test_the_spread_of_a_stock_ignores_a_cryptocurrency_next_to_it(method):
    mixed = _weekend_ohlc_risk()
    alone = _weekend_ohlc_risk(tickers=("SPY",))

    for period, rolling in (("monthly", None), ("quarterly", 2), ("weekly", None)):
        mixed_spread = mixed.get_bid_ask_spread(
            period=period, rolling=rolling, method=method
        )
        # The rows of the cryptocurrency's own periods (a first week holding a single
        # trading day of the stock) are NaN for the stock.
        for column in ("SPY", "Benchmark"):
            pd.testing.assert_series_equal(
                mixed_spread[column].dropna(),
                alone.get_bid_ask_spread(period=period, rolling=rolling, method=method)[
                    column
                ].dropna(),
            )
        assert mixed_spread["SPY"].notna().sum() > 0


def test_get_marginal_expected_shortfall(recorder, risk_module):
    recorder.capture(risk_module.get_marginal_expected_shortfall())
    recorder.capture(
        risk_module.get_marginal_expected_shortfall(period="quarterly", rolling=4)
    )

    # The average return of AAPL on the 5% worst days of the benchmark in 2021.
    returns = risk_module._historical_data["daily"]["Return"][["AAPL", "Benchmark"]]
    year = returns[returns.index.year == 2021].dropna()  # noqa: PLR2004
    tail = year["Benchmark"] <= np.percentile(year["Benchmark"], 5)
    assert np.isclose(
        risk_module.get_marginal_expected_shortfall(rounding=10).loc["2021", "AAPL"],
        year.loc[tail, "AAPL"].mean(),
    )


def test_get_srisk(recorder, risk_module):
    recorder.capture(risk_module.get_srisk())
    recorder.capture(
        risk_module.get_srisk(
            ticker="AAPL", decline=0.3, capital_ratio=0.055, period="quarterly"
        )
    )

    # SRISK of AAPL at the end of 2022 from the liabilities and shares of fiscal 2022 (which
    # ended in September 2022) and the last close of 2022.
    srisk = risk_module.get_srisk(rounding=10).loc["2022"]
    liabilities, _, shares, _ = risk_module._capital_shortfall_function()
    close = risk_module._get_prices("Close")["AAPL"]
    close = close[close.index.year == 2022].dropna().iloc[-1]  # noqa: PLR2004
    market_cap = close * shares.loc["AAPL", "2022"]
    lrmes = 0.6 ** srisk[("Beta", "AAPL")] - 1
    assert np.isclose(
        srisk[("Total Liabilities", "AAPL")], liabilities.loc["AAPL", "2022"]
    )
    assert np.isclose(srisk[("Market Cap", "AAPL")], market_cap)
    assert np.isclose(srisk[("LRMES", "AAPL")], lrmes)
    assert np.isclose(
        srisk[("SRISK", "AAPL")],
        0.08 * liabilities.loc["AAPL", "2022"] - 0.92 * (1 + lrmes) * market_cap,
    )


def test_get_srisk_takes_the_market_by_its_ticker(risk_module, monkeypatch):
    # The benchmark named as it trades (SPY) is the "Benchmark" column.
    pd.testing.assert_frame_equal(
        risk_module.get_srisk(market_ticker="SPY"), risk_module.get_srisk()
    )
    pd.testing.assert_frame_equal(
        risk_module.get_marginal_expected_shortfall(market_ticker="spy"),
        risk_module.get_marginal_expected_shortfall(),
    )

    # A market outside the Toolkit is retrieved for the same dates; here it has the returns of
    # the benchmark, which is then left out of the firms as a market index.
    requested = []

    def other_returns(tickers, period):
        requested.append((tickers, period))
        returns = risk_module._historical_data[period]["Return"]
        return returns[["Benchmark"]].set_axis(tickers, axis=1)

    monkeypatch.setattr(risk_module, "_other_returns_function", other_returns)
    pd.testing.assert_frame_equal(
        risk_module.get_srisk(market_ticker="^GSPC"), risk_module.get_srisk()
    )
    assert requested == [(["^GSPC"], "daily")]

    # Without a way to retrieve it, the missing market is reported.
    monkeypatch.setattr(risk_module, "_other_returns_function", None)
    assert risk_module.get_srisk(market_ticker="^GSPC").empty


def test_get_srisk_takes_a_fiscal_period_that_ends_on_a_weekend(risk_module):
    # A fiscal quarter that ends on Saturday 31 December 2022, after the last trading day of
    # the year, still belongs to 2022.
    liabilities, liability_dates, shares, share_dates = (
        risk_module._capital_shortfall_function()
    )
    weekend = liability_dates.copy()
    weekend.loc["AAPL", "2022"] = pd.Timestamp("2022-12-31")
    shifted = share_dates.copy()
    shifted.loc["AAPL", "2022"] = pd.Timestamp("2022-12-31")
    original = risk_module._capital_shortfall_function
    risk_module._capital_shortfall_function = lambda: (
        liabilities,
        weekend,
        shares,
        shifted,
    )
    try:
        srisk = risk_module.get_srisk(rounding=None).loc["2022"]
    finally:
        risk_module._capital_shortfall_function = original

    assert srisk[("Total Liabilities", "AAPL")] == liabilities.loc["AAPL", "2022"]
    close = risk_module._get_prices("Close")["AAPL"]
    close = close[close.index.year == 2022].dropna().iloc[-1]  # noqa: PLR2004
    assert np.isclose(srisk[("Market Cap", "AAPL")], close * shares.loc["AAPL", "2022"])


def test_the_marginal_expected_shortfall_of_a_stock_ignores_a_cryptocurrency_next_to_it():
    mixed = _weekend_risk()
    alone = _weekend_risk(tickers=("SPY",))
    mixed._tickers_source = ["BTC", "SPY", "Benchmark"]
    alone._tickers_source = ["SPY", "Benchmark"]

    pd.testing.assert_series_equal(
        mixed.get_marginal_expected_shortfall(rounding=None)["SPY"],
        alone.get_marginal_expected_shortfall(rounding=None)["SPY"],
    )
    # The cryptocurrency's weekend returns are compounded into the Monday of the market.
    assert mixed.get_marginal_expected_shortfall(rounding=None)["BTC"].notna().all()


def test_get_downside_deviation_semi_deviation(recorder, risk_module):
    recorder.capture(
        risk_module.get_downside_deviation(minimum_acceptable_return="mean")
    )
    recorder.capture(
        risk_module.get_downside_deviation(
            period="monthly", rolling=6, minimum_acceptable_return="mean"
        )
    )


def test_get_upside_deviation(recorder, risk_module):
    recorder.capture(risk_module.get_upside_deviation())
    recorder.capture(
        risk_module.get_upside_deviation(period="yearly", within_period=True)
    )
    recorder.capture(
        risk_module.get_upside_deviation(minimum_acceptable_return=0.001, rolling=20)
    )


def test_get_volatility_skewness(recorder, risk_module):
    recorder.capture(risk_module.get_volatility_skewness())
    recorder.capture(
        risk_module.get_volatility_skewness(
            period="yearly", within_period=True, method="variability"
        )
    )
    recorder.capture(risk_module.get_volatility_skewness(period="weekly", rolling=26))


def test_get_partial_moment(recorder, risk_module):
    recorder.capture(risk_module.get_partial_moment())
    recorder.capture(
        risk_module.get_partial_moment(period="yearly", within_period=True, order=0)
    )
    recorder.capture(
        risk_module.get_partial_moment(
            order=2, minimum_acceptable_return="mean", side="upper", root=True
        )
    )
    recorder.capture(
        risk_module.get_partial_moment(
            period="yearly", within_period=True, growth=True, order=3
        )
    )
    recorder.capture(risk_module.get_partial_moment(period="weekly", rolling=13))


def test_get_central_moment(recorder, risk_module):
    recorder.capture(risk_module.get_central_moment(rounding=12))
    recorder.capture(
        risk_module.get_central_moment(
            period="yearly", within_period=True, order=4, rounding=12
        )
    )
    recorder.capture(
        risk_module.get_central_moment(period="monthly", rolling=12, rounding=10)
    )


def test_get_worst_realization(recorder, risk_module):
    recorder.capture(risk_module.get_worst_realization())
    recorder.capture(
        risk_module.get_worst_realization(period="yearly", within_period=True)
    )
    recorder.capture(risk_module.get_worst_realization(period="weekly", rolling=4))


def test_get_gini_mean_difference(recorder, risk_module):
    recorder.capture(risk_module.get_gini_mean_difference())
    recorder.capture(
        risk_module.get_gini_mean_difference(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(risk_module.get_gini_mean_difference(period="monthly", rolling=6))


def test_get_tail_gini(recorder, risk_module):
    recorder.capture(risk_module.get_tail_gini())
    recorder.capture(
        risk_module.get_tail_gini(period="yearly", within_period=True, alpha=0.1)
    )
    recorder.capture(risk_module.get_tail_gini(rolling=60, points=50))


def test_get_risk_range(recorder, risk_module):
    recorder.capture(risk_module.get_risk_range())
    recorder.capture(risk_module.get_risk_range(measure="var", alpha=0.25))
    recorder.capture(
        risk_module.get_risk_range(
            period="yearly", within_period=True, measure="cvar", beta=0.1
        )
    )
    recorder.capture(risk_module.get_risk_range(measure="tail_gini", rolling=60))


def test_get_l_moment(recorder, risk_module):
    recorder.capture(risk_module.get_l_moment())
    recorder.capture(
        risk_module.get_l_moment(
            period="yearly", within_period=True, order=4, ratio=True
        )
    )
    recorder.capture(
        risk_module.get_l_moment(period="monthly", rolling=12, order=3, ratio=True)
    )


def test_get_l_moment_risk_measure(recorder, risk_module):
    recorder.capture(risk_module.get_l_moment_risk_measure())
    recorder.capture(
        risk_module.get_l_moment_risk_measure(
            period="yearly", within_period=True, method="crra"
        )
    )
    recorder.capture(
        risk_module.get_l_moment_risk_measure(order=6, method="mss", rolling=120)
    )


def test_get_skewness_kurtosis_ratio(recorder, risk_module):
    recorder.capture(risk_module.get_skewness_kurtosis_ratio())
    recorder.capture(
        risk_module.get_skewness_kurtosis_ratio(period="yearly", within_period=True)
    )
    recorder.capture(
        risk_module.get_skewness_kurtosis_ratio(period="monthly", rolling=12)
    )


def test_get_value_at_risk_kernel(recorder, risk_module):
    recorder.capture(risk_module.get_value_at_risk(method="kernel"))
    recorder.capture(
        risk_module.get_value_at_risk(
            period="yearly", within_period=True, method="kernel", alpha=0.01
        )
    )


def test_get_conditional_value_at_risk_weighted(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_value_at_risk(method="kernel"))
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            alpha=[0.05, 0.01], alpha_weights=[0.5, 0.5]
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="weekly", rolling=20, alpha=[0.1, 0.05, 0.025]
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="yearly",
            within_period=True,
            method="gaussian",
            alpha=[0.05, 0.01],
            alpha_weights=[0.75, 0.25],
        )
    )


def test_get_entropic_value_at_risk_historic(recorder, risk_module):
    recorder.capture(risk_module.get_entropic_value_at_risk(method="historic"))
    recorder.capture(
        risk_module.get_entropic_value_at_risk(
            period="yearly", within_period=True, method="historic", growth=True
        )
    )
    recorder.capture(
        risk_module.get_entropic_value_at_risk(period="weekly", rolling=26)
    )
    recorder.capture(
        risk_module.get_entropic_value_at_risk(rolling=60, method="historic", alpha=0.1)
    )


def test_get_entropic_risk_measure(recorder, risk_module):
    recorder.capture(risk_module.get_entropic_risk_measure())
    recorder.capture(
        risk_module.get_entropic_risk_measure(
            period="yearly", within_period=True, risk_aversion=0.01
        )
    )
    recorder.capture(
        risk_module.get_entropic_risk_measure(
            period="weekly", rolling=26, risk_aversion=0.05
        )
    )


def test_get_drawdown_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_drawdown_at_risk())
    recorder.capture(
        risk_module.get_drawdown_at_risk(
            period="yearly", within_period=True, alpha=0.1, compound_returns=False
        )
    )
    recorder.capture(risk_module.get_drawdown_at_risk(rolling=60))


def test_get_entropic_drawdown_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_entropic_drawdown_at_risk())
    recorder.capture(
        risk_module.get_entropic_drawdown_at_risk(
            period="yearly", within_period=True, growth=True
        )
    )
    recorder.capture(
        risk_module.get_entropic_drawdown_at_risk(
            period="weekly", rolling=26, compound_returns=False
        )
    )


def test_get_average_drawdown(recorder, risk_module):
    recorder.capture(risk_module.get_average_drawdown())
    recorder.capture(risk_module.get_average_drawdown(method="observation"))
    recorder.capture(
        risk_module.get_average_drawdown(
            period="yearly", within_period=True, compound_returns=False
        )
    )
    recorder.capture(risk_module.get_average_drawdown(rolling=60))


def test_get_drawdown_statistics(recorder, risk_module):
    recorder.capture(risk_module.get_drawdown_statistics())
    recorder.capture(risk_module.get_drawdown_statistics(period="weekly"))


def test_get_maximum_drawdown_rolling_and_uncompounded(recorder, risk_module):
    recorder.capture(risk_module.get_maximum_drawdown(rolling=60))
    recorder.capture(risk_module.get_maximum_drawdown(compound_returns=False))
    recorder.capture(
        risk_module.get_maximum_drawdown(
            period="yearly", within_period=True, compound_returns=False
        )
    )


def test_drawdown_measures_uncompounded(recorder, risk_module):
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(compound_returns=False)
    )
    recorder.capture(
        risk_module.get_conditional_drawdown_at_risk(rolling=60, compound_returns=False)
    )
    recorder.capture(risk_module.get_ulcer_index(window=None, compound_returns=False))


def test_get_risk_range_evar(recorder, risk_module):
    recorder.capture(risk_module.get_risk_range(measure="evar"))
    recorder.capture(
        risk_module.get_risk_range(
            period="yearly", within_period=True, measure="evar", beta=0.1
        )
    )


def test_get_srisk_with_a_window_longer_than_the_data_is_empty(risk_module):
    # Thirteen quarters of returns can't fill a window of twenty: the result is empty (the
    # error is logged), not an AttributeError on an index without periods.
    assert risk_module.get_srisk(period="quarterly", rolling=20).empty
    assert not risk_module.get_srisk(period="yearly", rolling=2).empty


def test_get_garch_family_variants(recorder, risk_module):
    recorder.capture(risk_module.get_garch(method="aparch", distribution="t"))
    recorder.capture(risk_module.get_garch(method="component", period="weekly"))
    recorder.capture(
        risk_module.get_garch(
            method="component", period="weekly", output="long_run_variance"
        )
    )
    recorder.capture(
        risk_module.get_garch(mean="arma", output="standardized_residuals")
    )
    with pytest.raises(InvalidArgumentError, match="distribution must be one of"):
        risk_module.get_garch(distribution="cauchy")
    with pytest.raises(InvalidArgumentError):
        risk_module.get_garch(method="gjr", output="long_run_variance")


def test_get_garch_forecast_family_variants(recorder, risk_module):
    recorder.capture(risk_module.get_garch_forecast(method="igarch"))
    recorder.capture(risk_module.get_garch_forecast(method="gjr", distribution="skewt"))


def test_get_garch_parameters_family_variants(recorder, risk_module):
    recorder.capture(
        risk_module.get_garch_parameters(method="tgarch", distribution="ged")
    )
    recorder.capture(risk_module.get_garch_parameters(method="igarch"))
    recorder.capture(
        risk_module.get_garch_parameters(
            distribution="johnson_su", mean="constant", diagnostics=True
        )
    )


def test_get_news_impact_curve(recorder, risk_module):
    recorder.capture(risk_module.get_news_impact_curve())
    recorder.capture(risk_module.get_news_impact_curve(method="gjr"))


def test_get_garch_simulation(recorder, risk_module):
    recorder.capture(risk_module.get_garch_simulation(simulations=1000))
    recorder.capture(
        risk_module.get_garch_simulation(
            simulations=1000, output="cumulative_returns", distribution="t"
        )
    )


def test_get_garch_figarch(recorder, risk_module):
    recorder.capture(risk_module.get_garch(method="figarch"))
    recorder.capture(risk_module.get_garch_forecast(method="figarch"))
    # The standard errors move by about 1e-6 when the prices move by 1e-12 (the fit stops
    # within its tolerance), so the parameters are recorded at three decimals.
    recorder.capture(
        risk_module.get_garch_parameters(method="figarch", diagnostics=True, rounding=3)
    )


def test_get_garch_arfima_mean(recorder, risk_module):
    recorder.capture(risk_module.get_garch(mean="arfima"))


def test_get_garch_simulation_bootstrap(recorder, risk_module):
    # Each bootstrap fit runs on resampled returns whose likelihood can be flat, so its fourth
    # decimal moves with the platform's floating point: recorded to three.
    recorder.capture(
        risk_module.get_garch_simulation(simulations=500, bootstrap_fits=2).round(3)
    )
    recorder.capture(
        risk_module.get_garch_simulation(
            simulations=500, bootstrap_fits=0, output="volatility"
        )
    )


def test_realized_garch_takes_the_daily_parkinson_variance(risk_module):
    # The realized GARCH fit itself (about 30 seconds for the three series) is tested on
    # simulated data in test_garch_fit.py; here its realized measure.
    measure = risk_module._get_realized_measure("realized", "daily")
    high = risk_module._get_prices("High")
    low = risk_module._get_prices("Low")
    expected = np.log(high / low) ** 2 / (4 * np.log(2))

    pd.testing.assert_frame_equal(measure, expected)
    assert risk_module._get_realized_measure("garch", "daily") is None
    with pytest.raises(InvalidArgumentError):
        risk_module.get_garch(method="realized", period="monthly")


def test_get_volatility_garman_klass_yang_zhang(recorder, risk_module):
    recorder.capture(
        risk_module.get_volatility(method="garman_klass_yang_zhang", period="monthly")
    )


def test_get_hurst_exponent(recorder, risk_module):
    recorder.capture(risk_module.get_hurst_exponent())
    recorder.capture(risk_module.get_hurst_exponent(q=[1, 2]))
    recorder.capture(risk_module.get_hurst_exponent(method="rescaled_range"))
    recorder.capture(
        risk_module.get_hurst_exponent(method="rescaled_range", corrected=False)
    )
    recorder.capture(risk_module.get_hurst_exponent(method="detrended_fluctuation"))
    recorder.capture(
        risk_module.get_hurst_exponent(method="detrended_fluctuation", rolling=252)
    )


def test_get_fractal_dimension(recorder, risk_module):
    recorder.capture(risk_module.get_fractal_dimension())
    recorder.capture(risk_module.get_fractal_dimension(method="katz"))
    recorder.capture(risk_module.get_fractal_dimension(method="petrosian"))
    recorder.capture(risk_module.get_fractal_dimension(rolling=100, period="weekly"))


def test_get_correlation_dimension(recorder, risk_module):
    recorder.capture(risk_module.get_correlation_dimension())
    recorder.capture(risk_module.get_correlation_dimension(embedding=2))


def test_get_lyapunov_exponent(recorder, risk_module):
    recorder.capture(risk_module.get_lyapunov_exponent())
    recorder.capture(risk_module.get_lyapunov_exponent(method="eckmann"))


def test_fractal_measures_use_log_returns_per_ticker(risk_module):
    alpha = risk_module.get_hurst_exponent(method="detrended_fluctuation", rounding=10)
    log_returns = np.log(risk_module._historical_data["daily"]["Adj Close"]).diff()

    for ticker in alpha.index:
        assert alpha[ticker] == pytest.approx(
            fractal_model.get_detrended_fluctuation(log_returns[ticker].dropna()),
            abs=1e-9,
        )


def test_get_var_backtest_all_tests(recorder, risk_module):
    result = risk_module.get_var_backtest(window=100, test="all")
    assert "Kupiec P-Value" in result.index
    assert "Tick Loss" in result.index
    recorder.capture(result)
    recorder.capture(
        risk_module.get_var_backtest(
            window=100, test=["duration", "mcneil_frey"], method="gaussian"
        )
    )


def test_get_var_backtest_garch(recorder, risk_module):
    recorder.capture(
        risk_module.get_var_backtest(
            method="garch", test="all", refit_every=63, window=504
        )
    )


def test_get_var_backtest_invalid_test(risk_module):
    with pytest.raises(InvalidArgumentError, match="'kupiec'"):
        risk_module.get_var_backtest(test="bad")


def test_get_value_at_risk_garch(recorder, risk_module):
    recorder.capture(risk_module.get_value_at_risk(method="garch"))
    recorder.capture(
        risk_module.get_value_at_risk(
            method="garch", garch_distribution="t", rolling=504, refit_every=63
        ).iloc[::21]
    )


def test_get_conditional_value_at_risk_garch(recorder, risk_module):
    recorder.capture(
        risk_module.get_conditional_value_at_risk(method="garch", garch_method="gjr")
    )


def test_get_acerbi_szekely_test_garch(recorder, risk_module):
    recorder.capture(
        risk_module.get_acerbi_szekely_test(method="garch", refit_every=63)
    )


def test_get_price_impact(recorder, risk_module):
    recorder.capture(risk_module.get_price_impact())
    for method in ("amihud", "hasbrouck", "roll"):
        recorder.capture(risk_module.get_price_impact(method=method))
    recorder.capture(risk_module.get_price_impact(period="quarterly", rolling=2))
    recorder.capture(risk_module.get_price_impact(method="amihud", scale=1e12))

    pd.testing.assert_frame_equal(
        risk_module.get_price_impact(), risk_module.get_price_impact(method="kyle")
    )
    with pytest.raises(InvalidArgumentError, match="'amihud'"):
        risk_module.get_price_impact(method="lambda")


def test_get_bekker_parkinson_volatility(recorder, risk_module):
    recorder.capture(risk_module.get_bekker_parkinson_volatility())
    recorder.capture(
        risk_module.get_bekker_parkinson_volatility(period="quarterly", rolling=2)
    )

    # Taking the spread out leaves at most the range-based (Parkinson) volatility on average.
    volatility = risk_module.get_bekker_parkinson_volatility(
        period="yearly", rounding=10
    )
    parkinson = risk_module.get_volatility(
        period="yearly", method="parkinson", rounding=10
    )
    assert (volatility.mean() <= parkinson.reindex(volatility.index).mean()).all()


def test_get_vpin(recorder, risk_module):
    result = risk_module.get_vpin()
    recorder.capture(result.dropna(how="all").iloc[::20])
    recorder.capture(
        risk_module.get_vpin(buckets=20, degrees_of_freedom=0.25)
        .dropna(how="all")
        .iloc[::20]
    )

    assert result.stack().dropna().between(0, 1).all()


def test_get_volatility_compounded(recorder, risk_module):
    compounded = risk_module.get_volatility(method="compounded")
    recorder.capture(compounded)
    recorder.capture(risk_module.get_volatility(method="compounded", period="monthly"))


def test_get_higher_moment_coherent_risk(recorder, risk_module):
    recorder.capture(risk_module.get_higher_moment_coherent_risk())
    recorder.capture(risk_module.get_higher_moment_coherent_risk(order=3, alpha=0.1))
    recorder.capture(
        risk_module.get_higher_moment_coherent_risk(period="weekly", rolling=52)
    )
    # Order 1 is the exact tail average of the returns, next to the interpolated historic CVaR.
    assert np.allclose(
        risk_module.get_higher_moment_coherent_risk(order=1, rounding=10),
        risk_module.get_conditional_value_at_risk(rounding=10),
        atol=2e-4,
    )


def test_get_expected_quadratic_shortfall(recorder, risk_module):
    recorder.capture(risk_module.get_expected_quadratic_shortfall(rounding=8))
    recorder.capture(
        risk_module.get_expected_quadratic_shortfall(
            period="monthly", within_period=True, rounding=8
        )
    )
    recorder.capture(
        risk_module.get_expected_quadratic_shortfall(rolling=60, alpha=0.1, rounding=8)
    )


def test_get_conditional_expected_drawdown(recorder, risk_module):
    recorder.capture(risk_module.get_conditional_expected_drawdown())
    recorder.capture(
        risk_module.get_conditional_expected_drawdown(
            period="yearly", within_period=True, window=10
        )
    )
    recorder.capture(risk_module.get_conditional_expected_drawdown(rolling=126))
    recorder.capture(
        risk_module.get_conditional_expected_drawdown(
            period="weekly", compound_returns=False
        )
    )
    # Every path lies within the history, so no path is deeper than the Maximum Drawdown.
    assert (
        risk_module.get_conditional_expected_drawdown(rounding=10)
        >= risk_module.get_maximum_drawdown(rounding=10) - 1e-12
    ).all()


def test_get_drawdown_table(recorder, risk_module):
    recorder.capture(risk_module.get_drawdown_table(top=3))
    recorder.capture(risk_module.get_drawdown_table(top=2, period="weekly"))


@pytest.mark.parametrize(
    "method",
    [
        "get_value_at_risk",
        "get_conditional_value_at_risk",
        "get_var_backtest",
        "get_acerbi_szekely_test",
    ],
)
def test_the_garch_options_are_checked_for_every_method(risk_module, method):
    for option in ("garch_method", "garch_distribution"):
        with pytest.raises(InvalidArgumentError, match=option):
            getattr(risk_module, method)(method="historic", **{option: "bogus"})


@pytest.mark.parametrize(
    "method",
    [
        "get_volatility",
        "get_value_at_risk",
        "get_maximum_drawdown",
        "get_skewness",
        "get_bid_ask_spread",
        "get_srisk",
    ],
)
def test_a_rolling_window_needs_two_periods(risk_module, method):
    for rolling in (1, 0, -3, 2.5):
        with pytest.raises(InvalidArgumentError, match="rolling is a number"):
            getattr(risk_module, method)(rolling=rolling)


def test_a_rolling_window_longer_than_the_data_is_logged_once(risk_module, caplog):
    with caplog.at_level("WARNING"):
        volatility = risk_module.get_volatility(period="monthly", rolling=1000)

    assert volatility.isna().all().all()
    assert caplog.text.count("is at least as long as the") == 1


def test_srisk_of_a_range_without_data_says_so(caplog):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    from financetoolkit import Toolkit

    after_the_data = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2023-06-01",
        end_date="2023-12-31",
        sleep_timer=False,
    )
    after_the_data._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )

    with caplog.at_level("ERROR"):
        assert after_the_data.risk.get_srisk().empty

    assert "The data holds no yearly period with daily returns" in caplog.text
    assert "None" not in caplog.text


# Robustness round 2: backtest windows, rolling minimums and constituents without data.


@pytest.mark.parametrize("method", ["get_var_backtest", "get_acerbi_szekely_test"])
def test_a_backtest_window_below_two_raises(risk_module, method):
    for window in (0, 1, 2.5):
        with pytest.raises(InvalidArgumentError, match="window is a number"):
            getattr(risk_module, method)(window=window)


def test_a_backtest_window_without_a_tail_observation_is_logged(risk_module, caplog):
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = risk_module.get_acerbi_szekely_test(window=2)

    assert caplog.text.count("holds no observation in the 0.05 tail") == 1
    # A CVaR of 0 in a two-return window gives NaN rather than an infinite statistic.
    assert not np.isinf(result.loc["Acerbi-Szekely Statistic"]).any()


def test_a_backtest_window_as_long_as_the_returns_is_logged(risk_module, caplog):
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = risk_module.get_var_backtest(window=2000)

    assert result.isna().all().all()
    assert caplog.text.count("no out-of-sample estimate is left") == 1


def test_the_rolling_window_counts_the_periods_with_a_return(risk_module, caplog):
    # The data starts on 2019-12-31: four years and three yearly returns.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        volatility = risk_module.get_volatility(period="yearly", rolling=4)

    assert volatility.isna().all().all()
    assert "the 3 yearly returns in the data" in caplog.text


@pytest.mark.parametrize(
    ("method", "arguments", "minimum"),
    [
        ("get_skewness", {"rolling": 2}, 3),
        ("get_kurtosis", {"rolling": 3}, 4),
        ("get_skewness_kurtosis_ratio", {"rolling": 3}, 4),
        ("get_hurst_exponent", {"rolling": 4}, 5),
        ("get_hurst_exponent", {"rolling": 100, "method": "rescaled_range"}, 120),
        ("get_fractal_dimension", {"rolling": 11}, 12),
        ("get_fractal_dimension", {"rolling": 2, "method": "katz"}, 3),
        ("get_conditional_expected_drawdown", {"rolling": 20}, 21),
    ],
)
def test_a_rolling_window_below_the_measure_minimum_raises(
    risk_module, method, arguments, minimum
):
    with pytest.raises(InvalidArgumentError, match=f"at least {minimum}"):
        getattr(risk_module, method)(**arguments)


def test_the_rolling_minimums_give_values(risk_module):
    assert risk_module.get_skewness(rolling=3).notna().any().any()
    assert risk_module.get_kurtosis(rolling=4).notna().any().any()
    assert risk_module.get_hurst_exponent(rolling=5).notna().any().any()
    assert risk_module.get_fractal_dimension(rolling=12).notna().any().any()


def test_a_portfolio_constituent_without_returns_is_named(caplog):
    from financetoolkit import Toolkit

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    historical = historical.drop(columns=["Return", "Cumulative Return"], level=0)
    for column in historical.columns.get_level_values(0).unique():
        historical[(column, "MSFT")] = np.nan
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )

    with caplog.at_level("WARNING", logger="financetoolkit"):
        marginal = toolkit.risk.get_marginal_value_at_risk()

    assert marginal.isna().all()
    assert "MSFT has no Return values" in caplog.text


# Robustness round 3: default backtest windows and per-ticker rolling windows.


@pytest.mark.parametrize("method", ["get_var_backtest", "get_acerbi_szekely_test"])
def test_a_default_backtest_window_as_long_as_the_returns_is_logged(
    risk_module, method, caplog
):
    # The default window is 20 yearly returns, while the data holds three.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = getattr(risk_module, method)(period="yearly")

    assert result.isna().all().all()
    assert caplog.text.count("A window of 20 yearly returns") == 1


def test_a_default_backtest_window_without_a_tail_observation_is_logged(
    risk_module, caplog
):
    # The default window of 20 monthly returns holds no observation in a 1% tail.
    with caplog.at_level("WARNING", logger="financetoolkit"):
        risk_module.get_var_backtest(period="monthly", alpha=0.01)

    assert caplog.text.count("holds no observation in the 0.01 tail") == 1


def test_a_rolling_window_names_the_tickers_with_fewer_returns(caplog):
    from financetoolkit import Toolkit

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle").drop(
        columns=["Return", "Cumulative Return"], level=0
    )
    # MSFT starts in June 2021, while AAPL and the Benchmark have the full range.
    before = historical.index < pd.Period("2021-06-01", "D")
    for column in historical.columns.get_level_values(0).unique():
        historical.loc[before, (column, "MSFT")] = np.nan
    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    with caplog.at_level("WARNING", logger="financetoolkit"):
        volatility = toolkit.risk.get_volatility(period="quarterly", rolling=8)

    assert volatility["MSFT"].isna().all()
    assert volatility["AAPL"].notna().any()
    assert caplog.text.count("the quarterly returns of MSFT (") == 1
    assert "AAPL (" not in caplog.text


def test_a_benchmark_shorter_than_the_rolling_window_is_named(caplog):
    from financetoolkit.risk.helpers import check_rolling

    years = pd.period_range("2019", "2023", freq="Y")
    returns = pd.DataFrame(
        {
            "AAPL": [np.nan, 0.1, 0.2, 0.3, 0.4],
            "MSFT": [np.nan, 0.1, 0.2, 0.3, 0.4],
            "Benchmark": np.nan,
        },
        index=years,
    )

    with caplog.at_level("WARNING", logger="financetoolkit"):
        check_rolling(2, {"yearly": pd.concat({"Return": returns}, axis=1)}, "yearly")

    assert "the yearly returns of Benchmark (0)" in caplog.text
    assert "AAPL (" not in caplog.text
