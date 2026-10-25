"""Risk Controller Tests""" ""
# pylint: disable=missing-function-docstring

import copy

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model

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
            period="monthly", within_period=False, distribution="evt"
        )
    )
    recorder.capture(
        risk_module.get_value_at_risk(
            period="monthly", within_period=False, distribution="cornish-fisher"
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
            period="monthly", within_period=False, distribution="cornish-fisher"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, distribution="evt"
        )
    )
    recorder.capture(
        risk_module.get_conditional_value_at_risk(
            period="monthly", within_period=False, distribution="studentt"
        )
    )


def test_get_conditional_value_at_risk_studentt_differs_from_var(recorder, risk_module):
    # Regression test: studentt used to dispatch to VaR, so CVaR must be as extreme.
    var = risk_module.get_value_at_risk(
        period="monthly", within_period=False, distribution="studentt"
    )
    cvar = risk_module.get_conditional_value_at_risk(
        period="monthly", within_period=False, distribution="studentt"
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


def test_get_garch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_garch_forecast())
    recorder.capture(risk_module.get_garch_forecast(period="monthly"))
    recorder.capture(risk_module.get_garch_forecast(period="yearly", growth=True))
    recorder.capture(
        risk_module.get_garch_forecast(period="yearly", growth=True, lag=[1, 2, 3])
    )


def test_get_garch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_garch_parameters())
    recorder.capture(risk_module.get_garch_parameters(period="monthly"))


def test_get_gjr_garch(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch())
    recorder.capture(risk_module.get_gjr_garch(period="monthly"))
    recorder.capture(risk_module.get_gjr_garch(period="yearly", growth=True))


def test_get_gjr_garch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch_forecast())
    recorder.capture(risk_module.get_gjr_garch_forecast(period="monthly"))


def test_get_gjr_garch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_gjr_garch_parameters())
    recorder.capture(risk_module.get_gjr_garch_parameters(period="monthly"))


@EGARCH_PLATFORM_DRIFT
def test_get_egarch(recorder, risk_module):
    recorder.capture(risk_module.get_egarch())
    recorder.capture(risk_module.get_egarch(period="monthly"))
    recorder.capture(risk_module.get_egarch(period="yearly", growth=True))


@EGARCH_PLATFORM_DRIFT
def test_get_egarch_forecast(recorder, risk_module):
    recorder.capture(risk_module.get_egarch_forecast())
    recorder.capture(risk_module.get_egarch_forecast(period="monthly"))


@EGARCH_PLATFORM_DRIFT
def test_get_egarch_parameters(recorder, risk_module):
    recorder.capture(risk_module.get_egarch_parameters())
    recorder.capture(risk_module.get_egarch_parameters(period="monthly"))


def test_get_var_backtest(recorder, risk_module):
    recorder.capture(risk_module.get_var_backtest(window_size=100))
    recorder.capture(
        risk_module.get_var_backtest(window_size=100, distribution="gaussian")
    )
    recorder.capture(risk_module.get_var_backtest(window_size=100, test="kupiec"))
    recorder.capture(
        risk_module.get_var_backtest(window_size=100, test="christoffersen")
    )


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
    recorder.capture(risk_module.get_ulcer_index(rolling=5))
    recorder.capture(risk_module.get_ulcer_index(period="monthly"))
    recorder.capture(risk_module.get_ulcer_index(growth=True))
    recorder.capture(risk_module.get_ulcer_index(growth=True, lag=[1, 2, 3]))


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
    # @handle_errors returns an empty Series rather than propagating the ValueError.
    result = risk_module.get_har_rv_forecast(estimator="bad")
    assert result.empty


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
        risk_module.get_marginal_value_at_risk(period="yearly", distribution="gaussian")
    )


def test_get_component_value_at_risk(recorder, risk_module):
    recorder.capture(risk_module.get_component_value_at_risk(period="yearly"))
    recorder.capture(
        risk_module.get_component_value_at_risk(
            weights={"AAPL": 0.7, "MSFT": 0.3}, period="yearly"
        )
    )
    recorder.capture(
        risk_module.get_component_value_at_risk(
            period="yearly", distribution="gaussian"
        )
    )


def test_get_acerbi_szekely_test(recorder, risk_module):
    recorder.capture(risk_module.get_acerbi_szekely_test(window_size=100))
    recorder.capture(
        risk_module.get_acerbi_szekely_test(window_size=100, distribution="gaussian")
    )


def _risk_with_late_listing(risk_module, listing_date="2021-01-01"):
    """A copy of the Risk module in which MSFT has no data before `listing_date`."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.risk.risk_controller import Risk  # noqa: PLC0415

    historical_data = {}
    for period, period_data in risk_module._historical_data.items():
        data = period_data.copy()
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

    full = risk_module.get_var_backtest(window_size=100, rounding=10)
    shortened = late_listing.get_var_backtest(window_size=100, rounding=10)

    assert full["AAPL"].equals(shortened["AAPL"])
    assert shortened["MSFT"].notna().all()


def test_get_ulcer_index_weekly_has_values(risk_module):
    ulcer_index = risk_module.get_ulcer_index(period="weekly")

    assert ulcer_index.iloc[1:].notna().all().all()


def test_get_value_at_risk_cf_is_not_a_distribution(risk_module):
    # "cornish-fisher" is the one name of the Cornish-Fisher VaR.
    result = risk_module.get_value_at_risk(distribution="cf")

    assert result.empty


def test_get_hurst_exponent_uses_log_prices(risk_module):
    hurst_exponent = risk_module.get_hurst_exponent(rounding=10)
    log_prices = np.log(risk_module._historical_data["daily"]["Adj Close"])

    for ticker in hurst_exponent.index:
        assert hurst_exponent[ticker] == pytest.approx(
            risk_model.get_hurst_exponent(log_prices[ticker]), abs=1e-9
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
