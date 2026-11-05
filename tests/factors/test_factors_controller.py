"""Factors Controller Tests"""

# pylint: disable=missing-function-docstring

import numpy as np
import pandas as pd

from financetoolkit.factors.factors_controller import Factors

TICKERS = [f"T{number}" for number in range(10)]


def daily_prices(seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    returns = rng.normal(0, 0.01, (len(dates), len(TICKERS) + 1))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.PeriodIndex(dates, freq="D"),
        columns=[*TICKERS, "Benchmark"],
    )
    return pd.concat({"Adj Close": prices}, axis=1)


class FakeRatios:
    def __init__(self, factor):
        self.factor = factor
        self.calls = []

    def get_price_to_earnings_ratio(self, show_daily=False):
        self.calls.append(show_daily)
        return self.factor


def factors_module(factor_modules=None, period_end_dates=None, tickers=None):
    historical = daily_prices()
    return Factors(
        tickers=[*(tickers or TICKERS), "Benchmark"],
        historical_data={"daily": historical},
        factor_modules=lambda: factor_modules or {},
        period_end_dates=lambda: period_end_dates,
        rounding=None,
        start_date="2020-01-01",
        end_date="2022-12-31",
    )


def future_monthly_return_factor():
    """A factor that knows each ticker's return over the next month: an IC of one."""
    prices = daily_prices()["Adj Close"][TICKERS]
    prices.index = prices.index.to_timestamp()
    month_end = prices.groupby(prices.index.to_period("M")).last()
    forward = month_end.shift(-1) / month_end - 1
    # Spread each month's value over its days, so the month-end sample picks it up.
    factor = forward.reindex(prices.index.to_period("M"))
    factor.index = prices.index
    return factor


def test_a_perfect_factor_has_an_ic_of_one_and_leaves_the_benchmark_out():
    module = factors_module()
    factor = future_monthly_return_factor()
    factor["Benchmark"] = 1.0

    ic = module.get_information_coefficient(factor, horizons=1)
    summary = module.get_information_coefficient(factor, horizons=1, summary=True)

    assert list(ic.columns) == ["1 Month"]
    assert np.allclose(ic["1 Month"].dropna(), 1)
    assert summary.loc["Mean IC", "1 Month"] == 1


def test_the_quantile_returns_of_a_perfect_factor_rise_with_the_quantile():
    module = factors_module()

    returns = module.get_quantile_returns(
        future_monthly_return_factor(), quantiles=5, horizons=[1]
    )

    assert returns["1 Month"].drop("Top Minus Bottom").is_monotonic_increasing
    assert returns.loc["Top Minus Bottom", "1 Month"] > 0


def test_a_named_ratio_is_calculated_daily():
    ratios = FakeRatios(future_monthly_return_factor())
    module = factors_module(factor_modules={"ratios": ratios})

    ic = module.get_information_coefficient("get_price_to_earnings_ratio", horizons=1)

    assert ratios.calls == [True]
    assert np.allclose(ic["1 Month"].dropna(), 1)


def test_statement_figures_apply_from_the_period_end_only():
    # A yearly figure per ticker: the 2021 figure may only be used after its period end.
    periods = pd.PeriodIndex(["2020", "2021"], freq="Y")
    period_factor = pd.DataFrame(
        [[float(number), float(-number)] for number in range(10)],
        index=TICKERS,
        columns=periods,
    )
    period_end_dates = pd.DataFrame(
        [[pd.Timestamp("2020-12-31"), pd.Timestamp("2021-12-31")]] * 10,
        index=TICKERS,
        columns=periods,
    )
    module = factors_module(period_end_dates=period_end_dates)

    autocorrelation = module.get_factor_rank_autocorrelation(period_factor)

    # The ranking reverses once, in December 2021 when the 2021 figures become known, and
    # not at the start of 2021.
    assert autocorrelation.loc["2021-12", "Rank Autocorrelation"] == -1
    assert (autocorrelation.drop(pd.Period("2021-12", freq="M")) == 1).all().all()
    assert autocorrelation.index[0] == pd.Period("2021-01", freq="M")


def test_turnover_follows_the_quantiles():
    module = factors_module()

    turnover = module.get_quantile_turnover(future_monthly_return_factor(), quantiles=5)

    assert list(turnover.columns) == [f"Quantile {number}" for number in range(1, 6)]
    assert ((turnover >= 0) & (turnover <= 1)).all().all()


def test_a_factor_of_two_tickers_returns_no_result():
    module = factors_module(tickers=TICKERS[:2])
    factor = future_monthly_return_factor()[TICKERS[:2]]

    assert module.get_information_coefficient(factor).empty


def test_an_unknown_method_name_returns_no_result():
    module = factors_module(factor_modules={"ratios": FakeRatios(None)})

    assert module.get_information_coefficient("get_unknown_ratio").empty
