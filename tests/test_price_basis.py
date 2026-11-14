"""Tests of the price basis: market values at the traded close, consistent OHLC in the technicals,
no look-ahead in the option inputs and an expected market return in the CAPM."""

import numpy as np
import pandas as pd
import pytest


def test_market_cap_uses_the_traded_close(test_toolkit):
    market_cap = test_toolkit.ratios.get_market_cap(rounding=None)
    shares = test_toolkit._income_statement.loc[:, "Weighted Average Shares Diluted", :]
    close = test_toolkit.get_historical_data(period="yearly")["Close"]["AAPL"]
    year = pd.Period("2021", freq="Y")

    # 177.57 is Apple's close at the end of 2021; the dividend-adjusted price is lower.
    assert market_cap.loc["AAPL", year] == pytest.approx(
        close.loc[year] * shares.loc["AAPL", year], rel=1e-6
    )


def test_cost_of_equity_uses_a_market_risk_premium(test_toolkit):
    default = test_toolkit.models.get_weighted_average_cost_of_capital(rounding=None)
    given = test_toolkit.models.get_weighted_average_cost_of_capital(
        market_risk_premium=0.05, rounding=None
    )

    # The cost of equity no longer follows each year's benchmark return down (2022), and a
    # given premium replaces the benchmark's average one.
    assert (default.loc["AAPL"].loc["Cost of Equity"].dropna() > 0).all()
    assert (
        not given.loc["AAPL"]
        .loc["Cost of Equity"]
        .equals(default.loc["AAPL"].loc["Cost of Equity"])
    )


def test_bounded_indicators_stay_within_their_bounds(test_toolkit):
    williams = test_toolkit.technicals.get_williams_percent_r().dropna(how="all")
    stochastic = test_toolkit.technicals.get_stochastic_oscillator().dropna(how="all")

    assert williams.min().min() >= -100 - 1e-9 and williams.max().max() <= 1e-9
    assert stochastic.min().min() >= -1e-9 and stochastic.max().max() <= 100 + 1e-9


def test_option_volatility_only_uses_the_year_before(test_toolkit):
    options = test_toolkit.options
    returns = options._daily_historical["Return"]["AAPL"]
    date = returns.index[400]

    expected = returns.iloc[400 - 251 : 401].std() * np.sqrt(252)

    assert options._volatility.loc[date, "AAPL"] == pytest.approx(expected)
    # The dividend yield of a date is between zero and a plausible yield.
    dividend_yield = options._daily_dividend_yield.loc[date, "AAPL"]
    assert 0 <= dividend_yield < 0.05  # noqa: PLR2004
    assert isinstance(options._dividend_yield_on("AAPL", date), float)


def test_options_inputs_do_not_change_with_later_data(test_toolkit):
    options = test_toolkit.options
    date = options._daily_historical.index[300]
    truncated = options._daily_historical["Return"].loc[:date]

    assert options._volatility.loc[date, "MSFT"] == pytest.approx(
        truncated["MSFT"].iloc[-252:].std() * np.sqrt(252)
    )
    assert isinstance(pd.Period(date), pd.Period)


def test_fmp_adjusted_close_is_not_rounded_to_two_decimals():
    from financetoolkit import fmp_model

    days = pd.period_range("2026-11-05", periods=4, freq="D")
    close = pd.Series([1.1219, 1.1141, 1.1224, 1.1165], index=days)
    rounded = close.round(2)

    # A currency pair pays no dividends: the adjusted close is the close itself.
    assert fmp_model._unrounded_adjusted_close(close, rounded, None).equals(close)
    no_dividends = pd.Series(0.0, index=days)
    assert fmp_model._unrounded_adjusted_close(close, rounded, no_dividends).equals(
        close
    )

    # A dividend of 0.2775 on the third day: the closes before it are multiplied by
    # 1 - 0.2775 / 24.87, so the ex-dividend return is 22.18 / (24.87 - 0.2775) - 1.
    shares = pd.Series([24.47, 24.87, 22.18, 22.49], index=days)
    dividends = pd.Series([0.0, 0.0, 0.2775, 0.0], index=days)
    adjusted = fmp_model._unrounded_adjusted_close(
        shares, shares.round(2) * 0.99, dividends
    )
    assert adjusted.iloc[2:].tolist() == [22.18, 22.49]
    assert adjusted.iloc[2] / adjusted.iloc[1] - 1 == pytest.approx(
        22.18 / (24.87 - 0.2775) - 1
    )
    assert adjusted.iloc[1] / adjusted.iloc[0] == pytest.approx(24.87 / 24.47)
