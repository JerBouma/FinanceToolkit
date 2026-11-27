"""Options Controller Tests""" ""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.utilities.error_model import InvalidArgumentError


def test_get_binomial_model(recorder, options_module):
    recorder.capture(options_module.get_binomial_model())
    recorder.capture(options_module.get_binomial_model(put_option=True))
    recorder.capture(options_module.get_binomial_model(american_option=True))
    recorder.capture(
        options_module.get_binomial_model(
            strike_price_range=0.10,
            strike_step_size=2,
            risk_free_rate=0.01,
            dividend_yield=0.005,
            time_steps=2,
            rounding=2,
        )
    )


def test_get_black_scholes_model(recorder, options_module):
    recorder.capture(options_module.get_black_scholes_model())
    recorder.capture(options_module.get_black_scholes_model(put_option=True))
    recorder.capture(
        options_module.get_black_scholes_model(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_stock_price_simulation(recorder, options_module):
    recorder.capture(options_module.get_stock_price_simulation())
    recorder.capture(options_module.get_stock_price_simulation(time_steps=5))


def test_the_stock_price_simulation_takes_a_fraction_of_a_year(options_module):
    half_year = options_module.get_stock_price_simulation(
        time_to_expiration=0.5, time_steps=2
    )

    start, end = half_year.columns[0], half_year.columns[-1]
    assert (end.to_timestamp() - start.to_timestamp()).days in (182, 183)


def test_get_implied_volatility(recorder, options_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live option chain data")
    recorder.capture(options_module.get_implied_volatility())
    recorder.capture(options_module.get_implied_volatility(put_option=True))


def test_get_option_chains(recorder, options_module, live_mode):
    if not live_mode:
        pytest.skip("Requires --live flag for live option chain data")
    recorder.capture(options_module.get_option_chains())
    recorder.capture(options_module.get_option_chains(put_option=True))


def test_collect_all_greeks(recorder, options_module):
    recorder.capture(options_module.collect_all_greeks())
    recorder.capture(options_module.collect_all_greeks(put_option=True))
    recorder.capture(
        options_module.collect_all_greeks(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_collect_first_order_greeks(recorder, options_module):
    recorder.capture(options_module.collect_first_order_greeks())
    recorder.capture(options_module.collect_first_order_greeks(put_option=True))
    recorder.capture(
        options_module.collect_first_order_greeks(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_delta(recorder, options_module):
    recorder.capture(options_module.get_delta())
    recorder.capture(options_module.get_delta(put_option=True))
    recorder.capture(
        options_module.get_delta(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_dual_delta(recorder, options_module):
    recorder.capture(options_module.get_dual_delta())
    recorder.capture(options_module.get_dual_delta(put_option=True))
    recorder.capture(
        options_module.get_dual_delta(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_vega(recorder, options_module):
    recorder.capture(options_module.get_vega())
    recorder.capture(
        options_module.get_vega(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_theta(recorder, options_module):
    recorder.capture(options_module.get_theta())
    recorder.capture(options_module.get_theta(put_option=True))
    recorder.capture(
        options_module.get_theta(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_rho(recorder, options_module):
    recorder.capture(options_module.get_rho())
    recorder.capture(options_module.get_rho(put_option=True))
    recorder.capture(
        options_module.get_rho(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_epsilon(recorder, options_module):
    recorder.capture(options_module.get_epsilon())
    recorder.capture(options_module.get_epsilon(put_option=True))
    recorder.capture(
        options_module.get_epsilon(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_lambda(recorder, options_module):
    recorder.capture(options_module.get_lambda())
    recorder.capture(options_module.get_lambda(put_option=True))
    recorder.capture(
        options_module.get_lambda(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_collect_second_order_greeks(recorder, options_module):
    recorder.capture(options_module.collect_second_order_greeks())
    recorder.capture(options_module.collect_second_order_greeks(put_option=True))
    recorder.capture(
        options_module.collect_second_order_greeks(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
        )
    )


def test_get_gamma(recorder, options_module):
    recorder.capture(options_module.get_gamma())
    recorder.capture(
        options_module.get_gamma(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_dual_gamma(recorder, options_module):
    recorder.capture(options_module.get_dual_gamma())
    recorder.capture(
        options_module.get_dual_gamma(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_vanna(recorder, options_module):
    recorder.capture(options_module.get_vanna())
    recorder.capture(
        options_module.get_vanna(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_charm(recorder, options_module):
    recorder.capture(options_module.get_charm())
    recorder.capture(options_module.get_charm(put_option=True))
    recorder.capture(
        options_module.get_charm(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_vomma(recorder, options_module):
    recorder.capture(options_module.get_vomma())
    recorder.capture(
        options_module.get_vomma(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_vera(recorder, options_module):
    recorder.capture(options_module.get_vera())
    recorder.capture(
        options_module.get_vera(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.01,
            rounding=2,
        )
    )


def test_get_veta(recorder, options_module):
    recorder.capture(options_module.get_veta())
    recorder.capture(
        options_module.get_veta(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_partial_derivative(recorder, options_module):
    recorder.capture(options_module.get_partial_derivative())
    recorder.capture(
        options_module.get_partial_derivative(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
        )
    )


def test_collect_third_order_greeks(recorder, options_module):
    recorder.capture(options_module.collect_third_order_greeks())
    recorder.capture(
        options_module.collect_third_order_greeks(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
        )
    )


def test_get_speed(recorder, options_module):
    recorder.capture(options_module.get_speed())
    recorder.capture(
        options_module.get_speed(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_zomma(recorder, options_module):
    recorder.capture(options_module.get_zomma())
    recorder.capture(
        options_module.get_zomma(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_color(recorder, options_module):
    recorder.capture(options_module.get_color())
    recorder.capture(
        options_module.get_color(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_ultima(recorder, options_module):
    recorder.capture(options_module.get_ultima())
    recorder.capture(
        options_module.get_ultima(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_put_call_parity(recorder, options_module):
    recorder.capture(options_module.get_put_call_parity())
    recorder.capture(
        options_module.get_put_call_parity(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=6,
        )
    )


def test_get_garman_kohlhagen(recorder, options_module):
    recorder.capture(options_module.get_garman_kohlhagen())
    recorder.capture(options_module.get_garman_kohlhagen(put_option=True))
    recorder.capture(
        options_module.get_garman_kohlhagen(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            foreign_risk_free_rate=0.02,
            rounding=2,
        )
    )


def test_get_binary_option(recorder, options_module):
    recorder.capture(options_module.get_binary_option())
    recorder.capture(options_module.get_binary_option(put_option=True))
    recorder.capture(
        options_module.get_binary_option(
            option_type="asset-or-nothing",
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_monte_carlo_option_price(recorder, options_module):
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            simulations=1_000, time_steps=10, seed=42
        )
    )
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            put_option=True, simulations=1_000, time_steps=10, seed=42
        )
    )
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            simulations=1_000,
            time_steps=10,
            seed=42,
            rounding=2,
        )
    )


def test_get_monte_carlo_option_price_reproducible(options_module):
    first = options_module.get_monte_carlo_option_price(
        simulations=500, time_steps=5, seed=123
    )
    second = options_module.get_monte_carlo_option_price(
        simulations=500, time_steps=5, seed=123
    )

    pd.testing.assert_frame_equal(first, second)


def test_get_monte_carlo_option_price_standard_error(options_module):
    prices, standard_errors = options_module.get_monte_carlo_option_price(
        simulations=500, time_steps=5, seed=123, show_standard_error=True
    )

    assert prices.shape == standard_errors.shape
    assert (standard_errors >= 0).all().all()


def test_get_barrier_option(recorder, options_module):
    recorder.capture(options_module.get_barrier_option())
    recorder.capture(options_module.get_barrier_option(put_option=True))
    recorder.capture(
        options_module.get_barrier_option(
            barrier_percentage=1.2,
            barrier_direction="up",
            knock_type="in",
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_asian_option(recorder, options_module):
    recorder.capture(options_module.get_asian_option())
    recorder.capture(options_module.get_asian_option(put_option=True))
    recorder.capture(
        options_module.get_asian_option(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_strategy_payoff(recorder, options_module):
    straddle_legs = [
        {"strike_price": 150, "put_option": False, "position": "long", "premium": 8},
        {"strike_price": 150, "put_option": True, "position": "long", "premium": 6},
    ]
    recorder.capture(options_module.get_strategy_payoff(legs=straddle_legs))

    covered_call_legs = [
        {"instrument": "stock", "position": "long", "premium": 150},
        {
            "strike_price": 160,
            "put_option": False,
            "position": "short",
            "premium": 3,
        },
    ]
    recorder.capture(
        options_module.get_strategy_payoff(
            legs=covered_call_legs, stock_price_range=0.3, stock_price_step_size=5
        )
    )


def test_get_strategy_payoff_invalid_legs(options_module):
    # Legs that are not a strategy are a mistake in the call, so they raise and say why.
    with pytest.raises(InvalidArgumentError, match="at least one leg"):
        options_module.get_strategy_payoff(legs=[])
    with pytest.raises(InvalidArgumentError, match="strike_price"):
        options_module.get_strategy_payoff(legs=[{"put_option": False}])
    with pytest.raises(InvalidArgumentError, match="'bogus'"):
        options_module.get_strategy_payoff(legs=[{"bogus": 1}])
    with pytest.raises(TypeError):
        options_module.get_strategy_payoff(legs=[{"strike_price": "100"}])
    with pytest.raises(InvalidArgumentError, match="stock_price_step_size"):
        options_module.get_strategy_payoff(stock_price_step_size=-1)
    with pytest.raises(InvalidArgumentError, match="stock_price_range"):
        options_module.get_strategy_payoff(stock_price_range=1.5)


def test_stock_price_on_a_past_date_is_the_traded_close(options_module):
    simulation = options_module.get_stock_price_simulation(
        start_date="2021-06-01", time_steps=1, rounding=6
    )
    historical = options_module._daily_historical.loc["2021-06-01"]

    # The Adj Close of a past date is lowered by the dividends paid after it.
    assert simulation.loc[("AAPL", "U"), "2021-06-01"] == pytest.approx(
        historical.loc[("Close", "AAPL")]
    )
    assert historical.loc[("Close", "AAPL")] > historical.loc[("Adj Close", "AAPL")]


def _prices_with_dividends(dividends, split_date="2021-06-01"):
    import numpy as np

    dates = pd.bdate_range("2020-01-01", "2022-12-30")
    rng = np.random.default_rng(3)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates
    )

    # A 2-for-1 split: the history before it is divided by two in both price series, and
    # the dividends per share before it by two as well.
    split = dates < pd.Timestamp(split_date)
    close[split] /= 2
    close = close.round(2)

    # The adjusted close scales every close before an ex-date by (1 - Dividend / Close
    # before the ex-date), as the data providers do.
    factor = pd.Series(1.0, index=dates)
    for ex_date, dividend in dividends.items():
        position = dates.get_loc(pd.Timestamp(ex_date))
        adjusted_dividend = dividend / 2 if split[position] else dividend
        factor.iloc[:position] *= 1 - adjusted_dividend / close.iloc[position - 1]
    adjusted_close = (close * factor).round(2)

    index = dates.to_period("D")
    return (
        close.to_frame("ABC").set_axis(index),
        adjusted_close.to_frame("ABC").set_axis(index),
    )


def test_trailing_dividend_yield_is_the_dividends_of_the_year_over_the_close():
    from financetoolkit.options import options_model

    dividends = {
        date: 1.0
        for date in pd.date_range("2020-02-03", "2022-12-01", freq="QS-FEB")
        + pd.offsets.BDay(2)
    }
    close, adjusted_close = _prices_with_dividends(dividends)

    dividend_yield = options_model.get_trailing_dividend_yield(close, adjusted_close)

    # The dividends per share in today's shares: 0.50 before the split, 1.00 after it.
    paid = pd.Series(
        [
            value / 2 if pd.Timestamp(date) < pd.Timestamp("2021-06-01") else value
            for date, value in dividends.items()
        ],
        index=pd.DatetimeIndex(list(dividends)).to_period("D"),
    )
    for date in ["2021-03-01", "2021-08-02", "2022-06-15", "2022-12-30"]:
        moment = pd.Period(date, "D")
        window = paid[(paid.index > moment - 365) & (paid.index <= moment)]
        expected = window.sum() / close.loc[moment, "ABC"]
        assert dividend_yield.loc[moment, "ABC"] == pytest.approx(expected, abs=2e-4)

    # A dividend only enters the yield from its ex-date on: removing the last dividend
    # leaves every earlier yield as it was.
    last = list(dividends)[-1]
    close_without, adjusted_without = _prices_with_dividends(
        {date: value for date, value in dividends.items() if date != last}
    )
    without = options_model.get_trailing_dividend_yield(close_without, adjusted_without)
    before = dividend_yield.index < pd.Timestamp(last).to_period("D")
    # Up to the rounding of the adjusted close to the cent, which the dividend rescales.
    pd.testing.assert_frame_equal(
        dividend_yield[before], without[before], check_exact=False, atol=5e-4, rtol=0
    )
    assert (
        dividend_yield.loc[pd.Timestamp(last).to_period("D"), "ABC"]
        > without.loc[pd.Timestamp(last).to_period("D"), "ABC"] + 0.004
    )


def test_get_barrier_option_refuses_a_breached_barrier(options_module):
    # An up barrier at or below the spot and a down barrier at or above it are breached
    # before the option starts, a mistake in the call that is raised.
    for direction, percentage, message in (
        ("up", 0.9, "above the current stock price"),
        ("up", 1, "above the current stock price"),
        ("down", 1.1, "below the current stock price"),
    ):
        with pytest.raises(InvalidArgumentError, match=message):
            options_module.get_barrier_option(
                barrier_direction=direction, barrier_percentage=percentage
            )


def test_the_treasury_yield_is_converted_to_continuous_compounding(
    options_module, test_toolkit
):
    import numpy as np

    from financetoolkit.options import options_model

    # 4% compounded twice a year grows to the same amount as 3.9605% compounded continuously.
    assert options_model.get_continuously_compounded_rate(0.04) == pytest.approx(
        2 * np.log(1.02)
    )
    assert np.exp(
        options_model.get_continuously_compounded_rate(0.05)
    ) == pytest.approx(1.025**2)

    # The option formulas receive the continuously compounded equivalent of the Toolkit's yield.
    quoted = test_toolkit._daily_risk_free_rate["Adj Close"].dropna()
    used = options_module._risk_free_rate.loc[quoted.index]
    pd.testing.assert_series_equal(used, 2 * np.log1p(quoted / 2), check_names=False)
    assert (used[quoted > 0] < quoted[quoted > 0]).all()


def test_a_stale_spot_next_to_live_chains_is_reported(options_module, monkeypatch):
    from financetoolkit.options import options_controller

    warnings = []
    monkeypatch.setattr(
        options_controller.logger, "warning", lambda *args: warnings.append(args)
    )
    monkeypatch.setattr(
        options_module, "_stale_spot_warning_shown", False, raising=False
    )

    # The recorded price history ends years before today.
    options_module._warn_if_spot_is_stale()
    options_module._warn_if_spot_is_stale()

    assert len(warnings) == 1
    assert "option chains" in warnings[0][0]


def test_trailing_dividend_yield_falls_back_to_the_dividends_column():
    from financetoolkit.options import options_model

    index = pd.period_range("2020-01-01", "2022-12-31", freq="D")
    close = pd.DataFrame({"ABC": 100.0}, index=index)
    dividends = pd.DataFrame({"ABC": 0.0}, index=index)
    for date in ["2021-03-01", "2021-06-01", "2021-09-01", "2021-12-01"]:
        dividends.loc[pd.Period(date, "D"), "ABC"] = 0.5

    # A history that lists dividends without adjusting the close for them.
    unadjusted = options_model.get_trailing_dividend_yield(close, close.copy())
    filled = options_model.get_trailing_dividend_yield(
        close, close.copy(), dividends=dividends
    )

    assert unadjusted.loc[pd.Period("2021-12-31", "D"), "ABC"] == 0
    assert filled.loc[pd.Period("2021-12-31", "D"), "ABC"] == pytest.approx(0.02)
    assert filled.loc[pd.Period("2021-04-01", "D"), "ABC"] == pytest.approx(0.005)


def test_the_dividend_yield_is_converted_to_continuous_compounding(options_module):
    import numpy as np

    ticker = options_module._tickers[0]
    date = options_module._daily_dividend_yield.index[-1]
    simple_yield = options_module._daily_dividend_yield.loc[date, ticker]

    assert options_module._dividend_yield_on(ticker, date) == pytest.approx(
        np.log1p(simple_yield)
    )


@pytest.fixture
def option_market(monkeypatch, options_module):
    """Option chains priced with Black-Scholes around each ticker's last price."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.options import options_model  # noqa: PLC0415
    from financetoolkit.options.pricing import black_scholes_model  # noqa: PLC0415

    today = pd.Timestamp.today().normalize()
    expiries = [
        (today + pd.Timedelta(days=days)).strftime("%Y-%m-%d") for days in (10, 24, 45)
    ]

    def chains(tickers, expiration_date, put_option=False):
        frames = {}
        days = (pd.Timestamp(expiration_date) - today).days
        for ticker in tickers:
            price = float(options_module._prices[ticker].dropna().iloc[-1])
            strikes = np.round(np.linspace(price * 0.6, price * 1.4, 81), 1)
            # A skew: lower strikes carry a higher volatility.
            volatility = 0.25 + 0.15 * (price - strikes) / price
            values = np.array(
                [
                    black_scholes_model.get_black_scholes(
                        stock_price=price,
                        strike_price=float(strike),
                        risk_free_rate=0.03,
                        volatility=float(vol),
                        time_to_expiration=days / 365,
                        put_option=put_option,
                    )
                    for strike, vol in zip(strikes, volatility, strict=True)
                ]
            )
            frames[ticker] = pd.DataFrame(
                {
                    "Bid": np.maximum(values - 0.02, 0.0),
                    "Ask": values + 0.02,
                    "Last Price": values,
                    "Volume": 50.0 + strikes % 7,
                    "Open Interest": (300.0 if put_option else 200.0)
                    * np.exp(-(((strikes - price) / (0.1 * price)) ** 2)),
                    "Implied Volatility": volatility,
                    # The sources quote the price of the underlying with the chain.
                    "Underlying Price": price,
                },
                index=pd.Index(strikes, name="Strike Price"),
            )
        result = pd.concat(frames)
        result.index.names = ["Ticker", "Strike Price"]
        return result

    monkeypatch.setattr(
        options_model, "get_option_expiry_dates", lambda ticker: expiries
    )
    monkeypatch.setattr(options_model, "get_option_chains", chains)
    # Valued at today's close in New York, so the time to expiration is the whole days the
    # chains are priced with and the results do not depend on the time the tests run.
    monkeypatch.setattr(
        options_module, "_valuation_time", today + pd.Timedelta(hours=16)
    )
    labels = {date: f"Expiry {number}" for number, date in enumerate(expiries, 1)}
    return options_module, labels


def stable(table: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """The result with the expiry dates, which depend on today, replaced by labels."""
    table = table.drop(columns="Days to Expiration", errors="ignore")
    if isinstance(table.index, pd.MultiIndex):
        table = table.rename(index=labels, level=1)
    return table.replace(labels)


def test_get_put_call_ratio(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_put_call_ratio(), labels))


def test_get_max_pain(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_max_pain(number_of_expirations=2), labels))


def test_get_gamma_exposure(recorder, option_market):
    module, _ = option_market
    recorder.capture(module.get_gamma_exposure(number_of_expirations=2))
    recorder.capture(
        module.get_gamma_exposure(number_of_expirations=1, by_strike=True)
        .groupby(level=0)
        .sum()
    )


def test_get_implied_volatility_term_structure(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_implied_volatility_term_structure(), labels))


def test_get_volatility_skew(recorder, option_market):
    module, labels = option_market
    skew = module.get_volatility_skew()
    recorder.capture(stable(skew, labels))
    # Puts carry the higher volatility in the simulated skew.
    assert (skew["25-Delta Risk Reversal"] < 0).all()


def test_get_implied_move(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_implied_move(), labels))


def test_the_implied_move_warns_about_a_leg_priced_from_its_last_trade(
    caplog, monkeypatch, option_market
):
    import logging  # noqa: PLC0415

    from financetoolkit.options import options_model  # noqa: PLC0415

    module, labels = option_market
    expiry = list(labels)[0]
    original = options_model.get_option_chains

    def stale_call(tickers, expiration_date, put_option=False):
        chains = original(tickers, expiration_date, put_option)
        chains["Last Trade Date"] = "2026-11-09"
        if not put_option:
            # Every call loses its quote; only the open interest keeps its last trade.
            chains[["Bid", "Ask", "Open Interest"]] = [0.0, 1.75, 1.0]
        return chains

    monkeypatch.setattr(options_model, "get_option_chains", stale_call)
    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        moves = module.get_implied_move(expiration_dates=[expiry])

    ticker = module._tickers[0]
    strike = f"{moves.loc[(ticker, expiry), 'Straddle Strike']:g}"
    assert (moves["Last Trade Legs"] == 1).all()
    assert (
        f"{ticker} {expiry}: the {strike} call has no two-sided quote and is priced at its "
        "last trade of 2026-11-09 (open interest 1), and the put's bid-ask spread is"
    ) in caplog.text
    assert "may not reflect the live market" in caplog.text

    def unquoted(tickers, expiration_date, put_option=False):
        return original(tickers, expiration_date, put_option).assign(Bid=0.0, Ask=0.0)

    monkeypatch.setattr(options_model, "get_option_chains", unquoted)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        module.get_implied_move(number_of_expirations=2)

    # One warning for every expiry instead of one per expiry.
    assert caplog.text.count("Outside US market hours") == 1
    assert f"{ticker} {expiry}" in caplog.text


def test_get_model_free_implied_volatility(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_model_free_implied_volatility(), labels))


def test_get_volatility_risk_premium(recorder, option_market):
    module, _ = option_market
    recorder.capture(module.get_volatility_risk_premium())


def test_the_earnings_move_uses_the_first_expiry_after_the_announcement(option_market):
    module, _ = option_market
    expiries = [
        (pd.Timestamp.today().normalize() + pd.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        for days in (10, 24, 45)
    ]
    # An expiry on the announcement day settles before an announcement after the close.
    module._earnings_function = lambda: dict.fromkeys(module._tickers, expiries[0])

    moves = module.get_implied_move(around_earnings=True)

    assert (moves["Days to Expiration"] > 10).all()


def test_the_earnings_move_answers_every_ticker_with_its_own_expiry(option_market):
    """Each ticker's dividend yield and expiry are kept, also with differing dates."""
    module, labels = option_market
    expiries = list(labels)
    # The cached dividend yields are calculated afresh, for all tickers at once.
    module.__dict__.pop("_dividend_yield", None)
    earnings = {
        ticker: expiries[number % 2] for number, ticker in enumerate(module._tickers)
    }
    module._earnings_function = lambda: earnings

    moves = module.get_implied_move(around_earnings=True)

    assert list(moves.index.get_level_values("Ticker")) == module._tickers
    for (ticker, expiration), earnings_date in moves["Earnings Date"].items():
        assert earnings_date == earnings[ticker]
        assert expiration == expiries[expiries.index(earnings[ticker]) + 1]
    assert moves["Implied Move"].notna().all()


def test_the_mcp_earnings_move_answers_every_ticker(monkeypatch, option_market):
    import asyncio  # noqa: PLC0415
    import types  # noqa: PLC0415

    from financetoolkit.mcp_server import mcp_controller  # noqa: PLC0415
    from financetoolkit.mcp_server.registry.provider_model import (  # noqa: PLC0415
        ToolkitProvider,
    )

    module, labels = option_market
    expiries = list(labels)
    module.__dict__.pop("_dividend_yield", None)
    module._earnings_function = lambda: dict(
        zip(module._tickers, expiries, strict=False)
    )
    monkeypatch.setattr(
        ToolkitProvider,
        "get_toolkit_instance",
        lambda self, *args, **kwargs: types.SimpleNamespace(options=module),
    )

    text = asyncio.run(
        mcp_controller.mcp.call_tool(
            "options",
            {
                "indicator": "get_implied_move",
                "tickers": ",".join(module._tickers),
                "around_earnings": True,
            },
        )
    )[0].text

    assert "Invalid input" not in text
    for ticker, earnings_date in zip(module._tickers, expiries, strict=False):
        assert ticker in text
        assert earnings_date in text


def test_the_volatility_risk_premium_in_variance_units(option_market):
    module, _ = option_market
    premium = module.get_volatility_risk_premium(rounding=12)

    expected = premium["Implied Volatility"] ** 2 - premium["Realized Volatility"] ** 2
    pd.testing.assert_series_equal(
        premium["Variance Risk Premium"], expected, check_names=False
    )


def test_the_market_metrics_use_the_price_quoted_with_the_chain(
    monkeypatch, option_market
):
    from financetoolkit.options import options_model  # noqa: PLC0415

    module, _ = option_market
    ticker = module._tickers[0]
    original = options_model.get_option_chains

    def quoted(tickers, expiration_date, put_option=False):
        return original(tickers, expiration_date, put_option).assign(
            **{"Underlying Price": 123.45}
        )

    monkeypatch.setattr(options_model, "get_option_chains", quoted)
    exposure = module.get_gamma_exposure(number_of_expirations=1, rounding=6)

    assert (exposure["Stock Price"] == 123.45).all()

    def unquoted(tickers, expiration_date, put_option=False):
        return original(tickers, expiration_date, put_option).drop(
            columns="Underlying Price"
        )

    monkeypatch.setattr(options_model, "get_option_chains", unquoted)
    exposure = module.get_gamma_exposure(number_of_expirations=1, rounding=6)

    # Without a quoted price the traded Close is used, not the dividend-adjusted one.
    close = module._daily_historical["Close"][ticker].dropna().iloc[-1]
    assert exposure.loc[ticker, "Stock Price"] == pytest.approx(close, abs=1e-6)


def test_the_market_metrics_warn_about_history_that_ends_before_today(
    caplog, option_market
):
    import logging  # noqa: PLC0415

    module, _ = option_market
    with caplog.at_level(logging.WARNING, logger="financetoolkit"):
        module.get_implied_move(number_of_expirations=1)
        module.get_volatility_risk_premium()

    assert "Leave end_date at today" in caplog.text
    assert "realized volatility covers the days up to that date" in caplog.text


def test_the_model_free_volatility_beyond_the_longest_expiry_raises(option_market):
    module, _ = option_market
    ticker = module._tickers[0]

    with pytest.raises(ValueError, match=f"longest usable expiry of {ticker}"):
        module._model_free_volatility(ticker, days=365)
    assert module.get_model_free_implied_volatility(days=365).empty


def test_the_model_free_volatility_extrapolates_a_short_target(option_market):
    module, _ = option_market
    ticker = module._tickers[0]

    # The nearest expiry is 10 days away, so 7 days is extrapolated from the two nearest.
    result = module._model_free_volatility(ticker, days=7)
    near = module._model_free_volatility(ticker, days=10)

    assert result["Near-Term Expiration"] == near["Near-Term Expiration"]
    assert result["Next-Term Expiration"] == near["Next-Term Expiration"]
    assert result["Model-Free Implied Volatility"] > 0


def test_an_expiry_without_a_chain_leaves_the_other_tickers(option_market, monkeypatch):
    module, _ = option_market
    good, bad = module._tickers[:2]
    expiry_inputs = module._expiry_inputs

    def without_chain(ticker, expiration_date):
        inputs = expiry_inputs(ticker, expiration_date)
        return (
            (pd.DataFrame(), pd.DataFrame(), *inputs[2:]) if ticker == bad else inputs
        )

    monkeypatch.setattr(module, "_expiry_inputs", without_chain)

    with pytest.raises(ValueError, match=f"{bad} has no option chain"):
        module._model_free_volatility(bad, days=30)
    assert not module._model_free_volatility(good, days=30).empty


def test_monte_carlo_paths_in_chunks_give_the_same_price_and_large_runs_are_refused():
    import numpy as np
    import pytest

    from financetoolkit.options import options_model

    kwargs = {
        "stock_price": 100.0,
        "strike_price": 100.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "time_to_expiration": 1.0,
        "simulations": 25_000,
        "time_steps": 50,
        "seed": 1,
    }

    # The price of all paths drawn at once, as before the paths were drawn in chunks.
    shocks = np.random.default_rng(1).standard_normal((25_000, 50))
    delta = 1 / 50
    log_paths = ((0.05 - 0.02) * delta + 0.2 * np.sqrt(delta) * shocks).sum(axis=1)
    expected = np.exp(-0.05) * np.maximum(100 * np.exp(log_paths) - 100, 0).mean()

    price, _ = options_model.get_monte_carlo_option_price(**kwargs)

    assert price == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="too large"):
        options_model.get_monte_carlo_option_price(
            **{**kwargs, "simulations": 1_000_000, "time_steps": 1_000}
        )


def test_get_strategy_payoff_defaults_to_an_at_the_money_straddle(options_module):
    payoff = options_module.get_strategy_payoff()
    date = options_module._daily_historical.index[-1]

    for ticker in payoff.columns:
        legs = options_module._at_the_money_straddle(ticker, date)
        premiums = sum(leg["premium"] for leg in legs)
        strike = legs[0]["strike_price"]
        ticker_payoff = payoff[ticker].dropna()

        # The loss is largest, both premiums, at the strike and shrinks one for one away from it.
        assert legs[0]["put_option"] is False and legs[1]["put_option"] is True
        assert premiums > 0
        assert ticker_payoff.min() >= -premiums - 1e-9
        nearest = ticker_payoff.index[abs(ticker_payoff.index - strike).argmin()]
        assert ticker_payoff[nearest] == pytest.approx(
            abs(nearest - strike) - premiums, abs=1e-3
        )


def _options_without_msft_prices_after(date):
    import numpy as np

    from financetoolkit import Toolkit

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    historical.loc[historical.index > pd.Period(date, "D"), (slice(None), "MSFT")] = (
        np.nan
    )
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
    return toolkit.options


def test_a_ticker_without_a_price_leaves_the_others_their_option_values(
    options_module, caplog
):
    options = _options_without_msft_prices_after("2022-06-01")

    black_scholes = options.get_black_scholes_model(start_date="2022-12-30")
    delta = options.get_delta(start_date="2022-12-30")

    assert black_scholes.index.unique(level=0).tolist() == ["AAPL"]
    assert delta.index.unique(level=0).tolist() == ["AAPL"]
    assert "MSFT has no price on 2022-12-30" in caplog.text
    pd.testing.assert_frame_equal(
        black_scholes,
        options_module.get_black_scholes_model(start_date="2022-12-30").loc[["AAPL"]],
    )


def test_a_start_date_without_prices_uses_the_last_trading_day_before_it(
    options_module,
):
    # 2022-12-31 is a Saturday and 2020-02-29 a Saturday as well (leap day).
    for weekend, trading_day in (
        ("2022-12-31", "2022-12-30"),
        ("2020-02-29", "2020-02-28"),
    ):
        pd.testing.assert_frame_equal(
            options_module.get_black_scholes_model(start_date=weekend),
            options_module.get_black_scholes_model(start_date=trading_day),
        )


def test_a_start_date_before_the_data_or_not_a_date_is_refused(options_module, caplog):
    assert options_module.get_black_scholes_model(start_date="2019-01-01").empty
    assert "lies before the historical data" in caplog.text
    caplog.clear()
    assert options_module.get_delta(start_date="not a date").empty
    assert "is not a valid date" in caplog.text


def test_the_time_to_expiration_runs_to_the_close_on_the_expiration_date(
    option_market, monkeypatch
):
    from financetoolkit.options.market import market_metrics_model  # noqa: PLC0415

    module, labels = option_market
    ticker = module._tickers[0]
    today = pd.Timestamp.today().normalize()
    # Six hours before today's close in New York, 10 days before the first expiry.
    monkeypatch.setattr(module, "_valuation_time", today + pd.Timedelta(hours=10))
    near, following = list(labels)[:2]

    time_to_expiration = module._expiry_inputs(ticker, near)[3]

    assert time_to_expiration == pytest.approx((10 * 24 + 6) / (365 * 24))

    # The model-free variance and its interpolation weights use that same time.
    variances, weights = [], []
    variance = market_metrics_model.get_model_free_variance
    interpolate = market_metrics_model.get_constant_maturity_volatility

    def recorded_variance(calls, puts, time, rate):
        variances.append(time)
        return variance(calls, puts, time, rate)

    def recorded_weights(near_variance, near_days, next_variance, next_days, days):
        weights.extend([near_days, next_days])
        return interpolate(near_variance, near_days, next_variance, next_days, days)

    monkeypatch.setattr(
        market_metrics_model, "get_model_free_variance", recorded_variance
    )
    monkeypatch.setattr(
        market_metrics_model, "get_constant_maturity_volatility", recorded_weights
    )
    result = module._model_free_volatility(ticker, days=20)

    assert result["Near-Term Expiration"] == near
    assert result["Next-Term Expiration"] == following
    assert variances[0] == pytest.approx(time_to_expiration)
    assert weights == pytest.approx([time * 365 for time in variances])


def test_an_expiry_that_has_closed_is_left_out(option_market, monkeypatch):
    module, labels = option_market
    first = list(labels)[0]
    # After the close on the first expiration date.
    monkeypatch.setattr(
        module, "_valuation_time", pd.Timestamp(first) + pd.Timedelta(hours=17)
    )

    ratio = module.get_put_call_ratio(expiration_dates=list(labels))

    assert first not in ratio.index.get_level_values("Expiration")
    assert set(ratio.index.get_level_values("Expiration")) == {*list(labels)[1:], "All"}

    # The gamma exposure leaves the closed expiry out too, instead of failing as a whole.
    gamma_exposure = module.get_gamma_exposure(expiration_dates=list(labels))
    assert not gamma_exposure.empty


def test_the_market_metrics_use_the_trailing_continuous_dividend_yield(option_market):
    import numpy as np  # noqa: PLC0415

    module, labels = option_market
    ticker = module._tickers[0]
    last_date = module._daily_historical.index[-1]

    dividend_yield = module._expiry_inputs(ticker, list(labels)[0])[5]

    assert dividend_yield == pytest.approx(module._dividend_yield_on(ticker, last_date))
    assert dividend_yield == pytest.approx(
        np.log1p(module._daily_dividend_yield.loc[last_date, ticker])
    )


def test_the_implied_volatility_is_solved_from_mid_prices_and_the_quoted_price(
    option_market, monkeypatch
):
    import numpy as np  # noqa: PLC0415

    from financetoolkit.options import options_model  # noqa: PLC0415

    module, labels = option_market
    ticker = module._tickers[0]
    expiry = list(labels)[1]
    original = options_model.get_option_chains

    def misleading(tickers, expiration_date, put_option=False):
        # A stale last price and a history that ends at another price than the quote.
        return original(tickers, expiration_date, put_option).assign(
            **{"Last Price": lambda chain: chain["Last Price"] * 1.5}
        )

    monkeypatch.setattr(options_model, "get_option_chains", misleading)
    monkeypatch.setitem(module.__dict__, "_prices", module._prices * 1.1)

    # The chains are priced with a rate of 3% and no dividends.
    implied_volatility = module.get_implied_volatility(
        expiration_date=expiry, risk_free_rate=0.03, dividend_yield=0.0, rounding=8
    ).loc[ticker]

    price = float(original([ticker], expiry).loc[ticker, "Underlying Price"].iloc[0])
    near_the_money = implied_volatility[
        (implied_volatility.index > 0.9 * price)
        & (implied_volatility.index < 1.1 * price)
    ]
    expected = 0.25 + 0.15 * (price - near_the_money.index.to_numpy()) / price
    assert len(near_the_money) > 5
    np.testing.assert_allclose(near_the_money.to_numpy(), expected, atol=2e-3)


def test_the_default_implied_volatility_expiry_is_a_day_or_more_away(
    option_market, monkeypatch
):
    from financetoolkit.options import options_model  # noqa: PLC0415

    module, labels = option_market
    today = pd.Timestamp.today().normalize()
    expiries = [today.strftime("%Y-%m-%d"), *labels]
    monkeypatch.setattr(
        options_model, "get_option_expiry_dates", lambda ticker: expiries
    )
    monkeypatch.setattr(module, "_valuation_time", today + pd.Timedelta(hours=10))

    assert module.get_implied_volatility().name == expiries[1]


def test_get_risk_neutral_moments(recorder, option_market):
    module, labels = option_market
    moments = module.get_risk_neutral_moments()
    recorder.capture(stable(moments, labels))
    # The simulated chains carry a higher volatility at lower strikes: a left skew.
    assert (moments["Skewness"] < 0).all()
    assert (moments["Volatility"] > 0).all()


def test_get_skew_index(recorder, option_market):
    module, labels = option_market
    skew = module.get_skew_index()
    recorder.capture(stable(skew, labels))
    assert (skew["SKEW Index"] > 100).all()
    assert skew["SKEW Index"].tolist() == pytest.approx(
        (100 - 10 * skew["Skewness"]).tolist(), abs=1e-3
    )


def test_the_skew_index_uses_the_terms_of_the_model_free_volatility(option_market):
    module, _ = option_market
    ticker = module._tickers[0]
    skew = module._skew_index(ticker, days=30)
    volatility = module._model_free_volatility(ticker, days=30)

    for column in ("Near-Term Expiration", "Next-Term Expiration", "Forward Price"):
        assert skew[column] == volatility[column]
    with pytest.raises(ValueError, match=f"longest usable expiry of {ticker}"):
        module._skew_index(ticker, days=365)
    assert module.get_skew_index(days=365).empty


def test_the_skew_index_reports_a_chain_without_bids(option_market, monkeypatch):
    # At the weekend Yahoo Finance reports a bid and ask of 0: the index is reported as
    # missing quotes, not computed from last prices.
    from financetoolkit.options import options_model  # noqa: PLC0415

    module, _ = option_market
    ticker = module._tickers[0]
    quoted = options_model.get_option_chains

    def closed(tickers, expiration_date, put_option=False):
        return quoted(tickers, expiration_date, put_option).assign(Bid=0.0, Ask=0.0)

    monkeypatch.setattr(options_model, "get_option_chains", closed)
    with pytest.raises(ValueError, match=f"{ticker} has no bid quotes"):
        module._skew_index(ticker, days=30)
    assert module.get_skew_index().empty
    assert module.get_model_free_implied_volatility().empty


def test_the_volatility_rolls_over_each_tickers_own_trading_days():
    import numpy as np

    from financetoolkit import Toolkit
    from financetoolkit.historical_model import fill_non_trading_days

    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")

    def options_on(data):
        toolkit = Toolkit(
            tickers=sorted(set(data.columns.get_level_values(1)) - {"Benchmark"}),
            historical=data,
            convert_currency=False,
            start_date="2019-12-31",
            end_date="2023-01-01",
            sleep_timer=False,
        )
        toolkit._daily_risk_free_rate = pd.read_pickle(
            "tests/datasets/risk_free_rate.pickle"
        )
        toolkit._daily_treasury_data = pd.read_pickle(
            "tests/datasets/treasury_data.pickle"
        )
        return toolkit.options

    # A ticker trading every calendar day next to the stocks: the shared dates gain the
    # weekends, on which the stocks have no Return.
    calendar = pd.period_range(historical.index[0], historical.index[-1], freq="D")
    with_crypto = historical.reindex(calendar)
    returns = np.random.default_rng(0).normal(0, 0.03, len(calendar))
    for column in historical.columns.get_level_values(0).unique():
        with_crypto[(column, "CRYPTO")] = (
            returns if column == "Return" else 100 * np.cumprod(1 + returns)
        )
    with_crypto = fill_non_trading_days(with_crypto.sort_index(axis=1))

    alone = options_on(historical)._volatility
    together = options_on(with_crypto)._volatility

    pd.testing.assert_frame_equal(
        together.loc[historical.index, ["AAPL", "MSFT"]],
        alone[["AAPL", "MSFT"]],
        check_freq=False,
        check_names=False,
        check_column_type=False,
    )
    # A weekend takes the volatility of the Friday before it.
    assert together.loc["2022-12-25", "AAPL"] == alone.loc["2022-12-23", "AAPL"]


EXPIRY_DATES = {
    # AAPL lists weekly options; MSFT only a monthly and a later expiry AAPL does not have.
    "AAPL": ["2026-11-27", "2026-12-04", "2026-12-18"],
    "MSFT": ["2026-12-18", "2027-01-15"],
}


@pytest.fixture
def chains(monkeypatch):
    """Serve the expiration dates above and a chain for every ticker on each of its dates."""
    from financetoolkit.options import options_model

    requested = []

    def fake_option_chains(tickers, expiration_date, put_option=False):  # noqa: ARG001
        requested.extend((ticker, expiration_date) for ticker in tickers)
        frames = {
            ticker: pd.DataFrame(
                {
                    "Bid": [1.0, 2.0],
                    "Ask": [1.1, 2.1],
                    "Change": [0.1, 0.2],
                    "Percent Change": [1.5, -3.36],
                    "Implied Volatility": [0.3, 0.25],
                    "Underlying Price": [100.0, 100.0],
                },
                index=pd.Index([95.0, 105.0], name="Strike"),
            )
            for ticker in tickers
            if expiration_date in EXPIRY_DATES[ticker]
        }
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames)
        result.index.names = ["Ticker", "Strike Price"]
        return result

    monkeypatch.setattr(
        options_model, "get_option_expiry_dates", lambda ticker: EXPIRY_DATES[ticker]
    )
    monkeypatch.setattr(options_model, "get_option_chains", fake_option_chains)
    return requested


def test_each_ticker_is_checked_against_its_own_expiration_dates(
    options_module, chains, caplog
):
    # A date only the second ticker lists is not refused for the first one's dates.
    with caplog.at_level("WARNING"):
        january = options_module.get_option_chains(expiration_date="2027-01-15")

    assert list(january.index.get_level_values(0).unique()) == ["MSFT"]
    assert set(january["Expiration Date"]) == {"2027-01-15"}
    assert january.name == "2027-01-15"
    assert chains == [("MSFT", "2027-01-15")]
    # The ticker without that date is left out with its own dates, not as an unknown symbol.
    assert (
        "No options data is available for AAPL expiring on 2027-01-15, so it is left out. "
        "Its expiration dates are 2026-11-27, 2026-12-04, 2026-12-18." in caplog.text
    )

    shared = options_module.get_option_chains(expiration_date="2026-12-18")
    assert list(shared.index.get_level_values(0).unique()) == ["AAPL", "MSFT"]


def test_each_ticker_shows_its_nearest_expiry_by_default(options_module, chains):
    nearest = options_module.get_option_chains()

    assert nearest.groupby(level=0)["Expiration Date"].first().to_dict() == {
        "AAPL": "2026-11-27",
        "MSFT": "2026-12-18",
    }
    assert list(nearest.columns[:3]) == ["Expiration Date", "Bid", "Ask"]
    assert "Underlying Price" not in nearest.columns
    # The expiries differ, so no single date names the chains.
    assert getattr(nearest, "name", None) is None


def test_the_expiration_dates_are_shown_per_ticker(options_module, chains):
    assert options_module.get_option_chains(show_expiration_dates=True) == EXPIRY_DATES
    assert options_module._all_expiration_dates() == [
        "2026-11-27",
        "2026-12-04",
        "2026-12-18",
        "2027-01-15",
    ]
    assert options_module._all_expiration_dates(shared_first=True) == ["2026-12-18"]


def test_an_expiration_date_no_ticker_lists_names_the_dates_of_each(
    options_module, chains, caplog
):
    with caplog.at_level("ERROR"):
        result = options_module.get_option_chains(expiration_date="2030-01-18")

    assert result.empty
    assert chains == []
    assert "AAPL: 2026-11-27, 2026-12-04, 2026-12-18; MSFT: 2026-12-18, 2027-01-15" in (
        caplog.text
    )


def test_a_ticker_without_options_is_left_out_by_name(
    options_module, chains, monkeypatch, caplog
):
    monkeypatch.setitem(EXPIRY_DATES, "MSFT", [])

    with caplog.at_level("WARNING"):
        nearest = options_module.get_option_chains()

    assert list(nearest.index.get_level_values(0).unique()) == ["AAPL"]
    assert nearest.name == "2026-11-27"
    assert (
        "No options data is available for MSFT, so it is left out: Yahoo Finance lists no options of it."
        in caplog.text
    )


def test_the_percent_change_of_an_option_is_a_decimal(options_module, chains):
    # Yahoo Finance quotes the change in percent (1.5 for 1.5%).
    nearest = options_module.get_option_chains()
    assert nearest["Percent Change"].tolist() == [0.015, -0.0336, 0.015, -0.0336]


def test_get_american_option(recorder, options_module):
    recorder.capture(options_module.get_american_option())
    recorder.capture(
        options_module.get_american_option(put_option=True, strike_price_range=0.10)
    )
    recorder.capture(
        options_module.get_american_option(
            method="barone_adesi_whaley",
            put_option=True,
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.03,
            dividend_yield=0.02,
            rounding=2,
        )
    )
    recorder.capture(
        options_module.get_american_option(
            method="ju_zhong", dividend_yield=0.06, expiration_time_range=10
        )
    )


def test_get_american_option_bjerksund_stensland(recorder, options_module):
    recorder.capture(options_module.get_american_option(method="bjerksund_stensland"))
    recorder.capture(
        options_module.get_american_option(
            method="bjerksund_stensland", put_option=True
        )
    )
    recorder.capture(
        options_module.get_american_option(
            method="bjerksund_stensland",
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.02,
            rounding=2,
        )
    )


def test_get_american_option_bjerksund_stensland_1993(recorder, options_module):
    recorder.capture(
        options_module.get_american_option(method="bjerksund_stensland_1993")
    )
    recorder.capture(
        options_module.get_american_option(
            method="bjerksund_stensland_1993", put_option=True
        )
    )


def test_get_american_option_qd_fixed_point(recorder, options_module):
    recorder.capture(
        options_module.get_american_option(
            method="qd_fixed_point",
            put_option=True,
            strike_price_range=0.05,
            expiration_time_range=5,
        )
    )


def test_get_binomial_model_variants(recorder, options_module):
    for method in (
        "jarrow_rudd",
        "tian",
        "trigeorgis",
        "leisen_reimer",
        "exact_inversion",
    ):
        recorder.capture(options_module.get_binomial_model(method=method, time_steps=5))
    recorder.capture(
        options_module.get_binomial_model(
            method="leisen_reimer", put_option=True, american_option=True
        )
    )


def test_the_strike_centred_trees_use_an_odd_number_of_steps(options_module):
    tree = options_module.get_binomial_model(method="exact_inversion", time_steps=4)
    # Four steps become five: six dates from the start date to expiration.
    assert len(tree.columns) == 6


def test_a_short_exact_inversion_tree_keeps_every_strike(options_module, caplog):
    # At 0.004 years the strikes far from the price have N(d2) of 1 in floating point, which
    # gave a probability of 1 and an error that emptied the result of every ticker.
    for method in ("exact_inversion", "leisen_reimer"):
        tree = options_module.get_binomial_model(
            method=method,
            time_to_expiration=0.004,
            time_steps=11,
            strike_price_range=0.5,
        )
        assert not tree.empty
        # Every strike has its value at the start date.
        start = tree.iloc[:, 0].groupby(level=[0, 1]).count()
        assert (start == 1).all()
        assert start.index.get_level_values(0).unique().tolist() == ["AAPL", "MSFT"]

    # The movements of these trees differ per strike, so they are shown per strike.
    with caplog.at_level("INFO"):
        options_module.get_binomial_model(
            method="exact_inversion", time_steps=5, show_input_info=True
        )
    assert "at a strike of" in caplog.text


def test_the_cev_model_near_an_elasticity_of_one(options_module):
    # The non-central chi-squared series no longer converges there: NaN prices at β = 0.9999.
    cev = options_module.get_cev_model(beta=0.9999, expiration_time_range=10)
    black_scholes = options_module.get_black_scholes_model(expiration_time_range=10)
    assert not cev.isna().any().any()
    assert (cev - black_scholes).abs().max().max() < 1e-3


def test_get_cev_model(recorder, options_module):
    recorder.capture(options_module.get_cev_model())
    recorder.capture(
        options_module.get_cev_model(
            put_option=True, beta=0.3, strike_price_range=0.10, rounding=2
        )
    )


def test_the_cev_model_with_an_elasticity_of_one_is_black_scholes(options_module):
    cev = options_module.get_cev_model(beta=1.0, expiration_time_range=10)
    black_scholes = options_module.get_black_scholes_model(expiration_time_range=10)
    pd.testing.assert_frame_equal(cev, black_scholes)


def test_get_black_76_model(recorder, options_module):
    recorder.capture(options_module.get_black_76_model())
    recorder.capture(
        options_module.get_black_76_model(
            put_option=True, risk_free_rate=0.03, expiration_time_range=10
        )
    )


def test_get_black_scholes_model_cash_dividends(recorder, options_module):
    recorder.capture(
        options_module.get_black_scholes_model(
            dividends="escrowed", expiration_time_range=60
        )
    )
    recorder.capture(
        options_module.get_black_scholes_model(
            dividends="haug_haug_lewis",
            put_option=True,
            strike_price_range=0.05,
            expiration_time_range=60,
        )
    )


def test_the_american_option_is_worth_at_least_the_european_option(options_module):
    american = options_module.get_american_option(
        put_option=True, method="ju_zhong", expiration_time_range=60
    )
    european = options_module.get_black_scholes_model(
        put_option=True, expiration_time_range=60
    )
    assert (american >= european - 1e-4).all().all()


def test_cash_dividends_replace_the_dividend_yield(options_module):
    # MSFT's next dividend of 0.68 is projected 47 days after the last date of 2022-12-30
    # (2022-11-16 plus the 91 days between its ex-dates); options expiring before it are
    # priced as without dividends, those on or after it lower.
    escrowed = options_module.get_black_scholes_model(
        dividends="escrowed", expiration_time_range=90
    ).loc["MSFT"]
    without = options_module.get_black_scholes_model(
        dividend_yield=0.0, expiration_time_range=90
    ).loc["MSFT"]
    assert (escrowed.iloc[:, :46] - without.iloc[:, :46]).abs().max().max() == 0
    assert (escrowed.iloc[:, -5:] < without.iloc[:, -5:]).all().all()

    with pytest.raises(InvalidArgumentError, match="only used with"):
        options_module.get_black_scholes_model(
            dividends="escrowed", dividend_yield=0.01
        )


def test_get_implied_volatility_american(recorder, option_market):
    module, labels = option_market
    european = module.get_implied_volatility(put_option=True)
    american = module.get_implied_volatility(put_option=True, american_option=True)
    recorder.capture(american.to_frame("American"))
    # The early exercise right is part of the American price, so the volatility that
    # reprices a put is lower when that right is counted, most for the deep in-the-money.
    common = american.index.intersection(european.index)
    assert (american.loc[common] <= european.loc[common] + 1e-4).all()


def test_get_asian_option_methods(recorder, options_module):
    for method in ("turnbull_wakeman", "curran"):
        recorder.capture(
            options_module.get_asian_option(method=method, expiration_time_range=10)
        )
    recorder.capture(
        options_module.get_asian_option(
            method="curran", fixings=12, put_option=True, expiration_time_range=10
        )
    )
    recorder.capture(
        options_module.get_asian_option(
            fixings=12, strike_price_range=0.10, expiration_time_range=5
        )
    )
    recorder.capture(
        options_module.get_asian_option(
            method="monte_carlo",
            fixings=12,
            simulations=2_000,
            seed=42,
            strike_price_range=0.10,
            expiration_time_range=5,
        )
    )


def test_get_barrier_option_variants(recorder, options_module):
    recorder.capture(
        options_module.get_barrier_option(
            barrier_direction="double", expiration_time_range=10
        )
    )
    recorder.capture(
        options_module.get_barrier_option(
            barrier_window="start", window_share=0.5, expiration_time_range=10
        )
    )
    recorder.capture(
        options_module.get_barrier_option(
            put_option=True,
            barrier_percentage=1.1,
            barrier_direction="up",
            knock_type="in",
            barrier_window="end",
            expiration_time_range=10,
        )
    )
    recorder.capture(
        options_module.get_barrier_option(
            soft_barrier_percentage=0.85,
            strike_price_range=0.10,
            expiration_time_range=5,
        )
    )
    recorder.capture(
        options_module.get_barrier_option(monitoring="daily", expiration_time_range=10)
    )


def test_get_lookback_option(recorder, options_module):
    recorder.capture(options_module.get_lookback_option())
    recorder.capture(options_module.get_lookback_option(put_option=True))
    recorder.capture(options_module.get_lookback_option(strike_type="floating"))
    recorder.capture(
        options_module.get_lookback_option(
            strike_type="floating", put_option=True, lookback_period=0.5
        )
    )
    recorder.capture(
        options_module.get_lookback_option(
            lookback_period=0.5,
            strike_price_range=0.10,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_touch_option(recorder, options_module):
    recorder.capture(options_module.get_touch_option())
    recorder.capture(
        options_module.get_touch_option(barrier_percentage=1.1, payment="hit")
    )
    recorder.capture(options_module.get_touch_option(option_type="no-touch"))
    recorder.capture(options_module.get_touch_option(option_type="double-no-touch"))
    recorder.capture(
        options_module.get_touch_option(
            option_type="double-one-touch", monitoring="daily", cash_payout=100
        )
    )


def test_get_chooser_option(recorder, options_module):
    recorder.capture(options_module.get_chooser_option())
    recorder.capture(options_module.get_chooser_option(choice_period=0.25))
    recorder.capture(
        options_module.get_chooser_option(
            choice_period=1,
            strike_price_range=0.10,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=2,
        )
    )


def test_get_forward_start_option(recorder, options_module):
    recorder.capture(options_module.get_forward_start_option())
    recorder.capture(
        options_module.get_forward_start_option(put_option=True, moneyness=1.1)
    )
    recorder.capture(options_module.get_forward_start_option(start_period=0, periods=3))
    recorder.capture(
        options_module.get_forward_start_option(
            start_period=0.25,
            periods=2,
            performance=True,
            expiration_time_range=5,
            dividend_yield=0.01,
            rounding=6,
        )
    )


def test_compound_style_options_refuse_invalid_choices(options_module):
    # Each invalid choice is a mistake in the call, raised with the valid values.
    for call, message in (
        (
            lambda: options_module.get_chooser_option(choice_period=1.5),
            "choice_period",
        ),
        (
            lambda: options_module.get_forward_start_option(start_period=1),
            "start_period",
        ),
        (
            lambda: options_module.get_forward_start_option(moneyness=0),
            "moneyness",
        ),
        (
            lambda: options_module.get_forward_start_option(periods=0),
            "periods",
        ),
    ):
        with pytest.raises(InvalidArgumentError, match=message):
            call()


def test_path_dependent_options_refuse_invalid_choices(options_module):
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        options_module.get_asian_option(method="levy")
    # Each invalid combination is a mistake in the call, raised with what to pass instead.
    for call, message in (
        (
            lambda: options_module.get_barrier_option(
                barrier_direction="double", rebate=1
            ),
            "rebate",
        ),
        (
            lambda: options_module.get_barrier_option(
                barrier_direction="double", barrier_window="start"
            ),
            "cannot be combined",
        ),
        (
            lambda: options_module.get_barrier_option(soft_barrier_percentage=1.05),
            "same side",
        ),
        (
            lambda: options_module.get_lookback_option(lookback_period=0),
            "lookback_period",
        ),
        (
            lambda: options_module.get_touch_option(
                option_type="no-touch", payment="hit"
            ),
            "pays at expiration",
        ),
        (
            lambda: options_module.get_touch_option(barrier_percentage=1),
            "touched already",
        ),
        (
            lambda: options_module.get_barrier_option(monitoring="hourly"),
            "monitoring must be one of",
        ),
        (
            lambda: options_module.get_touch_option(monitoring="hourly"),
            "monitoring must be one of",
        ),
        (
            lambda: options_module.get_barrier_option(barrier_percentage=-0.5),
            "barrier_percentage",
        ),
        (
            lambda: options_module.get_touch_option(barrier_percentage=0),
            "barrier_percentage",
        ),
    ):
        with pytest.raises(InvalidArgumentError, match=message):
            call()


def test_monte_carlo_asian_option_is_reproducible_and_close_to_curran(options_module):
    arguments = {"fixings": 12, "strike_price_range": 0.10, "expiration_time_range": 5}
    first = options_module.get_asian_option(
        method="monte_carlo", simulations=20_000, seed=1, **arguments
    )
    second = options_module.get_asian_option(
        method="monte_carlo", simulations=20_000, seed=1, **arguments
    )
    curran = options_module.get_asian_option(method="curran", **arguments)

    pd.testing.assert_frame_equal(first, second)
    assert (first - curran).abs().max().max() < 0.05


@pytest.mark.parametrize("method", ["svi", "ssvi", "sabr"])
def test_get_volatility_surface_methods(recorder, option_market, method):
    module, labels = option_market
    surface = module.get_volatility_surface(method=method, number_of_expirations=2)
    recorder.capture(surface.rename(columns=labels))
    # Every model follows the simulated skew: lower strikes carry the higher volatility.
    for ticker in surface.index.get_level_values(0).unique():
        smile = surface.loc[ticker].iloc[:, 0].dropna()
        assert smile.iloc[0] > smile.iloc[-1]


@pytest.mark.parametrize("method", ["ssvi", "sabr"])
def test_get_volatility_surface_parameters(recorder, option_market, method):
    module, labels = option_market
    parameters = module.get_volatility_surface(
        method=method,
        expiration_dates=list(labels)[1:],
        sabr_beta=0.5,
        show_parameters=True,
    )
    recorder.capture(stable(parameters, labels))
    if method == "sabr":
        assert (parameters["beta"] == 0.5).all()
    else:
        # One rho, eta and gamma for the whole surface of a ticker.
        for ticker in parameters.index.get_level_values(0).unique():
            assert parameters.loc[ticker, ["rho", "eta", "gamma"]].nunique().max() == 1
    with pytest.raises(InvalidArgumentError, match="method must be one of"):
        module.get_volatility_surface(method="heston")


def test_get_local_volatility_surface(recorder, option_market):
    module, labels = option_market
    local_volatility = module.get_local_volatility_surface(method="ssvi")
    recorder.capture(local_volatility.rename(columns=labels))
    assert list(local_volatility.columns) == list(labels)
    with pytest.raises(
        InvalidArgumentError, match="method must be one of 'svi' or 'ssvi'"
    ):
        module.get_local_volatility_surface(method="sabr")


def test_get_exchange_option(recorder, options_module):
    recorder.capture(options_module.get_exchange_option())
    recorder.capture(options_module.get_exchange_option(american_option=True))
    recorder.capture(
        options_module.get_exchange_option(
            second_ticker="MSFT",
            exchange_ratio=0.9,
            expiration_time_range=5,
            rounding=6,
        )
    )


def test_get_spread_option(recorder, options_module):
    recorder.capture(options_module.get_spread_option())
    recorder.capture(options_module.get_spread_option(method="kirk", put_option=True))
    recorder.capture(
        options_module.get_spread_option(
            second_ticker="AAPL",
            strike_price_range=0.1,
            strike_step_size=2.5,
            expiration_time_range=5,
            risk_free_rate=0.01,
            rounding=6,
        )
    )


def test_the_spread_option_at_a_strike_of_zero_is_the_exchange_option(
    options_module,
):
    spread = options_module.get_spread_option(rounding=10)
    exchange = options_module.get_exchange_option(rounding=10)

    for ticker in exchange.index:
        pd.testing.assert_series_equal(
            spread.loc[(ticker, 0)], exchange.loc[ticker], check_names=False, atol=1e-8
        )


def test_the_two_asset_options_reject_an_unknown_second_asset(options_module):
    assert options_module.get_exchange_option(second_ticker="TSLA").empty


def test_get_variance_swap(recorder, option_market):
    module, labels = option_market
    recorder.capture(stable(module.get_variance_swap(), labels))
    recorder.capture(
        stable(module.get_variance_swap(number_of_expirations=2, strike=0.3), labels)
    )


def test_the_variance_swap_is_close_to_the_vix_variance_and_values_a_traded_swap(
    option_market,
):
    module, _ = option_market
    swaps = module.get_variance_swap(rounding=None)
    # The strikes are a percent of the price apart, so the replication and the midpoint
    # rule of the Cboe methodology agree to well within a percent.
    np.testing.assert_allclose(
        swaps["Fair Variance"], swaps["Cboe Variance"], rtol=0.01
    )

    traded = module.get_variance_swap(
        number_of_expirations=1, strike=0.25, trade_date="2022-12-01", rounding=None
    ).iloc[0]
    assert 0 < traded["Elapsed Fraction"] < 1
    rate = float(module._risk_free_rate.dropna().iloc[-1])
    time = float(traded["Days to Expiration"]) / 365
    expected = np.exp(-rate * time) * (
        traded["Elapsed Fraction"] * traded["Realized Variance"]
        + (1 - traded["Elapsed Fraction"]) * traded["Fair Variance"]
        - 0.0625
    )
    # The table is rounded to four decimals.
    assert traded["Variance Swap Value"] == pytest.approx(expected, abs=2e-4)

    with pytest.raises(InvalidArgumentError, match="needs its strike"):
        module.get_variance_swap(trade_date="2022-12-01")


def test_get_delta_hedging_backtest(recorder, options_module):
    recorder.capture(options_module.get_delta_hedging_backtest())
    recorder.capture(
        options_module.get_delta_hedging_backtest(
            days=21,
            start_date="2022-06-01",
            moneyness=0.95,
            volatility=0.3,
            put_option=True,
            transaction_cost=0.001,
            rebalancing_days=5,
        )
    )


def test_the_delta_hedging_backtest_without_moves_in_volatility_breaks_even(
    options_module,
):
    # At the volatility the stock realized the seller roughly breaks even, against a clear
    # profit when the option is sold at twice that volatility.
    backtest = options_module.get_delta_hedging_backtest(days=126, rounding=None)
    fair = options_module.get_delta_hedging_backtest(
        days=126,
        volatility=float(backtest["Realized Volatility"].iloc[0]),
        rounding=None,
    )
    rich = options_module.get_delta_hedging_backtest(
        days=126,
        volatility=2 * float(backtest["Realized Volatility"].iloc[0]),
        rounding=None,
    )
    first = backtest.index[0]
    assert (
        abs(fair.loc[first, "Profit and Loss"]) < 0.25 * fair.loc[first, "Option Value"]
    )
    assert rich.loc[first, "Profit and Loss"] > 0.25 * rich.loc[first, "Option Value"]


def test_get_delta_hedging_simulation(recorder, options_module):
    recorder.capture(options_module.get_delta_hedging_simulation(simulations=2000))
    recorder.capture(
        options_module.get_delta_hedging_simulation(
            days=21,
            start_date="2022-06-01",
            moneyness=1.05,
            implied_volatility=0.35,
            put_option=True,
            transaction_cost=0.001,
            rebalancing_days=5,
            simulations=2000,
        )
    )


def test_get_heston_parameters(recorder, option_market):
    module, _ = option_market
    # The simulated expirations lie 10, 24 and 45 days away: the default leaves out the one
    # under 14 days and fits the other two, the nearest to every month of the grid.
    parameters = module.get_heston_parameters()
    # The fit stops where the least squares stops improving, which moves its last digits
    # with the platform's floating point, so it is recorded to two decimals. Two short
    # expirations hardly pin down κ and θ apart (κ ends near zero, where θ barely changes the
    # prices, and θ moved by 0.01 between Python versions), so only their combination is
    # recorded: the Feller ratio 2κθ/σ², to one decimal.
    recorder.capture(
        parameters.drop(columns=["kappa", "theta"]).round(2).round({"Feller Ratio": 1})
    )
    # The simulated chains have a downward skew: a negative correlation.
    assert (parameters["rho"] < 0).all()
    assert (parameters["Expirations"] == 2).all()
    assert (parameters["RMSE"] < 0.05).all()


def test_get_heston_parameters_names_the_parameters_at_a_bound(
    monkeypatch, option_market
):
    from financetoolkit.options.volatility import heston_model  # noqa: PLC0415

    module, _ = option_market
    fit = heston_model.get_heston_parameters

    def fit_at_the_upper_kappa(*args, **kwargs):
        return {**fit(*args, **kwargs), "kappa": heston_model.HESTON_BOUNDS[1][1]}

    monkeypatch.setattr(heston_model, "get_heston_parameters", fit_at_the_upper_kappa)
    parameters = module.get_heston_parameters()

    # The bound is data in the answer, so it reaches the MCP answer too, and the numbers keep
    # a numeric type next to it.
    assert (parameters["At Bound"] == "kappa").all()
    assert (parameters["kappa"] == 20).all()
    assert pd.api.types.is_float_dtype(parameters["RMSE"])


def test_get_heston_parameters_skips_a_ticker_without_quotes_on_its_dates(
    monkeypatch, option_market
):
    from financetoolkit.options import options_model  # noqa: PLC0415

    module, _ = option_market
    expiries = sorted(module._all_expiration_dates())
    chains = options_model.get_option_chains

    # AAPL lists only the 45-day expiry and MSFT only the 24-day one; the chain of the 45-day
    # expiry has no AAPL quotes. AAPL then has quotes only on the date MSFT selected, which
    # raised an IndexError for the whole call.
    monkeypatch.setattr(
        module,
        "_expiration_dates",
        lambda: {"AAPL": [expiries[2]], "MSFT": [expiries[1]]},
    )

    def chains_without_aapl(tickers, expiration_date, put_option=False):
        result = chains(tickers, expiration_date, put_option)
        if expiration_date == expiries[2]:
            result = result.drop(index="AAPL", level="Ticker")
        return result

    monkeypatch.setattr(options_model, "get_option_chains", chains_without_aapl)
    parameters = module.get_heston_parameters()

    assert list(parameters.index) == ["MSFT"]


def test_get_heston_nandi_parameters_keep_the_decimals_of_the_log_likelihood(
    options_module,
):
    parameters = options_module.get_heston_nandi_model(show_parameters=True)
    exact = options_module.get_heston_nandi_model(show_parameters=True, rounding=10)

    # Rounded to 4 significant digits a log-likelihood in the thousands lost its decimals,
    # the part fits are compared by.
    np.testing.assert_allclose(
        parameters["Log-Likelihood"], exact["Log-Likelihood"].round(4)
    )


def test_get_heston_nandi_model(recorder, options_module):
    # The GARCH fit stops where the likelihood stops improving, which moves its last digits
    # with the platform's floating point, so the prices are recorded to two decimals.
    recorder.capture(options_module.get_heston_nandi_model(rounding=2))
    recorder.capture(options_module.get_heston_nandi_model(put_option=True, rounding=2))
    parameters = options_module.get_heston_nandi_model(show_parameters=True)
    assert (parameters["Persistence"] < 1).all()
    assert (parameters["Next-Day Volatility"] > 0).all()


def test_get_american_option_finite_difference(recorder, options_module):
    recorder.capture(
        options_module.get_american_option(
            method="finite_difference", put_option=True, strike_price_range=0.10
        )
    )
    recorder.capture(
        options_module.get_american_option(
            method="finite_difference", dividend_yield=0.06, expiration_time_range=10
        )
    )


def test_get_american_option_finite_difference_matches_the_fixed_point(options_module):
    finite_difference = options_module.get_american_option(
        method="finite_difference",
        put_option=True,
        strike_price_range=0.10,
        expiration_time_range=8,
        rounding=8,
    )
    fixed_point = options_module.get_american_option(
        method="qd_fixed_point",
        put_option=True,
        strike_price_range=0.10,
        expiration_time_range=8,
        rounding=8,
    )

    assert (finite_difference - fixed_point).abs().max().max() < 1e-3


def test_get_monte_carlo_option_price_american_and_variance_reduction(
    recorder, options_module
):
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            put_option=True,
            strike_price_range=0.10,
            expiration_time_range=5,
            simulations=1_000,
            time_steps=10,
            seed=42,
            american_option=True,
        )
    )
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            simulations=1_024,
            time_steps=10,
            seed=42,
            variance_reduction="control_variate",
            sequence="sobol",
        )
    )
    recorder.capture(
        options_module.get_monte_carlo_option_price(
            expiration_time_range=5,
            simulations=1_000,
            time_steps=10,
            seed=42,
            variance_reduction="antithetic",
        )
    )


@pytest.fixture
def options_with_constant_prices(test_toolkit):
    # MSFT without any price moves, so its volatility is 0, next to AAPL with its own data.
    module = test_toolkit.options
    volatility = module._trailing_volatility.copy()
    volatility["MSFT"] = 0.0
    module.__dict__["_trailing_volatility"] = volatility
    return module


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_american_option", {}),
        ("get_barrier_option", {}),
        ("get_binomial_model", {"time_steps": 5}),
        ("get_cev_model", {}),
        ("get_chooser_option", {}),
        ("get_exchange_option", {"second_ticker": "AAPL"}),
        ("get_forward_start_option", {}),
        ("get_lookback_option", {}),
        ("get_spread_option", {"second_ticker": "AAPL"}),
        ("get_stock_price_simulation", {"time_steps": 5}),
        ("get_touch_option", {}),
    ],
)
def test_a_ticker_without_volatility_leaves_out_only_that_ticker(
    options_with_constant_prices, method, arguments, caplog
):
    result = getattr(options_with_constant_prices, method)(**arguments)

    tickers = set(result.index.get_level_values(0))
    if method in ("get_exchange_option", "get_spread_option"):
        # AAPL is the second asset, so MSFT is the only ticker priced against it.
        assert result.empty
    else:
        assert tickers == {"AAPL"}
    assert "for MSFT" in caplog.text


def test_the_stock_price_simulation_refuses_a_time_to_expiration_that_is_not_positive(
    options_module,
):
    # A negative time gave NaN movements, on which listing every path never ended.
    for time_to_expiration in (-1, 0, float("nan")):
        with pytest.raises(InvalidArgumentError, match="time_to_expiration"):
            options_module.get_stock_price_simulation(
                time_to_expiration=time_to_expiration
            )
    with pytest.raises(InvalidArgumentError, match="time_steps"):
        options_module.get_binomial_model(time_steps=0)


@pytest.mark.parametrize(
    "arguments",
    [
        {"strike_price_range": 1.5},
        {"strike_price_range": -0.1},
        {"strike_step_size": 0},
        {"strike_step_size": -5},
        {"strike_step_size": 0.001},
        {"expiration_time_range": 1},
        {"expiration_time_range": 0},
    ],
)
def test_the_greeks_refuse_an_invalid_grid(options_module, arguments):
    # Each raised a TypeError from combining the empty greeks, after logging once per greek.
    with pytest.raises(InvalidArgumentError):
        options_module.collect_first_order_greeks(**arguments)
    with pytest.raises(InvalidArgumentError):
        options_module.collect_all_greeks(**arguments)
    with pytest.raises(InvalidArgumentError):
        options_module.get_black_scholes_model(**arguments)


def test_the_greeks_are_empty_with_one_warning_when_none_can_be_calculated(
    test_toolkit, monkeypatch, caplog
):
    module = test_toolkit.options
    for greek in ("get_delta", "get_dual_delta", "get_vega", "get_theta", "get_rho"):
        monkeypatch.setattr(module, greek, lambda **_: pd.DataFrame())
    for greek in ("get_epsilon", "get_lambda"):
        monkeypatch.setattr(module, greek, lambda **_: pd.DataFrame())

    assert module.collect_all_greeks().empty
    assert caplog.text.count("None of the greeks could be calculated") == 1


@pytest.mark.parametrize(
    ("method", "arguments", "name"),
    [
        ("get_monte_carlo_option_price", {"simulations": 0}, "simulations"),
        ("get_monte_carlo_option_price", {"time_steps": 0}, "time_steps"),
        ("get_asian_option", {"simulations": 0}, "simulations"),
        ("get_asian_option", {"fixings": 0}, "fixings"),
        ("get_cev_model", {"beta": 1.5}, "beta"),
        ("get_chooser_option", {"choice_period": -0.1}, "choice_period"),
        ("get_delta_hedging_simulation", {"moneyness": 0}, "moneyness"),
        ("get_delta_hedging_simulation", {"simulations": 0}, "simulations"),
        ("get_delta_hedging_backtest", {"days": 0}, "days"),
        ("get_exchange_option", {"exchange_ratio": 0}, "exchange_ratio"),
        (
            "get_heston_nandi_model",
            {"expiration_time_range": 1},
            "expiration_time_range",
        ),
        ("get_put_call_ratio", {"number_of_expirations": 0}, "number_of_expirations"),
        ("get_model_free_implied_volatility", {"days": 0}, "days"),
        ("get_skew_index", {"days": 0}, "days"),
        ("get_risk_neutral_density", {"number_of_strikes": 2}, "number_of_strikes"),
    ],
)
def test_the_options_refuse_invalid_arguments(options_module, method, arguments, name):
    with pytest.raises(InvalidArgumentError, match=name):
        getattr(options_module, method)(**arguments)


def test_the_put_call_parity_says_why_its_z_scores_are_empty(options_module, caplog):
    result = options_module.get_put_call_parity(standardize=True)

    assert result.isna().all().all()
    assert "no Z-score to calculate" in caplog.text


def test_bjerksund_stensland_warns_about_a_negative_cost_of_carry(
    options_module, caplog
):
    options_module.get_american_option(
        method="bjerksund_stensland", risk_free_rate=-0.01, expiration_time_range=3
    )
    assert caplog.text.count('Use method="qd_plus"') == 1

    caplog.clear()
    options_module.get_american_option(
        method="bjerksund_stensland",
        risk_free_rate=0.05,
        dividend_yield=0.01,
        expiration_time_range=3,
    )
    assert "qd_plus" not in caplog.text


@pytest.mark.parametrize(
    ("method", "arguments", "name"),
    [
        ("get_stock_price_simulation", {"time_steps": 2000}, "time_steps"),
        ("get_stock_price_simulation", {"time_steps": 17}, "time_steps"),
        ("get_binomial_model", {"time_steps": 2000}, "time_steps"),
        ("get_delta_hedging_simulation", {"volatility": -0.2}, "volatility"),
        (
            "get_delta_hedging_simulation",
            {"implied_volatility": -0.1},
            "implied_volatility",
        ),
        (
            "get_delta_hedging_simulation",
            {"transaction_cost": -0.01},
            "transaction_cost",
        ),
        ("get_delta_hedging_backtest", {"volatility": -0.2}, "volatility"),
        ("get_delta_hedging_backtest", {"transaction_cost": -0.01}, "transaction_cost"),
    ],
)
def test_the_options_refuse_invalid_arguments_once_for_every_ticker(
    options_module, caplog, method, arguments, name
):
    with pytest.raises(InvalidArgumentError, match=name):
        getattr(options_module, method)(**arguments)
    # Raised before any ticker is priced, rather than logged per ticker.
    assert "for AAPL" not in caplog.text and "for MSFT" not in caplog.text


def test_the_delta_hedging_simulation_leaves_out_a_ticker_without_prices(
    test_toolkit, caplog
):
    module = test_toolkit.options
    prices = module._prices.copy()
    prices["MSFT"] = np.nan
    module.__dict__["_prices"] = prices

    result = module.get_delta_hedging_simulation(simulations=100)

    assert list(result.index) == ["AAPL"]
    assert "No delta hedging simulation for MSFT" in caplog.text


def test_only_expired_expiries_give_one_warning_for_the_volatility_surface(
    option_market, monkeypatch, caplog
):
    module, labels = option_market
    # Valued two months on, every expiry of the chains has expired.
    monkeypatch.setattr(
        module, "_valuation_time", module._valuation_time + pd.Timedelta(days=60)
    )

    assert module.get_volatility_surface(expiration_dates=list(labels)).empty
    assert caplog.text.count("None of the expiries") == 1


def test_a_history_too_short_for_a_volatility_is_said_once_per_call(
    test_toolkit, caplog
):
    module = test_toolkit.options
    module.__dict__["_daily_historical"] = module._daily_historical.iloc[:2]

    values = module.get_black_scholes_model()
    assert values.isna().all().all()
    assert caplog.text.count("too few for a volatility") == 1

    # A later call on the same Options says it again, as its values are NaN as well.
    caplog.clear()
    module.get_delta()
    assert caplog.text.count("too few for a volatility") == 1

    # The delta hedging names each ticker it leaves out itself, so not again for the call.
    caplog.clear()
    module.get_delta_hedging_simulation(simulations=10)
    assert "too few for a volatility" not in caplog.text
    assert "No delta hedging simulation for AAPL: it has too few prices" in caplog.text


def test_a_ticker_without_a_price_is_said_once_for_all_greeks(test_toolkit, caplog):
    module = test_toolkit.options
    prices = module._prices.copy()
    prices.iloc[-1, prices.columns.get_loc("MSFT")] = np.nan
    module.__dict__["_prices"] = prices

    module.collect_all_greeks()

    assert caplog.text.count("MSFT has no price on") == 1


def test_the_options_take_numpy_whole_numbers(options_module):
    # Counts read from a DataFrame are numpy integers.
    assert not options_module.get_monte_carlo_option_price(
        time_steps=np.int64(3), simulations=np.int64(200)
    ).empty
    with pytest.raises(InvalidArgumentError, match="whole number"):
        options_module.get_binomial_model(time_steps=10.0)


def test_delta_hedging_at_zero_volatility_sells_the_discounted_intrinsic_value(
    options_module,
):
    import numpy as np

    from financetoolkit.utilities.error_model import InvalidArgumentError

    # At zero volatility the option is worth its discounted intrinsic value and the delta
    # steps from 0 to 1, so both methods return finite numbers rather than an error.
    backtest = options_module.get_delta_hedging_backtest(volatility=0.0, rounding=None)
    simulation = options_module.get_delta_hedging_simulation(
        volatility=0.0, implied_volatility=0.0, simulations=200, rounding=None
    )

    assert np.isfinite(backtest["Profit and Loss"].to_numpy(dtype=float)).all()
    assert np.isfinite(simulation["Option Value"].to_numpy(dtype=float)).all()
    assert (simulation["Option Value"] >= 0).all()

    # Only a negative volatility is refused, once for every ticker.
    with pytest.raises(InvalidArgumentError, match="volatility must be 0 or above"):
        options_module.get_delta_hedging_backtest(volatility=-0.1)
    with pytest.raises(
        InvalidArgumentError, match="implied_volatility must be 0 or above"
    ):
        options_module.get_delta_hedging_simulation(implied_volatility=-0.1)


def test_a_ticker_without_any_price_is_not_named_for_a_short_history(
    test_toolkit, caplog
):
    module = test_toolkit.options
    historical = module._daily_historical.astype(float)
    historical.loc[:, (slice(None), "MSFT")] = np.nan
    module.__dict__["_daily_historical"] = historical

    with caplog.at_level("WARNING", logger="financetoolkit"):
        module.get_black_scholes_model()

    # MSFT has no price at all, which its own message says, rather than too few returns.
    assert "MSFT has fewer than" not in caplog.text
    assert "too few for a volatility" not in caplog.text


def test_the_delta_hedging_backtest_leaves_out_a_ticker_without_a_volatility(
    test_toolkit, caplog
):
    module = test_toolkit.options
    historical = module._daily_historical.astype(float)
    # Thirty prices cover a hedge of 20 days, but not the 21 returns a volatility needs.
    historical.loc[historical.index[:-30], (slice(None), "MSFT")] = np.nan
    module.__dict__["_daily_historical"] = historical

    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = module.get_delta_hedging_backtest(days=20)

    assert list(result.index) == ["AAPL"]
    assert caplog.text.count("No delta hedging backtest for MSFT") == 1
    assert "too few for a volatility" not in caplog.text
