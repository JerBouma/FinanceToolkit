"""Options Controller Tests""" ""
import pandas as pd
import pytest

# pylint: disable=missing-function-docstring


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
            timesteps=2,
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
    recorder.capture(options_module.get_stock_price_simulation(timesteps=5))


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


def test_get_bjerksund_stensland(recorder, options_module):
    recorder.capture(options_module.get_bjerksund_stensland())
    recorder.capture(options_module.get_bjerksund_stensland(put_option=True))
    recorder.capture(
        options_module.get_bjerksund_stensland(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
            dividend_yield=0.02,
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
            barrier_percentage=0.8,
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
    with pytest.raises(ValueError):
        options_module.get_strategy_payoff(legs=[])

    with pytest.raises(TypeError):
        options_module.get_strategy_payoff(legs=[{"put_option": False}])


@pytest.fixture
def option_market(monkeypatch, options_module):
    """Option chains priced with Black-Scholes around each ticker's last price."""
    import numpy as np  # noqa: PLC0415

    from financetoolkit.options import (  # noqa: PLC0415
        black_scholes_model,
        options_model,
    )

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
    with pytest.raises(ValueError, match="longest usable expiry"):
        module.get_model_free_implied_volatility(days=365)


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
