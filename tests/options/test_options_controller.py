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
