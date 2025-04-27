"""Options Controller Tests""" ""
import pandas as pd

from financetoolkit import Toolkit

historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
risk_free_rate = pd.read_pickle("tests/datasets/risk_free_rate.pickle")
treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

toolkit = Toolkit(
    tickers=["AAPL", "MSFT"],
    historical=historical,
    convert_currency=False,
    start_date="2019-12-31",
    end_date="2023-01-01",
    sleep_timer=False,
)

toolkit._daily_risk_free_rate = risk_free_rate
toolkit._daily_treasury_data = treasury_data

options_module = toolkit.options

# pylint: disable=missing-function-docstring


def test_get_binomial_model(recorder):
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


def test_get_black_scholes_model(recorder):
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


def test_collect_all_greeks(recorder):
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


def test_collect_first_order_greeks(recorder):
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


def test_get_delta(recorder):
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


def test_get_dual_delta(recorder):
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


def test_get_vega(recorder):
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


def test_get_theta(recorder):
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


def test_get_rho(recorder):
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


def test_get_epsilon(recorder):
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


def test_get_lambda(recorder):
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


def test_collect_second_order_greeks(recorder):
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


def test_get_gamma(recorder):
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


def test_get_dual_gamma(recorder):
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


def test_get_vanna(recorder):
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


def test_get_charm(recorder):
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


def test_get_vomma(recorder):
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


def test_get_vera(recorder):
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


def test_get_veta(recorder):
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


def test_get_partial_derivative(recorder):
    recorder.capture(options_module.get_partial_derivative())
    recorder.capture(
        options_module.get_partial_derivative(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
        )
    )


def test_collect_third_order_greeks(recorder):
    recorder.capture(options_module.collect_third_order_greeks())
    recorder.capture(
        options_module.collect_third_order_greeks(
            strike_price_range=0.10,
            strike_step_size=2,
            expiration_time_range=5,
            risk_free_rate=0.01,
        )
    )


def test_get_speed(recorder):
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


def test_get_zomma(recorder):
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


def test_get_color(recorder):
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


def test_get_ultima(recorder):
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
