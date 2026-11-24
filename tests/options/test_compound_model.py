"""Compound-Style Exotic Option Model Tests"""

import numpy as np
import pytest
from scipy.special import ndtr

from financetoolkit.options.pricing import compound_model

# QuantLib 1.44 reference prices for S = 100, r = 5%, q = 2% and σ = 30% (Actual/365, whole
# days), from the analytic engines named in each test.
QUANTLIB = {"stock_price": 100.0, "risk_free_rate": 0.05, "volatility": 0.3}
DIVIDEND = {"dividend_yield": 0.02}


def days(number: int) -> float:
    return number / 365


def discounted_expectation(payoff, decision_time: float) -> float:
    """
    e^(-rt) * E[payoff(S(t))] under the QUANTLIB inputs, by the trapezoid rule over 400,001
    points of the standard normal variable within 12 deviations. The payoff has kinks at the
    exercise boundaries, which cost the rule at most about 1e-9.
    """
    z = np.linspace(-12, 12, 400_001)
    price = 100.0 * np.exp(
        (0.05 - 0.02 - 0.5 * 0.3**2) * decision_time + 0.3 * np.sqrt(decision_time) * z
    )
    density = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    return float(
        np.exp(-0.05 * decision_time) * np.trapezoid(payoff(price) * density, z)
    )


def black_scholes(price, strike: float, time: float, put: bool):
    """The Black-Scholes value under the QUANTLIB inputs, for a price or an array of prices."""
    deviation = 0.3 * np.sqrt(time)
    d1 = (np.log(price / strike) + (0.05 - 0.02 + 0.045) * time) / deviation
    call = price * np.exp(-0.02 * time) * ndtr(d1) - strike * np.exp(
        -0.05 * time
    ) * ndtr(d1 - deviation)

    return (
        call - price * np.exp(-0.02 * time) + strike * np.exp(-0.05 * time)
        if put
        else call
    )


# Haug (2007), The Complete Guide to Option Pricing Formulas, tabulated to four decimals. His
# compound (21.1965) and complex chooser (6.0508) values lie up to 1.5e-4 from the integral of
# the payoff at the decision time (21.19635 and 6.05072), so they are checked to 2e-4.
def test_the_values_match_haug():
    assert compound_model.get_chooser_option(
        50, 50, 0.08, 0.25, 0.5, 0.25
    ) == pytest.approx(6.1071, abs=2e-4)
    assert compound_model.get_chooser_option(
        50, 55, 0.1, 0.35, 0.5, 0.25, 0.05, 48, 0.5833
    ) == pytest.approx(6.0508, abs=2e-4)
    assert compound_model.get_compound_option(
        500, 520, 50, 0.08, 0.35, 0.5, 0.25, 0.03, "put_on_call"
    ) == pytest.approx(21.1965, abs=2e-4)
    assert compound_model.get_forward_start_option(
        60, 0.08, 0.3, 1, 0.25, 1.1, 0.04
    ) == pytest.approx(4.4064, abs=2e-4)
    assert compound_model.get_extendible_option(
        80, 90, 82, 0.1, 0.3, 0.5, 0.75, method="writer"
    ) == pytest.approx(6.8238, abs=2e-4)
    assert compound_model.get_quanto_option(
        100, 105, 0.08, 0.05, 0.2, 0.1, 0.3, 0.5, 0.04, fixed_exchange_rate=1.5
    ) == pytest.approx(5.3280, abs=2e-4)


@pytest.mark.parametrize(
    "strike, expected",
    [(90, 16.3243251955), (100, 14.1308533594), (110, 16.2739454799)],
)
def test_the_simple_chooser_matches_quantlib(strike, expected):
    # AnalyticSimpleChooserEngine, choosing after 90 days, expiring after 180 days.
    assert compound_model.get_chooser_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=float(strike),
        time_to_expiration=days(180),
        choice_time=days(90),
    ) == pytest.approx(expected, abs=1e-9)


def test_the_complex_chooser_matches_the_integral_of_its_choice():
    # A call (105, 180 days) or a put (95, 270 days), chosen after 90 days.
    value = compound_model.get_chooser_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=105.0,
        time_to_expiration=days(180),
        choice_time=days(90),
        put_strike_price=95.0,
        put_time_to_expiration=days(270),
    )
    expected = discounted_expectation(
        lambda price: np.maximum(
            black_scholes(price, 105, days(90), False),
            black_scholes(price, 95, days(180), True),
        ),
        days(90),
    )

    assert value == pytest.approx(expected, abs=1e-9)


def test_the_chooser_limits():
    inputs = {**QUANTLIB, **DIVIDEND, "strike_price": 100.0, "time_to_expiration": 0.5}
    call = black_scholes(100, 100, 0.5, False)
    put = black_scholes(100, 100, 0.5, True)

    # Choosing today takes the larger, choosing at expiration holds both (a straddle).
    assert compound_model.get_chooser_option(**inputs, choice_time=0) == pytest.approx(
        max(call, put), abs=1e-12
    )
    assert compound_model.get_chooser_option(
        **inputs, choice_time=0.5
    ) == pytest.approx(call + put, abs=1e-9)
    # The complex chooser with the same strike and expiration is the simple chooser.
    assert compound_model.get_chooser_option(
        **inputs, choice_time=0.25, put_strike_price=100.0, put_time_to_expiration=0.5
    ) == pytest.approx(
        compound_model.get_chooser_option(**inputs, choice_time=0.25), abs=1e-12
    )
    assert compound_model.get_chooser_option(
        **{**inputs, "strike_price": 100.0000001},
        choice_time=0.25,
        put_strike_price=100.0,
    ) == pytest.approx(
        compound_model.get_chooser_option(**inputs, choice_time=0.25), abs=1e-6
    )


# AnalyticCompoundOptionEngine with a compound strike of 5 after 90 days on an option with
# strike 100 expiring after 180 days. QuantLib's bivariate normal distribution is accurate to
# about 1e-5 here, so the exact value comes from integrating the payoff at the compound's
# expiration.
QUANTLIB_COMPOUNDS = {
    "call_on_call": 5.3429991473,
    "call_on_put": 3.8905747485,
    "put_on_call": 1.2885801309,
    "put_on_put": 1.2903044152,
}


@pytest.mark.parametrize("option_type", list(QUANTLIB_COMPOUNDS))
def test_the_compound_options_match_quantlib_and_their_integral(option_type):
    value = compound_model.get_compound_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=100.0,
        compound_strike_price=5.0,
        time_to_expiration=days(180),
        compound_time_to_expiration=days(90),
        option_type=option_type,
    )
    sign = 1 if option_type.startswith("call") else -1
    put = option_type.endswith("put")
    expected = discounted_expectation(
        lambda price: np.maximum(
            sign * (black_scholes(price, 100, days(90), put) - 5.0), 0.0
        ),
        days(90),
    )

    assert value == pytest.approx(QUANTLIB_COMPOUNDS[option_type], abs=5e-5)
    assert value == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("put", [False, True])
def test_the_compound_parity(put):
    # A call on an option minus a put on it is the option minus the discounted compound strike.
    inputs = {
        **QUANTLIB,
        **DIVIDEND,
        "strike_price": 95.0,
        "compound_strike_price": 6.0,
        "time_to_expiration": 1.0,
        "compound_time_to_expiration": 0.4,
    }
    underlying = "put" if put else "call"
    call = compound_model.get_compound_option(
        **inputs, option_type=f"call_on_{underlying}"
    )
    put_value = compound_model.get_compound_option(
        **inputs, option_type=f"put_on_{underlying}"
    )

    assert call - put_value == pytest.approx(
        black_scholes(100, 95, 1.0, put) - 6.0 * np.exp(-0.05 * 0.4), abs=1e-9
    )


def test_a_compound_strike_above_the_highest_put_value():
    # A put is worth at most K2 * e^(-r(T2 - T1)) at T1: a call on it at a higher compound
    # strike is worthless and a put on it is the discounted strike minus the put.
    inputs = {
        **QUANTLIB,
        **DIVIDEND,
        "strike_price": 50.0,
        "compound_strike_price": 60.0,
        "time_to_expiration": 1.0,
        "compound_time_to_expiration": 0.5,
    }

    assert compound_model.get_compound_option(
        **inputs, option_type="call_on_put"
    ) == pytest.approx(0.0, abs=1e-12)
    assert compound_model.get_compound_option(
        **inputs, option_type="put_on_put"
    ) == pytest.approx(60 * np.exp(-0.05 * 0.5) - black_scholes(100, 50, 1.0, True))


@pytest.mark.parametrize(
    "put, expected", [(False, 7.1416515666), (True, 14.5792712516)]
)
def test_the_forward_start_options_match_quantlib(put, expected):
    # ForwardEuropeanEngine, strike 110% of the price after 90 days, expiring after 360 days.
    assert compound_model.get_forward_start_option(
        **QUANTLIB,
        **DIVIDEND,
        time_to_expiration=days(360),
        start_time=days(90),
        moneyness=1.1,
        put_option=put,
    ) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "put, cliquet, performance",
    [(False, 24.8576689716, 0.2458533734), (True, 21.9457594789, 0.2170532968)],
)
def test_the_cliquets_match_quantlib(put, cliquet, performance):
    # AnalyticCliquetEngine and AnalyticPerformanceEngine, at the money, resets after 0, 90,
    # 180 and 270 days, expiring after 360 days.
    inputs = {
        **QUANTLIB,
        **DIVIDEND,
        "time_to_expiration": days(360),
        "reset_times": [0.0, days(90), days(180), days(270)],
        "put_option": put,
    }

    assert compound_model.get_cliquet_option(**inputs) == pytest.approx(
        cliquet, abs=1e-9
    )
    assert compound_model.get_cliquet_option(
        **inputs, performance=True
    ) == pytest.approx(performance, abs=1e-9)


def test_a_cliquet_with_one_reset_is_a_forward_start_option():
    inputs = {**QUANTLIB, **DIVIDEND, "time_to_expiration": 1.0, "moneyness": 0.95}

    assert compound_model.get_cliquet_option(
        **inputs, reset_times=[0.3]
    ) == pytest.approx(
        compound_model.get_forward_start_option(**inputs, start_time=0.3), abs=1e-12
    )
    # Starting today, it is the vanilla option at 95% of the price.
    assert compound_model.get_forward_start_option(
        **inputs, start_time=0
    ) == pytest.approx(black_scholes(100, 95, 1.0, False), abs=1e-12)


@pytest.mark.parametrize(
    "put, extended_strike, writer",
    [(False, 105.0, 9.5751458117), (True, 95.0, 8.0105594135)],
)
def test_the_writer_extendible_options_match_quantlib(put, extended_strike, writer):
    # AnalyticWriterExtensibleOptionEngine, strike 100 after 180 days, extended to 270 days.
    assert compound_model.get_extendible_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=100.0,
        extended_strike_price=extended_strike,
        time_to_expiration=days(180),
        extended_time_to_expiration=days(270),
        put_option=put,
        method="writer",
    ) == pytest.approx(writer, abs=1e-9)


@pytest.mark.parametrize("fee", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("put", [False, True])
def test_the_holder_extendible_options_match_the_integral_of_the_choice(put, fee):
    # At t1 the holder takes the most of exercising, extending for the fee or nothing.
    extended_strike = 95.0 if put else 105.0
    value = compound_model.get_extendible_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=100.0,
        extended_strike_price=extended_strike,
        time_to_expiration=days(180),
        extended_time_to_expiration=days(270),
        put_option=put,
        extension_fee=fee,
    )
    sign = -1 if put else 1
    expected = discounted_expectation(
        lambda price: np.maximum.reduce(
            [
                sign * (price - 100.0),
                black_scholes(price, extended_strike, days(90), put) - fee,
                np.zeros_like(price),
            ]
        ),
        days(180),
    )

    assert value == pytest.approx(expected, abs=1e-8)


def test_the_holder_extendible_call_matches_quantlib():
    # AnalyticHolderExtensibleOptionEngine for a call with a fee of 1; QuantLib's bivariate
    # normal distribution is accurate to about 1e-5. Its put prices below the vanilla put,
    # which an option to extend cannot be, so the put is checked against the integral only.
    assert compound_model.get_extendible_option(
        **QUANTLIB,
        **DIVIDEND,
        strike_price=100.0,
        extended_strike_price=105.0,
        time_to_expiration=days(180),
        extended_time_to_expiration=days(270),
        extension_fee=1.0,
    ) == pytest.approx(9.4466352258, abs=5e-5)


def test_a_holder_never_extends_when_the_fee_exceeds_the_extended_option():
    inputs = {
        **QUANTLIB,
        **DIVIDEND,
        "strike_price": 100.0,
        "extended_strike_price": 105.0,
        "time_to_expiration": 0.5,
        "extended_time_to_expiration": 0.75,
    }

    assert compound_model.get_extendible_option(
        **inputs, extension_fee=50.0
    ) == pytest.approx(black_scholes(100, 100, 0.5, False), abs=1e-12)


@pytest.mark.parametrize(
    "put, quanto, forward_start",
    [(False, 11.3249510634, 6.2180365637), (True, 11.2310205785, 15.6687391558)],
)
def test_the_quanto_options_match_quantlib(put, quanto, forward_start):
    # QuantoEuropeanEngine and QuantoForwardEuropeanEngine with a foreign rate of 3%, an
    # exchange rate volatility of 10% and a correlation of 0.3, strike 100 (forward start:
    # 110% of the price after 90 days), expiring after 360 days.
    inputs = {
        **QUANTLIB,
        **DIVIDEND,
        "foreign_risk_free_rate": 0.03,
        "exchange_rate_volatility": 0.1,
        "correlation": 0.3,
        "time_to_expiration": days(360),
        "put_option": put,
    }

    assert compound_model.get_quanto_option(
        **inputs, strike_price=100.0
    ) == pytest.approx(quanto, abs=1e-9)
    assert compound_model.get_quanto_option(
        **inputs, strike_price=110.0, start_time=days(90)
    ) == pytest.approx(forward_start, abs=1e-9)
    assert compound_model.get_quanto_option(
        **inputs, strike_price=100.0, fixed_exchange_rate=1.25
    ) == pytest.approx(1.25 * quanto, abs=1e-9)


def test_the_invalid_inputs_are_refused():
    with pytest.raises(ValueError, match="choice_time"):
        compound_model.get_chooser_option(100, 100, 0.05, 0.3, 0.5, 0.6)
    with pytest.raises(ValueError, match="option_type"):
        compound_model.get_compound_option(100, 100, 5, 0.05, 0.3, 1, 0.5, 0, "call")
    with pytest.raises(ValueError, match="compound_time_to_expiration"):
        compound_model.get_compound_option(100, 100, 5, 0.05, 0.3, 1, 1.5)
    with pytest.raises(ValueError, match="start_time"):
        compound_model.get_forward_start_option(100, 0.05, 0.3, 1, 1.5)
    with pytest.raises(ValueError, match="moneyness"):
        compound_model.get_forward_start_option(100, 0.05, 0.3, 1, 0.5, 0)
    with pytest.raises(ValueError, match="reset_times"):
        compound_model.get_cliquet_option(100, 0.05, 0.3, 1, [0.5, 0.25])
    with pytest.raises(ValueError, match="reset_times"):
        compound_model.get_cliquet_option(100, 0.05, 0.3, 1, [])
    with pytest.raises(ValueError, match="method"):
        compound_model.get_extendible_option(
            100, 100, 105, 0.05, 0.3, 0.5, 1, method="both"
        )
    with pytest.raises(ValueError, match="extension_fee"):
        compound_model.get_extendible_option(
            100, 100, 105, 0.05, 0.3, 0.5, 1, extension_fee=-1
        )
    with pytest.raises(ValueError, match="times"):
        compound_model.get_extendible_option(100, 100, 105, 0.05, 0.3, 1, 0.5)
    with pytest.raises(ValueError, match="correlation"):
        compound_model.get_quanto_option(100, 100, 0.05, 0.03, 0.3, 0.1, 1.5, 1)
    with pytest.raises(ValueError, match="volatility"):
        compound_model.get_chooser_option(100, 100, 0.05, 0, 0.5, 0.25)
    with pytest.raises(TypeError, match="stock_price"):
        compound_model.get_chooser_option("100", 100, 0.05, 0.3, 0.5, 0.25)


@pytest.mark.parametrize("choice_share", [0.0, 0.5, 1.0])
def test_a_simple_chooser_on_a_grid_equals_the_price_per_option(choice_share):
    strikes, times = np.meshgrid(
        np.arange(70.0, 135.0, 5.0), np.array([0, 1, 30, 182, 365]) / 365, indexing="ij"
    )
    inputs = {**QUANTLIB, **DIVIDEND}

    grid = compound_model.get_chooser_option(
        strike_price=strikes,
        time_to_expiration=times,
        choice_time=choice_share * times,
        **inputs,
    )
    per_option = np.array(
        [
            compound_model.get_chooser_option(
                strike_price=float(strike),
                time_to_expiration=float(time),
                choice_time=choice_share * float(time),
                **inputs,
            )
            for strike, time in zip(strikes.ravel(), times.ravel(), strict=True)
        ]
    ).reshape(strikes.shape)

    np.testing.assert_allclose(grid, per_option, rtol=0, atol=1e-12)
