"""Multi-Asset Option Model Tests"""

import numpy as np
import pytest
from scipy.special import ndtr

from financetoolkit.options.multiasset import multi_asset_model
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes

CORRELATION_MATRIX = [[1, 0.5, 0.3], [0.5, 1, 0.6], [0.3, 0.6, 1]]


def black_scholes(price, strike, rate, volatility, time, dividend=0.0, put=False):
    return float(
        get_black_scholes(
            price,
            strike,
            risk_free_rate=rate,
            volatility=volatility,
            time_to_expiration=time,
            dividend_yield=dividend,
            put_option=put,
        )
    )


def exact_spread_call(s1, s2, k, r, v1, v2, rho, t, q1, q2):
    """
    e^(-rT) * E[max(S1 - S2 - K, 0)], conditioning on S2: given S2 the first price is
    lognormal, so the inner expectation is a Black value, integrated by the trapezoid rule
    over 200,001 points of the standard normal variable within 12 deviations.
    """
    z = np.linspace(-12, 12, 200_001)
    density = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    first_forward = s1 * np.exp((r - q1) * t)
    second = s2 * np.exp((r - q2 - 0.5 * v2**2) * t + v2 * np.sqrt(t) * z)
    conditional = first_forward * np.exp(
        rho * v1 * np.sqrt(t) * z - 0.5 * rho**2 * v1**2 * t
    )
    deviation = v1 * np.sqrt((1 - rho**2) * t)
    strike = second + k
    # Where S2 + K is not positive the call is always exercised.
    positive = np.maximum(strike, 1e-300)
    d1 = np.log(conditional / positive) / deviation + 0.5 * deviation
    call = np.where(
        strike > 0,
        conditional * ndtr(d1) - positive * ndtr(d1 - deviation),
        conditional - strike,
    )

    return float(np.exp(-r * t) * np.trapezoid(call * density, z))


# Haug (2007), The Complete Guide to Option Pricing Formulas, tabulated to four decimals: Kirk's
# spread call on two futures (F1 = 28, F2 = 20, K = 7) and Zhang's two-asset correlation call.
def test_the_values_match_haug():
    assert multi_asset_model.get_spread_option(
        28, 20, 7, 0.05, 0.29, 0.36, 0.42, 0.25, 0.05, 0.05, method="kirk"
    ) == pytest.approx(2.1670, abs=5e-5)
    assert multi_asset_model.get_two_asset_correlation_option(
        52, 65, 50, 70, 0.1, 0.2, 0.3, 0.75, 0.5
    ) == pytest.approx(4.7073, abs=5e-5)


# QuantLib 1.44 reference prices (Actual/365, whole days). The closed forms agree to about
# 1e-13; the two-asset correlation engine's bivariate normal is accurate to about 2e-5, so that
# one is checked to 5e-5 (direct integration of the payoff agrees with this model to 1e-8).
# The American values are QuantLib's AnalyticAmericanMargrabeEngine, the Bjerksund-Stensland
# (1993) lower bound, and a 999-step Leisen-Reimer tree of the call it equals (an American call
# on S1 with strike S2, a rate of q2 and a dividend yield of q1).
@pytest.mark.parametrize(
    "inputs, european, lower_bound, tree",
    [
        (
            (110, 100, 0.35, 0.2, -0.5, 730 / 365, 0.03, 0.05),
            32.939027535045156,
            33.105612241878404,
            33.21240131661149,
        ),
        (
            (90, 100, 0.15, 0.4, 0.7, 1.0, 0.03, 0.0),
            6.409268140589177,
            6.489355072496679,
            6.549382861305851,
        ),
        (
            (100, 100, 0.3, 0.2, 0.4, 1.0, 0.05, 0.01),
            None,
            None,
            9.64551875415208,
        ),
    ],
)
def test_the_exchange_option_matches_quantlib_and_the_tree(
    inputs, european, lower_bound, tree
):
    if european is not None:
        assert multi_asset_model.get_exchange_option(*inputs) == pytest.approx(
            european, abs=1e-10
        )
    american = multi_asset_model.get_exchange_option(*inputs, american_option=True)
    # QD+ lies within 0.1% of the tree, where the 1993 approximation is up to 1% below it.
    assert american == pytest.approx(tree, rel=1.1e-3)
    if lower_bound is not None:
        assert american > lower_bound


@pytest.mark.parametrize(
    "inputs, put, kirk, bjerksund_stensland",
    [
        (
            (120, 100, 20, 0.05, 0.4, 0.25, 0.3, 1.0, 0.03, 0.02),
            False,
            17.843054996419138,
            17.84106925666204,
        ),
        (
            (120, 100, 20, 0.05, 0.4, 0.25, 0.3, 1.0, 0.03, 0.02),
            True,
            18.43404679128796,
            18.432061051530862,
        ),
        (
            (100, 100, -10, 0.05, 0.2, 0.5, -0.5, 91 / 365, 0.0, 0.0),
            False,
            18.29520241989323,
            18.266085444348057,
        ),
        (
            (100, 100, -10, 0.05, 0.2, 0.5, -0.5, 91 / 365, 0.0, 0.0),
            True,
            8.41908619758755,
            8.389969222042373,
        ),
    ],
)
def test_the_spread_option_matches_quantlib(inputs, put, kirk, bjerksund_stensland):
    assert multi_asset_model.get_spread_option(
        *inputs, put_option=put, method="kirk"
    ) == pytest.approx(kirk, abs=1e-10)
    assert multi_asset_model.get_spread_option(
        *inputs, put_option=put
    ) == pytest.approx(bjerksund_stensland, abs=1e-10)


def test_bjerksund_stensland_is_closer_to_the_exact_spread_price_than_kirk():
    inputs = (100, 100, -10, 0.05, 0.2, 0.5, -0.5, 91 / 365, 0.0, 0.0)
    exact = exact_spread_call(*inputs)

    bjerksund_stensland = multi_asset_model.get_spread_option(*inputs)
    kirk = multi_asset_model.get_spread_option(*inputs, method="kirk")

    # Bjerksund and Stensland give a lower bound of the exact price.
    assert bjerksund_stensland <= exact
    assert abs(bjerksund_stensland - exact) < abs(kirk - exact)


def test_the_spread_option_at_a_strike_of_zero_is_the_exchange_option():
    inputs = (105, 95, 0.0, 0.04, 0.3, 0.2, 0.4, 0.75, 0.01, 0.03)
    exchange = multi_asset_model.get_exchange_option(
        105, 95, 0.3, 0.2, 0.4, 0.75, 0.01, 0.03
    )

    for method in ("bjerksund_stensland", "kirk"):
        assert multi_asset_model.get_spread_option(
            *inputs, method=method
        ) == pytest.approx(exchange, abs=1e-12)


@pytest.mark.parametrize(
    "put, expected", [(False, 3.3628813223892777), (True, 1.9784595094478048)]
)
def test_the_two_asset_correlation_option_matches_quantlib(put, expected):
    assert multi_asset_model.get_two_asset_correlation_option(
        52, 75, 55, 70, 0.1, 0.2, 0.5, -0.6, 1.0, 0.03, 0.02, put_option=put
    ) == pytest.approx(expected, abs=5e-5)


def test_the_two_asset_correlation_option_without_a_first_condition_is_a_vanilla():
    # A first strike far below the price is always met, which leaves a vanilla on S2.
    for put in (False, True):
        value = multi_asset_model.get_two_asset_correlation_option(
            52,
            65,
            1e-6 if not put else 1e6,
            70,
            0.1,
            0.2,
            0.3,
            0.75,
            0.5,
            put_option=put,
        )
        assert value == pytest.approx(
            black_scholes(65, 70, 0.1, 0.3, 0.5, put=put), abs=1e-10
        )


@pytest.mark.parametrize(
    "option_type, expected",
    [
        ("call_on_max", 25.02815832534565),
        ("call_on_min", 0.944623102075766),
        ("put_on_max", 2.935576943844744),
        ("put_on_min", 26.13734863803721),
    ],
)
def test_the_rainbow_option_matches_quantlib(option_type, expected):
    assert multi_asset_model.get_rainbow_option(
        100, 90, 98, 0.05, 0.3, 0.45, -0.5, 1.0, 0.05, 0.02, option_type=option_type
    ) == pytest.approx(expected, abs=1e-10)


def test_the_best_of_and_worst_of_add_up_to_the_two_vanillas():
    inputs = (100, 105, 98, 0.05, 0.11, 0.16, 0.63, 0.5, 0.05, 0.02)
    for put in (False, True):
        kind = "put" if put else "call"
        best = multi_asset_model.get_rainbow_option(
            *inputs, option_type=f"{kind}_on_max"
        )
        worst = multi_asset_model.get_rainbow_option(
            *inputs, option_type=f"{kind}_on_min"
        )
        vanillas = black_scholes(100, 98, 0.05, 0.11, 0.5, 0.05, put) + black_scholes(
            105, 98, 0.05, 0.16, 0.5, 0.02, put
        )
        assert best + worst == pytest.approx(vanillas, abs=1e-12)


def test_the_basket_of_one_asset_is_black_scholes():
    expected = black_scholes(100, 95, 0.05, 0.3, 1.5, 0.02, False)
    for method in ("levy", "gentle"):
        assert multi_asset_model.get_basket_option(
            [100], [1], 95, 0.05, [0.3], [[1]], 1.5, [0.02], method=method
        ) == pytest.approx(expected, abs=1e-10)


# QuantLib's ChoiBasketEngine (λ = 20), accurate to about 1e-6, for a basket of three assets.
BASKET = {
    "stock_prices": [100, 90, 110],
    "weights": [0.5, 0.3, 0.2],
    "strike_price": 100,
    "risk_free_rate": 0.05,
    "volatilities": [0.2, 0.3, 0.4],
    "correlation_matrix": CORRELATION_MATRIX,
    "time_to_expiration": 730 / 365,
    "dividend_yields": [0.0, 0.02, 0.01],
}


@pytest.mark.parametrize(
    "put, choi", [(False, 15.27309971805401), (True, 8.251155851788035)]
)
def test_the_basket_approximations_are_close_to_choi(put, choi):
    # Levy's lognormal is within 2% here and Gentle's geometric basket within 5%, as Gentle is
    # the less accurate of the two at these volatilities.
    levy = multi_asset_model.get_basket_option(**BASKET, put_option=put)
    gentle = multi_asset_model.get_basket_option(
        **BASKET, put_option=put, method="gentle"
    )

    assert levy == pytest.approx(choi, rel=0.02)
    assert gentle == pytest.approx(choi, rel=0.05)
    assert levy == pytest.approx(
        15.404728783758948 if not put else 8.382784917493563, abs=1e-10
    )
    assert gentle == pytest.approx(
        14.903599785170922 if not put else 7.881655918905536, abs=1e-10
    )


@pytest.mark.parametrize(
    "put, choi", [(False, 15.27309971805401), (True, 8.251155851788035)]
)
def test_the_monte_carlo_basket_matches_choi(put, choi):
    price, standard_error = multi_asset_model.get_monte_carlo_basket_option(
        **BASKET, put_option=put, simulations=200_000, seed=7
    )

    assert abs(price - choi) < 4 * standard_error
    assert standard_error < 0.01


def test_the_monte_carlo_basket_is_reproducible_with_a_seed():
    first = multi_asset_model.get_monte_carlo_basket_option(**BASKET, seed=3)
    second = multi_asset_model.get_monte_carlo_basket_option(**BASKET, seed=3)

    assert first == second
    assert multi_asset_model.get_monte_carlo_basket_option(**BASKET) == (
        multi_asset_model.get_monte_carlo_basket_option(**BASKET)
    )


def test_a_negative_weight_prices_without_the_control_variate():
    price, standard_error = multi_asset_model.get_monte_carlo_basket_option(
        [100, 100], [1, -0.5], 40, 0.05, [0.3, 0.2], [[1, 0.5], [0.5, 1]], 1.0, seed=1
    )

    assert price > 0 and standard_error > 0
    with pytest.raises(ValueError, match="gentle"):
        multi_asset_model.get_basket_option(
            [100, 100],
            [1, -0.5],
            40,
            0.05,
            [0.3, 0.2],
            [[1, 0.5], [0.5, 1]],
            1.0,
            method="gentle",
        )


def test_the_options_at_expiration_are_worth_their_payoff():
    assert multi_asset_model.get_exchange_option(110, 100, 0.3, 0.2, 0.5, 0) == 10
    assert (
        multi_asset_model.get_spread_option(
            110, 100, 5, 0.05, 0.3, 0.2, 0.5, 0, put_option=True
        )
        == 0
    )
    assert (
        multi_asset_model.get_rainbow_option(
            110, 100, 105, 0.05, 0.3, 0.2, 0.5, 0, option_type="put_on_min"
        )
        == 5
    )
    assert (
        multi_asset_model.get_two_asset_correlation_option(
            110, 100, 105, 90, 0.05, 0.3, 0.2, 0.5, 0
        )
        == 10
    )
    assert multi_asset_model.get_monte_carlo_basket_option(
        [100, 100], [1, 1], 150, 0.05, [0.3, 0.2], [[1, 0.5], [0.5, 1]], 0
    ) == (50, 0.0)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: multi_asset_model.get_exchange_option(100, 100, 0.2, 0.2, 1.0, 1.0),
            "move as one",
        ),
        (
            lambda: multi_asset_model.get_exchange_option(100, 100, 0.2, 0.2, 1.5, 1.0),
            "correlation",
        ),
        (
            lambda: multi_asset_model.get_spread_option(
                100, 50, -60, 0.0, 0.2, 0.2, 0.5, 1.0
            ),
            "forward",
        ),
        (
            lambda: multi_asset_model.get_spread_option(
                100, 50, 0, 0.0, 0.2, 0.2, 0.5, 1.0, method="pearson"
            ),
            "method",
        ),
        (
            lambda: multi_asset_model.get_rainbow_option(
                100, 90, 95, 0.05, 0.2, 0.3, 0.5, 1.0, option_type="best"
            ),
            "option_type",
        ),
        (
            lambda: multi_asset_model.get_basket_option(
                [100, 90], [1, 1], 95, 0.05, [0.2, 0.3], [[1, 0.5], [0.4, 1]], 1.0
            ),
            "symmetric",
        ),
        (
            lambda: multi_asset_model.get_monte_carlo_basket_option(
                [100, 90], [1, 1], 95, 0.05, [0.2, 0.3], [[1, 1], [1, 1]], 1.0
            ),
            "positive definite",
        ),
    ],
)
def test_invalid_inputs_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_non_numeric_inputs_raise():
    with pytest.raises(TypeError):
        multi_asset_model.get_exchange_option("100", 100, 0.2, 0.2, 0.5, 1.0)


def test_the_exchange_option_refuses_a_second_price_of_zero():
    with pytest.raises(ValueError, match="second_stock_price must be above 0"):
        multi_asset_model.get_exchange_option(
            first_stock_price=100.0,
            second_stock_price=0.0,
            first_volatility=0.2,
            second_volatility=0.3,
            correlation=0.5,
            time_to_expiration=1.0,
        )
