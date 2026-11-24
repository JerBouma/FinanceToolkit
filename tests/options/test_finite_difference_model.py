"""Finite Difference Model Tests"""

import numpy as np
import pytest

from financetoolkit.options.numerical import finite_difference_model
from financetoolkit.options.pricing.american_model import get_qd_fixed_point
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.options.pricing.exotics_model import get_barrier_option

INPUTS = {
    "stock_price": 100.0,
    "strike_price": 100.0,
    "risk_free_rate": 0.05,
    "volatility": 0.2,
    "time_to_expiration": 1.0,
    "dividend_yield": 0.02,
}


def _put(**kwargs):
    return finite_difference_model.get_finite_difference_option(
        **{**INPUTS, "put_option": True, **kwargs}
    )


def test_crank_nicolson_with_rannacher_steps_is_second_order():
    exact = get_black_scholes(100, 100, 0.05, 0.2, 1.0, 0.02, True)
    errors = [
        abs(_put(price_steps=steps, time_steps=steps) - exact)
        for steps in (100, 200, 400)
    ]

    # Doubling the steps in price and time divides the error by four.
    assert 3.6 < errors[0] / errors[1] < 4.4
    assert 3.6 < errors[1] / errors[2] < 4.4
    assert errors[2] < 1e-3


def test_the_implicit_scheme_is_first_order_in_time():
    exact = get_black_scholes(100, 100, 0.05, 0.2, 1.0, 0.02, True)
    errors = [
        abs(_put(scheme="implicit", price_steps=800, time_steps=steps) - exact)
        for steps in (50, 100, 200)
    ]

    assert 1.7 < errors[0] / errors[1] < 2.3
    assert 1.7 < errors[1] / errors[2] < 2.3


def test_the_explicit_scheme_needs_small_time_steps():
    with pytest.raises(ValueError, match="unstable"):
        _put(scheme="explicit", price_steps=200, time_steps=200)

    exact = get_black_scholes(100, 100, 0.05, 0.2, 1.0, 0.02, True)
    assert _put(scheme="explicit", price_steps=200, time_steps=2000) == pytest.approx(
        exact, abs=3e-3
    )


def test_european_values_match_black_scholes_and_the_references():
    # QuantLib's FdBlackScholesVanillaEngine (800 x 800, Crank-Nicolson) gives 6.330109 and
    # FinancePy's black_scholes_fd (800 x 800) 6.329916 for this put.
    value = _put(price_steps=800, time_steps=800)

    assert value == pytest.approx(6.330080627549918, abs=2e-4)
    assert value == pytest.approx(6.330108843412492, abs=2e-4)
    assert value == pytest.approx(6.329916315042896, abs=2e-4)


def test_american_put_matches_the_fixed_point_and_the_references():
    exact = get_qd_fixed_point(100, 100, 0.05, 0.2, 1.0, 0.02, True)
    value = _put(american_option=True, price_steps=800, time_steps=800)

    # QuantLib's FdBlackScholesVanillaEngine gives 6.660162 and FinancePy's
    # black_scholes_fd 6.660009 on 800 x 800 grids.
    assert value == pytest.approx(exact, abs=5e-4)
    assert value == pytest.approx(6.6601622132166485, abs=5e-4)
    assert value == pytest.approx(6.660009260122683, abs=5e-4)


def test_american_call_with_a_high_dividend_matches_the_fixed_point():
    value = finite_difference_model.get_finite_difference_option(
        100, 100, 0.03, 0.25, 1.0, 0.07, american_option=True
    )

    # QuantLib's FdBlackScholesVanillaEngine (800 x 800) gives 8.163973.
    assert value == pytest.approx(
        get_qd_fixed_point(100, 100, 0.03, 0.25, 1.0, 0.07), abs=2e-3
    )
    assert value == pytest.approx(8.163973457388666, abs=2e-3)


def test_psor_and_brennan_schwartz_solve_the_same_problem():
    brennan_schwartz = _put(american_option=True, price_steps=100, time_steps=50)
    psor = _put(
        american_option=True,
        price_steps=100,
        time_steps=50,
        exercise_method="psor",
    )

    assert psor == pytest.approx(brennan_schwartz, abs=1e-9)


def test_an_american_call_without_dividends_is_european():
    european = finite_difference_model.get_finite_difference_option(
        100, 110, 0.05, 0.3, 0.5
    )
    american = finite_difference_model.get_finite_difference_option(
        100, 110, 0.05, 0.3, 0.5, american_option=True
    )

    assert american == pytest.approx(european, abs=1e-10)


def test_one_solve_gives_every_strike_and_expiration():
    strikes = np.array([90.0, 100.0, 110.0])
    times = np.array([0.0, 30 / 365, 0.5, 1.0])
    values = finite_difference_model.get_finite_difference_option_values(
        100, strikes, times, 0.04, 0.25, 0.03, True, True, time_steps=400
    )

    np.testing.assert_allclose(values[:, 0], [0.0, 0.0, 10.0])
    for row, strike in enumerate(strikes):
        for column, time in enumerate(times[1:], start=1):
            assert values[row, column] == pytest.approx(
                get_qd_fixed_point(100, strike, 0.04, 0.25, time, 0.03, True),
                # The 30-day expiration gets about 33 of the 400 time steps.
                abs=1.5e-3,
            )


@pytest.mark.parametrize("direction, barrier", [("down", 90.0), ("up", 120.0)])
@pytest.mark.parametrize("knock_type", ["out", "in"])
@pytest.mark.parametrize("put_option", [False, True])
def test_barrier_options_with_a_rebate_match_reiner_rubinstein(
    direction, barrier, knock_type, put_option
):
    closed_form = get_barrier_option(
        100, 100, barrier, 0.05, 0.2, 1.0, 0.02, put_option, direction, knock_type, 3.0
    )
    value = finite_difference_model.get_finite_difference_barrier_option(
        100,
        100,
        barrier,
        0.05,
        0.2,
        1.0,
        0.02,
        put_option,
        direction,
        knock_type,
        3.0,
    )

    # QuantLib's FdBlackScholesBarrierEngine (800 x 800) is within 3e-3 of the closed
    # forms on these options; this grid is within 3e-4.
    assert value == pytest.approx(closed_form, abs=3e-4)


def test_a_barrier_already_hit_knocks_the_option_out_or_in():
    knocked_out = finite_difference_model.get_finite_difference_barrier_option(
        85, 100, 90, 0.05, 0.2, 1.0, rebate=2.5
    )
    knocked_in = finite_difference_model.get_finite_difference_barrier_option(
        85, 100, 90, 0.05, 0.2, 1.0, knock_type="in"
    )

    assert knocked_out == 2.5
    assert knocked_in == pytest.approx(get_black_scholes(85, 100, 0.05, 0.2, 1.0))


@pytest.mark.parametrize(
    "put_option, dividend_yield, quantlib",
    [
        (False, 0.0, 12.836375431569438),
        (False, 0.03, 11.116584468191432),
        (True, 0.0, 7.814106701412249),
        (True, 0.03, 8.899879951682575),
    ],
)
def test_shout_option_matches_quantlib(put_option, dividend_yield, quantlib):
    value = finite_difference_model.get_finite_difference_shout_option(
        100,
        100,
        0.05,
        0.2,
        1.0,
        dividend_yield,
        put_option,
        time_steps=400,
    )

    # QuantLib's FdBlackScholesShoutEngine on an 800 x 800 grid; this 400 x 400 grid is
    # within 1.3e-3 of it and 200 x 200 within 3e-4.
    assert value == pytest.approx(quantlib, abs=3e-3)
    assert value > get_black_scholes(
        100, 100, 0.05, 0.2, 1.0, dividend_yield, put_option
    )


def test_values_are_stable_under_tiny_input_changes():
    for function in (
        lambda price: _put(stock_price=price, american_option=True),
        lambda price: finite_difference_model.get_finite_difference_barrier_option(
            price, 100, 90, 0.05, 0.2, 1.0, rebate=3
        ),
        lambda price: finite_difference_model.get_finite_difference_shout_option(
            price, 100, 0.05, 0.2, 1.0
        ),
    ):
        assert function(100 + 1e-12) == pytest.approx(function(100.0), abs=1e-9)


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="scheme"):
        _put(scheme="hopscotch")
    with pytest.raises(ValueError, match="exercise_method"):
        _put(exercise_method="penalty")
    with pytest.raises(ValueError, match="volatility"):
        _put(volatility=0.0)
    with pytest.raises(ValueError, match="barrier_direction"):
        finite_difference_model.get_finite_difference_barrier_option(
            100, 100, 90, 0.05, 0.2, 1.0, barrier_direction="sideways"
        )


def test_option_values_stay_precise_a_day_from_expiration_over_long_ranges():
    # One grid for every expiration from a day to a year was 2.6e-2 off a day from
    # expiration at a volatility of 100%; solved in groups of expirations whose longest is
    # at most four times their shortest, every value from a day to a quarter is within 1e-3.
    strikes = np.arange(75.0, 126.0, 5.0)
    times = np.arange(1, 92) / 365
    for volatility in (0.3, 1.0):
        values = finite_difference_model.get_finite_difference_option_values(
            100.0,
            strikes,
            times,
            0.04,
            volatility,
            0.01,
            put_option=True,
            time_steps=400,
        )
        exact = np.array(
            [
                [
                    float(
                        get_black_scholes(
                            100.0, strike, 0.04, volatility, time, 0.01, True
                        )
                    )
                    for time in times
                ]
                for strike in strikes
            ]
        )
        assert np.abs(values - exact).max() < 1e-3
