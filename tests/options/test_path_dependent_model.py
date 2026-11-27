"""Path-Dependent Option Model Tests"""

import numpy as np
import pytest

from financetoolkit.options.helpers import bivariate_normal_cdf
from financetoolkit.options.pricing import (
    black_scholes_model,
    exotics_model,
    path_dependent_model,
)

# QuantLib 1.44 reference prices for S = 100, r = 5%, q = 2%, σ = 30% and 360 days to expiration
# (Actual/365), from the analytic engines named in each test; the windows and fixings fall on
# whole days (180 days, every 30 days), so the dates match exactly.
QUANTLIB = {"stock_price": 100.0, "risk_free_rate": 0.05, "volatility": 0.3}
QUANTLIB_TIME = {"time_to_expiration": 360 / 365, "dividend_yield": 0.02}

# Strike, put, fixed, floating, partial fixed watched from 180 days and partial floating watched
# up to 180 days (AnalyticContinuous(Partial)Fixed/FloatingLookbackEngine).
QUANTLIB_LOOKBACKS = [
    (90, False, 36.1813132597, 22.3745890631, 30.4223590475, 19.8610501691),
    (110, False, 18.4318129214, 22.3745890631, 16.4795900866, 19.8610501691),
    (90, True, 11.2369532632, 23.803893213, 10.0803573043, 19.3571990548),
    (110, True, 29.03479251, 23.803893213, 24.4657758033, 19.3571990548),
]

# Strike, put, Turnbull-Wakeman and geometric with 12 fixings every 30 days, continuous Turnbull-Wakeman (the
# Levy engine, which gives the same price for an average that starts today) and
# the control variate Monte Carlo price with 1,000,000 paths and its standard error
# (TurnbullWakemanAsianEngine, AnalyticDiscreteGeometricAveragePriceAsianEngine,
# ContinuousArithmeticAsianLevyEngine and MCDiscreteArithmeticAPEngine).
QUANTLIB_ASIANS = [
    (90, False, 13.6119575787, 13.0301437188, 13.2133344229, 13.539468, 0.000731),
    (100, False, 7.8285501929, 7.3824947877, 7.3567839098, 7.791521, 0.000743),
    (110, True, 12.0408181263, 12.4536958421, 11.7323751379, 12.054475, 0.000462),
]


@pytest.mark.parametrize("case", QUANTLIB_LOOKBACKS)
def test_the_lookbacks_match_quantlib(case):
    strike, put, fixed, floating, partial_fixed, partial_floating = case
    inputs = {
        **QUANTLIB,
        **QUANTLIB_TIME,
        "strike_price": float(strike),
        "put_option": put,
    }

    assert path_dependent_model.get_lookback_option(**inputs) == pytest.approx(
        fixed, abs=1e-9
    )
    assert path_dependent_model.get_lookback_option(
        **inputs, strike_type="floating"
    ) == pytest.approx(floating, abs=1e-9)
    assert path_dependent_model.get_lookback_option(
        **inputs, lookback_period=0.5
    ) == pytest.approx(partial_fixed, abs=1e-9)
    assert path_dependent_model.get_lookback_option(
        **inputs, strike_type="floating", lookback_period=0.5
    ) == pytest.approx(partial_floating, abs=1e-9)


@pytest.mark.parametrize("case", QUANTLIB_ASIANS)
def test_the_asian_options_match_quantlib(case):
    strike, put, turnbull_wakeman, geometric, continuous, monte_carlo, error = case
    inputs = {
        **QUANTLIB,
        **QUANTLIB_TIME,
        "strike_price": float(strike),
        "put_option": put,
    }

    assert path_dependent_model.get_turnbull_wakeman(
        **inputs, fixings=12
    ) == pytest.approx(turnbull_wakeman, abs=1e-9)
    assert path_dependent_model.get_geometric_asian_option(
        **inputs, fixings=12
    ) == pytest.approx(geometric, abs=1e-9)
    assert path_dependent_model.get_turnbull_wakeman(**inputs) == pytest.approx(
        continuous, abs=1e-6
    )
    # Curran is an approximation: within 0.01 of the simulated price at this volatility.
    assert path_dependent_model.get_curran(**inputs, fixings=12) == pytest.approx(
        monte_carlo, abs=0.01
    )
    prices, errors = path_dependent_model.get_monte_carlo_asian_options(
        **{key: value for key, value in inputs.items() if key != "strike_price"},
        strike_prices=np.array([float(strike)]),
        fixings=12,
        simulations=200_000,
        seed=7,
    )
    assert abs(prices[0] - monte_carlo) < 4 * np.hypot(errors[0], error)


def test_the_barriers_and_touches_match_quantlib():
    inputs = {**QUANTLIB, **QUANTLIB_TIME}

    # AnalyticDoubleBarrierEngine, barriers 80 and 120.
    for strike, put, knock_out, knock_in in (
        (95.0, False, 0.4501941933, 14.9240575135),
        (105.0, True, 0.6652438496, 12.0327448148),
    ):
        for knock_type, expected in (("out", knock_out), ("in", knock_in)):
            assert path_dependent_model.get_double_barrier_option(
                **inputs,
                strike_price=strike,
                lower_barrier=80.0,
                upper_barrier=120.0,
                put_option=put,
                knock_type=knock_type,
            ) == pytest.approx(expected, abs=1e-9)

    # AnalyticPartialTimeBarrierOptionEngine (calls, start window of 180 days), whose bivariate
    # normal distribution is an approximation accurate to about 1e-5 of the price.
    for strike, direction, knock_type, barrier, expected in (
        (100.0, "down", "out", 90.0, 8.7456405597),
        (95.0, "up", "in", 110.0, 13.6455870012),
    ):
        assert path_dependent_model.get_partial_time_barrier_option(
            **inputs,
            strike_price=strike,
            barrier=barrier,
            barrier_direction=direction,
            knock_type=knock_type,
            barrier_window="start",
            window_share=0.5,
        ) == pytest.approx(expected, abs=1e-4)

    # AnalyticSoftBarrierEngine: a down-and-out call and an up-and-in put.
    assert path_dependent_model.get_soft_barrier_option(
        **inputs, strike_price=100.0, first_barrier=85.0, second_barrier=95.0
    ) == pytest.approx(8.2733695224, abs=1e-9)
    assert path_dependent_model.get_soft_barrier_option(
        **inputs,
        strike_price=100.0,
        first_barrier=105.0,
        second_barrier=115.0,
        put_option=True,
        barrier_direction="up",
        knock_type="in",
    ) == pytest.approx(4.2117638955, abs=1e-9)

    # AnalyticDigitalAmericanEngine (at the hit and at expiration) and AnalyticBinaryBarrierEngine.
    for barrier, at_hit, at_expiration, no_touch in (
        (85.0, 0.5912019635, 0.5723125403, 0.3795686344),
        (120.0, 0.514602198, 0.4989315387, 0.452949636),
    ):
        assert path_dependent_model.get_touch_option(
            **inputs, barrier=barrier, payment="hit"
        ) == pytest.approx(at_hit, abs=1e-9)
        assert path_dependent_model.get_touch_option(
            **inputs, barrier=barrier
        ) == pytest.approx(at_expiration, abs=1e-9)
        assert path_dependent_model.get_touch_option(
            **inputs, barrier=barrier, option_type="no-touch"
        ) == pytest.approx(no_touch, abs=1e-9)

    # AnalyticDoubleBarrierBinaryEngine, barriers 85 and 115.
    for option_type, expected in (
        ("double-no-touch", 0.0099711082),
        ("double-one-touch", 0.9419100665),
    ):
        assert path_dependent_model.get_double_touch_option(
            **inputs, lower_barrier=85.0, upper_barrier=115.0, option_type=option_type
        ) == pytest.approx(expected, abs=1e-9)


def test_the_double_barrier_with_the_strike_outside_the_corridor():
    """Only the prices between the barriers pay: a call struck below the lower barrier pays from
    the lower barrier on. Checked against a Brownian-bridge simulation (200,000 paths of 400 steps,
    12.9001 ± 0.0225), where the formula with the strike in place of the lower barrier gives 12.79.
    """
    assert path_dependent_model.get_double_barrier_option(
        stock_price=100.0,
        strike_price=80.0,
        lower_barrier=85.0,
        upper_barrier=115.0,
        risk_free_rate=0.01,
        volatility=0.15,
        time_to_expiration=0.5,
        dividend_yield=0.03,
    ) == pytest.approx(12.9007, abs=1e-4)


INPUTS = {
    "stock_price": 100.0,
    "risk_free_rate": 0.05,
    "volatility": 0.3,
    "time_to_expiration": 1.0,
    "dividend_yield": 0.02,
}


def test_the_bivariate_normal_distribution_is_exact():
    """Owen's T gives M(0, 0; ρ) = 1/4 + arcsin(ρ) / (2π) and the limits at infinity."""
    for correlation in (-0.9, -0.3, 0.0, 0.5, 0.95):
        assert bivariate_normal_cdf(0, 0, correlation) == pytest.approx(
            0.25 + np.arcsin(correlation) / (2 * np.pi), abs=1e-15
        )
    assert bivariate_normal_cdf(np.inf, 0.3, 0.2) == pytest.approx(0.6179114221889526)
    assert bivariate_normal_cdf(1.0, 1.0, 1.0) == pytest.approx(0.8413447460685429)
    assert bivariate_normal_cdf(-np.inf, 1.0, 0.5) == 0.0


@pytest.mark.parametrize("put_option", [False, True])
def test_a_continuous_geometric_average_is_kemna_vorst(put_option):
    """Without fixings the geometric Asian option is the existing Kemna-Vorst price."""
    for strike_price in (80.0, 100.0, 120.0):
        assert path_dependent_model.get_geometric_asian_option(
            **INPUTS, strike_price=strike_price, put_option=put_option
        ) == pytest.approx(
            exotics_model.get_asian_option(
                **INPUTS, strike_price=strike_price, put_option=put_option
            ),
            rel=1e-12,
        )


def test_many_fixings_approach_the_continuous_average():
    """With many fixings the discrete approximations approach the continuous ones."""
    for function in (
        path_dependent_model.get_geometric_asian_option,
        path_dependent_model.get_turnbull_wakeman,
        path_dependent_model.get_curran,
    ):
        continuous = function(**INPUTS, strike_price=100.0)
        discrete = function(**INPUTS, strike_price=100.0, fixings=20_000)
        assert discrete == pytest.approx(continuous, rel=1e-4)


def test_the_arithmetic_average_lies_above_the_geometric_one():
    """A ≥ G, so the arithmetic call is worth more and the arithmetic put less."""
    for strike_price in (80.0, 100.0, 120.0):
        for fixings in (None, 12):
            geometric_call = path_dependent_model.get_geometric_asian_option(
                **INPUTS, strike_price=strike_price, fixings=fixings
            )
            geometric_put = path_dependent_model.get_geometric_asian_option(
                **INPUTS, strike_price=strike_price, fixings=fixings, put_option=True
            )
            curran_call = path_dependent_model.get_curran(
                **INPUTS, strike_price=strike_price, fixings=fixings
            )
            curran_put = path_dependent_model.get_curran(
                **INPUTS, strike_price=strike_price, fixings=fixings, put_option=True
            )
            assert curran_call > geometric_call
            assert curran_put < geometric_put


def test_the_monte_carlo_asian_option_is_seeded_and_near_curran():
    """The control variate keeps the simulation within a few standard errors of Curran."""
    strike_prices = np.array([90.0, 100.0, 110.0])
    first = path_dependent_model.get_monte_carlo_asian_options(
        **INPUTS, strike_prices=strike_prices, fixings=12, simulations=20_000, seed=3
    )
    second = path_dependent_model.get_monte_carlo_asian_options(
        **INPUTS, strike_prices=strike_prices, fixings=12, simulations=20_000, seed=3
    )
    np.testing.assert_array_equal(first[0], second[0])

    for strike_price, price, error in zip(strike_prices, *first, strict=True):
        curran = path_dependent_model.get_curran(
            **INPUTS, strike_price=float(strike_price), fixings=12
        )
        assert abs(price - curran) < max(5 * error, 0.01)

    with pytest.raises(ValueError, match="too large"):
        path_dependent_model.get_monte_carlo_asian_options(
            **INPUTS, strike_prices=strike_prices, fixings=10_000, simulations=10_000
        )


def test_a_full_window_is_the_regular_barrier_option():
    """A window over (almost) the whole life gives the Reiner-Rubinstein price."""
    for put_option in (False, True):
        for direction, barrier in (("down", 90.0), ("up", 110.0)):
            regular = exotics_model.get_barrier_option(
                **INPUTS,
                strike_price=100.0,
                barrier=barrier,
                put_option=put_option,
                barrier_direction=direction,
            )
            for window in ("start", "end"):
                partial = path_dependent_model.get_partial_time_barrier_option(
                    **INPUTS,
                    strike_price=100.0,
                    barrier=barrier,
                    put_option=put_option,
                    barrier_direction=direction,
                    barrier_window=window,
                    window_share=0.999999,
                )
                assert partial == pytest.approx(regular, abs=2e-3)


def test_in_out_parity_of_the_new_barriers():
    """Knock-in plus knock-out is the vanilla option for every new barrier."""
    for put_option in (False, True):
        vanilla = black_scholes_model.get_black_scholes(
            100.0, 95.0, 0.05, 0.3, 1.0, 0.02, put_option
        )
        prices = {
            knock_type: (
                path_dependent_model.get_double_barrier_option(
                    **INPUTS,
                    strike_price=95.0,
                    lower_barrier=80.0,
                    upper_barrier=120.0,
                    put_option=put_option,
                    knock_type=knock_type,
                ),
                path_dependent_model.get_partial_time_barrier_option(
                    **INPUTS,
                    strike_price=95.0,
                    barrier=110.0,
                    put_option=put_option,
                    barrier_direction="up",
                    knock_type=knock_type,
                    barrier_window="end",
                ),
                path_dependent_model.get_soft_barrier_option(
                    **INPUTS,
                    strike_price=95.0,
                    first_barrier=85.0,
                    second_barrier=92.0,
                    put_option=put_option,
                    knock_type=knock_type,
                ),
            )
            for knock_type in ("in", "out")
        }
        for knock_in, knock_out in zip(prices["in"], prices["out"], strict=True):
            assert knock_in + knock_out == pytest.approx(vanilla, rel=1e-10)


def test_a_double_barrier_far_away_is_the_vanilla_option():
    assert path_dependent_model.get_double_barrier_option(
        **INPUTS, strike_price=100.0, lower_barrier=1.0, upper_barrier=1e5
    ) == pytest.approx(
        black_scholes_model.get_black_scholes(100.0, 100.0, 0.05, 0.3, 1.0, 0.02),
        rel=1e-12,
    )


def test_one_touch_and_no_touch_add_up_to_the_discounted_cash():
    for barrier in (80.0, 125.0):
        one_touch = path_dependent_model.get_touch_option(**INPUTS, barrier=barrier)
        no_touch = path_dependent_model.get_touch_option(
            **INPUTS, barrier=barrier, option_type="no-touch"
        )
        at_hit = path_dependent_model.get_touch_option(
            **INPUTS, barrier=barrier, payment="hit"
        )
        assert one_touch + no_touch == pytest.approx(np.exp(-0.05))
        # Being paid at the hit is paid sooner, so worth more.
        assert at_hit > one_touch

    double_one_touch = path_dependent_model.get_double_touch_option(
        **INPUTS, lower_barrier=85.0, upper_barrier=115.0
    )
    double_no_touch = path_dependent_model.get_double_touch_option(
        **INPUTS, lower_barrier=85.0, upper_barrier=115.0, option_type="double-no-touch"
    )
    assert double_one_touch + double_no_touch == pytest.approx(np.exp(-0.05))


def test_the_discrete_monitoring_shift_moves_the_barrier_away():
    """Broadie-Glasserman-Kou: daily monitoring moves H by e^(±0.5826 σ √(1/252))."""
    shift = np.exp(0.5826 * 0.3 * np.sqrt(1 / 252))
    assert path_dependent_model.get_discrete_monitoring_barrier(
        110.0, 100.0, 0.3, 1 / 252
    ) == pytest.approx(110.0 * shift, rel=1e-4)
    assert path_dependent_model.get_discrete_monitoring_barrier(
        90.0, 100.0, 0.3, 1 / 252
    ) == pytest.approx(90.0 / shift, rel=1e-4)
    assert (
        path_dependent_model.get_discrete_monitoring_barrier(90.0, 100.0, 0.3, 0)
        == 90.0
    )


def test_lookbacks_at_a_zero_cost_of_carry_are_continuous():
    """The formulas divide by b, so b = 0 is taken as the average around it."""
    for strike_type in ("fixed", "floating"):
        for put_option in (False, True):
            at_zero = path_dependent_model.get_lookback_option(
                **{**INPUTS, "dividend_yield": 0.05},
                strike_price=100.0,
                put_option=put_option,
                strike_type=strike_type,
            )
            nearby = path_dependent_model.get_lookback_option(
                **{**INPUTS, "dividend_yield": 0.0501},
                strike_price=100.0,
                put_option=put_option,
                strike_type=strike_type,
            )
            assert at_zero == pytest.approx(nearby, rel=1e-3)


def test_the_payoffs_at_expiration():
    """At zero time to expiration every option is worth its payoff on today's price."""
    expired = {**INPUTS, "time_to_expiration": 0.0}
    assert path_dependent_model.get_curran(**expired, strike_price=90.0) == 10.0
    assert (
        path_dependent_model.get_lookback_option(**expired, strike_price=90.0) == 10.0
    )
    assert (
        path_dependent_model.get_lookback_option(
            **expired, strike_price=90.0, strike_type="floating"
        )
        == 0.0
    )
    assert path_dependent_model.get_touch_option(**expired, barrier=90.0) == 0.0
    assert (
        path_dependent_model.get_touch_option(
            **expired, barrier=90.0, option_type="no-touch"
        )
        == 1.0
    )


def test_touch_options_tend_to_the_deterministic_path_at_a_low_volatility():
    # The powers (H / S)^(μ ± λ) overflowed from a volatility of 0.1%. Without volatility the
    # price follows S * e^((r - q)t), which reaches 103 at t = ln(1.03) / 0.04 and never 110.
    hit_time = np.log(1.03) / 0.04
    for volatility in (1e-6, 1e-8, 1e-12):
        assert path_dependent_model.get_touch_option(
            100, 103, 0.05, volatility, 1, 0.01, payment="hit"
        ) == pytest.approx(np.exp(-0.05 * hit_time), rel=1e-9)
        assert path_dependent_model.get_touch_option(
            100, 103, 0.05, volatility, 1, 0.01
        ) == pytest.approx(np.exp(-0.05), rel=1e-9)
        assert path_dependent_model.get_touch_option(
            100, 103, 0.05, volatility, 1, 0.01, option_type="no-touch"
        ) == pytest.approx(0.0, abs=1e-12)
        assert path_dependent_model.get_touch_option(
            100, 110, 0.05, volatility, 1, 0.01, payment="hit"
        ) == pytest.approx(0.0, abs=1e-12)


def test_lookback_options_tend_to_the_deterministic_path_at_a_low_volatility():
    # The term in (S / E)^(-2b/σ²) gave NaN from a volatility of 1%. Without volatility the
    # maximum is S * e^(bT) and the minimum S.
    forward = 100 * np.exp(0.04)
    discount = np.exp(-0.05)
    for volatility in (1e-6, 1e-10):
        assert path_dependent_model.get_lookback_option(
            100, 95, 0.05, volatility, 1, 0.01
        ) == pytest.approx(discount * (forward - 95), rel=1e-9)
        assert path_dependent_model.get_lookback_option(
            100, 95, 0.05, volatility, 1, 0.01, strike_type="floating"
        ) == pytest.approx(discount * (forward - 100), rel=1e-9)

    with pytest.raises(ValueError, match="partial-time lookback can not be evaluated"):
        path_dependent_model.get_lookback_option(
            100, 95, 0.05, 1e-3, 1, 0.01, lookback_period=0.5
        )


GRID_STRIKES, GRID_TIMES = np.meshgrid(
    np.arange(60.0, 150.0, 10.0), np.array([0, 1, 30, 182, 365]) / 365, indexing="ij"
)


def _per_option(function, **inputs):
    """The formula evaluated once per strike and expiration of the grid."""
    return np.array(
        [
            [
                function(
                    strike_price=float(strike), time_to_expiration=float(time), **inputs
                )
                for strike, time in zip(row_k, row_t, strict=True)
            ]
            for row_k, row_t in zip(GRID_STRIKES, GRID_TIMES, strict=True)
        ]
    )


@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize("knock_type", ["in", "out"])
@pytest.mark.parametrize(
    ("barrier_direction", "barrier"), [("down", 90.0), ("up", 112.0)]
)
def test_barrier_variants_on_a_grid_equal_the_price_per_option(
    put_option, knock_type, barrier_direction, barrier
):
    common = {
        "stock_price": 100.0,
        "risk_free_rate": 0.05,
        "volatility": 0.3,
        "dividend_yield": 0.02,
        "put_option": put_option,
        "knock_type": knock_type,
    }
    cases = [
        (
            exotics_model.get_barrier_option,
            {
                **common,
                "barrier": barrier,
                "barrier_direction": barrier_direction,
                "rebate": 2.0,
            },
        ),
        (
            path_dependent_model.get_double_barrier_option,
            {**common, "lower_barrier": 90.0, "upper_barrier": 112.0},
        ),
        (
            path_dependent_model.get_soft_barrier_option,
            {
                **common,
                "first_barrier": barrier,
                "second_barrier": barrier
                * (0.95 if barrier_direction == "down" else 1.05),
                "barrier_direction": barrier_direction,
            },
        ),
    ] + [
        (
            path_dependent_model.get_partial_time_barrier_option,
            {
                **common,
                "barrier": barrier,
                "barrier_direction": barrier_direction,
                "barrier_window": window,
                "window_share": 0.4,
            },
        )
        for window in ("start", "end")
    ]

    for function, inputs in cases:
        grid = function(
            strike_price=GRID_STRIKES, time_to_expiration=GRID_TIMES, **inputs
        )
        assert grid.shape == GRID_STRIKES.shape
        np.testing.assert_allclose(
            grid, _per_option(function, **inputs), rtol=0, atol=1e-10
        )
        assert isinstance(
            function(strike_price=100.0, time_to_expiration=0.5, **inputs), float
        )


@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize("fixings", [None, 12])
def test_geometric_asian_options_on_a_grid_equal_the_price_per_option(
    put_option, fixings
):
    inputs = {
        "stock_price": 100.0,
        "risk_free_rate": 0.05,
        "volatility": 0.3,
        "dividend_yield": 0.02,
        "put_option": put_option,
    }

    np.testing.assert_allclose(
        path_dependent_model.get_geometric_asian_option(
            strike_price=GRID_STRIKES,
            time_to_expiration=GRID_TIMES,
            fixings=fixings,
            **inputs,
        ),
        _per_option(
            path_dependent_model.get_geometric_asian_option, fixings=fixings, **inputs
        ),
        rtol=0,
        atol=1e-12,
    )
    if fixings is None:
        np.testing.assert_allclose(
            exotics_model.get_asian_option(
                strike_price=GRID_STRIKES[:, 1:],
                time_to_expiration=GRID_TIMES[:, 1:],
                **inputs,
            ),
            _per_option(path_dependent_model.get_geometric_asian_option, **inputs)[
                :, 1:
            ],
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize("strike_type", ["fixed", "floating"])
@pytest.mark.parametrize("lookback_period", [1.0, 0.5])
def test_lookback_options_on_a_grid_equal_the_price_per_option(
    put_option, strike_type, lookback_period
):
    inputs = {
        "stock_price": 100.0,
        "risk_free_rate": 0.05,
        "volatility": 0.3,
        "dividend_yield": 0.02,
        "put_option": put_option,
        "strike_type": strike_type,
        "lookback_period": lookback_period,
    }

    np.testing.assert_allclose(
        path_dependent_model.get_lookback_option(
            strike_price=GRID_STRIKES, time_to_expiration=GRID_TIMES, **inputs
        ),
        _per_option(path_dependent_model.get_lookback_option, **inputs),
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "function",
    [path_dependent_model.get_turnbull_wakeman, path_dependent_model.get_curran],
)
@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize("fixings", [None, 1, 12])
def test_arithmetic_asian_approximations_on_strikes_equal_the_price_per_option(
    function, put_option, fixings
):
    """All strikes of an expiration at once, including the deep in-the-money strikes of
    Curran's non-positive adjusted strike."""
    strikes = np.arange(20.0, 200.0, 5.0)
    for time in (0.0, 1 / 365, 0.5, 2.0):
        inputs = (0.05, 0.3, time, 0.02, put_option, fixings)
        np.testing.assert_allclose(
            function(100.0, strikes, *inputs),
            [function(100.0, float(strike), *inputs) for strike in strikes],
            rtol=0,
            atol=1e-10,
        )


@pytest.mark.parametrize("fixings", [None, 12])
def test_the_curran_price_is_never_negative_far_from_the_money(fixings):
    # Far out of the money the first-order expansion of Curran's threshold breaks down,
    # which gave the forward minus the strike: -951,133 for a strike of 1,000,000.
    arguments = {
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "time_to_expiration": 1.0,
        "dividend_yield": 0.02,
        "fixings": fixings,
    }
    strikes = np.array([300.0, 1_000.0, 1e6])
    calls = path_dependent_model.get_curran(100.0, strikes, **arguments)
    np.testing.assert_allclose(calls, 0.0, atol=1e-12)
    assert (calls >= 0).all()
    for stock_price in (1e-8, 1.0, 10.0):
        assert path_dependent_model.get_curran(
            stock_price, 100.0, **arguments
        ) == pytest.approx(0.0, abs=1e-12)
        # The put is then the discounted strike minus the discounted average.
        assert path_dependent_model.get_curran(
            stock_price, 100.0, put_option=True, **arguments
        ) == pytest.approx(
            path_dependent_model.get_turnbull_wakeman(
                stock_price, 100.0, put_option=True, **arguments
            )
        )
    # Deep in the money the call is the discounted average minus the strike, as before.
    assert path_dependent_model.get_curran(100.0, 1e-6, **arguments) == pytest.approx(
        path_dependent_model.get_turnbull_wakeman(100.0, 1e-6, **arguments)
    )
    assert (
        path_dependent_model.get_curran(100.0, 1.0, put_option=True, **arguments) >= 0
    )


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (exotics_model.get_barrier_option, {"stock_price": 0.0}),
        (exotics_model.get_barrier_option, {"stock_price": -10.0}),
        (exotics_model.get_barrier_option, {"strike_price": 0.0}),
        (exotics_model.get_barrier_option, {"barrier": 0.0}),
        (path_dependent_model.get_partial_time_barrier_option, {"stock_price": 0.0}),
        (path_dependent_model.get_partial_time_barrier_option, {"barrier": 0.0}),
        (path_dependent_model.get_touch_option, {"stock_price": 0.0}),
        (path_dependent_model.get_turnbull_wakeman, {"strike_price": 0.0}),
        (exotics_model.get_asian_option, {"strike_price": 0.0}),
    ],
)
def test_the_barrier_and_average_formulas_refuse_prices_of_zero(function, arguments):
    import inspect

    base = {
        "stock_price": 100.0,
        "strike_price": 100.0,
        "barrier": 90.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "time_to_expiration": 1.0,
    }
    parameters = inspect.signature(function).parameters
    call = {name: value for name, value in base.items() if name in parameters}
    with pytest.raises(ValueError, match="must be above 0"):
        function(**{**call, **arguments})


@pytest.mark.parametrize(
    ("function", "name"),
    [
        (path_dependent_model.get_double_barrier_option, "lower_barrier"),
        (path_dependent_model.get_double_touch_option, "lower_barrier"),
    ],
)
def test_the_double_barriers_refuse_a_lower_barrier_of_zero(function, name):
    import inspect

    base = {
        "stock_price": 100.0,
        "strike_price": 100.0,
        "lower_barrier": 0.0,
        "upper_barrier": 120.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "time_to_expiration": 1.0,
    }
    parameters = inspect.signature(function).parameters
    with pytest.raises(ValueError, match=f"{name} must be above 0"):
        function(**{key: value for key, value in base.items() if key in parameters})
