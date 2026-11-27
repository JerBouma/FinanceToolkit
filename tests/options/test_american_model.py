"""American Option Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.options import options_model
from financetoolkit.options.pricing import american_model, black_scholes_model

# Each case: stock price, strike price, risk-free rate, dividend yield, volatility, days to
# expiration, put option and the QuantLib 1.44 prices of the BaroneAdesiWhaley, the
# BjerksundStensland (1993), the JuQuadratic and the QdPlus engine and the high precision
# QdFp engine (the exact American value).
QUANTLIB_CASES = [
    (
        100,
        100,
        0.05,
        0.03,
        0.25,
        365,
        True,
        8.903019,
        8.81523,
        8.857302,
        8.888598,
        8.882701,
    ),
    (
        100,
        90,
        0.08,
        0.12,
        0.20,
        182,
        False,
        10.641644,
        10.634515,
        10.666961,
        10.700632,
        10.684835,
    ),
    (
        100,
        110,
        0.03,
        0.00,
        0.40,
        730,
        True,
        25.484162,
        25.334144,
        25.394693,
        25.498243,
        25.47633,
    ),
    (
        100,
        100,
        0.02,
        0.06,
        0.30,
        91,
        False,
        5.507515,
        5.476649,
        5.505318,
        5.517845,
        5.515384,
    ),
]


@pytest.mark.parametrize("case", QUANTLIB_CASES)
def test_the_american_approximations_match_quantlib(case):
    stock, strike, rate, dividend, volatility, days, put = case[:7]
    barone_adesi_whaley, bjerksund_stensland, ju_zhong, qd_plus, exact = case[7:]
    inputs = {
        "stock_price": float(stock),
        "strike_price": float(strike),
        "risk_free_rate": rate,
        "volatility": volatility,
        "time_to_expiration": days / 365,
        "dividend_yield": dividend,
        "put_option": put,
    }

    assert american_model.get_barone_adesi_whaley(**inputs) == pytest.approx(
        barone_adesi_whaley, abs=1e-4
    )
    assert american_model.get_ju_zhong(**inputs) == pytest.approx(ju_zhong, abs=1e-4)
    assert american_model.get_qd_plus(**inputs) == pytest.approx(qd_plus, abs=1e-4)
    assert black_scholes_model.get_bjerksund_stensland(
        **inputs, method="1993"
    ) == pytest.approx(bjerksund_stensland, abs=1e-6)
    # QD+ is the closest of the approximations to the exact American value.
    assert american_model.get_qd_plus(**inputs) == pytest.approx(exact, rel=3e-3)
    # The QD fixed point is the exact value, up to the six decimals recorded here.
    assert american_model.get_qd_fixed_point(**inputs) == pytest.approx(exact, abs=1e-6)


def test_the_dispatcher_selects_the_approximation_and_refuses_others():
    inputs = (100.0, 100.0, 0.05, 0.25, 1.0, 0.03, True)
    assert american_model.get_american_option(
        *inputs, method="ju_zhong"
    ) == american_model.get_ju_zhong(*inputs)
    assert american_model.get_american_option(*inputs) == american_model.get_qd_plus(
        *inputs
    )

    with pytest.raises(ValueError, match="method must be one of"):
        american_model.get_american_option(*inputs, method="binomial")
    with pytest.raises(ValueError, match="'2002' or '1993'"):
        black_scholes_model.get_bjerksund_stensland(*inputs, method="2010")


@pytest.mark.parametrize(
    "method", ["qd_plus", "barone_adesi_whaley", "ju_zhong", "qd_fixed_point"], ids=str
)
def test_without_early_exercise_the_american_option_is_european(method):
    # A call without dividends and a put at a zero rate are never exercised early.
    for put, rate, dividend in ((False, 0.05, 0.0), (True, 0.0, 0.02)):
        european = float(
            black_scholes_model.get_black_scholes(
                100.0, 105.0, rate, 0.3, 0.5, dividend, put_option=put
            )
        )
        american = american_model.get_american_option(
            100.0, 105.0, rate, 0.3, 0.5, dividend, put_option=put, method=method
        )
        assert american == pytest.approx(max(european, 5.0 if put else 0.0), abs=1e-12)


@pytest.mark.parametrize(
    "method", ["qd_plus", "barone_adesi_whaley", "ju_zhong", "qd_fixed_point"], ids=str
)
def test_the_american_option_is_worth_at_least_the_european_and_the_exercise_value(
    method,
):
    for strike in (60.0, 100.0, 160.0):
        for rate, dividend in ((-0.01, 0.05), (0.05, -0.02), (0.08, 0.3)):
            for put in (False, True):
                for days in (0, 1, 30, 3650):
                    american = american_model.get_american_option(
                        100.0, strike, rate, 0.6, days / 365, dividend, put, method
                    )
                    european = float(
                        black_scholes_model.get_black_scholes(
                            100.0, strike, rate, 0.6, days / 365, dividend, put
                        )
                    )
                    exercise = max((strike - 100) if put else (100 - strike), 0.0)
                    assert np.isfinite(american)
                    assert american >= max(european, exercise) - 1e-9


def test_the_qd_fixed_point_matches_a_finite_difference_grid():
    # A Crank-Nicolson grid of QuantLib's FdBlackScholesVanillaEngine with 2,000, 4,000 and
    # 8,000 steps in time and price converges at first order here (the error halves with the
    # step), so it is extrapolated as 2 * V(8000) - V(4000): 8.882702, 18.410675, 5.457208,
    # each within 1e-6 of the QD fixed point.
    cases = [
        ((100.0, 100.0, 0.05, 0.25, 1.0, 0.03), 8.8827020),
        ((100.0, 110.0, 0.05, 0.30, 2.0, 0.00), 18.4106751),
        ((100.0, 90.0, 0.08, 0.40, 182 / 365, 0.02), 5.4572081),
    ]
    for inputs, extrapolated in cases:
        assert american_model.get_qd_fixed_point(
            *inputs, put_option=True
        ) == pytest.approx(extrapolated, abs=1e-6)


def test_the_qd_fixed_point_falls_back_to_fp_b_where_fp_a_diverges():
    # At a volatility of 0.5% with a negative dividend yield FP-A oscillates and diverges;
    # FP-B converges, close to the QD+ value 0.0065691.
    value = american_model.get_qd_fixed_point(
        100.0, 100.0, 0.05, 0.005, 1.0, -0.02, True
    )
    assert value == pytest.approx(0.0065691, abs=1e-6)
    # Thirty years at 20% with a negative yield: QD+ gives 3.22355.
    value = american_model.get_qd_fixed_point(100.0, 100.0, 0.2, 0.2, 30.0, -0.02, True)
    assert value == pytest.approx(3.2235, abs=1e-4)


def test_the_qd_fixed_point_is_stable_under_tiny_input_changes():
    # The iteration stops at a tolerance, so a change in the last bits of an input must not
    # move the price beyond the bits that tolerance leaves.
    base = american_model.get_qd_fixed_point(100.0, 100.0, 0.05, 0.25, 1.0, 0.03, True)
    for bump in (1e-12, -1e-12):
        bumped = american_model.get_qd_fixed_point(
            100.0 * (1 + bump), 100.0, 0.05, 0.25, 1.0, 0.03, True
        )
        assert bumped == pytest.approx(base, abs=1e-10)


def test_a_deep_in_the_money_put_is_exercised_right_away():
    # Below the QD+ boundary the put is worth its exercise value exactly, where QuantLib's
    # QdPlusAmericanEngine integrates beyond it (20.0434 against the exact 20.0000).
    assert (
        american_model.get_qd_plus(100.0, 120.0, 0.05, 0.15, 2.0, put_option=True)
        == 20.0
    )


def test_the_american_implied_volatility_reprices_the_option():
    for method in ("qd_plus", "barone_adesi_whaley"):
        for put in (False, True):
            price = american_model.get_american_option(
                100.0, 95.0, 0.05, 0.35, 0.5, 0.04, put, method
            )
            volatility = american_model.get_american_implied_volatility(
                price, 100.0, 95.0, 0.05, 0.5, 0.04, put, method
            )
            assert volatility == pytest.approx(0.35, abs=1e-8)

    # QuantLib's QdPlusAmericanEngine price of the first case inverts to its volatility.
    assert american_model.get_american_implied_volatility(
        8.888598, 100.0, 100.0, 0.05, 1.0, 0.03, True
    ) == pytest.approx(0.25, abs=1e-5)


def test_the_american_implied_volatility_of_an_impossible_price_is_nan():
    # At or below the exercise value, and above the stock price for a call.
    assert np.isnan(
        american_model.get_american_implied_volatility(
            20.0, 100.0, 120.0, 0.05, 1.0, put_option=True
        )
    )
    assert np.isnan(
        american_model.get_american_implied_volatility(150.0, 100.0, 100.0, 0.05, 1.0)
    )
    assert np.isnan(
        american_model.get_american_implied_volatility(5.0, 100.0, 100.0, 0.05, 0.0)
    )
    assert np.isnan(
        american_model.get_american_implied_volatility(np.nan, 100.0, 100.0, 0.05, 1.0)
    )


def test_the_cash_dividend_prices_match_quantlib():
    # QuantLib 1.44 AnalyticDividendEuropeanEngine (escrowed) and CashDividendEuropeanEngine
    # with the Spot model (Haug-Haug-Lewis); S = 100, r = 5%, σ = 30%, T = 1.
    cases = [
        (100.0, False, [30], [2.0], 13.013298, 13.032528),
        (95.0, True, [45, 136, 227, 318], [0.8] * 4, 8.203891, 8.381214),
    ]
    for strike, put, days, amounts, escrowed, haug_haug_lewis in cases:
        times = np.array(days) / 365
        assert black_scholes_model.get_black_scholes_cash_dividends(
            100.0, strike, 0.05, 0.3, 1.0, times, amounts, put, "escrowed"
        ) == pytest.approx(escrowed, abs=1e-6)
        assert black_scholes_model.get_black_scholes_cash_dividends(
            100.0, strike, 0.05, 0.3, 1.0, times, amounts, put, "haug_haug_lewis"
        ) == pytest.approx(haug_haug_lewis, abs=1e-5)


def test_cash_dividends_after_expiration_are_left_out():
    european = float(
        black_scholes_model.get_black_scholes(100.0, 100.0, 0.05, 0.3, 0.1)
    )
    for method in ("escrowed", "haug_haug_lewis"):
        assert black_scholes_model.get_black_scholes_cash_dividends(
            100.0, 100.0, 0.05, 0.3, 0.1, [0.5], [2.0], method=method
        ) == pytest.approx(european, abs=1e-12)

    # The escrowed method also prices arrays of expirations, each with its own dividends.
    values = black_scholes_model.get_black_scholes_cash_dividends(
        100.0, 100.0, 0.05, 0.3, np.array([0.1, 1.0]), [0.5], [2.0]
    )
    assert values[0] == pytest.approx(european, abs=1e-12)
    assert values[1] == pytest.approx(
        black_scholes_model.get_black_scholes_cash_dividends(
            100.0, 100.0, 0.05, 0.3, 1.0, [0.5], [2.0]
        )
    )

    with pytest.raises(ValueError, match="same length"):
        black_scholes_model.get_black_scholes_cash_dividends(
            100.0, 100.0, 0.05, 0.3, 1.0, [0.5], [2.0, 1.0], method="haug_haug_lewis"
        )


def test_the_projected_dividends_repeat_the_latest_dividend():
    dates = pd.to_datetime(["2024-02-09", "2024-05-10", "2024-08-12", "2024-11-08"])
    dividends = pd.Series([0.24, 0.25, 0.25, 0.25], index=dates)

    projected = options_model.get_projected_dividends(
        dividends, pd.Timestamp("2024-12-31"), horizon_days=200
    )

    # The median interval between the ex-dates is 91 days: 2025-02-07 and 2025-05-09.
    assert projected.index.tolist() == [38, 129]
    assert projected.tolist() == [0.25, 0.25]

    # The dividends after the pricing date are not known yet on it: a single dividend is
    # taken to be annual.
    assert options_model.get_projected_dividends(
        dividends, pd.Timestamp("2024-03-01"), horizon_days=400
    ).to_dict() == {344: 0.24}
    # A stock that stopped paying projects nothing.
    assert options_model.get_projected_dividends(
        dividends, pd.Timestamp("2025-12-31"), horizon_days=200
    ).empty


# A put at r ≤ 0 with q < r is exercised between two boundaries. A Crank-Nicolson grid of
# 4,000 by 4,000 points (finite_difference_model) gives these values; the European put is
# 11.1716, 10.6424 and 13.8994, which the single boundary methods returned.
@pytest.mark.parametrize(
    "stock, rate, dividend, volatility, years, reference",
    [
        (100, -0.01, -0.03, 0.3, 1.0, 11.25204),
        (100, 0.0, -0.03, 0.3, 1.0, 10.82493),
        (90, -0.02, -0.05, 0.2, 2.0, 14.43965),
    ],
)
@pytest.mark.parametrize(
    "method", ["qd_plus", "barone_adesi_whaley", "ju_zhong", "qd_fixed_point"]
)
def test_an_american_put_with_two_exercise_boundaries(
    stock, rate, dividend, volatility, years, reference, method
):
    put = american_model.get_american_option(
        stock, 100, rate, volatility, years, dividend, True, method
    )
    assert put == pytest.approx(reference, abs=2e-3)
    # The call with the stock and strike price and the rates swapped is the same value.
    call = american_model.get_american_option(
        100, stock, dividend, volatility, years, rate, False, method
    )
    assert call == pytest.approx(put, abs=1e-12)


def test_an_american_option_needs_positive_prices():
    with pytest.raises(ValueError, match="must be positive"):
        american_model.get_qd_fixed_point(100, 0, 0.04, 0.2, 1, 0.02, True)


@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize(
    ("risk_free_rate", "dividend_yield", "volatility"),
    [
        (0.05, 0.03, 0.25),
        (0.05, 0.0, 0.05),
        (0.12, 0.2, 0.8),
        (-0.01, -0.02, 0.3),
        (0.0, 0.03, 0.3),
    ],
)
def test_qd_plus_on_a_grid_equals_the_price_per_option(
    put_option, risk_free_rate, dividend_yield, volatility
):
    """The boundaries of all expirations solved together give the same prices as one by one."""
    strikes, times = np.meshgrid(
        np.arange(60.0, 150.0, 10.0),
        np.array([0, 1, 7, 91, 365, 730]) / 365,
        indexing="ij",
    )
    inputs = {
        "stock_price": 100.0,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "put_option": put_option,
    }

    grid = american_model.get_qd_plus(
        strike_price=strikes, time_to_expiration=times, **inputs
    )
    american_model._qd_plus_unit_boundary.cache_clear()
    per_option = np.array(
        [
            american_model.get_qd_plus(
                strike_price=float(strike), time_to_expiration=float(time), **inputs
            )
            for strike, time in zip(strikes.ravel(), times.ravel(), strict=True)
        ]
    ).reshape(strikes.shape)

    np.testing.assert_allclose(grid, per_option, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "function", [american_model.get_barone_adesi_whaley, american_model.get_ju_zhong]
)
@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize(
    ("risk_free_rate", "dividend_yield", "volatility"),
    [(0.05, 0.03, 0.25), (0.12, 0.2, 0.8), (-0.01, -0.02, 0.3), (0.0, 0.03, 0.3)],
)
def test_the_quadratic_approximations_on_a_grid_equal_the_price_per_option(
    function, put_option, risk_free_rate, dividend_yield, volatility
):
    strikes, times = np.meshgrid(
        np.arange(60.0, 150.0, 10.0),
        np.array([0, 1, 7, 91, 365, 730]) / 365,
        indexing="ij",
    )
    inputs = (risk_free_rate, volatility)

    grid = function(100.0, strikes, *inputs, times, dividend_yield, put_option)
    per_option = np.array(
        [
            function(
                100.0, float(strike), *inputs, float(time), dividend_yield, put_option
            )
            for strike, time in zip(strikes.ravel(), times.ravel(), strict=True)
        ]
    ).reshape(strikes.shape)

    np.testing.assert_allclose(grid, per_option, rtol=0, atol=1e-10)
