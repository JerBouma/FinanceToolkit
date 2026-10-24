"""Short-Rate Models and Callable Bond Valuation Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome import short_rate_model

# pylint: disable=missing-function-docstring

ZERO_MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
ZERO_RATES = np.array(
    [0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.049, 0.05]
)


def test_vasicek_calibration_recovers_the_parameters():
    paths = short_rate_model.simulate_short_rate(
        "vasicek", 0.5, 0.04, 0.01, 0.02, years=200, steps_per_year=52, paths=1, seed=1
    )
    parameters = short_rate_model.calibrate_short_rate_model(
        pd.Series(paths[0]), "vasicek", periods_per_year=52
    )

    assert parameters["Long-Run Mean"] == pytest.approx(0.04, abs=0.005)
    assert parameters["Volatility"] == pytest.approx(0.01, rel=0.05)
    assert parameters["Speed of Mean Reversion"] == pytest.approx(0.5, rel=0.3)
    assert parameters["Half-Life"] == pytest.approx(
        np.log(2) / parameters["Speed of Mean Reversion"]
    )


def test_cir_calibration_rejects_a_rate_at_zero():
    rates = pd.Series(
        np.r_[np.linspace(0.02, 0.0, 20), np.linspace(0.001, 0.02, 20)],
        index=pd.period_range("2020-01-01", periods=40, freq="D"),
    )
    with pytest.raises(ValueError, match="2020-01-20"):
        short_rate_model.calibrate_short_rate_model(rates, "cir")


@pytest.mark.parametrize("model", ["vasicek", "cir"])
def test_model_yields_start_at_the_short_rate(model):
    yields = short_rate_model.get_model_yields(
        np.array([1e-4, 1, 30.0]),
        model,
        0.3,
        0.05,
        0.02 if model == "vasicek" else 0.1,
        0.03,
    )

    assert yields[0] == pytest.approx(0.03, abs=1e-5)
    # Mean reversion pulls the long end towards the long-run mean of 5%.
    assert 0.03 < yields[1] < yields[2] < 0.05


def test_cir_simulation_stays_at_or_above_zero():
    paths = short_rate_model.simulate_short_rate(
        "cir", 0.2, 0.01, 0.2, 0.005, years=5, paths=500
    )
    assert (paths >= 0).all()


@pytest.mark.parametrize("steps_per_year", [1, 12])
def test_hull_white_reprices_the_zero_curve(steps_per_year):
    continuous = np.log1p(ZERO_RATES)
    mean = short_rate_model.hull_white_mean(
        ZERO_MATURITIES, continuous, 0.1, 0.01, 0.04, 10, steps_per_year
    )
    paths = short_rate_model.simulate_short_rate(
        "hull_white", 0.1, mean, 0.01, 0.04, 10, steps_per_year, 20000, 0
    )

    dt = 1 / steps_per_year
    model_discount = np.exp(-np.cumsum(paths[:, :-1] * dt, axis=1)).mean(axis=0)
    times = np.arange(1, paths.shape[1]) * dt
    market_discount = np.exp(-np.interp(times, ZERO_MATURITIES, continuous) * times)

    assert model_discount == pytest.approx(market_discount, abs=1.5e-3)


def test_option_adjusted_spread_of_a_straight_bond():
    straight = short_rate_model.get_option_adjusted_spread(
        price=100.0,
        coupon_rate=0.05,
        years_to_maturity=10,
        maturities=ZERO_MATURITIES,
        zero_rates=ZERO_RATES,
        call_price=None,
    )
    tree = short_rate_model._ho_lee_tree(
        np.exp(
            -np.log1p(np.interp(np.arange(1, 21) / 2, ZERO_MATURITIES, ZERO_RATES))
            * np.arange(1, 21)
            / 2
        ),
        0.01,
        0.5,
    )
    model_price = short_rate_model.price_callable_bond(tree, 0.05, 2)
    repriced = short_rate_model.get_option_adjusted_spread(
        price=model_price,
        coupon_rate=0.05,
        years_to_maturity=10,
        maturities=ZERO_MATURITIES,
        zero_rates=ZERO_RATES,
        call_price=None,
    )

    assert straight["Option Value"] == pytest.approx(0, abs=1e-8)
    assert repriced["Option-Adjusted Spread"] == pytest.approx(0, abs=1e-6)
    assert straight["Effective Convexity"] > 0


def test_a_call_lowers_the_spread_and_the_convexity():
    common = dict(
        price=101.0,
        coupon_rate=0.055,
        years_to_maturity=10,
        maturities=ZERO_MATURITIES,
        zero_rates=ZERO_RATES,
    )
    straight = short_rate_model.get_option_adjusted_spread(**common, call_price=None)
    callable_bond = short_rate_model.get_option_adjusted_spread(
        **common, call_price=100.0, years_to_first_call=1
    )

    # The investor is short the call, so at the same price the callable bond's spread over
    # the curve is lower once the option is valued.
    assert callable_bond["Option-Adjusted Spread"] < straight["Option-Adjusted Spread"]
    assert callable_bond["Option Value"] > 0
    assert callable_bond["Effective Duration"] < straight["Effective Duration"]
    assert callable_bond["Effective Convexity"] < 0


def test_implied_default_probability():
    result = short_rate_model.get_implied_default_probability(0.012, 0.4, (1, 5))

    assert result["Hazard Rate"] == pytest.approx(0.02)
    assert result["1Y Default Probability"] == pytest.approx(1 - np.exp(-0.02))
    assert result["5Y Default Probability"] == pytest.approx(1 - np.exp(-0.1))
    with pytest.raises(ValueError):
        short_rate_model.get_implied_default_probability(0.01, 1.0)
