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
    steps = short_rate_model.STEPS_PER_PERIOD
    times = np.arange(1, 20 * steps + 1) / (2 * steps)
    tree = short_rate_model._ho_lee_tree(
        np.exp(-np.log1p(np.interp(times, ZERO_MATURITIES, ZERO_RATES)) * times),
        0.01,
        1 / (2 * steps),
    )
    model_price = short_rate_model.price_callable_bond(
        tree, 0.05, 2, steps_per_period=steps
    )
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


def test_vasicek_calibration_is_exact_for_monthly_data():
    # The Euler mapping a = -β/Δt understates a fast speed on monthly data (about 1.84 for
    # 2); the exact AR(1) mapping recovers it.
    speeds = []
    for seed in range(10):
        path = short_rate_model.simulate_short_rate(
            "vasicek", 2.0, 0.04, 0.01, 0.04, 300, 240, 1, seed
        )[0][::20]
        speeds.append(
            short_rate_model.calibrate_short_rate_model(
                pd.Series(path), "vasicek", periods_per_year=12
            )["Speed of Mean Reversion"]
        )

    assert np.mean(speeds) == pytest.approx(2.0, rel=0.03)


def test_option_adjusted_spread_matches_a_fine_tree():
    # QuantLib's TreeCallableFixedRateBondEngine (Hull-White with a near-zero speed, which is
    # Ho-Lee, on 400 steps) gives an OAS of 0.005704 for this bond; a tree of 320 steps per
    # coupon period gives an effective convexity of -102.5 over a 100 basis point shift. A
    # coarse tree of one step per coupon gave a positive convexity.
    result = short_rate_model.get_option_adjusted_spread(
        price=98.0,
        coupon_rate=0.05,
        years_to_maturity=10,
        maturities=np.array([1, 2, 5, 10, 20, 30.0]),
        zero_rates=np.array([0.036, 0.035, 0.037, 0.041, 0.047, 0.047]),
        call_price=100.0,
        years_to_first_call=2,
    )

    assert result["Option-Adjusted Spread"] == pytest.approx(0.005704, abs=2e-5)
    assert result["Effective Convexity"] == pytest.approx(-102.5, rel=0.02)


def test_option_adjusted_spread_between_coupon_dates():
    # A straight bond priced at the value of its cash flows on the curve (clean, so without
    # the interest accrued since the last coupon) has an OAS of zero.
    years, coupon_rate = 9.75, 0.05
    times = np.arange(0.25, years + 1e-9, 0.5)
    continuous = np.log1p(np.interp(times, ZERO_MATURITIES, ZERO_RATES))
    dirty = float(
        np.sum(2.5 * np.exp(-continuous * times))
        + 100 * np.exp(-continuous[-1] * years)
    )
    clean = dirty - 2.5 * 0.5

    result = short_rate_model.get_option_adjusted_spread(
        price=clean,
        coupon_rate=coupon_rate,
        years_to_maturity=years,
        maturities=ZERO_MATURITIES,
        zero_rates=ZERO_RATES,
        call_price=None,
    )

    assert result["Option-Adjusted Spread"] == pytest.approx(0, abs=1e-6)
    assert result["Straight Bond Price"] == pytest.approx(clean, abs=1e-6)


def test_cir_calibration_recovers_the_volatility_of_a_fast_rate():
    # Exact CIR transitions (a scaled noncentral chi-square) of a fast reverting rate on
    # monthly data. The variance of a step also has a term b σ² (1 - φ)² / (2a) that does not
    # scale with the rate; leaving it out overstates σ by about 4% here.
    rng = np.random.default_rng(3)
    speed, mean, volatility, dt = 2.0, 0.04, 0.05, 1 / 12
    scale = volatility**2 * (1 - np.exp(-speed * dt)) / (4 * speed)
    degrees = 4 * speed * mean / volatility**2
    rates = np.empty(24_001)
    rates[0] = mean
    for step in range(1, len(rates)):
        rates[step] = scale * rng.noncentral_chisquare(
            degrees, rates[step - 1] * np.exp(-speed * dt) / scale
        )
    parameters = short_rate_model.calibrate_short_rate_model(
        pd.Series(rates), "cir", periods_per_year=12
    )

    assert parameters["Volatility"] == pytest.approx(volatility, rel=0.01)
    assert parameters["Speed of Mean Reversion"] == pytest.approx(speed, rel=0.05)
    assert parameters["Long-Run Mean"] == pytest.approx(mean, rel=0.02)


def test_effective_convexity_is_negative_at_a_high_volatility():
    # Over a 25 basis point shift the convexity of this callable bond at a 2% volatility is
    # lattice noise that can even turn positive; over 100 basis points it stays close to that
    # of a tree of 320 steps per coupon period (-43.1).
    result = short_rate_model.get_option_adjusted_spread(
        price=100.0,
        coupon_rate=0.05,
        years_to_maturity=10,
        maturities=np.array([1, 2, 5, 10, 20, 30.0]),
        zero_rates=np.array([0.036, 0.035, 0.037, 0.041, 0.047, 0.047]),
        volatility=0.02,
        call_price=100.0,
        years_to_first_call=2,
    )

    assert result["Effective Convexity"] == pytest.approx(-43.1, rel=0.08)


def test_effective_duration_shifts_the_curve():
    # A parallel shift of the continuously compounded curve moves every Ho-Lee node by the
    # same amount, so rebuilding the tree on a shifted curve gives the price of the base tree
    # at the spread plus the shift, which is what the effective duration and convexity use.
    steps = short_rate_model.STEPS_PER_PERIOD
    dt = 1 / (2 * steps)
    times = np.arange(1, 20 * steps + 1) * dt
    continuous = np.log1p(np.interp(times, ZERO_MATURITIES, ZERO_RATES))
    result = short_rate_model.get_option_adjusted_spread(
        price=101.0,
        coupon_rate=0.05,
        years_to_maturity=10,
        maturities=ZERO_MATURITIES,
        zero_rates=ZERO_RATES,
        call_price=100.0,
        years_to_first_call=2,
    )
    oas = result["Option-Adjusted Spread"]

    def price(shift):
        tree = short_rate_model._ho_lee_tree(
            np.exp(-(continuous + shift) * times), 0.01, dt
        )
        return short_rate_model.price_callable_bond(
            tree, 0.05, 2, 100.0, 100.0, int(np.ceil(2 / dt - 1e-9)), oas, steps
        )

    up, down = price(0.01), price(-0.01)
    assert result["Effective Duration"] == pytest.approx(
        (down - up) / (2 * 101.0 * 0.01), rel=1e-9
    )
    assert result["Effective Convexity"] == pytest.approx(
        (down + up - 2 * 101.0) / (101.0 * 0.01**2), rel=1e-9
    )
