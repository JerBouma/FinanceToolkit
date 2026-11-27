"""Short-Rate Models and Callable Bond Valuation Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.short_rates import (
    short_rate_model,
    trinomial_tree_model,
)

ZERO_MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
ZERO_RATES = np.array(
    [0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.049, 0.05]
)


def test_vasicek_calibration_recovers_the_parameters():
    paths = short_rate_model.simulate_short_rate(
        "vasicek",
        0.5,
        0.04,
        0.01,
        0.02,
        years=200,
        steps_per_year=52,
        simulations=1,
        seed=1,
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
        "cir", 0.2, 0.01, 0.2, 0.005, years=5, simulations=500
    )
    assert (paths >= 0).all()


@pytest.mark.parametrize("steps_per_year", [1, 12])
@pytest.mark.parametrize("scheme", ["euler", "exact"])
def test_hull_white_reprices_the_zero_curve(steps_per_year, scheme):
    continuous = np.log1p(ZERO_RATES)
    mean = short_rate_model.hull_white_mean(
        ZERO_MATURITIES, continuous, 0.1, 0.01, 0.04, 10, steps_per_year, scheme
    )
    paths = short_rate_model.simulate_short_rate(
        "hull_white", 0.1, mean, 0.01, 0.04, 10, steps_per_year, 20000, 0, scheme
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
    tree = trinomial_tree_model.build_short_rate_tree(
        pd.Series(ZERO_RATES, index=ZERO_MATURITIES),
        np.arange(1, 21) / 2,
        0.0,
        0.01,
        "ho_lee",
        2 * short_rate_model.STEPS_PER_PERIOD,
    )
    model_price = trinomial_tree_model.price_callable_bond(tree, 0.05, 10)
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
    # same amount, so the tree on a shifted curve prices the bond as the base tree does at the
    # spread plus the shift.
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
    tree = trinomial_tree_model.build_short_rate_tree(
        pd.Series(ZERO_RATES, index=ZERO_MATURITIES),
        np.arange(1, 21) / 2,
        0.0,
        0.01,
        "ho_lee",
        2 * short_rate_model.STEPS_PER_PERIOD,
    )

    def price(shift):
        return trinomial_tree_model.price_callable_bond(
            tree, 0.05, 10, 2, 100.0, 100.0, 2, spread=oas + shift
        )

    up, down = price(0.01), price(-0.01)
    assert result["Effective Duration"] == pytest.approx(
        (down - up) / (2 * 101.0 * 0.01), rel=1e-9
    )
    assert result["Effective Convexity"] == pytest.approx(
        (down + up - 2 * 101.0) / (101.0 * 0.01**2), rel=1e-9
    )


def test_option_adjusted_spread_refuses_a_tree_that_is_too_large():
    with pytest.raises(ValueError, match="too large"):
        short_rate_model.get_option_adjusted_spread(
            price=100,
            coupon_rate=0.05,
            years_to_maturity=30,
            maturities=np.array([1.0, 30.0]),
            zero_rates=np.array([0.04, 0.045]),
            frequency=12,
        )


def test_cir_yields_hold_for_a_high_speed_and_a_small_volatility():
    # A speed of 34.78 makes e^(γT) overflow at 30 years (NaN before). The expected yields
    # are the closed form as written, evaluated with 60 digits (mpmath).
    yields = short_rate_model.get_model_yields(
        np.array([20.0, 30.0]), "cir", 34.781365, 0.0004, 0.02, 0.0004
    )
    np.testing.assert_allclose(
        yields, [0.00039999993401289009, 0.00039999993396535768], rtol=1e-12
    )

    # A volatility near zero gives the Vasicek yield without volatility, rather than losing
    # the precision of 2ab/σ² times the log of a number close to 1 (0.020866 at 5 years).
    for maturity in (0.25, 5.0, 30.0):
        cir = short_rate_model.get_model_yields(
            np.array([maturity]), "cir", 0.3, 0.04, 1e-8, 0.03
        )
        vasicek = short_rate_model.get_model_yields(
            np.array([maturity]), "vasicek", 0.3, 0.04, 1e-8, 0.03
        )
        np.testing.assert_allclose(cir, vasicek, rtol=1e-9)


def test_cir_yields_match_the_closed_form():
    # The closed form with e^(γT) as written, at maturities where it does not overflow.
    a, b, sigma, rate = 1.2, 0.02, 0.1, 0.05
    maturities = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    gamma = np.sqrt(a**2 + 2 * sigma**2)
    growth = np.exp(gamma * maturities) - 1
    denominator = (gamma + a) * growth + 2 * gamma
    log_a = (2 * a * b / sigma**2) * np.log(
        2 * gamma * np.exp((a + gamma) * maturities / 2) / denominator
    )
    expected = -(log_a - 2 * growth / denominator * rate) / maturities

    np.testing.assert_allclose(
        short_rate_model.get_model_yields(maturities, "cir", a, b, sigma, rate),
        expected,
        rtol=1e-12,
    )


# QuantLib 1.44 references: Vasicek(r0=0.03, a=0.1, b=0.04, σ=0.01, λ=0),
# CoxIngersollRoss(r0=0.035, θ=0.03, k=0.15, σ=0.05), HullWhite(a=0.08, σ=0.012) and
# ExtendedCoxIngersollRoss(θ=0.03, k=0.15, σ=0.05, x0=0.035) on a flat 3.5% annually
# compounded curve (SimpleDayCounter, so the times are whole years).
FLAT_CURVE = pd.Series([0.035, 0.035], index=[0.0, 60.0])
OPTION_MODELS = {
    "vasicek": {"speed": 0.1, "mean": 0.04, "volatility": 0.01, "rate": 0.03},
    "cir": {"speed": 0.15, "mean": 0.03, "volatility": 0.05, "rate": 0.035},
    "hull_white": {"speed": 0.08, "volatility": 0.012, "zero_curve": FLAT_CURVE},
    "cir++": {
        "speed": 0.15,
        "mean": 0.03,
        "volatility": 0.05,
        "rate": 0.035,
        "zero_curve": FLAT_CURVE,
    },
}


@pytest.mark.parametrize(
    "model, expiry, maturity, strike, call, put",
    [
        ("vasicek", 1, 5, 0.85, 0.030226802572760, 0.001890984377922),
        ("vasicek", 2, 10, 0.7, 0.064351401313878, 0.002341758372779),
        ("cir", 1, 5, 0.85, 0.027070974942089, 0.001555611745225),
        ("cir", 2, 10, 0.72, 0.054997436202918, 0.001743764967735),
        ("hull_white", 1, 5, 0.87, 0.013960519977186, 0.012567063263589),
        ("hull_white", 0.5, 1, 0.98, 0.003424366634379, 0.000528238633813),
        ("cir++", 1, 5, 0.87, 0.009389768713008, 0.007996311999411),
        ("cir++", 2, 10, 0.72, 0.040295035043587, 0.003503925597625),
    ],
)
def test_zero_coupon_bond_option_matches_quantlib(
    model, expiry, maturity, strike, call, put
):
    parameters = OPTION_MODELS[model]
    assert short_rate_model.get_zero_coupon_bond_option(
        expiry, maturity, strike, model=model, **parameters
    ) == pytest.approx(call, abs=1e-11)
    assert short_rate_model.get_zero_coupon_bond_option(
        expiry, maturity, strike, model=model, option_type="put", **parameters
    ) == pytest.approx(put, abs=1e-11)


@pytest.mark.parametrize(
    "model, expiry, tenor, fixed_rate, frequency, payer, receiver",
    [
        ("vasicek", 1, 5, 0.035, 1, 0.009586798853611, 0.018806706951932),
        ("vasicek", 2, 10, 0.04, 2, 0.009118178276158, 0.058681386258359),
        ("cir", 1, 5, 0.035, 1, 0.007812742077058, 0.015318332955597),
        ("cir", 5, 5, 0.03, 1, 0.018332726058618, 0.013089929001639),
        ("hull_white", 1, 5, 0.035, 1, 0.017218056313670, 0.017218056313667),
        ("hull_white", 2, 10, 0.04, 2, 0.018332736162563, 0.059844841530483),
        ("cir++", 1, 5, 0.035, 1, 0.010930081539062, 0.010930081539062),
        ("cir++", 2, 10, 0.04, 2, 0.006013328486922, 0.047525435065543),
    ],
)
def test_swaption_matches_quantlib_jamshidian(
    model, expiry, tenor, fixed_rate, frequency, payer, receiver
):
    # QuantLib solves for the critical rate to a looser tolerance, hence 1e-8.
    parameters = OPTION_MODELS[model]
    common = {"fixed_frequency": frequency, "model": model, **parameters}
    assert short_rate_model.get_swaption(
        expiry, tenor, fixed_rate, **common
    ) == pytest.approx(payer, abs=1e-8)
    assert short_rate_model.get_swaption(
        expiry, tenor, fixed_rate, swaption_type="receiver", **common
    ) == pytest.approx(receiver, abs=1e-8)


def test_swaption_payer_less_receiver_is_the_forward_swap():
    for model, parameters in OPTION_MODELS.items():
        payer = short_rate_model.get_swaption(
            2, 5, 0.04, model=model, fixed_frequency=2, **parameters
        )
        receiver = short_rate_model.get_swaption(
            2,
            5,
            0.04,
            model=model,
            fixed_frequency=2,
            swaption_type="receiver",
            **parameters,
        )
        times = 2 + np.arange(1, 11) / 2
        discount = short_rate_model._discount(
            model,
            parameters["speed"],
            parameters.get("mean"),
            parameters["volatility"],
            parameters.get("rate"),
            parameters.get("zero_curve"),
            np.concatenate([[2.0], times]),
        )
        forward = discount[0] - discount[-1] - 0.04 / 2 * discount[1:].sum()
        assert payer - receiver == pytest.approx(forward, abs=1e-12)


def test_caps_and_floors_match_quantlib():
    # QuantLib's AnalyticCapFloorEngine with HullWhite: quarterly caplets from 3 months to
    # 3 years at 3.5%.
    resets = np.arange(1, 12) * 0.25
    common = {"model": "hull_white", **OPTION_MODELS["hull_white"]}
    cap = short_rate_model.get_caplet(resets, resets + 0.25, 0.035, **common)
    floor = short_rate_model.get_caplet(
        resets, resets + 0.25, 0.035, option_type="floorlet", **common
    )

    assert cap.sum() == pytest.approx(0.012896276712935, abs=1e-12)
    assert floor.sum() == pytest.approx(0.014062464849151, abs=1e-12)
    # Cap - floor = the swap paying 3.5% against the floating rate.
    discount = (1.035) ** -np.append(resets, resets[-1] + 0.25)
    swap = discount[0] - discount[-1] - 0.035 * 0.25 * discount[1:].sum()
    assert cap.sum() - floor.sum() == pytest.approx(swap, abs=1e-12)


def test_option_models_check_their_inputs():
    with pytest.raises(ValueError, match="fitted to a curve"):
        short_rate_model.get_zero_coupon_bond_option(
            1, 5, 0.9, 0.1, 0.01, model="hull_white"
        )
    with pytest.raises(ValueError, match="implies its own curve"):
        short_rate_model.get_zero_coupon_bond_option(
            1, 5, 0.9, 0.1, 0.01, 0.04, 0.03, FLAT_CURVE, model="vasicek"
        )
    with pytest.raises(ValueError, match="expire after today"):
        short_rate_model.get_zero_coupon_bond_option(5, 5, 0.9, 0.1, 0.01, 0.04, 0.03)
    with pytest.raises(ValueError, match="model must be one of"):
        short_rate_model.get_swaption(1, 5, 0.03, 0.1, 0.01, 0.04, 0.03, model="g2++")


def test_cir_swaption_without_a_critical_rate_is_the_forward_swap():
    # A fixed rate so high that the coupon bond is worth more than 1 even at a zero rate
    # is not possible; one so low that it stays below 1 at a zero rate makes the payer
    # swaption a forward swap that is always exercised.
    parameters = OPTION_MODELS["cir"]
    payer = short_rate_model.get_swaption(1, 5, 0.0, model="cir", **parameters)
    receiver = short_rate_model.get_swaption(
        1, 5, 0.0, model="cir", swaption_type="receiver", **parameters
    )
    discount = short_rate_model._discount(
        "cir", 0.15, 0.03, 0.05, 0.035, None, np.array([1.0, 6.0])
    )

    assert payer == pytest.approx(discount[0] - discount[1], abs=1e-14)
    assert receiver == 0.0


def test_cir_plus_plus_shift_fits_the_curve():
    # The shift integrated over [0, t] is ln(P_CIR(0, t) / P(0, t)), so CIR++ reprices the
    # market's discount factors.
    parameters = OPTION_MODELS["cir"]
    curve = pd.Series(ZERO_RATES, index=ZERO_MATURITIES)
    times = np.linspace(0, 10, 20001)
    shift = short_rate_model.get_cir_plus_plus_shift(times, curve, **parameters)
    integral = np.concatenate(
        [[0], np.cumsum((shift[1:] + shift[:-1]) / 2 * np.diff(times))]
    )
    cir = short_rate_model._discount("cir", 0.15, 0.03, 0.05, 0.035, None, times)
    market = (1 + np.interp(times, ZERO_MATURITIES, ZERO_RATES)) ** -times

    assert cir * np.exp(-integral) == pytest.approx(market, abs=1e-5)
    assert short_rate_model.get_cir_plus_plus_shift(
        0.0, FLAT_CURVE, **parameters
    ) == pytest.approx(np.log(1.035) - 0.035, abs=1e-9)


def test_feller_ratio():
    assert short_rate_model.get_feller_ratio(0.15, 0.03, 0.05) == pytest.approx(3.6)
    assert short_rate_model.get_feller_ratio(0.1, 0.01, 0.1) < 1


@pytest.mark.parametrize("volatility", [0.05, 0.2])
def test_exact_cir_scheme_has_the_moments_of_the_model(volatility):
    # FinancePy's meanr and variancer of cir_montecarlo give the same moments; at
    # σ = 0.2 the Feller condition fails (ratio 0.15) and the rate still stays at or above
    # zero without being cut off.
    paths = short_rate_model.simulate_short_rate(
        "cir", 0.15, 0.03, volatility, 0.035, 5, 1, 200_000, 1, scheme="exact"
    )
    decay = np.exp(-0.15 * 5)
    mean = 0.035 * decay + 0.03 * (1 - decay)
    variance = 0.035 * volatility**2 * (
        decay - decay**2
    ) / 0.15 + 0.03 * volatility**2 * (1 - decay) ** 2 / (2 * 0.15)

    assert (paths >= 0).all()
    assert paths[:, -1].mean() == pytest.approx(
        mean, abs=4 * np.sqrt(variance / 200_000)
    )
    assert paths[:, -1].var(ddof=1) == pytest.approx(variance, rel=0.02)


def test_exact_vasicek_scheme_is_exact_at_any_step():
    paths = short_rate_model.simulate_short_rate(
        "vasicek", 0.1, 0.04, 0.01, 0.03, 10, 1, 200_000, 2, scheme="exact"
    )
    decay = np.exp(-0.1 * 10)
    variance = 0.01**2 * (1 - decay**2) / 0.2

    assert paths[:, -1].mean() == pytest.approx(
        0.03 * decay + 0.04 * (1 - decay), abs=4 * np.sqrt(variance / 200_000)
    )
    assert paths[:, -1].var(ddof=1) == pytest.approx(variance, rel=0.02)
    with pytest.raises(ValueError, match="scheme"):
        short_rate_model.simulate_short_rate(
            "vasicek", 0.1, 0.04, 0.01, 0.03, scheme="milstein"
        )


@pytest.mark.parametrize(
    "volatility, coupon_rate, put_price, quantlib",
    [(0.01, 0.04, 100.0, 108.1444956619319), (0.015, 0.03, 98.0, 103.63507325372478)],
)
def test_puttable_bond_matches_quantlib(volatility, coupon_rate, put_price, quantlib):
    # QuantLib's TreeCallableFixedRateBondEngine (800 steps) on a HullWhite tree with
    # a = 1e-6 (Ho-Lee) and a flat 3.5% curve: a 10-year semi-annual bond puttable at the
    # put price on every coupon date from year 3. The trees have different steps, so the
    # prices agree within a cent per 100 of face value.
    price = trinomial_tree_model.get_callable_bond_price(
        coupon_rate,
        10,
        pd.Series([0.035, 0.035], index=[1, 30]),
        0.0,
        volatility,
        "ho_lee",
        call_price=None,
        put_price=put_price,
        years_to_first_put=3,
    )["Price"]

    assert price == pytest.approx(quantlib, abs=0.01)


def test_a_put_raises_the_spread_and_the_convexity():
    common = {
        "price": 104.0,
        "coupon_rate": 0.04,
        "years_to_maturity": 10,
        "maturities": ZERO_MATURITIES,
        "zero_rates": ZERO_RATES,
        "call_price": None,
    }
    straight = short_rate_model.get_option_adjusted_spread(**common)
    puttable = short_rate_model.get_option_adjusted_spread(
        **common, put_price=100.0, years_to_first_put=3
    )

    # The investor pays for the put, so the spread over the curve is higher and the
    # option value (the call less the put) negative.
    assert puttable["Option-Adjusted Spread"] > straight["Option-Adjusted Spread"]
    assert puttable["Option Value"] < 0
    assert puttable["Effective Duration"] < straight["Effective Duration"]
    assert puttable["Effective Convexity"] > straight["Effective Convexity"]


@pytest.mark.parametrize("model", ["black_karasinski", "black_derman_toy"])
def test_lognormal_option_adjusted_spread_on_a_low_curve(model):
    # Shifted down by 100 basis points a flat 0.8% curve has negative forward rates, which the
    # lognormal models cannot fit: the duration then shifts by half the lowest forward rate.
    result = short_rate_model.get_option_adjusted_spread(
        101,
        0.04,
        10,
        np.array([0.5, 30]),
        np.array([0.008, 0.008]),
        0.2,
        call_price=100,
        years_to_first_call=2,
        model=model,
        speed=0.05 if model == "black_karasinski" else 0.0,
    )
    assert all(np.isfinite(value) for value in result.values())
    assert 0 < result["Effective Duration"] < 10


def test_vasicek_bond_coefficients_hold_their_precision_as_the_speed_goes_to_zero():
    from decimal import Decimal, getcontext  # noqa: PLC0415

    # The closed form ln A = (B - τ)(a²b - σ²/2) / a² - σ²B² / (4a), evaluated with 60
    # digits, against the series below aτ = 0.1 and the closed form above it. In double
    # precision the closed form gave ln A(5) = 2.615 at a = 1e-7 instead of 0.002083.
    getcontext().prec = 60
    mean, volatility = 0.04, 0.01

    def reference(speed: float, time: float) -> tuple[float, float]:
        if speed == 0:
            return volatility**2 * time**3 / 6, time
        a, t = Decimal(repr(speed)), Decimal(repr(time))
        b, sigma = Decimal(repr(mean)), Decimal(repr(volatility))
        big_b = (1 - (-a * t).exp()) / a
        log_a = (big_b - t) * (a * a * b - sigma * sigma / 2) / (
            a * a
        ) - sigma**2 * big_b**2 / (4 * a)
        return float(log_a), float(big_b)

    for speed in (
        0.0,
        1e-9,
        1e-7,
        1e-5,
        1e-4,
        1e-3,
        0.0199,
        0.02,
        0.0201,
        0.1,
        0.5,
        2.0,
    ):
        for time in (0.25, 1.0, 5.0, 30.0):
            log_a, big_b = short_rate_model._bond_coefficients(
                "vasicek", speed, mean, volatility, np.array([time])
            )
            expected_log_a, expected_big_b = reference(speed, time)
            # The series switch lies at aτ = 0.1 (a = 0.02 at five years).
            assert log_a[0] == pytest.approx(expected_log_a, rel=1e-11, abs=1e-18)
            assert big_b[0] == pytest.approx(expected_big_b, rel=1e-14)


def test_short_rate_options_take_their_limit_at_a_speed_of_zero():
    zero_curve = pd.Series([0.03, 0.032, 0.035, 0.037], index=[1, 2, 5, 10])
    for model, inputs, volatility in (
        ("vasicek", {"mean": 0.04, "rate": 0.03}, 0.01),
        ("hull_white", {"zero_curve": zero_curve}, 0.01),
        ("cir", {"mean": 0.04, "rate": 0.03}, 0.05),
        ("cir++", {"mean": 0.04, "rate": 0.03, "zero_curve": zero_curve}, 0.05),
    ):
        # NaN at a speed of 0 and, for Vasicek, 2.38 (above the bond price) at 1e-7.
        prices = [
            short_rate_model.get_zero_coupon_bond_option(
                1, 5, 0.85, speed, volatility, model=model, **inputs
            )
            for speed in (0.0, 1e-8, 1e-6)
        ]
        assert np.all(np.isfinite(prices))
        np.testing.assert_allclose(prices, prices[0], rtol=1e-5)
        swaptions = [
            short_rate_model.get_swaption(
                1, 5, 0.04, speed, volatility, model=model, **inputs
            )
            for speed in (0.0, 1e-8, 0.0199, 0.02, 0.0201)
        ]
        assert np.all(np.isfinite(swaptions))
        assert swaptions[0] == pytest.approx(swaptions[1], rel=1e-6)
        # Continuous across the switch of the series at aτ = 0.1.
        assert swaptions[3] == pytest.approx(
            0.5 * (swaptions[2] + swaptions[4]), rel=1e-6
        )

    # Vasicek at a = 0 is the Merton (Ho-Lee) model: σ_P = σ(S - T)√T.
    p_expiry = np.exp(0.01**2 / 6 - 0.03)
    p_maturity = np.exp(0.01**2 * 125 / 6 - 0.15)
    sigma_p = 0.01 * 4
    h = np.log(p_maturity / (0.85 * p_expiry)) / sigma_p + sigma_p / 2
    from scipy.stats import norm  # noqa: PLC0415

    assert short_rate_model.get_zero_coupon_bond_option(
        1, 5, 0.85, 0.0, 0.01, mean=0.04, rate=0.03
    ) == pytest.approx(
        p_maturity * norm.cdf(h) - 0.85 * p_expiry * norm.cdf(h - sigma_p), rel=1e-12
    )


def test_cir_options_need_a_volatility():
    zero_curve = pd.Series([0.03, 0.037], index=[1, 10])
    for model, inputs in (
        ("cir", {"mean": 0.04, "rate": 0.03}),
        ("cir++", {"mean": 0.04, "rate": 0.03, "zero_curve": zero_curve}),
    ):
        with pytest.raises(ValueError, match="needs a volatility above zero"):
            short_rate_model.get_zero_coupon_bond_option(
                1, 5, 0.85, 0.1, 0.0, model=model, **inputs
            )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"model": "vasicekk"}, "model"),
        ({"steps_per_year": 0}, "steps_per_year"),
        ({"simulations": 0}, "simulations"),
    ],
)
def test_simulate_short_rate_refuses_impossible_arguments(arguments, message):
    settings = {
        "model": "vasicek",
        "speed": 0.3,
        "mean": 0.04,
        "volatility": 0.01,
        "rate": 0.03,
        "years": 1.0,
        "simulations": 10,
    }

    with pytest.raises(ValueError, match=message):
        short_rate_model.simulate_short_rate(**{**settings, **arguments})


def test_calibrating_a_flat_history_says_why():
    rates = pd.Series(0.03, index=pd.period_range("2024-01-01", periods=300, freq="D"))

    with pytest.raises(ValueError, match="does not move"):
        short_rate_model.calibrate_short_rate_model(rates)
