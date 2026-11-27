"""Bond Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.bonds import bond_model

TAYLOR_APPROXIMATION_TOLERANCE = 0.0001


def test_get_bond_price_from_curve_matches_flat_yield(recorder):
    flat_curve = pd.Series({1: 0.08, 2: 0.08, 3: 0.08, 5: 0.08, 10: 0.08})
    price_from_curve = bond_model.get_bond_price_from_curve(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        spot_rates=flat_curve,
        frequency=1,
    )
    price_from_ytm = bond_model.get_bond_price(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        yield_to_maturity=0.08,
        frequency=1,
    )
    recorder.capture(bool(round(price_from_curve, 6) == round(price_from_ytm, 6)))


def test_get_z_spread(recorder):
    benchmark = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 10: 0.05})
    recorder.capture(
        bond_model.get_z_spread(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            bond_price=96.038065,
            spot_rates=benchmark,
            frequency=1,
        )
    )


def test_get_z_spread_recovers_synthetic_spread(recorder):
    benchmark = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 10: 0.05})
    true_spread = 0.015
    synthetic_price = bond_model.get_bond_price_from_curve(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        spot_rates=benchmark + true_spread,
        frequency=1,
    )
    recovered_spread = bond_model.get_z_spread(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        bond_price=synthetic_price,
        spot_rates=benchmark,
        frequency=1,
    )
    recorder.capture(bool(round(recovered_spread, 3) == round(true_spread, 3)))


def test_get_z_spread_type_error():
    with pytest.raises(TypeError):
        bond_model.get_z_spread(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            bond_price=96,
            spot_rates=[0.03, 0.04, 0.05],
        )


def test_get_bond_equivalent_yield(recorder):
    recorder.capture(
        bond_model.get_bond_equivalent_yield(discount_yield=0.05, days_to_maturity=182)
    )


def test_get_bond_equivalent_yield_matches_hand_calculation(recorder):
    bey = bond_model.get_bond_equivalent_yield(
        discount_yield=0.05, days_to_maturity=182
    )
    expected = 365 * 0.05 / (360 - 182 * 0.05)
    recorder.capture(bool(round(bey, 10) == round(expected, 10)))


def test_get_bond_equivalent_yield_type_error():
    with pytest.raises(TypeError):
        bond_model.get_bond_equivalent_yield(
            discount_yield="0.05", days_to_maturity=182
        )


def test_get_key_rate_duration(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 7: 0.048, 10: 0.05})
    recorder.capture(
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=spot_rates,
            key_rate_maturity=5,
            frequency=1,
        )
    )


def test_get_key_rate_duration_sums_to_approx_effective_duration(recorder):
    spot_rates = pd.Series({1: 0.03, 2: 0.035, 3: 0.04, 5: 0.045, 7: 0.048, 10: 0.05})
    total_krd = sum(
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=spot_rates,
            key_rate_maturity=maturity,
            frequency=1,
        )
        for maturity in spot_rates.index
    )
    approx_flat_rate = float(np.interp(5, spot_rates.index, spot_rates.to_numpy()))
    effective_duration = bond_model.get_effective_duration(
        par_value=100,
        coupon_rate=0.05,
        years_to_maturity=5,
        yield_to_maturity=approx_flat_rate,
        frequency=1,
    )
    # Summed key rate durations should approximate the parallel-shift duration.
    recorder.capture(bool(abs(total_krd - effective_duration) < 1))


def test_get_key_rate_duration_type_error():
    with pytest.raises(TypeError):
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            spot_rates=[0.03, 0.04, 0.05],
            key_rate_maturity=5,
        )


def test_get_key_rate_duration_invalid_key_rate_maturity():
    spot_rates = pd.Series({1: 0.03, 2: 0.04, 3: 0.05})
    with pytest.raises(ValueError):
        bond_model.get_key_rate_duration(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=3,
            spot_rates=spot_rates,
            key_rate_maturity=99,
        )


def test_get_taylor_price_change(recorder):
    recorder.capture(
        bond_model.get_taylor_price_change(
            par_value=100,
            coupon_rate=0.05,
            years_to_maturity=5,
            yield_to_maturity=0.08,
            frequency=1,
            yield_change=0.01,
        )
    )


def test_get_taylor_price_change_matches_actual_repricing(recorder):
    par_value, coupon_rate, years_to_maturity, ytm, frequency = 100, 0.05, 5, 0.08, 1
    yield_change = 0.001

    taylor_pct = bond_model.get_taylor_price_change(
        par_value, coupon_rate, years_to_maturity, ytm, frequency, yield_change
    )

    price_before = bond_model.get_bond_price(
        par_value, coupon_rate, years_to_maturity, ytm, frequency
    )
    price_after = bond_model.get_bond_price(
        par_value, coupon_rate, years_to_maturity, ytm + yield_change, frequency
    )
    actual_pct = (price_after - price_before) / price_before

    # For a small yield shift the Taylor approximation should match the reprice.
    recorder.capture(
        bool(abs(taylor_pct - actual_pct) < TAYLOR_APPROXIMATION_TOLERANCE)
    )


def test_periods_are_rounded_not_truncated():
    """0.29 years of 100 periods is 28.999999999999996 in floating point, not 28."""
    assert bond_model.get_number_of_periods(0.29, 100) == 29  # noqa: PLR2004
    assert bond_model.get_bond_price(100, 0.05, 0.29, 0.05, 100) == pytest.approx(
        bond_model.get_bond_price(100, 0.05, 29 / 100, 0.05, 100)
    )


def test_yield_to_maturity_from_a_guess_of_zero():
    """A guess of 0 starts the secant method from two distinct points."""
    price = bond_model.get_bond_price(1000, 0.05, 10, 0.06, 2)

    assert bond_model.get_yield_to_maturity(
        1000, 0.05, 10, price, 2, guess=0
    ) == pytest.approx(0.06, abs=1e-6)


def test_macaulay_duration_matches_the_price_periods():
    """The principal is weighted at the same moment the price discounts it to."""
    # Duration is a sensitivity of the full price, the clean price plus accrued interest.
    price = bond_model.get_bond_price(
        100, 0.05, 2.75, 0.05, 2
    ) + bond_model.get_accrued_interest(100, 0.05, 2.75, 2)
    shock = 1e-6
    effective_duration = (
        bond_model.get_bond_price(100, 0.05, 2.75, 0.05 - shock, 2)
        - bond_model.get_bond_price(100, 0.05, 2.75, 0.05 + shock, 2)
    ) / (2 * shock * price)
    modified_duration = bond_model.get_macaulays_duration(100, 0.05, 2.75, 0.05, 2) / (
        1 + 0.05 / 2
    )

    assert modified_duration == pytest.approx(effective_duration, rel=1e-6)


def test_fractional_maturity_matches_quantlib():
    """2.5 years of annual coupons: QuantLib (30/360) gives dirty 100.2043, clean 97.7043."""
    assert bond_model.get_accrued_fraction(2.5, 1) == pytest.approx(0.5)
    assert bond_model.get_accrued_interest(100, 0.05, 2.5, 1) == pytest.approx(2.5)
    np.testing.assert_allclose(
        bond_model.get_cash_flow_periods(2.5, 1), [0.5, 1.5, 2.5]
    )
    assert bond_model.get_bond_price(100, 0.05, 2.5, 0.06, 1) == pytest.approx(
        97.70426717046045, abs=1e-9
    )
    assert bond_model.get_macaulays_duration(100, 0.05, 2.5, 0.06, 1) == pytest.approx(
        2.3573474352559565, abs=1e-9
    )


def test_fractional_maturity_semi_annual_matches_quantlib():
    """2.75 years of semi-annual coupons, reference values from QuantLib (30/360)."""
    assert bond_model.get_bond_price(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        101.27289635, abs=1e-6
    )
    assert bond_model.get_macaulays_duration(
        100, 0.05, 2.75, 0.045, 2
    ) == pytest.approx(2.57435708, abs=1e-6)
    assert bond_model.get_modified_duration(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        2.51770864, abs=1e-6
    )
    assert bond_model.get_convexity(100, 0.05, 2.75, 0.045, 2) == pytest.approx(
        7.851, abs=1e-3
    )


def test_macaulay_duration_never_exceeds_the_maturity():
    for maturity in (0.5, 2.5, 7.25, 10.5):
        assert (
            bond_model.get_macaulays_duration(100, 0.05, maturity, 0.06, 1)
            <= maturity + 1e-12
        )


def test_bond_at_its_coupon_yield_prices_near_par_between_coupon_dates():
    """The clean price excludes the accrued coupon, so it stays near par, not par plus accrued."""
    assert bond_model.get_bond_price(100, 0.05, 2.5, 0.05, 1) == pytest.approx(
        100, abs=0.05
    )
    assert bond_model.get_bond_price(100, 0.04, 3.3, 0.04, 4) == pytest.approx(
        100, abs=0.05
    )


def test_yield_to_maturity_inverts_a_fractional_maturity_price():
    price = bond_model.get_bond_price(100, 0.05, 2.75, 0.045, 2)

    assert bond_model.get_yield_to_maturity(
        100, 0.05, 2.75, price, 2, tolerance=1e-10
    ) == pytest.approx(0.045, abs=1e-9)


def test_curve_price_uses_fractional_times_on_a_flat_curve():
    """On a flat curve the curve price equals the yield price, also between coupon dates."""
    flat_curve = pd.Series({0.25: 0.06, 1: 0.06, 5: 0.06, 10: 0.06})

    for maturity, frequency in ((2.5, 1), (2.75, 2), (7.3, 4)):
        assert bond_model.get_bond_price_from_curve(
            100, 0.05, maturity, flat_curve, frequency
        ) == pytest.approx(
            bond_model.get_bond_price(100, 0.05, maturity, 0.06, frequency)
        )

    # A bond priced on the curve has a Z-spread of zero.
    price = bond_model.get_bond_price_from_curve(
        100, 0.05, 2.5, pd.Series({1: 0.03, 2: 0.035, 5: 0.04}), 1
    )
    assert bond_model.get_z_spread(
        100,
        0.05,
        2.5,
        price,
        pd.Series({1: 0.03, 2: 0.035, 5: 0.04}),
        1,
        tolerance=1e-10,
    ) == pytest.approx(0, abs=1e-8)


def test_yield_to_maturity_is_nan_when_no_yield_gives_the_price():
    # A price of ten times par has no realistic yield; the steps settled at 5.5% before,
    # where the bond is worth 96.
    assert np.isnan(bond_model.get_yield_to_maturity(100, 0.05, 10, 1000, 2))

    ytm = bond_model.get_yield_to_maturity(100, 0.05, 10, 95, 2)
    assert bond_model.get_bond_price(100, 0.05, 10, ytm, 2) == pytest.approx(
        95, abs=0.01
    )

    with pytest.raises(ValueError, match="frequency"):
        bond_model.get_bond_price(100, 0.05, 10, 0.05, 0)


def test_get_current_yield_is_the_coupon_over_the_price():
    assert bond_model.get_current_yield(
        par_value=1000, coupon_rate=0.05, bond_price=950
    ) == pytest.approx(50 / 950)


def test_get_effective_yield_compounds_the_coupon_payments():
    assert bond_model.get_effective_yield(0.06, 2) == pytest.approx(1.03**2 - 1)
    assert bond_model.get_effective_yield(0.06, 1) == pytest.approx(0.06)

    with pytest.raises(ValueError, match="positive"):
        bond_model.get_effective_yield(0.06, 0)


def test_dollar_duration_and_dv01_of_a_zero_coupon_bond():
    # A two-year zero at 5%: price 100 / 1.05^2, Macaulay duration 2, modified 2 / 1.05.
    price = 100 / 1.05**2
    modified_duration = 2 / 1.05

    dollar_duration = bond_model.get_dollar_duration(100, 0.0, 2, 0.05, 1)
    assert dollar_duration == pytest.approx(modified_duration * price / 100)

    # DV01 is the price change of one basis point: the dollar duration over 100.
    dv01 = bond_model.get_dv01(100, 0.0, 2, 0.05, 1)
    assert dv01 == pytest.approx(modified_duration * price * 0.0001, rel=1e-6)
    assert dv01 == pytest.approx(dollar_duration / 100, rel=1e-6)


def test_yield_to_call_matches_quantlib_and_excel_yield():
    """
    A 6% semi-annual bond maturing 2036-11-15, settled 2026-12-03 (30/360: 9.95 years) at
    106, callable at 103. QuantLib's bondYield (Compounded, Semiannual, 30/360) of the bond
    redeemed at 103 on the call date, and Excel's YIELD with the call date as maturity.
    """
    # On a coupon date (2031-11-15): QuantLib 0.051503727095507046, Excel YIELD 0.05150372709550677.  # noqa: E501
    assert bond_model.get_yield_to_call(
        100, 0.06, 9.95, 4.95, 103, 106, 2, tolerance=1e-12
    ) == pytest.approx(0.051503727095507046, abs=1e-12)

    # A month after a coupon date (2031-12-15) the call pays a sixth of a coupon accrued on top.  # noqa: E501
    assert bond_model.get_yield_to_call(
        100, 0.06, 9.95, 5.033333333333332, 103, 106, 2, tolerance=1e-12
    ) == pytest.approx(0.051609214458571855, abs=1e-12)


def test_yield_to_call_of_a_closed_form_price():
    """26 semi-annual coupons of 55 and a call at 1,055, priced at 4.5% a period, yield 9%."""
    rate = 0.045
    price = 55 * (1 - (1 + rate) ** -26) / rate + 1055 / (1 + rate) ** 26

    assert bond_model.get_yield_to_call(1000, 0.11, 18, 13, 1055, price, 2) == (
        pytest.approx(0.09, abs=1e-6)
    )


def test_yield_to_call_at_maturity_at_par_is_the_yield_to_maturity():
    assert bond_model.get_yield_to_call(100, 0.05, 7.3, 7.3, 100, 97, 2) == (
        pytest.approx(
            bond_model.get_yield_to_maturity(100, 0.05, 7.3, 97, 2), abs=1e-12
        )
    )

    with pytest.raises(ValueError, match="years to call"):
        bond_model.get_yield_to_call(100, 0.05, 10, 0, 100, 97, 2)
    with pytest.raises(ValueError, match="years to call"):
        bond_model.get_yield_to_call(100, 0.05, 10, 11, 100, 97, 2)


def test_yield_to_call_before_the_next_coupon_date():
    """Called before its next coupon: the dirty price buys the call price plus accrued."""
    # 9.9 years of semi-annual coupons: 0.2 of the period elapsed, 0.3 at the call in 0.05 years.  # noqa: E501
    ytc = bond_model.get_yield_to_call(100, 0.06, 9.9, 0.05, 100, 100.5, 2)

    assert (100 + 3 * 0.3) / (1 + ytc / 2) ** 0.1 == pytest.approx(
        100.5 + 3 * 0.2, rel=1e-3
    )
    assert ytc < 0


def test_yield_to_worst():
    """The lowest of the yield to maturity and the yields to every call date."""
    schedule = [(5, 103), (7, 101.5), (9, 100)]
    yields = [bond_model.get_yield_to_maturity(100, 0.06, 10, 106, 2)] + [
        bond_model.get_yield_to_call(100, 0.06, 10, years, price, 106, 2)
        for years, price in schedule
    ]

    # A premium bond is worst to a call, here the one in seven years.
    assert bond_model.get_yield_to_worst(100, 0.06, 10, schedule, 106, 2) == min(yields)
    assert min(yields) == yields[2]
    assert bond_model.get_yield_to_worst(100, 0.06, 10, dict(schedule), 106, 2) == min(
        yields
    )

    # A discount bond callable at par is worst to maturity.
    assert bond_model.get_yield_to_worst(
        100, 0.04, 10, [(5, 100)], 95, 2
    ) == bond_model.get_yield_to_maturity(100, 0.04, 10, 95, 2)

    # Calls that have passed or fall on or after maturity are skipped.
    assert bond_model.get_yield_to_worst(
        100, 0.06, 10, [(-1, 100), (0, 100), (10, 100), (12, 100)], 106, 2
    ) == bond_model.get_yield_to_maturity(100, 0.06, 10, 106, 2)

    # A yield that cannot be solved leaves the worst unknown.
    assert np.isnan(bond_model.get_yield_to_worst(100, 0.05, 10, [(5, 100)], 1000, 2))


# QuantLib 1.44 AmortizingFixedRateBond with BondFunctions on a 30/360 schedule (clean price,
# accrued interest, Macaulay and modified duration, convexity), at the yield compounded at
# the coupon frequency.
AMORTIZING_REFERENCES = [
    (
        {"coupon_rate": 0.06, "years_to_maturity": 10, "frequency": 2},
        {"yield_to_maturity": 0.07},
        (
            95.52967445926615,
            0.0,
            4.682549490563951,
            4.524202406341982,
            30.233095235186354,
        ),
    ),
    (
        {"coupon_rate": 0.05, "years_to_maturity": 10, "frequency": 1},
        {"yield_to_maturity": 0.045, "sinking_fund": {6: 10, 7: 10, 8: 10, 9: 10}},
        (103.61957795241, 0.0, 7.467736528762714, 7.14615935766767, 65.07087777027013),
    ),
    (
        {"coupon_rate": 0.05, "years_to_maturity": 4.75, "frequency": 2},
        {"yield_to_maturity": 0.06},
        (
            97.66621485113603,
            1.25,
            2.378248899140061,
            2.3089795137282145,
            8.387805981879483,
        ),
    ),
]


@pytest.mark.parametrize("terms, inputs, reference", AMORTIZING_REFERENCES)
def test_amortizing_bond_measures_match_quantlib(terms, inputs, reference):
    """Annuity and sinking fund bonds, on and between coupon dates, against QuantLib."""
    measures = bond_model.get_amortizing_bond_measures(100, **terms, **inputs)

    price, accrued, macaulay, modified, convexity = reference
    assert measures["Price"] == pytest.approx(price, abs=1e-10)
    assert measures["Accrued Interest"] == pytest.approx(accrued, abs=1e-10)
    assert measures["Macaulay's Duration"] == pytest.approx(macaulay, abs=1e-10)
    assert measures["Modified Duration"] == pytest.approx(modified, abs=1e-10)
    assert measures["Convexity"] == pytest.approx(convexity, rel=1e-10)
    assert measures["DV01"] == pytest.approx(
        measures["Modified Duration"] * measures["Dirty Price"] * 1e-4, rel=1e-6
    )

    # The price solves back to the yield.
    solved = bond_model.get_amortizing_bond_measures(
        100,
        **terms,
        bond_price=measures["Price"],
        sinking_fund=inputs.get("sinking_fund"),
    )
    assert solved["Yield to Maturity"] == pytest.approx(
        inputs["yield_to_maturity"], abs=1e-10
    )


def test_amortizing_cash_flows():
    """The schedules repay the par value; the annuity pays equal amounts."""
    annuity = bond_model.get_amortizing_cash_flows(100, 0.06, 3, 2)
    assert annuity["Cash Flow"].to_numpy() == pytest.approx(
        np.full(6, 100 * 0.03 / (1 - 1.03**-6))
    )
    assert annuity["Principal"].sum() == pytest.approx(100)
    assert annuity["Interest"].iloc[1] == pytest.approx(
        0.03 * (100 - annuity["Principal"].iloc[0])
    )

    linear = bond_model.get_amortizing_cash_flows(100, 0.06, 2, 2, "linear")
    assert linear["Principal"].tolist() == pytest.approx([25] * 4)
    assert linear["Outstanding"].tolist() == pytest.approx([100, 75, 50, 25])

    sinking = bond_model.get_amortizing_cash_flows(
        100, 0.05, 5, 1, sinking_fund={3.1: 20, 4: 30}
    )
    assert sinking["Principal"].tolist() == pytest.approx([0, 0, 20, 30, 50])

    bullet = bond_model.get_amortizing_cash_flows(100, 0.05, 2.5, 2, "bullet")
    assert bullet["Cash Flow"].tolist() == pytest.approx(
        bond_model.get_bond_cash_flows(100, 0.05, 2.5, 2).tolist()
    )

    with pytest.raises(ValueError):
        bond_model.get_amortizing_cash_flows(100, 0.05, 5, 1, "balloon")
    with pytest.raises(ValueError):
        bond_model.get_amortizing_cash_flows(100, 0.05, 5, 1, sinking_fund={2: 120})
    with pytest.raises(ValueError):
        bond_model.get_amortizing_cash_flows(100, 0.05, 5, 1, sinking_fund={6: 10})


def test_amortizing_bullet_matches_the_bond_measures():
    """A bullet schedule gives the measures of the bond functions."""
    measures = bond_model.get_amortizing_bond_measures(
        100, 0.05, 7.3, 2, 0.06, amortization="bullet"
    )

    assert measures["Price"] == pytest.approx(
        bond_model.get_bond_price(100, 0.05, 7.3, 0.06, 2), abs=1e-12
    )
    assert measures["Macaulay's Duration"] == pytest.approx(
        bond_model.get_macaulays_duration(100, 0.05, 7.3, 0.06, 2), abs=1e-12
    )
    assert measures["DV01"] == pytest.approx(
        bond_model.get_dv01(100, 0.05, 7.3, 0.06, 2), abs=1e-12
    )
    # The analytic convexity against the 1 basis point difference of get_convexity.
    assert measures["Convexity"] == pytest.approx(
        bond_model.get_convexity(100, 0.05, 7.3, 0.06, 2), rel=1e-6
    )
    assert measures["Weighted Average Life"] == pytest.approx(7.3)

    with pytest.raises(ValueError):
        bond_model.get_amortizing_bond_measures(100, 0.05, 7, 2)


def test_amortizing_bond_weighted_average_life(recorder):
    """An annuity returns its principal well before maturity."""
    measures = bond_model.get_amortizing_bond_measures(100, 0.06, 10, 2, 0.07)
    principal = bond_model.get_amortizing_cash_flows(100, 0.06, 10, 2)["Principal"]

    assert measures["Weighted Average Life"] == pytest.approx(
        (principal.index * principal).sum() / 100
    )
    assert measures["Weighted Average Life"] < 10
    recorder.capture(measures.round(8))


@pytest.mark.parametrize("frequency", [0, -1])
def test_bond_functions_refuse_a_frequency_below_one(frequency):
    for function, arguments in [
        (bond_model.get_accrued_interest, (100, 0.05, 2.75, frequency)),
        (bond_model.get_convexity, (100, 0.05, 5, 0.05, frequency)),
        (bond_model.get_effective_duration, (100, 0.05, 5, 0.05, frequency)),
        (bond_model.get_yield_to_call, (100, 0.05, 5, 3, 100, 98, frequency)),
    ]:
        with pytest.raises(ValueError, match="frequency must be at least 1"):
            function(*arguments)
