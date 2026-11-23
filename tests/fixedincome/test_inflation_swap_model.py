"""Inflation Swap Model Tests

The reference values come from QuantLib 1.44: a ZeroInflationCurve and a YoYInflationCurve on a
30/360 day count (so whole months are exact twelfths of a year), with no observation lag, the
nominal curve's discount factors at every year, a ZeroCouponInflationSwap, a
YearOnYearInflationSwap and a ZeroInflationIndex forecast with MultiplicativePriceSeasonality.
"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.inflation import inflation_swap_model

NOMINAL = pd.Series([0.040, 0.042, 0.045], index=[1, 5, 10])
REAL = pd.Series([0.015, 0.018, 0.021], index=[1, 5, 10])


def zero_inflation_curve() -> pd.Series:
    return inflation_swap_model.get_zero_coupon_inflation_swap_rate(
        NOMINAL, REAL, range(1, 11)
    )


def test_zero_coupon_inflation_swap_rate_matches_quantlib():
    rates = zero_inflation_curve()

    # QuantLib's fairRate on the forward index ratios P_real / P_nominal.
    assert rates[[1, 5, 10]].tolist() == pytest.approx(
        [0.024630541871921485, 0.023575638506876162, 0.02350636630754166], rel=1e-14
    )
    # The Fisher relation of the two zero rates.
    assert rates[5] == pytest.approx((1.042 / 1.018) - 1, rel=1e-14)
    with pytest.raises(ValueError, match="positive"):
        inflation_swap_model.get_zero_coupon_inflation_swap_rate(NOMINAL, REAL, [0])


def test_year_on_year_inflation_swap_rate_matches_quantlib():
    rates = inflation_swap_model.get_year_on_year_inflation_swap_rate(
        NOMINAL, zero_inflation_curve(), [1, 5, 10]
    )

    assert rates.tolist() == pytest.approx(
        [0.024630541871921575, 0.023619692781693243, 0.023539654131637083], rel=1e-13
    )
    with pytest.raises(ValueError, match="whole numbers"):
        inflation_swap_model.get_year_on_year_inflation_swap_rate(
            NOMINAL, zero_inflation_curve(), [2.5]
        )


def test_forward_year_on_year_rates_compound_to_the_zero_coupon_rates():
    curve = zero_inflation_curve()
    forwards = inflation_swap_model.get_forward_year_on_year_inflation_rates(curve, 10)

    assert np.prod(1 + forwards.to_numpy()) == pytest.approx(
        (1 + curve[10]) ** 10, rel=1e-14
    )
    assert forwards[1] == pytest.approx(curve[1], rel=1e-14)


@pytest.mark.parametrize(
    "method, maturity, value, inflation_leg, fixed_leg",
    [
        ("zero-coupon", 1, 15678.28723001515, 236832.1333838604, 221153.84615384525),
        ("zero-coupon", 10, 40100.5338717713, 1684211.850520784, 1644111.3166490127),
        ("year-on-year", 5, 27487.11100034509, 1047675.7781662073, 1020188.6671658622),
        ("year-on-year", 10, 43107.10547315166, 1880327.2206509942, 1837220.1151778426),
    ],
)
def test_inflation_swap_value_matches_quantlib(
    method, maturity, value, inflation_leg, fixed_leg
):
    swap = inflation_swap_model.get_inflation_swap_value(
        NOMINAL, zero_inflation_curve(), 0.023, maturity, method=method
    )

    # QuantLib's NPV of the inflation receiver (a "Receiver" ZeroCouponInflationSwap and a
    # "Payer" YearOnYearInflationSwap, as QuantLib names the two sides).
    assert swap["Value"] == pytest.approx(value, rel=1e-10)
    assert swap["Inflation Leg"] == pytest.approx(inflation_leg, rel=1e-12)
    assert swap["Fixed Leg"] == pytest.approx(fixed_leg, rel=1e-12)
    receiver = inflation_swap_model.get_inflation_swap_value(
        NOMINAL,
        zero_inflation_curve(),
        0.023,
        maturity,
        method=method,
        is_receiver=True,
    )
    assert receiver["Value"] == pytest.approx(-value, rel=1e-10)


def test_inflation_swap_value_at_the_fair_rate_is_zero():
    curve = zero_inflation_curve()
    zero_coupon = inflation_swap_model.get_inflation_swap_value(
        NOMINAL, curve, curve[7], 7
    )
    year_on_year = inflation_swap_model.get_inflation_swap_value(
        NOMINAL,
        curve,
        inflation_swap_model.get_year_on_year_inflation_swap_rate(NOMINAL, curve, 7)[7],
        7,
        method="year-on-year",
    )

    assert zero_coupon["Value"] == pytest.approx(0, abs=1e-6)
    assert year_on_year["Value"] == pytest.approx(0, abs=1e-6)


def test_running_inflation_swaps():
    curve = zero_inflation_curve()
    # A zero-coupon swap started two years ago, the index 5% up since.
    running = inflation_swap_model.get_inflation_swap_value(
        NOMINAL, curve, 0.02, 3, index_ratio=1.05, years_elapsed=2
    )
    discount = (1 + np.interp(3, [1, 5, 10], [0.04, 0.042, 0.045])) ** -3
    assert running["Inflation Leg"] == pytest.approx(
        10_000_000 * (1.05 * (1 + curve[3]) ** 3 - 1) * discount, rel=1e-12
    )
    assert running["Fixed Leg"] == pytest.approx(
        10_000_000 * (1.02**5 - 1) * discount, rel=1e-12
    )
    # A year-on-year swap half way through its year pays that year's inflation in half a year.
    year_on_year = inflation_swap_model.get_inflation_swap_value(
        NOMINAL, curve, 0.02, 2.5, method="year-on-year", index_ratio=1.01
    )
    growth = (1 + np.interp([0.5, 1.5, 2.5], curve.index, curve)) ** np.array(
        [0.5, 1.5, 2.5]
    )
    rates = np.array(
        [1.01 * growth[0] - 1, growth[1] / growth[0] - 1, growth[2] / growth[1] - 1]
    )
    discounts = (1 + np.interp([0.5, 1.5, 2.5], NOMINAL.index, NOMINAL)) ** -np.array(
        [0.5, 1.5, 2.5]
    )
    assert year_on_year["Inflation Leg"] == pytest.approx(
        10_000_000 * np.sum(rates * discounts), rel=1e-12
    )
    with pytest.raises(ValueError, match="method"):
        inflation_swap_model.get_inflation_swap_value(
            NOMINAL, curve, 0.02, 3, method="cpi"
        )


FACTORS = pd.Series(
    [0.998, 1.001, 1.003, 1.004, 1.004, 1.003, 1.001, 1.0, 0.999, 0.998, 0.996, 0.993],
    index=range(1, 13),
)


def test_forward_inflation_index_matches_quantlib_seasonality():
    forward = inflation_swap_model.get_forward_inflation_index(
        320.0,
        pd.Series([0.030, 0.026, 0.024], index=[1, 3, 5]),
        months=13,
        base_month=10,
        seasonal_factors=FACTORS,
    )

    # QuantLib's ZeroInflationIndex fixings of November 2024 to November 2025, from a base
    # of 320 in October 2024.
    assert forward.tolist() == pytest.approx(
        [
            320.146342186201,
            319.96923500664644,
            322.3734629688745,
            324.13997136160674,
            325.58861752137466,
            326.7170222477784,
            327.5227945638451,
            328.00353166754354,
            328.15681888311764,
            328.6375046214829,
            329.118565351811,
            329.6,
            329.6929286408601,
        ],
        rel=1e-13,
    )


def test_forward_inflation_index_without_seasonality_and_errors():
    curve = pd.Series([0.03], index=[1])
    forward = inflation_swap_model.get_forward_inflation_index(100.0, curve, months=24)

    assert forward[24] == pytest.approx(100 * 1.03**2, rel=1e-14)
    with pytest.raises(ValueError, match="base month"):
        inflation_swap_model.get_forward_inflation_index(100.0, curve, base_month=13)
    with pytest.raises(ValueError, match="every calendar month"):
        inflation_swap_model.get_forward_inflation_index(
            100.0, curve, seasonal_factors=FACTORS.iloc[:11]
        )


def test_inflation_seasonality_recovers_the_seasonal_pattern():
    months = pd.period_range("2015-01", "2024-12", freq="M")
    pattern = FACTORS / np.exp(np.log(FACTORS).mean())
    index = pd.Series(
        100 * 1.002 ** np.arange(len(months)) * pattern[months.month].to_numpy(),
        index=months,
    )
    factors = inflation_swap_model.get_inflation_seasonality(index)

    assert factors.to_numpy() == pytest.approx(pattern.to_numpy(), rel=1e-12)
    assert np.prod(factors) == pytest.approx(1.0, rel=1e-14)
    with pytest.raises(ValueError, match="every calendar month"):
        inflation_swap_model.get_inflation_seasonality(index.iloc[:12])


def test_inflation_seasonality_skips_gaps():
    months = pd.period_range("2015-01", "2024-12", freq="M")
    index = pd.Series(100 * 1.002 ** np.arange(len(months)), index=months)
    # A missing month is not a change of two months.
    factors = inflation_swap_model.get_inflation_seasonality(index.drop(months[30]))

    assert factors.to_numpy() == pytest.approx(np.ones(12), rel=1e-12)
