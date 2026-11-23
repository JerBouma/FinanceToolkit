"""Convertible Bond Model Tests"""

import numpy as np
import pytest
from scipy.stats import norm

from financetoolkit.fixedincome.bonds.convertible_model import (
    get_convertible_bond_value,
)

# QuantLib 1.44 ConvertibleZeroCouponBond / ConvertibleFixedCouponBond with the
# BinomialCRRConvertibleEngine without a credit spread (where it and the Tsiveriotis-Fernandes
# split are the same tree), flat continuously compounded curves on Actual/365, par 100 and a
# conversion ratio of 2: (stock price, coupon, years, rate, volatility, dividend yield,
# frequency, call price, steps, QuantLib value). The call is callable every day.
QUANTLIB_REFERENCES = [
    (50, 0.0, 5, 0.04, 0.3, 0.0, 1, None, 2000, 115.83383685650512),
    (40, 0.0, 5, 0.04, 0.3, 0.02, 1, None, 2000, 98.67440353638807),
    (30, 0.05, 7, 0.03, 0.25, 0.01, 2, None, 2100, 119.88692366221281),
    (50, 0.0, 5, 0.04, 0.3, 0.0, 1, 110, 2000, 104.9807683668034),
]


@pytest.mark.parametrize(
    "stock, coupon, years, rate, volatility, dividend, frequency, call, steps, reference",
    QUANTLIB_REFERENCES,
)
def test_convertible_matches_quantlib_without_a_credit_spread(
    stock, coupon, years, rate, volatility, dividend, frequency, call, steps, reference
):
    value = get_convertible_bond_value(
        stock,
        2,
        100,
        coupon,
        years,
        rate,
        volatility,
        0.0,
        dividend,
        frequency,
        call_price=call,
        steps=steps,
    )
    # The trees place the coupons on their steps differently, a few cents at most.
    assert value["Convertible Value"] == pytest.approx(reference, abs=0.02)


@pytest.mark.parametrize("stock, spread", [(30, 0.03), (40, 0.05), (45, 0.02)])
def test_tsiveriotis_fernandes_split_matches_its_closed_form(stock, spread):
    """A zero-coupon convertible without dividends is held to maturity: the shares are worth
    a call discounted at the risk-free rate, the redemption is discounted with the spread.
    """
    strike, years, rate, volatility = 50, 5, 0.04, 0.3
    d1 = (np.log(stock / strike) + (rate + volatility**2 / 2) * years) / (
        volatility * np.sqrt(years)
    )
    d2 = d1 - volatility * np.sqrt(years)
    closed_form = 2 * stock * norm.cdf(d1) + 100 * np.exp(
        -(rate + spread) * years
    ) * norm.cdf(-d2)

    # Averaged over an odd and even number of steps, as the tree oscillates around it.
    value = np.mean(
        [
            get_convertible_bond_value(
                stock, 2, 100, 0, years, rate, volatility, spread, steps=steps
            )["Convertible Value"]
            for steps in (2000, 2001)
        ]
    )
    assert value == pytest.approx(closed_form, abs=0.15)


def test_convertible_bounds_and_components(recorder):
    value = get_convertible_bond_value(
        40, 2, 100, 0.03, 5, 0.04, 0.3, 0.03, 0.01, 2, call_price=120, call_start=2
    )

    assert value["Convertible Value"] == pytest.approx(
        value["Equity Component"] + value["Debt Component"]
    )
    assert value["Convertible Value"] >= max(
        value["Bond Floor"], value["Conversion Value"]
    )
    assert 0 < value["Delta"] < 2
    assert value["Gamma"] > 0
    recorder.capture(value.round(5))


def test_credit_spread_puts_and_calls_move_the_value():
    base = get_convertible_bond_value(40, 2, 100, 0.03, 5, 0.04, 0.3, 0.02)
    wider = get_convertible_bond_value(40, 2, 100, 0.03, 5, 0.04, 0.3, 0.06)
    callable_bond = get_convertible_bond_value(
        40, 2, 100, 0.03, 5, 0.04, 0.3, 0.02, call_price=105
    )
    puttable = get_convertible_bond_value(
        40, 2, 100, 0.03, 5, 0.04, 0.3, 0.06, put_price=100, put_dates=[2]
    )

    assert wider["Convertible Value"] < base["Convertible Value"]
    assert callable_bond["Convertible Value"] < base["Convertible Value"]
    assert puttable["Convertible Value"] > wider["Convertible Value"]

    # Deep out of the money it is the straight bond; deep in the money, without coupons
    # and with a high dividend yield, it is converted today.
    floor = get_convertible_bond_value(1, 2, 100, 0.03, 5, 0.04, 0.3, 0.02)
    assert floor["Convertible Value"] == pytest.approx(floor["Bond Floor"], rel=1e-6)
    shares = get_convertible_bond_value(500, 2, 100, 0.0, 5, 0.04, 0.3, 0.02, 0.1)
    assert shares["Convertible Value"] == pytest.approx(1000, rel=1e-6)

    with pytest.raises(ValueError):
        get_convertible_bond_value(40, 2, 100, 0.03, 5, 0.04, 0.0, 0.02)
