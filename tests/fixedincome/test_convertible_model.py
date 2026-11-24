"""Convertible Bond Model Tests"""

import numpy as np
import pytest
from scipy.stats import norm

from financetoolkit.fixedincome.bonds.convertible_model import (
    get_convertible_bond_value,
)

# QuantLib 1.44 ConvertibleZeroCouponBond / ConvertibleFixedCouponBond with the
# BinomialCRRConvertibleEngine without a credit spread (where it and the Tsiveriotis-Fernandes
# split value the same bond), flat continuously compounded curves on Actual/365, par 100 and a
# conversion ratio of 2: (stock price, coupon, years, rate, volatility, dividend yield,
# frequency, call price, steps, QuantLib value). QuantLib calls on call dates only, so a
# callable bond is compared with a binomial tree called on every step instead (below).
QUANTLIB_REFERENCES = [
    (50, 0.0, 5, 0.04, 0.3, 0.0, 1, None, 2000, 115.83383685650512),
    (40, 0.0, 5, 0.04, 0.3, 0.02, 1, None, 2000, 98.67440353638807),
    (30, 0.05, 7, 0.03, 0.25, 0.01, 2, None, 2100, 119.88692366221281),
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


def test_callable_convertible_is_stable_in_the_steps():
    """The call caps the value at max(Call Price, κS); with that stock price on a node of the
    tree the value no longer swings with the steps. On a Cox-Ross-Rubinstein tree it swings
    between 102.85 and 103.65 from 200 to 1,000 steps for this bond.
    """
    values = [
        get_convertible_bond_value(
            100, 1, 100, 0.03, 5, 0.04, 0.3, 0.02, call_price=105, steps=steps
        )["Convertible Value"]
        for steps in (200, 300, 500, 700, 800, 1000)
    ]
    assert max(values) - min(values) < 0.01

    # A Cox-Ross-Rubinstein tree with the same call on every step converges slowly to the
    # same value: 104.9762 at 2,000 steps, 104.7119 at 16,000 and 104.7077 at 24,000.
    callable_bond = get_convertible_bond_value(
        50, 2, 100, 0.0, 5, 0.04, 0.3, 0.0, 0.0, 1, call_price=110
    )
    assert callable_bond["Convertible Value"] == pytest.approx(104.7077, abs=0.01)


def test_convertible_refuses_a_bond_that_has_matured():
    with pytest.raises(ValueError, match="years_to_maturity must be positive"):
        get_convertible_bond_value(40, 2, 100, 0.03, 0, 0.04, 0.3, 0.02)
