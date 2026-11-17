"""Derivative Model Tests""" ""

import math

import pytest

from financetoolkit.fixedincome.bonds import derivative_model


def test_get_annuity_factor(recorder):
    recorder.capture(
        derivative_model._get_annuity_factor(
            risk_free_rate=0.03,
            years_to_maturity=1,
            tenor=5,
            payment_frequency=2,
        )
    )


def test_get_annuity_factor_scales_with_tenor(recorder):
    short_tenor = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=1, payment_frequency=2
    )
    long_tenor = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=10, payment_frequency=2
    )
    recorder.capture(bool(long_tenor > short_tenor))


def test_get_black_price_receiver(recorder):
    recorder.capture(
        derivative_model.get_black_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.2,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=True,
        )
    )


def test_get_black_price_payer(recorder):
    recorder.capture(
        derivative_model.get_black_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.2,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=False,
        )
    )


def test_get_black_price_default_tenor_matches_maturity(recorder):
    with_default_tenor = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=5,
        risk_free_rate=0.03,
        notional=10_000_000,
        is_receiver=True,
    )
    with_explicit_tenor = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=5,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=5,
        is_receiver=True,
    )
    recorder.capture(with_default_tenor == with_explicit_tenor)


def test_get_black_price_scales_with_tenor(recorder):
    short_tenor_price, _ = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=1,
        is_receiver=True,
    )
    long_tenor_price, _ = derivative_model.get_black_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.2,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=10,
        is_receiver=True,
    )
    recorder.capture(bool(long_tenor_price > short_tenor_price))


def test_get_bachelier_price_receiver(recorder):
    recorder.capture(
        derivative_model.get_bachelier_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.01,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=True,
        )
    )


def test_get_bachelier_price_payer(recorder):
    recorder.capture(
        derivative_model.get_bachelier_price(
            forward_rate=0.035,
            strike_rate=0.03,
            volatility=0.01,
            years_to_maturity=1,
            risk_free_rate=0.03,
            notional=10_000_000,
            tenor=5,
            payment_frequency=2,
            is_receiver=False,
        )
    )


def test_get_bachelier_price_scales_with_tenor(recorder):
    short_tenor_price, _ = derivative_model.get_bachelier_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.01,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=1,
        is_receiver=True,
    )
    long_tenor_price, _ = derivative_model.get_bachelier_price(
        forward_rate=0.035,
        strike_rate=0.03,
        volatility=0.01,
        years_to_maturity=1,
        risk_free_rate=0.03,
        notional=10_000_000,
        tenor=10,
        is_receiver=True,
    )
    recorder.capture(bool(long_tenor_price > short_tenor_price))


def test_swaptions_without_volatility_or_time_pay_their_intrinsic_value():
    _, black_payoff = derivative_model.get_black_price(0.03, 0.03, 0.2, 0, 0.03)
    _, bachelier_payoff = derivative_model.get_bachelier_price(
        0.03, 0.02, 0.0, 1, 0.03, is_receiver=False
    )

    assert black_payoff == 0
    assert abs(bachelier_payoff - 0.01) < 1e-12


def test_get_annuity_factor_ends_with_a_short_period():
    # A 1.25-year swap paying twice a year accrues 0.5, 0.5 and a last 0.25 year, paid at
    # the end of the tenor: 2.5, 3.0 and 3.25 years from today.
    annuity = derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=2, tenor=1.25, payment_frequency=2
    )
    expected = (
        0.5 * math.exp(-0.03 * 2.5)
        + 0.5 * math.exp(-0.03 * 3.0)
        + 0.25 * math.exp(-0.03 * 3.25)
    )
    assert annuity == pytest.approx(expected)

    # A tenor shorter than one period is a single payment accruing the tenor.
    assert derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=0.3, payment_frequency=2
    ) == pytest.approx(0.3 * math.exp(-0.03 * 1.3))

    # A whole number of periods in floating point noise has no short period.
    assert derivative_model._get_annuity_factor(
        risk_free_rate=0.03, years_to_maturity=1, tenor=0.1 * 3, payment_frequency=10
    ) == pytest.approx(0.1 * sum(math.exp(-0.03 * (1 + 0.1 * k)) for k in (1, 2, 3)))
