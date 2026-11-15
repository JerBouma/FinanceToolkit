"""Derivative Model Tests""" ""

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
