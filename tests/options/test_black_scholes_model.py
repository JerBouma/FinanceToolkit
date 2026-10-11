"""Black Scholes Model Tests"""

# pylint: disable=missing-function-docstring


from financetoolkit.options import black_scholes_model


def test_black_scholes_at_expiration_is_the_intrinsic_value():
    assert black_scholes_model.get_black_scholes(100, 100, 0.05, 0.2, 0) == 0
    assert (
        black_scholes_model.get_black_scholes(110, 100, 0.05, 0.2, 0) == 10
    )  # noqa: PLR2004
