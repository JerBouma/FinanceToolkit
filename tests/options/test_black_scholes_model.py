"""Black Scholes Model Tests"""

# pylint: disable=missing-function-docstring


from financetoolkit.options import black_scholes_model


def test_black_scholes_at_expiration_is_the_intrinsic_value():
    assert black_scholes_model.get_black_scholes(100, 100, 0.05, 0.2, 0) == 0
    assert (
        black_scholes_model.get_black_scholes(110, 100, 0.05, 0.2, 0) == 10
    )  # noqa: PLR2004


def test_bivariate_normal_distribution_matches_scipy():
    import pytest
    from scipy.stats import multivariate_normal

    for x, y, rho in [
        (0.3, -0.5, 0.2),
        (1.0, 0.5, 0.6),
        (-0.2, 0.4, 0.95),
        (0.5, 0.1, -0.97),
        (1.2, -0.3, -0.5),
        (-1.5, -2.0, 0.0),
    ]:
        assert black_scholes_model._bivariate_normal_cdf(x, y, rho) == pytest.approx(
            multivariate_normal.cdf([x, y], cov=[[1, rho], [rho, 1]]), abs=1e-7
        )


def test_bjerksund_stensland_2002_against_a_finite_difference_american_price():
    """
    The reference prices come from QuantLib's FdBlackScholesVanillaEngine (800 time and
    1600 space steps) for American options; the 2002 approximation sits slightly below
    them and closer than the single-boundary 1993 approximation, whose prices are listed
    alongside.
    """
    cases = [
        # S, K, r, q, sigma, T, put, finite difference, 1993 approximation
        (100.0, 100.0, 0.05, 0.03, 0.25, 1.0, True, 8.88229, 8.81523),
        (100.0, 100.0, 0.05, 0.03, 0.25, 1.0, False, 10.55077, 10.54944),
        (90.0, 100.0, 0.06, 0.0, 0.3, 2.0, True, 16.55625, 16.42289),
        (110.0, 100.0, 0.03, 0.07, 0.2, 1.0, False, 12.18167, 12.08543),
        (42.0, 40.0, 0.04, 0.08, 0.35, 274 / 365, False, 5.31036, 5.27197),
    ]

    for S, K, r, q, sigma, T, put, finite_difference, approximation_1993 in cases:
        price = black_scholes_model.get_bjerksund_stensland(S, K, r, sigma, T, q, put)

        assert price <= finite_difference + 1e-4
        assert abs(price - finite_difference) < abs(
            approximation_1993 - finite_difference
        )


def test_bjerksund_stensland_at_expiration_is_the_exercise_value():
    assert black_scholes_model.get_bjerksund_stensland(110.0, 100.0, 0.05, 0.2, 0) == 10
    assert (
        black_scholes_model.get_bjerksund_stensland(
            110.0, 100.0, 0.05, 0.2, 0, 0.0, True
        )
        == 0
    )
