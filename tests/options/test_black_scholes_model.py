"""Black Scholes Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.options.pricing import black_scholes_model


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


def test_bjerksund_stensland_is_never_below_the_european_or_exercise_value():
    import itertools
    import math

    for S, r, q, sigma, T, put in itertools.product(
        (50.0, 100.0, 150.0),
        (-0.01, 0.0, 0.02, 0.08),
        (0.0, 0.03, 0.12),
        (0.05, 0.3, 1.0),
        (1 / 365, 0.25, 2.0),
        (False, True),
    ):
        price = black_scholes_model.get_bjerksund_stensland(
            S, 100.0, r, sigma, T, q, put
        )
        european = float(
            black_scholes_model.get_black_scholes(S, 100.0, r, sigma, T, q, put)
        )
        exercise = max(100.0 - S, 0.0) if put else max(S - 100.0, 0.0)

        assert math.isfinite(price)
        assert price >= european - 1e-8
        assert price >= exercise - 1e-8


def test_bjerksund_stensland_stays_finite_and_close_at_low_volatility():
    """
    Finite-difference references from QuantLib's FdBlackScholesVanillaEngine (400 time and
    800 space steps). At low volatility with a negative cost of carry the boundary of the
    approximation falls below the strike, where a binomial tree takes over.
    """
    cases = [
        # S, K, r, q, sigma, T, put, finite difference
        (100.0, 100.0, 0.08, 0.0, 0.05, 2.0, True, 0.5654),
        (100.0, 100.0, 0.02, 0.12, 0.05, 2.0, False, 0.4513),
        (50.0, 100.0, 0.02, 0.12, 0.05, 2.0, False, 0.0),
    ]

    for S, K, r, q, sigma, T, put, finite_difference in cases:
        price = black_scholes_model.get_bjerksund_stensland(S, K, r, sigma, T, q, put)
        assert abs(price - finite_difference) < 5e-3

    for sigma in (0.001, 0.005, 0.01):
        price = black_scholes_model.get_bjerksund_stensland(
            100.0, 100.0, 0.05, sigma, 1.0, 0.0, True
        )
        assert price == price and price >= 0  # noqa: PLR0124


def test_options_without_volatility_or_time_are_worth_their_discounted_intrinsic_value():
    import numpy as np
    import pytest

    from financetoolkit.options.greeks import greeks_model

    # At the money at expiration, and in the money without volatility.
    assert black_scholes_model.get_black_scholes(100, 100, 0, 0, 1) == 0
    assert black_scholes_model.get_black_scholes(100, 100, 0.05, 0.2, 0) == 0
    assert black_scholes_model.get_black_scholes(110, 100, 0.05, 0, 1) == pytest.approx(
        110 - 100 * np.exp(-0.05)
    )
    assert black_scholes_model.get_black_scholes(
        90, 100, 0.05, 0, 1, put_option=True
    ) == pytest.approx(100 * np.exp(-0.05) - 90)
    assert np.isnan(greeks_model.get_lambda(90, 100, 0.0, 0.2, 0))

    with pytest.raises(ValueError, match="volatility must be positive"):
        black_scholes_model.get_bjerksund_stensland(
            100, 100, 0.05, 0, 1, put_option=True
        )


def test_get_d1_and_d2_by_hand():
    # ln(S/K) = 0, (r + sigma^2 / 2) T = 0.07 over sigma sqrt(T) = 0.2.
    d1 = black_scholes_model.get_d1(100, 100, 0.05, 0.2, 1)
    assert d1 == pytest.approx(0.35)
    assert black_scholes_model.get_d2(d1, 0.2, 1) == pytest.approx(0.15)

    # A dividend yield lowers the forward: (0.05 - 0.05 + 0.02) / 0.2.
    assert black_scholes_model.get_d1(100, 100, 0.05, 0.2, 1, 0.05) == pytest.approx(
        0.1
    )
    # Four months: sigma sqrt(T) = 0.2 / sqrt(3).
    assert black_scholes_model.get_d2(0.5, 0.2, 1 / 3) == pytest.approx(
        0.5 - 0.2 / np.sqrt(3)
    )


def test_get_d1_without_uncertainty_is_the_limit_on_either_side_of_the_strike():
    stock_prices = pd.Series([110.0, 90.0, 100.0])

    d1 = black_scholes_model.get_d1(stock_prices, 100, 0.0, 0.0, 1)

    assert isinstance(d1, pd.Series)
    assert d1.tolist() == [np.inf, -np.inf, 0.0]
    assert black_scholes_model.get_d1(100, 100, 0.05, 0.2, 0) == 0.0


def test_greeks_without_uncertainty_are_their_limits_off_the_strike():
    from financetoolkit.options.greeks import greeks_model

    rate, dividend, volatility = 0.05, 0.02, 0.2

    def greek(name, stock_price, time, vol=volatility, **kwargs):
        return float(
            getattr(greeks_model, name)(
                stock_price, 100.0, time, rate, vol, dividend, **kwargs
            )
        )

    curvature = [
        "get_gamma",
        "get_dual_gamma",
        "get_vanna",
        "get_vomma",
        "get_vera",
        "get_veta",
        "get_second_order_partial_derivative",
        "get_speed",
        "get_zomma",
        "get_color",
        "get_ultima",
    ]
    # At expiration (in and out of the money) and without volatility the limit is the
    # value a moment before expiration: 0 for the Greeks of the curvature.
    for name in curvature:
        for stock_price in (90.0, 110.0):
            assert greek(name, stock_price, 0.0) == pytest.approx(
                greek(name, stock_price, 1e-9), abs=1e-9
            ), name
        assert greek(name, 110.0, 0.5, vol=0.0) == 0.0, name

    # Theta keeps the carry of the intrinsic value, Charm the dividend yield.
    for stock_price, put_option in ((110.0, False), (90.0, True), (90.0, False)):
        for name in ("get_theta", "get_charm"):
            assert greek(
                name, stock_price, 0.0, put_option=put_option
            ) == pytest.approx(
                greek(name, stock_price, 1e-9, put_option=put_option), abs=1e-6
            ), name
    assert greek("get_theta", 110.0, 0.0) == pytest.approx(
        (-rate * 100.0 + dividend * 110.0) / 365
    )
    assert greek("get_charm", 90.0, 0.0, put_option=True) == pytest.approx(-dividend)

    # At the strike the limits are infinite and stay so.
    assert np.isinf(greek("get_gamma", 100.0, 0.0))
    assert np.isnan(greek("get_second_order_partial_derivative", 100.0, 0.0))

    # Arrays keep their shape.
    gamma = greeks_model.get_gamma(
        np.array([90.0, 110.0]), 100.0, np.array([0.0, 0.5]), rate, volatility, dividend
    )
    assert gamma[0] == 0.0
    assert gamma[1] == pytest.approx(
        float(greeks_model.get_gamma(110.0, 100.0, 0.5, rate, volatility, dividend))
    )


def test_black_76_matches_quantlib():
    # QuantLib 1.44 blackFormula with the standard deviation σ√t and discount e^(-rt).
    assert black_scholes_model.get_black_76(
        70.0, 75.0, 0.04, 0.3, 0.5, put_option=True
    ) == pytest.approx(8.766689044573, abs=1e-11)
    assert black_scholes_model.get_black_76(
        2000.0, 2100.0, 0.03, 0.18, 0.25
    ) == pytest.approx(33.870178357681, abs=1e-9)


def test_black_76_is_black_scholes_on_the_forward_and_handles_arrays():
    forward = 100 * np.exp((0.05 - 0.02) * 1.5)
    for put in (False, True):
        assert black_scholes_model.get_black_76(
            forward, 105.0, 0.05, 0.25, 1.5, put
        ) == pytest.approx(
            black_scholes_model.get_black_scholes(
                100.0, 105.0, 0.05, 0.25, 1.5, 0.02, put
            ),
            abs=1e-12,
        )

    strikes = np.array([90.0, 100.0, 110.0])
    values = black_scholes_model.get_black_76(100.0, strikes, 0.03, 0.2, 0.0)
    np.testing.assert_allclose(values, [10.0, 0.0, 0.0])


def test_the_black_76_implied_volatility_reprices_the_option():
    for put in (False, True):
        price = black_scholes_model.get_black_76(70.0, 75.0, 0.04, 0.3, 0.5, put)
        assert black_scholes_model.get_black_76_implied_volatility(
            price, 70.0, 75.0, 0.04, 0.5, put
        ) == pytest.approx(0.3, abs=1e-8)

    # Above the discounted futures price no volatility can give a call's price.
    assert np.isnan(
        black_scholes_model.get_black_76_implied_volatility(70.0, 70.0, 75.0, 0.04, 0.5)
    )


def test_black_76_at_a_zero_strike_is_the_discounted_forward():
    call = black_scholes_model.get_black_76(100.0, 0.0, 0.04, 0.2, 0.5)
    put = black_scholes_model.get_black_76(100.0, 0.0, 0.04, 0.2, 0.5, put_option=True)
    assert call == pytest.approx(100 * np.exp(-0.02), rel=1e-14)
    assert put == 0


# A grid of strikes and expirations around a price of 100, with the expiration day itself.
GRID_STRIKES, GRID_TIMES = np.meshgrid(
    np.arange(50.0, 155.0, 5.0),
    np.array([0, 1, 7, 30, 91, 182, 365, 730]) / 365,
    indexing="ij",
)


def _per_option(function, **inputs):
    """The formula evaluated once per strike and expiration of the grid."""
    return np.array(
        [
            [
                function(strike_price=strike, time_to_expiration=time, **inputs)
                for strike, time in zip(row_k, row_t, strict=True)
            ]
            for row_k, row_t in zip(GRID_STRIKES, GRID_TIMES, strict=True)
        ]
    )


@pytest.mark.parametrize("method", ["2002", "1993"])
@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize(
    ("risk_free_rate", "dividend_yield", "volatility"),
    [(0.05, 0.03, 0.25), (0.05, 0.0, 0.02), (-0.01, 0.02, 0.4), (0.03, 0.08, 0.8)],
)
def test_bjerksund_stensland_on_a_grid_equals_the_price_per_option(
    method, put_option, risk_free_rate, dividend_yield, volatility
):
    inputs = {
        "stock_price": 100.0,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
        "put_option": put_option,
        "method": method,
    }

    grid = black_scholes_model.get_bjerksund_stensland(
        strike_price=GRID_STRIKES, time_to_expiration=GRID_TIMES, **inputs
    )

    assert grid.shape == GRID_STRIKES.shape
    np.testing.assert_allclose(
        grid,
        _per_option(black_scholes_model.get_bjerksund_stensland, **inputs),
        rtol=0,
        atol=1e-10,
    )
    assert isinstance(
        black_scholes_model.get_bjerksund_stensland(
            strike_price=100.0, time_to_expiration=1.0, **inputs
        ),
        float,
    )


def test_bjerksund_stensland_keeps_its_reference_prices():
    """Haug (2007): the 2002 put of S = K = 100, r = 5%, q = 3%, σ = 25%, T = 1 is 8.8439."""
    assert black_scholes_model.get_bjerksund_stensland(
        100.0, 100.0, 0.05, 0.25, 1.0, 0.03, put_option=True
    ) == pytest.approx(8.8439, abs=1e-4)
    assert black_scholes_model.get_bjerksund_stensland(
        100.0, 100.0, 0.05, 0.25, 1.0, 0.03, put_option=True, method="1993"
    ) == pytest.approx(8.8152, abs=1e-4)


def test_the_bivariate_normal_distribution_on_arrays_matches_scipy():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(3)
    first, second = rng.normal(0, 2, 60), rng.normal(0, 2, 60)
    # Every branch: the 3, 6 and 10 point rules and the transformed integral near ±1.
    correlation = np.tile([0.0, 0.2, -0.5, 0.8, 0.93, -0.97, 0.9999], 9)[:60]

    values = black_scholes_model._bivariate_normal_cdf(first, second, correlation)

    for x, y, rho, value in zip(first, second, correlation, values, strict=True):
        expected = multivariate_normal(
            mean=[0, 0], cov=[[1, rho], [rho, 1]], allow_singular=True
        ).cdf([x, y])
        assert value == pytest.approx(expected, abs=1e-8)
        assert value == black_scholes_model._bivariate_normal_cdf(x, y, rho)


@pytest.mark.parametrize("put_option", [False, True])
def test_parity_garman_kohlhagen_and_binary_options_on_a_grid(put_option):
    common = {"stock_price": 100.0, "risk_free_rate": 0.04, "volatility": 0.3}

    for function, inputs in (
        (
            black_scholes_model.get_garman_kohlhagen,
            {**common, "foreign_risk_free_rate": 0.02, "put_option": put_option},
        ),
        (
            black_scholes_model.get_binary_option,
            {**common, "dividend_yield": 0.01, "put_option": put_option},
        ),
        (
            black_scholes_model.get_binary_option,
            {
                **common,
                "dividend_yield": 0.01,
                "put_option": put_option,
                "option_type": "asset-or-nothing",
            },
        ),
        (
            black_scholes_model.get_put_call_parity,
            {"stock_price": 100.0, "risk_free_rate": 0.04, "dividend_yield": 0.01},
        ),
    ):
        np.testing.assert_allclose(
            function(
                strike_price=GRID_STRIKES, time_to_expiration=GRID_TIMES, **inputs
            ),
            _per_option(function, **inputs),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (black_scholes_model.get_black_scholes, {"volatility": -0.2}),
        (black_scholes_model.get_black_scholes, {"stock_price": -1.0}),
        (black_scholes_model.get_black_scholes, {"strike_price": -5.0}),
        (black_scholes_model.get_black_scholes, {"time_to_expiration": -0.1}),
        (black_scholes_model.get_garman_kohlhagen, {"volatility": -0.2}),
        (black_scholes_model.get_black_76, {"volatility": -0.2}),
        (black_scholes_model.get_black_76, {"forward_price": -70.0}),
    ],
)
def test_the_black_scholes_family_refuses_negative_inputs(function, arguments):
    # A negative volatility gave a negative option value instead of an error.
    import inspect

    base = {
        "stock_price": 100.0,
        "forward_price": 100.0,
        "strike_price": 100.0,
        "risk_free_rate": 0.05,
        "foreign_risk_free_rate": 0.02,
        "volatility": 0.2,
        "time_to_expiration": 1.0,
    }
    parameters = inspect.signature(function).parameters
    call = {name: value for name, value in base.items() if name in parameters}
    with pytest.raises(ValueError, match="must be 0 or above"):
        function(**{**call, **arguments})


def test_the_lambda_of_a_worthless_option_is_nan():
    from financetoolkit.options.greeks import greeks_model

    # Far out of the money the call is worth 0.0, which gave a lambda of inf.
    assert black_scholes_model.get_black_scholes(100, 1e6, 0.05, 0.2, 1) == 0
    assert np.isnan(greeks_model.get_lambda(100, 1e6, 1, 0.05, 0.2))
    values = greeks_model.get_lambda(100, np.array([100.0, 1e6]), 1, 0.05, 0.2)
    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_bjerksund_stensland_falls_to_the_european_value_with_a_negative_rate():
    # The limitation the documentation and the controller's warning describe.
    arguments = (100.0, 100.0, -0.02, 0.01, 1.0, -0.03, True)
    european = black_scholes_model.get_black_scholes(
        100.0, 100.0, -0.02, 0.01, 1.0, -0.03, put_option=True
    )
    assert black_scholes_model.get_bjerksund_stensland(*arguments) == pytest.approx(
        european
    )
    from financetoolkit.options.pricing.american_model import get_qd_plus

    assert get_qd_plus(*arguments) == pytest.approx(0.1669, abs=1e-3)


def test_the_strike_density_refuses_prices_of_zero():
    from financetoolkit.options.greeks import greeks_model

    for stock_price, strike_price in ((0.0, 100.0), (100.0, 0.0)):
        with pytest.raises(ValueError, match="must be above 0"):
            greeks_model.get_second_order_partial_derivative(
                stock_price, strike_price, 1.0, 0.05, 0.2, 0.0
            )


def test_lambda_is_nan_when_the_option_price_underflows(monkeypatch):
    from financetoolkit.options.greeks import greeks_model

    # Near expiry and far out of the money the price can underflow to a subnormal number
    # (about 1e-320) while the delta does not, which made delta * S / price infinite.
    monkeypatch.setattr(
        greeks_model.black_scholes_model,
        "get_black_scholes",
        lambda **kwargs: np.array([10.0, 5e-324]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        greeks_model,
        "get_delta",
        lambda **kwargs: np.array([0.5, 1e-10]),  # noqa: ARG005
    )

    values = greeks_model.get_lambda(400, np.array([400.0, 480.0]), 1 / 365, 0.05, 0.2)

    assert values[0] == pytest.approx(20.0)
    assert np.isnan(values[1])
