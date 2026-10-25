"""Exotics Model Tests"""

# pylint: disable=missing-function-docstring

import pytest

from financetoolkit.options import exotics_model


def test_barrier_option_once_the_barrier_is_breached():
    """A breached knock-out pays its rebate and a breached knock-in is a regular option."""
    from financetoolkit.options import black_scholes_model

    arguments = {
        "stock_price": 90,
        "strike_price": 100,
        "barrier": 95,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "time_to_expiration": 1,
        "barrier_direction": "down",
    }

    assert (
        exotics_model.get_barrier_option(**arguments, knock_type="out", rebate=1) == 1
    )
    assert exotics_model.get_barrier_option(
        **arguments, knock_type="in"
    ) == pytest.approx(black_scholes_model.get_black_scholes(90, 100, 0.05, 0.2, 1))


def test_options_do_not_change_numpy_error_handling():
    """Ignoring division warnings inside a calculation does not leak into the caller."""
    import numpy as np

    before = np.geterr()
    exotics_model.get_barrier_option(100, 100, 95, 0.05, 0.2, 1)

    assert np.geterr() == before


def test_svi_fit_keeps_the_total_variance_non_negative():
    """Gatheral's a + b * sigma * sqrt(1 - rho^2) >= 0 holds, even for a smile touching zero."""
    import numpy as np

    from financetoolkit.options import svi_model

    log_moneyness = np.linspace(-0.4, 0.4, 15)
    smiles = [
        0.02 * np.abs(log_moneyness - 0.05),  # a V that touches zero total variance
        0.001 + 0.05 * log_moneyness**2 - 0.01 * log_moneyness,
        np.maximum(0.01 - 0.03 * log_moneyness, 1e-5),  # a steep one-sided skew
    ]

    for total_variance in smiles:
        parameters = svi_model.get_svi_parameters(log_moneyness, total_variance)
        minimum = parameters["a"] + parameters["b"] * parameters["sigma"] * np.sqrt(
            1 - parameters["rho"] ** 2
        )

        assert minimum >= -1e-12
        assert (
            svi_model.get_svi_total_variance(np.linspace(-3, 3, 601), **parameters)
            >= -1e-12
        ).all()


def test_risk_neutral_density_recovers_the_lognormal_density():
    """
    A flat smile gives a lognormal density. The three-point second difference is accurate
    at every grid point, the second and penultimate included.
    """
    import numpy as np
    from scipy.stats import lognorm

    from financetoolkit.options import risk_neutral_density_model

    stock_price, risk_free_rate, dividend_yield, time, volatility = (
        100.0,
        0.04,
        0.01,
        0.5,
        0.25,
    )
    forward = stock_price * np.exp((risk_free_rate - dividend_yield) * time)
    flat_smile = {
        "a": volatility**2 * time,
        "b": 1e-12,
        "rho": 0.0,
        "m": 0.0,
        "sigma": 0.1,
    }

    for number_of_strikes in (50, 200):
        density = risk_neutral_density_model.get_risk_neutral_density(
            stock_price,
            forward,
            time,
            risk_free_rate,
            dividend_yield,
            flat_smile,
            0.5,
            number_of_strikes,
        )
        expected = lognorm.pdf(
            density.index,
            s=volatility * np.sqrt(time),
            scale=forward * np.exp(-0.5 * volatility**2 * time),
        )

        assert len(density) == number_of_strikes - 2
        np.testing.assert_allclose(
            density.to_numpy(), expected, rtol=0.01, atol=0.005 * expected.max()
        )

        if number_of_strikes == 200:  # noqa: PLR2004
            # The points next to the edges carried a bias of about 20% with a 2h stencil.
            relative_error = density.to_numpy() / expected - 1
            assert abs(relative_error[0]) < 0.01  # noqa: PLR2004
            assert abs(relative_error[-1]) < 0.01  # noqa: PLR2004
