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
