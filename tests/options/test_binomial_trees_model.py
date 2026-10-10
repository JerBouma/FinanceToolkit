"""Binomial Trees Model Tests"""

# pylint: disable=missing-function-docstring

import pytest

from financetoolkit.options import binomial_trees_model


def test_a_risk_neutral_probability_outside_zero_and_one_is_refused():
    """A 1% volatility over a one year step cannot carry a 10% rate."""
    up, down = binomial_trees_model.calculate_up_and_down_movements(0.01, 1)

    with pytest.raises(ValueError, match="not between 0 and 1"):
        binomial_trees_model.calculate_risk_neutral_probability(0.10, 0, 1, up, down)


def test_a_regular_risk_neutral_probability():
    up, down = binomial_trees_model.calculate_up_and_down_movements(0.2, 1 / 12)

    assert (
        0
        < binomial_trees_model.calculate_risk_neutral_probability(
            0.05, 0, 1 / 12, up, down
        )
        < 1
    )
