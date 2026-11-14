"""Binomial Trees Model Tests"""

import pytest

from financetoolkit.options.pricing import binomial_trees_model


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


def test_binomial_tree_refuses_too_many_time_steps():
    with pytest.raises(ValueError, match="too large"):
        binomial_trees_model.calculate_stock_prices(
            100, 1.01, 0.99, binomial_trees_model.MAX_TIMESTEPS + 1
        )


def test_listing_every_path_is_limited_to_a_few_time_steps():
    """Every path of the tree is 2 ** timesteps of them, so only short trees are listed."""
    paths = binomial_trees_model.calculate_stock_prices(
        100, 1.1, 1 / 1.1, 3, show_unique_combinations=False
    )
    assert len(paths) == 2**3

    with pytest.raises(ValueError, match="show_unique_combinations=True"):
        binomial_trees_model.calculate_stock_prices(
            100, 1.1, 1 / 1.1, 40, show_unique_combinations=False
        )


def test_a_strike_grid_too_large_for_memory_is_refused():
    import pandas as pd

    from financetoolkit.options import helpers

    prices = pd.Series({"BRK-A": 700_000.0, "AAPL": 230.0})

    assert len(helpers.define_strike_prices(["AAPL"], prices, 5, 0.25)["AAPL"]) > 0
    with pytest.raises(ValueError, match="larger strike_step_size"):
        helpers.define_strike_prices(["BRK-A"], prices, 1, 0.25)
