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


def test_the_strike_grid_covers_the_whole_range_in_positive_steps():
    import pandas as pd

    from financetoolkit.options import helpers

    def strikes(price, step, strike_price_range):
        return helpers.define_strike_prices(
            ["A"], pd.Series({"A": price}, name="date"), step, strike_price_range
        )["A"]

    # Both ends of the range are included, rounded outwards to the step.
    assert strikes(100.0, 5, 0.25) == list(range(75, 130, 5))
    assert strikes(129.93, 5, 0.25)[0] == 95
    assert strikes(129.93, 5, 0.25)[-1] == 165
    # A price below the step still gets a positive strike, never 0.
    assert strikes(3.0, 5, 0.25) == [5]
    assert strikes(0.8, 1, 0.25) == [1]
    # Fractional steps.
    assert strikes(100.0, 2.5, 0.1) == [90.0 + 2.5 * step for step in range(9)]
    assert strikes(100.0, 0.1, 0.001) == [99.9, 100.0, 100.1]
    # Without a range the nearest strike.
    assert strikes(101.0, 5, 0) == [100]


def _tree_price(method, timesteps=101, put=False, american=False, **inputs):
    values = {
        "stock_price": 100.0,
        "strike_price": 95.0,
        "years": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.3,
        "dividend_yield": 0.02,
    } | inputs
    tree = binomial_trees_model.get_option_payoffs(
        timesteps=timesteps,
        put_option=put,
        american_option=american,
        method=method,
        **values,
    )
    return float(tree.iloc[0, 0])


@pytest.mark.parametrize(
    ("method", "quantlib"),
    [
        # QuantLib 1.44 BinomialJR, BinomialTian, BinomialTrigeorgis and BinomialLR
        # VanillaEngine, 101 steps, S = 100, K = 95, r = 5%, q = 2%, σ = 30%, T = 1, call.
        ("jarrow_rudd", 15.456778992325),
        ("tian", 15.480407654184),
        ("trigeorgis", 15.485910886585),
        ("leisen_reimer", 15.464163550654),
    ],
    ids=str,
)
def test_the_tree_variants_match_quantlib(method, quantlib):
    assert _tree_price(method) == pytest.approx(quantlib, abs=1e-8)


def test_the_exact_inversion_tree_prices_a_european_option_at_black_scholes():
    # The exact inverse of the binomial distribution puts N(d2) and N(d1) on the tree, so
    # the European price is the Black-Scholes price at every odd number of steps. QuantLib's
    # BinomialJ4VanillaEngine, Joshi's fourth order expansion of that inverse, gives
    # 15.464211426910 at 101 steps against Black-Scholes 15.464211545469.
    black_scholes = 15.46421154546934
    for timesteps in (5, 25, 101):
        assert _tree_price("exact_inversion", timesteps) == pytest.approx(
            black_scholes, abs=1e-10
        )
    assert _tree_price("exact_inversion") == pytest.approx(15.464211426910, abs=2e-7)


def test_the_cox_ross_rubinstein_tree_is_the_default():
    default = binomial_trees_model.get_option_payoffs(100.0, 95.0, 1.0, 10, 0.05, 0.3)
    explicit = binomial_trees_model.get_option_payoffs(
        100.0, 95.0, 1.0, 10, 0.05, 0.3, method="cox_ross_rubinstein"
    )
    assert default.equals(explicit)


def test_the_strike_centred_trees_need_an_odd_number_of_steps():
    for method in ("leisen_reimer", "exact_inversion"):
        with pytest.raises(ValueError, match="odd number of time steps"):
            _tree_price(method, timesteps=10)

    with pytest.raises(ValueError, match="method must be one of"):
        _tree_price("eqp")


def test_a_tree_without_a_valid_probability_is_refused():
    # At a volatility this small e^(σ²Δt) rounds to 1, so Tian's up and down movements are
    # equal and the probability is 0 / 0.
    with pytest.raises(ValueError, match="does not give a valid tree"):
        _tree_price("tian", timesteps=11, volatility=1e-10)
