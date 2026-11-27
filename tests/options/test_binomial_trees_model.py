"""Binomial Trees Model Tests"""

import numpy as np
import pytest
from scipy.special import betainc, ndtr

from financetoolkit.options.pricing import binomial_trees_model, black_scholes_model


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
    """Every path of the tree is 2 ** time_steps of them, so only short trees are listed."""
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


def _tree_price(method, time_steps=101, put=False, american=False, **inputs):
    values = {
        "stock_price": 100.0,
        "strike_price": 95.0,
        "years": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.3,
        "dividend_yield": 0.02,
    } | inputs
    tree = binomial_trees_model.get_option_payoffs(
        time_steps=time_steps,
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
    for time_steps in (5, 25, 101):
        assert _tree_price("exact_inversion", time_steps) == pytest.approx(
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
            _tree_price(method, time_steps=10)

    with pytest.raises(ValueError, match="method must be one of"):
        _tree_price("eqp")


def test_a_tree_without_a_valid_probability_is_refused():
    # At a volatility this small e^(σ²Δt) rounds to 1, so Tian's up and down movements are
    # equal and the probability is 0 / 0.
    with pytest.raises(ValueError, match="does not give a valid tree"):
        _tree_price("tian", time_steps=11, volatility=1e-10)


@pytest.mark.parametrize("method", ["leisen_reimer", "exact_inversion"])
@pytest.mark.parametrize(
    "strike, years, volatility, put_option",
    [
        (50, 0.004, 0.3, False),
        (150, 0.004, 0.3, True),
        (150, 0.004, 0.3, False),
        (60, 0.004, 0.01, False),
        (90, 1.0, 0.01, True),
    ],
)
def test_a_strike_with_a_certain_outcome_gets_its_value(
    method, strike, years, volatility, put_option
):
    # N(d2) rounds to 1 (or 0) here, which put p at 1 and gave the tree no down movement.
    value = binomial_trees_model.get_option_payoffs(
        stock_price=100,
        strike_price=strike,
        years=years,
        time_steps=11,
        risk_free_rate=0.04,
        volatility=volatility,
        put_option=put_option,
        method=method,
    ).iloc[0, 0]
    expected = black_scholes_model.get_black_scholes(
        100, strike, 0.04, volatility, years, put_option=put_option
    )
    assert value == pytest.approx(expected, abs=1e-9)


def test_the_strike_centred_trees_need_a_positive_strike():
    with pytest.raises(ValueError, match="must be positive"):
        binomial_trees_model.calculate_tree_movements(
            "exact_inversion", 100, 0, 0.04, 0, 0.2, 1, 11
        )


def test_the_exact_inversion_of_a_tail_far_below_double_precision():
    # N(—25) is about 3e-138, where scipy's inverse of the incomplete beta function with three
    # up moves gives NaN.
    probability, complement = binomial_trees_model._exact_binomial_inversion(25, 5)
    assert probability == 1
    assert betainc(3, 3, complement) == pytest.approx(ndtr(-25), rel=1e-12)


def _backward_induction(
    stock_price,
    strike_price,
    years,
    time_steps,
    rate,
    up,
    down,
    probability,
    put,
    american,
):
    """A plain scalar backward induction of a recombining tree, node by node."""
    discount = np.exp(-rate * years / time_steps)
    sign = -1 if put else 1
    values = [
        max(
            sign
            * (stock_price * down**downs * up ** (time_steps - downs) - strike_price),
            0,
        )
        for downs in range(time_steps + 1)
    ]
    for step in range(time_steps - 1, -1, -1):
        values = [
            discount
            * (probability * values[downs] + (1 - probability) * values[downs + 1])
            for downs in range(step + 1)
        ]
        if american:
            values = [
                max(
                    value,
                    sign
                    * (stock_price * down**downs * up ** (step - downs) - strike_price),
                )
                for downs, value in enumerate(values)
            ]
    return values[0]


@pytest.mark.parametrize("method", ["cox_ross_rubinstein", "tian", "leisen_reimer"])
@pytest.mark.parametrize("put_option", [False, True])
@pytest.mark.parametrize("american_option", [False, True])
def test_the_trees_of_all_strikes_at_once_match_a_scalar_backward_induction(
    method, put_option, american_option
):
    strikes = np.arange(80.0, 125.0, 5.0)
    time_steps = 21
    trees, ups, downs, probabilities = binomial_trees_model.get_option_payoff_trees(
        stock_price=100.0,
        strike_prices=strikes,
        years=0.75,
        time_steps=time_steps,
        risk_free_rate=0.04,
        volatility=0.3,
        dividend_yield=0.01,
        put_option=put_option,
        american_option=american_option,
        method=method,
    )

    assert trees.shape == (len(strikes), time_steps + 1, time_steps + 1)
    # Node k exists from step k on.
    assert np.isnan(trees[:, 1, 0]).all()
    assert not np.isnan(trees[:, 1, 1]).any()
    for position, strike in enumerate(strikes):
        assert trees[position, 0, 0] == pytest.approx(
            _backward_induction(
                100.0,
                strike,
                0.75,
                time_steps,
                0.04,
                ups[position],
                downs[position],
                probabilities[position],
                put_option,
                american_option,
            ),
            abs=1e-10,
        )
        single = binomial_trees_model.get_option_payoffs(
            100.0,
            float(strike),
            0.75,
            time_steps,
            0.04,
            0.3,
            0.01,
            put_option,
            american_option,
            method=method,
        )
        np.testing.assert_array_equal(single.to_numpy(), trees[position])


def test_a_grid_of_more_option_values_than_the_cap_is_refused(monkeypatch):
    from financetoolkit.options import helpers

    prices = __import__("pandas").Series({"AAPL": 200.0})
    monkeypatch.setattr(helpers, "MAX_OPTION_VALUES", 1_000)

    # 41 strikes of 200 ± 10% in steps of 1, with 24 expirations each.
    assert (
        len(helpers.define_strike_prices(["AAPL"], prices, 1, 0.1, 24)["AAPL"]) == 41
    )  # noqa: PLR2004
    with pytest.raises(
        ValueError, match="41 strike prices with 25 values each, 1,025 option values"
    ):
        helpers.define_strike_prices(["AAPL"], prices, 1, 0.1, values_per_strike=25)


@pytest.mark.parametrize(
    ("volatility", "time_delta"),
    [(0.2, -0.1), (0.2, 0.0), (0.2, np.nan), (0.0, 0.1), (-0.2, 0.1), (np.nan, 0.1)],
)
def test_the_movements_need_a_positive_volatility_and_step(volatility, time_delta):
    with pytest.raises(ValueError, match="finite number above 0"):
        binomial_trees_model.calculate_up_and_down_movements(volatility, time_delta)


@pytest.mark.timeout(10)
def test_listing_every_path_ends_whatever_the_movements():
    # NaN movements (from a negative time step) never equal themselves, on which the
    # listing of every path used to loop without end.
    paths = binomial_trees_model.calculate_stock_prices(
        100, np.nan, np.nan, 3, show_unique_combinations=False
    )
    assert paths.shape == (8, 4)
    assert list(paths.index[[0, -1]]) == ["UUU", "DDD"]

    paths = binomial_trees_model.calculate_stock_prices(
        100, 1.1, 0.9, 2, show_unique_combinations=False
    )
    assert paths.loc["UD"].tolist() == pytest.approx([100, 110, 99])
