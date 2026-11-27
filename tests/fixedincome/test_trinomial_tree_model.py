"""Short-Rate Tree Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.fixedincome.short_rates import (
    short_rate_model,
    trinomial_tree_model,
)

ZERO_CURVE = pd.Series([0.036, 0.035, 0.037, 0.041, 0.047], index=[1, 2, 5, 10, 30])
MODELS = [
    ("hull_white", 0.05, 0.01),
    ("black_karasinski", 0.05, 0.2),
    ("black_derman_toy", 0.0, 0.2),
    ("ho_lee", 0.0, 0.01),
]


@pytest.mark.parametrize("model, speed, volatility", MODELS)
def test_tree_reprices_the_zero_curve(model, speed, volatility):
    # Uneven steps: the dates 0.3 and 0.35 split the grid.
    tree = trinomial_tree_model.build_short_rate_tree(
        ZERO_CURVE, [0.3, 0.35, 4.0], speed, volatility, model, steps_per_year=12
    )
    model_discount = np.array([arrow.sum() for arrow in tree["Arrow-Debreu"]])

    assert model_discount == pytest.approx(
        discount_factors(ZERO_CURVE, tree["Times"]), abs=1e-14
    )
    for probabilities in tree["Probabilities"]:
        assert probabilities.min() > 0
        assert probabilities.sum(axis=1) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("model, speed, volatility", MODELS)
def test_tree_matches_the_mean_and_variance_of_every_step(model, speed, volatility):
    tree = trinomial_tree_model.build_short_rate_tree(
        ZERO_CURVE, [2.0], speed, volatility, model, steps_per_year=10
    )
    dt = 0.1
    for step in (3, 15):
        nodes, next_nodes = tree["Nodes"][step], tree["Nodes"][step + 1]
        spacing = (
            np.sqrt(3)
            * volatility
            * np.sqrt(-np.expm1(-2 * speed * dt) / (2 * speed) if speed else dt)
        )
        states, next_states = nodes * spacing, next_nodes * spacing
        branches = tree["Branches"][step]
        probabilities = tree["Probabilities"][step]
        targets = np.column_stack(
            [next_states[branches + offset] for offset in (-1, 0, 1)]
        )
        mean = np.sum(probabilities * targets, axis=1)
        variance = np.sum(probabilities * targets**2, axis=1) - mean**2

        assert mean == pytest.approx(states * np.exp(-speed * dt), abs=1e-14)
        assert variance == pytest.approx(spacing**2 / 3, rel=1e-9)


def test_lognormal_tree_needs_positive_forward_rates():
    inverted = pd.Series([0.01, -0.005], index=[1, 5])
    with pytest.raises(ValueError, match="positive forward rate"):
        trinomial_tree_model.build_short_rate_tree(
            inverted, [5.0], 0.1, 0.2, "black_karasinski"
        )
    # A normal model fits the same curve.
    trinomial_tree_model.build_short_rate_tree(inverted, [5.0], 0.1, 0.01)


def test_tree_rejects_bad_inputs():
    with pytest.raises(ValueError, match="model must be one of"):
        trinomial_tree_model.build_short_rate_tree(ZERO_CURVE, [1.0], 0.1, 0.01, "cir")
    with pytest.raises(ValueError, match="volatility"):
        trinomial_tree_model.build_short_rate_tree(ZERO_CURVE, [1.0], 0.1, 0.0)
    with pytest.raises(ValueError, match="too large"):
        trinomial_tree_model.build_short_rate_tree(
            ZERO_CURVE, [60.0], 0.1, 0.01, steps_per_year=100
        )


def test_short_rate_tree_table():
    rates = trinomial_tree_model.get_short_rate_tree(ZERO_CURVE, 1, 0.1, 0.01)

    assert rates.shape == (4, 7)
    # The rate of the first step is the continuously compounded forward rate of the step.
    assert rates.loc[0, 0] == pytest.approx(-np.log(1.036**-0.25) / 0.25, abs=1e-14)


# QuantLib 1.44 (`CallableFixedRateBond` with the `TreeCallableFixedRateBondEngine` and
# `Swaption` with the `TreeSwaptionEngine`, 1,000 and 1,100 steps, on a `DiscountCurve` of this
# curve's monthly discount factors with a 30/360 day count, so that the trees have the same
# dates): a 10-year 5% semi-annual bond callable or puttable at 100 on the coupon dates from
# year 2, and a co-terminal 1-year into 10-year swaption at 4.5% with annual fixed payments.
# With 100 steps a year the trees are the same, so the prices agree to rounding for
# Hull-White and within QuantLib's root-finding accuracy for Black-Karasinski.
@pytest.mark.parametrize(
    "model, volatility, call_price, put_price, quantlib, tolerance",
    [
        ("hull_white", 0.01, 100.0, None, 101.23475074544731, 1e-9),
        ("hull_white", 0.01, None, 100.0, 110.8512027517043, 1e-9),
        ("black_karasinski", 0.2, 100.0, None, 101.64947505420766, 1e-5),
    ],
)
def test_callable_and_puttable_bonds_match_quantlib(
    model, volatility, call_price, put_price, quantlib, tolerance
):
    result = trinomial_tree_model.get_callable_bond_price(
        0.05,
        10,
        ZERO_CURVE,
        0.05,
        volatility,
        model,
        call_price=call_price,
        years_to_first_call=2,
        put_price=put_price,
        years_to_first_put=2,
        steps_per_year=100,
    )

    assert result["Price"] == pytest.approx(quantlib, abs=tolerance)
    # The straight bond is the value of its payments on the curve.
    payments = np.arange(0.5, 10.001, 0.5)
    straight = 2.5 * discount_factors(ZERO_CURVE, payments).sum() + 100 * (
        discount_factors(ZERO_CURVE, 10.0)
    )
    assert result["Straight Bond Price"] == pytest.approx(straight, abs=1e-10)


@pytest.mark.parametrize(
    "model, volatility, swaption_type, quantlib, tolerance",
    [
        ("hull_white", 0.01, "payer", 0.04352896620104271, 1e-12),
        ("black_karasinski", 0.2, "receiver", 0.04820034330662937, 1e-7),
    ],
)
def test_bermudan_swaption_matches_quantlib(
    model, volatility, swaption_type, quantlib, tolerance
):
    price = trinomial_tree_model.get_bermudan_swaption(
        1,
        10,
        0.045,
        ZERO_CURVE,
        0.05,
        volatility,
        model,
        swaption_type,
        steps_per_year=100,
    )

    assert price == pytest.approx(quantlib, abs=tolerance)


def test_european_swaption_on_the_tree_matches_the_closed_form():
    # Jamshidian's decomposition on Hull-White; the tree converges to it as the steps
    # shrink (2e-5 off at 100 steps a year, QuantLib's tree as well).
    closed_form = short_rate_model.get_swaption(
        1, 5, 0.04, 0.05, 0.01, zero_curve=ZERO_CURVE, model="hull_white"
    )
    tree = trinomial_tree_model.get_bermudan_swaption(
        1, 5, 0.04, ZERO_CURVE, 0.05, 0.01, exercise="european", steps_per_year=200
    )
    bermudan = trinomial_tree_model.get_bermudan_swaption(
        1, 5, 0.04, ZERO_CURVE, 0.05, 0.01
    )

    assert tree == pytest.approx(closed_form, rel=2e-3)
    assert bermudan > tree


def test_black_derman_toy_puttable_bond_matches_financepy():
    # FinancePy 1.1.2's `BDTTree.callable_puttable_bond_tree` (1,000 binomial steps) gives
    # 111.17404 for the bond puttable at 100 from year 2. Its callable bond value (40.74, below
    # the 100 the issuer pays on a call) is not a reference.
    price = trinomial_tree_model.get_callable_bond_price(
        0.05,
        10,
        ZERO_CURVE,
        0.0,
        0.2,
        "black_derman_toy",
        call_price=None,
        put_price=100.0,
        years_to_first_put=2,
        steps_per_year=100,
    )["Price"]

    assert price == pytest.approx(111.17404289514172, abs=1e-4)


@pytest.mark.parametrize("model, speed, volatility", MODELS)
def test_bond_option_put_call_parity_and_early_exercise(model, speed, volatility):
    arguments = (2, 100, 0.04, 10, ZERO_CURVE, speed, volatility, model)
    call = trinomial_tree_model.get_bond_option(*arguments, "call", "european")
    put = trinomial_tree_model.get_bond_option(*arguments, "put", "european")
    american_call = trinomial_tree_model.get_bond_option(*arguments, "call")
    american_put = trinomial_tree_model.get_bond_option(*arguments, "put")

    # The tree reprices every zero-coupon bond, so the European call less the put is the
    # forward value of the bond less the strike, both discounted, to rounding.
    payments = np.arange(2.5, 10.001, 0.5)
    forward = (
        2 * discount_factors(ZERO_CURVE, payments).sum()
        + 100 * discount_factors(ZERO_CURVE, 10.0)
        - 100 * discount_factors(ZERO_CURVE, 2.0)
    )
    assert call - put == pytest.approx(forward, abs=1e-10)
    assert american_call >= call
    assert american_put >= put


def test_bond_option_matches_financepy():
    # FinancePy 1.1.2's `HWTree.bond_option` (1,000 steps): 2.36277 and 3.44679 for the
    # European call and put, 3.63929 for the American put, struck at 100 at 2 years on a
    # 4% semi-annual 10-year bond. FinancePy places the coupons on its nearest step, so the
    # prices agree within half a cent. Its American call (3.243) exercises neither on the
    # clean price (2.847 here) nor on the dirty price (3.754).
    arguments = (2, 100, 0.04, 10, ZERO_CURVE, 0.05, 0.01)
    european_call = trinomial_tree_model.get_bond_option(
        *arguments, option_type="call", exercise="european", steps_per_year=100
    )
    american_put = trinomial_tree_model.get_bond_option(
        *arguments, option_type="put", steps_per_year=100
    )

    assert european_call == pytest.approx(2.362772829331799, abs=5e-3)
    assert american_put == pytest.approx(3.639290952151691, abs=5e-3)


def test_tree_prices_are_stable_under_tiny_perturbations():
    base = trinomial_tree_model.get_callable_bond_price(
        0.05, 10, ZERO_CURVE, 0.05, 0.2, "black_karasinski", years_to_first_call=2
    )["Price"]
    perturbed = trinomial_tree_model.get_callable_bond_price(
        0.05 * (1 + 1e-12),
        10 + 1e-12,
        ZERO_CURVE * (1 + 1e-12),
        0.05 * (1 + 1e-12),
        0.2 * (1 + 1e-12),
        "black_karasinski",
        years_to_first_call=2 + 1e-12,
    )["Price"]

    assert perturbed == pytest.approx(base, abs=1e-8)


@pytest.mark.parametrize("model, speed, volatility", MODELS)
def test_option_adjusted_spread_by_model(model, speed, volatility):
    # A bond priced on the model's own tree has an OAS of zero, for every model.
    maturities, zero_rates = ZERO_CURVE.index.to_numpy(float), ZERO_CURVE.to_numpy()
    price = trinomial_tree_model.get_callable_bond_price(
        0.05,
        9.75,
        ZERO_CURVE,
        speed,
        volatility,
        model,
        years_to_first_call=2,
        steps_per_year=40,
    )["Price"]
    result = short_rate_model.get_option_adjusted_spread(
        price,
        0.05,
        9.75,
        maturities,
        zero_rates,
        volatility,
        call_price=100.0,
        years_to_first_call=2,
        model=model,
        speed=speed,
    )

    assert result["Option-Adjusted Spread"] == pytest.approx(0, abs=1e-9)
    assert result["Option Value"] > 0
    assert result["Effective Convexity"] < 0
