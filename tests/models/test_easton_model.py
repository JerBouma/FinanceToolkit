"""Easton Implied Cost of Capital Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models.valuation import easton_model


def test_implied_cost_of_capital_solves_eastons_valuation_equation():
    rng = np.random.default_rng(0)
    eps_one = rng.uniform(0.5, 10, 500)
    eps_two = eps_one + rng.uniform(0.01, 3, 500)
    price = rng.uniform(5, 500, 500)
    dividends = rng.uniform(0, 0.8, 500) * eps_one

    rate = easton_model.get_implied_cost_of_capital(
        pd.Series(eps_one), pd.Series(eps_two), pd.Series(price), pd.Series(dividends)
    ).to_numpy()

    # P0 = EPS1 / r + (EPS2 + r * DPS1 - (1 + r) * EPS1) / r^2
    abnormal_growth = eps_two + rate * dividends - (1 + rate) * eps_one
    implied_price = eps_one / rate + abnormal_growth / rate**2

    assert np.max(np.abs(implied_price / price - 1)) < 1e-12


def test_implied_cost_of_capital_peg_and_mpeg_by_hand():
    mpeg = easton_model.get_implied_cost_of_capital(5.0, 6.0, 100.0, 2.0)
    peg = easton_model.get_implied_cost_of_capital(5.0, 6.0, 100.0, 2.0, modified=False)

    assert mpeg == pytest.approx(0.01 + np.sqrt(0.01**2 + 0.01))
    assert peg == pytest.approx(0.1)


@pytest.mark.parametrize("eps_two, price", [(5.0, 100.0), (4.0, 100.0), (6.0, 0.0)])
def test_implied_cost_of_capital_is_nan_without_a_positive_root(eps_two, price):
    assert np.isnan(easton_model.get_implied_cost_of_capital(5.0, eps_two, price, 2.0))
    assert np.isnan(
        easton_model.get_implied_cost_of_capital(
            pd.Series([5.0]), pd.Series([eps_two]), pd.Series([price]), pd.Series([2.0])
        ).iloc[0]
    )


def test_implied_cost_of_capital_takes_scalar_earnings_with_series_prices():
    result = easton_model.get_implied_cost_of_capital(
        5.0, 6.0, pd.Series([100.0, 0.0]), pd.Series([2.0, 2.0])
    )

    assert result.iloc[0] == pytest.approx(0.01 + np.sqrt(0.01**2 + 0.01))
    assert np.isnan(result.iloc[1])
