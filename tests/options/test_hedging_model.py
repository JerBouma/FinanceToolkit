"""Delta Hedging Model Tests"""

import numpy as np
import pytest

from financetoolkit.options.greeks import greeks_model
from financetoolkit.options.market import hedging_model

STOCK_PRICE, STRIKE, TIME, RATE, VOLATILITY = 100.0, 100.0, 0.25, 0.05, 0.2


def simulate(**kwargs):
    arguments = {
        "stock_price": STOCK_PRICE,
        "strike_price": STRIKE,
        "time_to_expiration": TIME,
        "risk_free_rate": RATE,
        "volatility": VOLATILITY,
        "simulations": 20_000,
        "seed": 1,
    }
    return hedging_model.get_delta_hedging_simulation(**{**arguments, **kwargs})


def test_the_hedging_error_shrinks_with_the_square_root_of_the_rebalances():
    # Boyle and Emanuel (1980), Derman and Kamal (1999): the standard deviation is about
    # sqrt(π / 4) vega σ / sqrt(N), and the mean stays near zero.
    deviations = []
    for steps in (16, 64, 256):
        profit = simulate(steps=steps, simulations=10_000)["Profit and Loss"]
        approximation = hedging_model.get_hedging_error_approximation(
            STOCK_PRICE, STRIKE, TIME, RATE, VOLATILITY, steps
        )
        # Discrete hedging leaves a small bias of order Δt next to the noise of the mean.
        assert abs(profit.mean()) < 0.05 * profit.std()
        assert profit.std() == pytest.approx(approximation, rel=0.05)
        deviations.append(profit.std())
    assert deviations[0] / deviations[1] == pytest.approx(2, rel=0.05)
    assert deviations[1] / deviations[2] == pytest.approx(2, rel=0.05)


def test_the_simulation_matches_financepy():
    # FinancePy 1.1.2 simulate_hedge_paths (40,000 paths, 63 steps, seed 7) reports the
    # profit and loss at expiration: mean and standard deviation per case below.
    growth = np.exp(RATE * TIME)
    cases = {
        (False, 100.0, 0.2): (-0.000477, 0.431108),
        (True, 95.0, 0.3): (-1.701756, 1.079797),
    }
    for (put_option, strike, volatility), (mean, deviation) in cases.items():
        profit = (
            growth
            * simulate(
                strike_price=strike,
                volatility=volatility,
                implied_volatility=0.2,
                put_option=put_option,
                steps=63,
                simulations=40_000,
                seed=7,
            )["Profit and Loss"]
        )
        error = deviation / np.sqrt(40_000)
        assert abs(profit.mean() - mean) < 4 * error
        assert profit.std() == pytest.approx(deviation, rel=0.02)


def test_selling_at_the_leland_volatility_covers_the_rebalancing_costs():
    # Leland (1985): with a round-trip cost of 2k the option priced at
    # σ² (1 + sqrt(2 / π) 2k / (σ sqrt(Δt))) pays for the costs of rebalancing; what remains
    # is the cost of buying the first hedge and of the trade at expiration.
    cost, steps = 0.002, 63
    leland = VOLATILITY * np.sqrt(
        1 + np.sqrt(2 / np.pi) * 2 * cost / (VOLATILITY * np.sqrt(TIME / steps))
    )
    result = simulate(
        implied_volatility=leland, transaction_cost=cost, steps=steps, seed=2
    )
    unhedged = simulate(transaction_cost=cost, steps=steps, seed=2)
    first_delta = greeks_model.get_delta(STOCK_PRICE, STRIKE, TIME, RATE, leland)
    first_trade = first_delta * STOCK_PRICE * cost
    assert unhedged["Profit and Loss"].mean() < -0.5
    assert result["Profit and Loss"].mean() == pytest.approx(-first_trade, abs=0.03)


def test_a_higher_realized_volatility_costs_the_seller():
    lower = simulate(volatility=0.15, implied_volatility=0.2, steps=252)
    higher = simulate(volatility=0.25, implied_volatility=0.2, steps=252)
    assert lower["Profit and Loss"].mean() > 0 > higher["Profit and Loss"].mean()


def test_a_path_without_moves_earns_the_time_value():
    # A stock that stays at the strike ends out of the money with the hedge sold off along
    # the way, so the seller keeps about the option value less the cost of carry.
    prices = np.full(64, STOCK_PRICE)
    result = hedging_model.get_delta_hedging_profit_and_loss(
        prices, STRIKE, TIME, 0.0, VOLATILITY
    ).iloc[0]
    assert result["Payoff"] == 0
    assert result["Profit and Loss"] == pytest.approx(result["Option Value"])


def test_dividends_are_paid_on_the_shares_held():
    prices = np.linspace(100.0, 110.0, 11)
    paid = np.zeros(11)
    paid[5] = 1.0
    without = hedging_model.get_delta_hedging_profit_and_loss(
        prices, STRIKE, 0.1, 0.0, VOLATILITY, dividends=np.zeros(11)
    ).iloc[0]
    with_dividend = hedging_model.get_delta_hedging_profit_and_loss(
        prices, STRIKE, 0.1, 0.0, VOLATILITY, dividends=paid
    ).iloc[0]
    assert 0 < with_dividend["Profit and Loss"] - without["Profit and Loss"] < 1


def test_the_simulation_is_seeded_and_checks_its_inputs():
    first = simulate(simulations=100, steps=10)
    second = simulate(simulations=100, steps=10)
    assert first.equals(second)
    with pytest.raises(ValueError, match="positive"):
        hedging_model.get_delta_hedging_profit_and_loss(
            np.array([100.0, -1.0]), STRIKE, TIME, RATE, VOLATILITY
        )
    with pytest.raises(ValueError, match="rebalancing interval"):
        hedging_model.get_delta_hedging_profit_and_loss(
            np.array([100.0, 101.0]),
            STRIKE,
            TIME,
            RATE,
            VOLATILITY,
            rebalancing_interval=0,
        )
