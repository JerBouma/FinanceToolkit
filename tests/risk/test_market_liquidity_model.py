"""Market Liquidity Model Tests"""

import numpy as np
import pandas as pd

from financetoolkit.risk import market_liquidity_model

# pylint: disable=missing-function-docstring

ROLL_SPREAD_RECOVERY_TOLERANCE = 0.05


def test_get_amihud_illiquidity(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    dollar_volume = pd.Series([1e6, 2e6, 1.5e6])
    result = market_liquidity_model.get_amihud_illiquidity(returns, dollar_volume)
    expected = np.mean([0.01 / 1e6, 0.02 / 2e6, 0.03 / 1.5e6]) * 1_000_000
    assert np.isclose(result, expected)
    recorder.capture(round(result, 6))


def test_get_amihud_illiquidity_zero_volume_day(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    dollar_volume = pd.Series([1e6, 0.0, 1.5e6])
    result = market_liquidity_model.get_amihud_illiquidity(returns, dollar_volume)
    expected = np.mean([0.01 / 1e6, 0.03 / 1.5e6]) * 1_000_000
    assert np.isclose(result, expected)
    recorder.capture(round(result, 6))


def test_get_amihud_illiquidity_dataframe(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    dollar_volume = pd.Series([1e6, 2e6, 1.5e6])
    returns_df = pd.DataFrame({"AAPL": returns, "MSFT": returns})
    dollar_volume_df = pd.DataFrame({"AAPL": dollar_volume, "MSFT": dollar_volume})
    recorder.capture(
        market_liquidity_model.get_amihud_illiquidity(
            returns_df, dollar_volume_df
        ).round(6)
    )


def test_get_amihud_illiquidity_multi_period(recorder):
    idx = pd.MultiIndex.from_product([["2020Q1", "2020Q2"], range(3)])
    returns_mp = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03, 0.02, -0.01, 0.015],
            "MSFT": [0.01, -0.02, 0.03, 0.02, -0.01, 0.015],
        },
        index=idx,
    )
    dv_mp = pd.DataFrame(
        {
            "AAPL": [1e6, 2e6, 1.5e6, 1e6, 2e6, 1.5e6],
            "MSFT": [1e6, 2e6, 1.5e6, 1e6, 2e6, 1.5e6],
        },
        index=idx,
    )
    recorder.capture(
        market_liquidity_model.get_amihud_illiquidity(returns_mp, dv_mp).round(6)
    )


def test_get_amihud_illiquidity_invalid_type():
    try:
        market_liquidity_model.get_amihud_illiquidity(1, 2)  # type: ignore
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_roll_spread_bid_ask_bounce(recorder):
    # A constant efficient price with pure bid-ask bounce should recover the spread.
    rng = np.random.default_rng(1)
    n = 5000
    efficient_price = 100.0
    spread = 0.5
    side = rng.choice([-1, 1], size=n)
    observed = pd.Series(efficient_price + side * spread / 2)

    result = market_liquidity_model.get_roll_spread(observed)
    assert result["Valid Estimate"]
    assert abs(result["Roll Spread"] - spread) < ROLL_SPREAD_RECOVERY_TOLERANCE
    recorder.capture(result.round(4))


def test_get_roll_spread_positive_autocovariance(recorder):
    # A momentum path has positive lag-1 autocovariance, invalid for Roll's model.
    changes = [1, 1, 1, 1, -1, -1, -1, -1] * 10
    prices = pd.Series(np.cumsum([100, *changes]))
    result = market_liquidity_model.get_roll_spread(prices)
    assert not result["Valid Estimate"]
    assert pd.isna(result["Roll Spread"])
    recorder.capture(result.round(4))


def test_get_roll_spread_dataframe(recorder):
    rng = np.random.default_rng(1)
    n = 500
    side = rng.choice([-1, 1], size=n)
    observed = pd.Series(100.0 + side * 0.25)
    prices_df = pd.DataFrame({"AAPL": observed, "MSFT": observed})
    recorder.capture(market_liquidity_model.get_roll_spread(prices_df).round(4))


def test_get_roll_spread_too_few_observations(recorder):
    prices = pd.Series([100.0, 101.0, 100.5])
    recorder.capture(market_liquidity_model.get_roll_spread(prices))


def test_get_roll_spread_invalid_type():
    try:
        market_liquidity_model.get_roll_spread(1)  # type: ignore
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass
