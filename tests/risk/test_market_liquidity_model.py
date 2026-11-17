"""Market Liquidity Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk.liquidity import market_liquidity_model

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


def test_get_roll_spread_relative_spread_is_a_decimal():
    rng = np.random.default_rng(0)
    efficient_price = 100 + np.cumsum(rng.normal(0, 0.05, 3000))
    bounce = rng.choice([-0.25, 0.25], size=3000)
    prices = pd.Series(efficient_price + bounce)

    result = market_liquidity_model.get_roll_spread(prices)

    assert "Roll Spread (%)" not in result.index
    assert np.isclose(
        result["Relative Roll Spread"], result["Roll Spread"] / prices.mean()
    )
    assert result["Relative Roll Spread"] < 0.01


def corwin_schultz_two_day_spread(first, second):
    """The two-day spread written out per the paper, with the overnight adjustment."""
    high_1, low_1, close_1 = first
    high_2, low_2, _ = second

    if low_2 > close_1:
        high_2, low_2 = high_2 - (low_2 - close_1), close_1
    elif high_2 < close_1:
        high_2, low_2 = close_1, low_2 + (close_1 - high_2)

    beta = np.log(high_1 / low_1) ** 2 + np.log(high_2 / low_2) ** 2
    gamma = np.log(max(high_1, high_2) / min(low_1, low_2)) ** 2
    denominator = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denominator - np.sqrt(
        gamma / denominator
    )

    return max(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0)


def corwin_schultz_prices():
    dates = pd.PeriodIndex(
        ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"], freq="D"
    )
    # 30 January opens above the previous close (a gap up) and 1 and 2 February close at
    # their High in a steady rise, which gives a negative two-day spread.
    return pd.DataFrame(
        {
            "High": [101.0, 104.0, 103.5, 101.0, 102.0],
            "Low": [99.0, 101.5, 102.0, 100.0, 101.0],
            "Close": [100.5, 103.0, 102.5, 101.0, 102.0],
        },
        index=dates,
    )


def test_get_corwin_schultz_spread_matches_the_paper(recorder):
    prices = corwin_schultz_prices()
    rows = list(prices.itertuples(index=False))

    result = market_liquidity_model.get_corwin_schultz_spread(
        prices["High"], prices["Low"], prices["Close"], "monthly"
    )

    # The pair 31 January - 1 February spans two months and is left out.
    january = np.mean(
        [
            corwin_schultz_two_day_spread(rows[0], rows[1]),
            corwin_schultz_two_day_spread(rows[1], rows[2]),
        ]
    )
    february = corwin_schultz_two_day_spread(rows[3], rows[4])

    assert np.isclose(result[pd.Period("2024-01", "M")], january)
    assert february == 0
    assert result[pd.Period("2024-02", "M")] == 0
    recorder.capture(result.round(6))


def test_get_corwin_schultz_spread_recovers_the_spread():
    # Intraday random walks quoted with a 1% spread: the High is at the ask, the Low at the
    # bid and the Close at either, with overnight jumps on some days.
    rng = np.random.default_rng(7)
    dates = pd.period_range("2021-01-01", periods=250, freq="D")
    spread = 0.01
    price = 100.0
    high, low, close = [], [], []

    for _ in dates:
        path = price * np.exp(np.cumsum(rng.normal(0, 0.01 / np.sqrt(390), 390)))
        if rng.random() < 0.3:  # noqa: PLR2004
            path *= np.exp(rng.normal(0, 0.01))
        high.append(path.max() * (1 + spread / 2))
        low.append(path.min() * (1 - spread / 2))
        close.append(path[-1] * (1 + rng.choice([-1, 1]) * spread / 2))
        price = path[-1]

    result = market_liquidity_model.get_corwin_schultz_spread(
        pd.Series(high, index=dates),
        pd.Series(low, index=dates),
        pd.Series(close, index=dates),
        "yearly",
    )

    assert abs(result.iloc[0] - spread) < 0.002  # noqa: PLR2004


def test_get_corwin_schultz_spread_own_trading_days_and_rolling(recorder):
    prices = corwin_schultz_prices()
    # A date only the other ticker traded is not part of a pair.
    gapped = prices.copy()
    gapped.loc[pd.Period("2024-01-30", "D")] = np.nan
    high = pd.DataFrame({"A": prices["High"], "B": gapped["High"]})
    low = pd.DataFrame({"A": prices["Low"], "B": gapped["Low"]})
    close = pd.DataFrame({"A": prices["Close"], "B": gapped["Close"]})
    rows = list(prices.itertuples(index=False))

    result = market_liquidity_model.get_corwin_schultz_spread(
        high, low, close, "monthly"
    )

    assert np.isclose(
        result.loc[pd.Period("2024-01", "M"), "B"],
        corwin_schultz_two_day_spread(rows[0], rows[2]),
    )

    rolling = market_liquidity_model.get_corwin_schultz_spread(
        high, low, close, "monthly", rolling=2
    )

    # The mean of the three two-day spreads of both months, not of the two monthly means.
    assert np.isnan(rolling.loc[pd.Period("2024-01", "M"), "A"])
    assert np.isclose(
        rolling.loc[pd.Period("2024-02", "M"), "A"],
        result.loc[pd.Period("2024-01", "M"), "A"] * 2 / 3,
    )
    recorder.capture(rolling.round(6))


def test_get_share_turnover(recorder):
    dates = pd.period_range("2024-01-29", "2024-02-02", freq="D")
    volume = pd.DataFrame(
        {"A": [10.0, 20.0, 30.0, 40.0, 50.0], "B": [5.0, np.nan, 5.0, 5.0, 0.0]},
        index=dates,
    )
    # A's share count is unknown on its first day, so January is incomplete.
    shares = pd.DataFrame(
        {"A": [np.nan, 100.0, 100.0, 200.0, 200.0], "B": [50.0] * 5}, index=dates
    )

    result = market_liquidity_model.get_share_turnover(volume, shares, "monthly")

    assert np.isnan(result.loc[pd.Period("2024-01", "M"), "A"])
    assert np.isclose(result.loc[pd.Period("2024-02", "M"), "A"], 40 / 200 + 50 / 200)
    assert np.isclose(result.loc[pd.Period("2024-01", "M"), "B"], 5 / 50 + 5 / 50)
    assert np.isclose(result.loc[pd.Period("2024-02", "M"), "B"], 5 / 50)

    rolling = market_liquidity_model.get_share_turnover(
        volume, shares, "monthly", rolling=2
    )

    assert np.isclose(rolling.loc[pd.Period("2024-02", "M"), "B"], (0.2 + 0.1) / 2)
    recorder.capture(result.round(6))
    recorder.capture(rolling.round(6))


def idiosyncratic_data():
    rng = np.random.default_rng(3)
    dates = pd.period_range("2024-01-01", "2024-04-30", freq="D")
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 3)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML"],
    )
    returns = (
        0.0002 + factors @ np.array([1.1, 0.3, -0.2]) + rng.normal(0, 0.015, len(dates))
    )
    return returns.rename("A"), factors


def residual_volatility(returns, factors):
    """The residual standard deviation from the normal equations."""
    design = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    coefficients = np.linalg.solve(design.T @ design, design.T @ returns.to_numpy())
    return np.std(returns.to_numpy() - design @ coefficients, ddof=1)


def test_get_idiosyncratic_volatility(recorder):
    returns, factors = idiosyncratic_data()
    # A month with fewer than 15 returns is NaN.
    returns[
        (returns.index >= pd.Period("2024-02-10", "D"))
        & (returns.index <= pd.Period("2024-02-29", "D"))
    ] = np.nan

    result = market_liquidity_model.get_idiosyncratic_volatility(
        returns, factors, "monthly"
    )

    march = (returns.index >= pd.Period("2024-03-01", "D")) & (
        returns.index <= pd.Period("2024-03-31", "D")
    )
    assert np.isclose(
        result[pd.Period("2024-03", "M")],
        residual_volatility(returns[march], factors[march]) * np.sqrt(252 / 12),
    )
    assert np.isnan(result[pd.Period("2024-02", "M")])

    annualized = market_liquidity_model.get_idiosyncratic_volatility(
        returns, factors, "monthly", annualize=True
    )
    assert np.isclose(
        annualized[pd.Period("2024-03", "M")],
        result[pd.Period("2024-03", "M")] * np.sqrt(12),
    )
    recorder.capture(result.round(6))


def test_get_idiosyncratic_volatility_rolling_and_capm(recorder):
    returns, factors = idiosyncratic_data()
    frame = pd.DataFrame({"A": returns, "B": returns * 0.5})

    rolling = market_liquidity_model.get_idiosyncratic_volatility(
        frame, factors, "monthly", rolling=2
    )

    window = (returns.index >= pd.Period("2024-02-01", "D")) & (
        returns.index <= pd.Period("2024-03-31", "D")
    )
    # The first window that holds two months of history ends in February.
    assert rolling.index[0] == pd.Period("2024-02", "M")
    assert np.isclose(
        rolling.loc[pd.Period("2024-03", "M"), "A"],
        residual_volatility(returns[window], factors[window]) * np.sqrt(252 / 12),
    )
    assert np.allclose(rolling["B"], rolling["A"] * 0.5, equal_nan=True)

    capm = market_liquidity_model.get_idiosyncratic_volatility(
        returns, factors[["Mkt-RF"]], "quarterly"
    )
    first_quarter = returns.index <= pd.Period("2024-03-31", "D")
    assert np.isclose(
        capm[pd.Period("2024Q1", "Q")],
        residual_volatility(returns[first_quarter], factors[["Mkt-RF"]][first_quarter])
        * np.sqrt(252 / 4),
    )
    recorder.capture(rolling.round(6))
    recorder.capture(capm.round(6))


def test_liquidity_measures_invalid_input():
    with pytest.raises(TypeError):
        market_liquidity_model.get_corwin_schultz_spread(1, 2, 3)  # type: ignore
    with pytest.raises(TypeError):
        market_liquidity_model.get_share_turnover(1, 2)  # type: ignore
    with pytest.raises(ValueError, match="Period"):
        market_liquidity_model.get_idiosyncratic_volatility(
            pd.Series(dtype=float), pd.DataFrame(), "daily"
        )
