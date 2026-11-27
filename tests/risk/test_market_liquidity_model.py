"""Market Liquidity Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.risk.liquidity import market_liquidity_model
from financetoolkit.utilities.error_model import InvalidArgumentError

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
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        market_liquidity_model.get_idiosyncratic_volatility(
            pd.Series(dtype=float), pd.DataFrame(), "daily"
        )


def test_get_share_turnover_rolling_spans_a_period_without_trading_days():
    dates = pd.period_range("2024-01-01", "2024-04-30", freq="D")
    volume = pd.Series(10.0, index=dates)
    shares = pd.Series(100.0, index=dates)
    # A halt: February has no trading day in the data at all.
    traded = (dates < pd.Period("2024-02-01", "D")) | (
        dates > pd.Period("2024-02-29", "D")
    )

    rolling = market_liquidity_model.get_share_turnover(
        volume[traded], shares[traded], "monthly", rolling=2
    )

    # The windows that hold February are NaN, not the mean of January and March.
    assert rolling.index.tolist() == list(
        pd.period_range("2024-01", "2024-04", freq="M")
    )
    assert rolling.iloc[:3].isna().all()
    assert np.isclose(rolling[pd.Period("2024-04", "M")], (31 + 30) * 0.1 / 2)


def test_get_idiosyncratic_volatility_runs_up_to_the_last_return():
    returns, factors = idiosyncratic_data()
    # The factors are published up to February only.
    published = factors[factors.index <= pd.Period("2024-02-29", "D")]

    for rolling in (None, 2):
        result = market_liquidity_model.get_idiosyncratic_volatility(
            returns, published, "monthly", rolling=rolling
        )

        assert result.index[-1] == pd.Period("2024-04", "M")
        assert result.loc[pd.Period("2024-03", "M") :].isna().all()
        assert not np.isnan(result[pd.Period("2024-02", "M")])


def abdi_ranaldo_two_day_spread(first, second):
    """The two-day corrected spread of equation (11) of the paper, written out."""
    high_1, low_1, close_1 = np.log(first)
    high_2, low_2, _ = np.log(second)
    squared = 4 * (close_1 - (high_1 + low_1) / 2) * (close_1 - (high_2 + low_2) / 2)

    return np.sqrt(max(squared, 0))


def test_get_abdi_ranaldo_spread_matches_the_paper(recorder):
    prices = corwin_schultz_prices()
    rows = list(prices.itertuples(index=False))

    result = market_liquidity_model.get_abdi_ranaldo_spread(
        prices["High"], prices["Low"], prices["Close"], "monthly"
    )

    # The pair 31 January - 1 February spans two months and is left out.
    january = np.mean(
        [
            abdi_ranaldo_two_day_spread(rows[0], rows[1]),
            abdi_ranaldo_two_day_spread(rows[1], rows[2]),
        ]
    )
    assert np.isclose(result[pd.Period("2024-01", "M")], january)
    assert np.isclose(
        result[pd.Period("2024-02", "M")],
        abdi_ranaldo_two_day_spread(rows[3], rows[4]),
    )

    # A day only the other ticker traded is not part of a pair, and a rolling window
    # averages every two-day spread in it.
    gapped = prices.copy()
    gapped.loc[pd.Period("2024-01-30", "D")] = np.nan
    frames = {
        column: pd.DataFrame({"A": prices[column], "B": gapped[column]})
        for column in ["High", "Low", "Close"]
    }
    both = market_liquidity_model.get_abdi_ranaldo_spread(
        frames["High"], frames["Low"], frames["Close"], "monthly"
    )
    assert np.isclose(
        both.loc[pd.Period("2024-01", "M"), "B"],
        abdi_ranaldo_two_day_spread(rows[0], rows[2]),
    )
    rolling = market_liquidity_model.get_abdi_ranaldo_spread(
        frames["High"], frames["Low"], frames["Close"], "monthly", rolling=2
    )
    assert np.isnan(rolling.loc[pd.Period("2024-01", "M"), "A"])
    assert np.isclose(
        rolling.loc[pd.Period("2024-02", "M"), "A"],
        (2 * january + result[pd.Period("2024-02", "M")]) / 3,
    )
    recorder.capture(result.round(6))
    recorder.capture(rolling.round(6))


def simulated_quotes(spread, volatility, seed=7, days=250):
    """
    Intraday random walks of 390 trades a day, each at the bid or the ask of a quoted
    spread, with overnight jumps on some days: the Open is the first trade, the High and
    Low the highest and lowest and the Close the last.
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2021-01-01", periods=days, freq="D")
    price = 100.0
    rows = []

    for _ in dates:
        path = price * np.exp(np.cumsum(rng.normal(0, volatility / np.sqrt(390), 390)))
        if rng.random() < 0.3:  # noqa: PLR2004
            path *= np.exp(rng.normal(0, volatility))
        trades = path * (1 + rng.choice([-1, 1], 390) * spread / 2)
        rows.append([trades[0], trades.max(), trades.min(), trades[-1]])
        price = path[-1]

    return pd.DataFrame(rows, index=dates, columns=["Open", "High", "Low", "Close"])


def test_get_abdi_ranaldo_spread_recovers_the_spread():
    # The average of the two-day square roots is close to the spread when the volatility
    # is small relative to it (and lower when it is not, see the docstring).
    prices = simulated_quotes(spread=0.01, volatility=0.002)

    result = market_liquidity_model.get_abdi_ranaldo_spread(
        prices["High"], prices["Low"], prices["Close"], "yearly"
    )

    assert abs(result.iloc[0] - 0.01) < 0.001  # noqa: PLR2004


def edge_reference(prices, keep=None):
    """
    The squared EDGE spread written out as in the authors' `bidask` package (equations
    23 to 25 of the paper), over the pairs of consecutive rows marked in `keep`.
    """
    o, h, lo, c = (np.log(prices[column].to_numpy()) for column in prices.columns)
    m = (h + lo) / 2
    h1, l1, c1, m1 = h[:-1], lo[:-1], c[:-1], m[:-1]
    o, h, lo, m = o[1:], h[1:], lo[1:], m[1:]
    keep = np.ones(len(o), dtype=bool) if keep is None else keep
    r1, r2, r3, r4, r5 = (
        value[keep] for value in (m - o, o - m1, m - c1, c1 - m1, o - c1)
    )
    tau = ((h != lo) | (lo != c1))[keep].astype(float)
    po = np.mean(tau * (o != h)[keep]) + np.mean(tau * (o != lo)[keep])
    pc = np.mean(tau * (c1 != h1)[keep]) + np.mean(tau * (c1 != l1)[keep])
    pt = tau.mean()
    d1 = r1 - r1.mean() / pt * tau
    d3 = r3 - r3.mean() / pt * tau
    d5 = r5 - r5.mean() / pt * tau
    x1 = -4 / po * d1 * r2 - 4 / pc * d3 * r4
    x2 = -4 / po * d1 * r5 - 4 / pc * d5 * r4
    e1, e2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(), x2.var()

    return (v2 * e1 + v1 * e2) / (v1 + v2)


def test_get_edge_spread_matches_the_reference(recorder):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    prices = {column: historical[column] for column in ["Open", "High", "Low", "Close"]}

    result = market_liquidity_model.get_edge_spread(*prices.values(), "monthly")

    # Each month on its own days, a negative squared spread set to 0 (equation 26).
    for ticker in result.columns:
        days = pd.DataFrame({column: prices[column][ticker] for column in prices})
        for month, month_days in days.groupby(days.index.asfreq("M")):
            if len(month_days) < 3:  # noqa: PLR2004
                continue
            assert result.loc[month, ticker] == pytest.approx(
                np.sqrt(max(edge_reference(month_days), 0)), abs=1e-15
            )
    assert (result == 0).any().any()
    assert (result > 0).any().any()

    # A rolling window takes every pair of its periods together, apart from the pair across
    # the month end.
    rolling = market_liquidity_model.get_edge_spread(
        *prices.values(), "monthly", rolling=2
    )
    days = pd.DataFrame({column: prices[column]["AAPL"] for column in prices})
    window = days.loc["2021-01-01":"2021-02-28"]
    months = window.index.asfreq("M")
    squared = edge_reference(window, keep=np.asarray(months[1:] == months[:-1]))
    assert rolling.loc[pd.Period("2021-02", "M"), "AAPL"] == pytest.approx(
        np.sqrt(max(squared, 0)), abs=1e-15
    )
    assert rolling.iloc[0].isna().all()
    recorder.capture(result.round(6))
    recorder.capture(rolling.round(6))


def test_get_edge_spread_recovers_the_spread():
    prices = simulated_quotes(spread=0.01, volatility=0.01)

    result = market_liquidity_model.get_edge_spread(
        prices["Open"], prices["High"], prices["Low"], prices["Close"], "yearly"
    )

    assert abs(result.iloc[0] - 0.01) < 0.002  # noqa: PLR2004


def test_get_edge_spread_without_price_changes_is_nan():
    # A month of flat days (no trades repeat the previous Close) has no estimate.
    dates = pd.period_range("2024-01-01", periods=10, freq="D")
    flat = pd.Series(100.0, index=dates)

    result = market_liquidity_model.get_edge_spread(flat, flat, flat, flat)

    assert result.isna().all()
    with pytest.raises(TypeError):
        market_liquidity_model.get_edge_spread(1, 2, 3, 4)  # type: ignore
    with pytest.raises(TypeError):
        market_liquidity_model.get_abdi_ranaldo_spread(1, 2, 3)  # type: ignore


def daily_bars(days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Weekday Close, High, Low and Volume of a random walk with a bid-ask bounce."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=days).to_period("D")
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
    close = np.round(close + rng.choice([-0.05, 0.05], days), 2)

    return pd.DataFrame(
        {
            "High": close * (1 + np.abs(rng.normal(0, 0.01, days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.01, days))),
            "Close": close,
            "Volume": np.round(rng.lognormal(14, 0.5, days)),
        },
        index=dates,
    )


def test_get_bekker_parkinson_volatility_matches_the_formula(recorder):
    # Three days in one month without an overnight gap (each Close lies in the next day's range).
    dates = pd.period_range("2024-01-02", periods=3, freq="D")
    high = pd.Series([101.0, 102.0, 101.5], index=dates)
    low = pd.Series([99.0, 99.5, 99.8], index=dates)
    close = pd.Series([100.5, 100.0, 100.0], index=dates)

    result = market_liquidity_model.get_bekker_parkinson_volatility(high, low, close)

    k2 = np.sqrt(8 / np.pi)
    denominator = 3 - 2 * np.sqrt(2)
    sigmas = []
    for first, second in ((0, 1), (1, 2)):
        beta = np.log(high.iloc[first] / low.iloc[first]) ** 2 + (
            np.log(high.iloc[second] / low.iloc[second]) ** 2
        )
        gamma = (
            np.log(
                max(high.iloc[first], high.iloc[second])
                / min(low.iloc[first], low.iloc[second])
            )
            ** 2
        )
        sigma = (np.sqrt(beta / 2) - np.sqrt(beta)) / (k2 * denominator) + np.sqrt(
            gamma / (k2**2 * denominator)
        )
        sigmas.append(max(sigma, 0))
    days_per_period = market_liquidity_model.get_days_per_period(high, "monthly")

    assert result.iloc[0] == pytest.approx(
        np.mean(sigmas) * np.sqrt(days_per_period), rel=1e-12
    )
    recorder.capture(result.round(6))


def test_get_bekker_parkinson_volatility_rolling_and_dataframe(recorder):
    bars = daily_bars()
    frame = pd.DataFrame({"A": bars["High"], "B": bars["High"] * 2})

    monthly = market_liquidity_model.get_bekker_parkinson_volatility(
        frame,
        pd.DataFrame({"A": bars["Low"], "B": bars["Low"] * 2}),
        pd.DataFrame({"A": bars["Close"], "B": bars["Close"] * 2}),
    )
    rolling = market_liquidity_model.get_bekker_parkinson_volatility(
        bars["High"], bars["Low"], bars["Close"], "monthly", rolling=2
    )

    # The volatility is that of the log price, so it does not change with the price level.
    pd.testing.assert_series_equal(monthly["A"], monthly["B"], check_names=False)
    assert (monthly["A"] >= 0).all()
    assert np.isnan(rolling.iloc[0])
    with pytest.raises(TypeError):
        market_liquidity_model.get_bekker_parkinson_volatility(1, 2, 3)  # type: ignore
    recorder.capture(monthly.round(6))
    recorder.capture(rolling.round(6))


def test_price_impact_lambdas_recover_an_exact_impact():
    # Prices built from λ times the signed order flow are fitted exactly by each regression.
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=60).to_period("D")
    signs = rng.choice([-1.0, 1.0], len(dates))
    volume = pd.Series(rng.lognormal(12, 0.5, len(dates)), index=dates)

    kyle_prices = pd.Series(
        100 + np.cumsum(np.r_[0, 2e-6 * (signs * volume.to_numpy())[1:]]), index=dates
    )
    log_returns = np.r_[0, 1e-4 * (signs * np.sqrt(volume.to_numpy()))[1:]]
    hasbrouck_prices = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=dates)

    kyle = market_liquidity_model.get_kyle_lambda(
        kyle_prices, volume, "monthly", scale=1
    )
    hasbrouck = market_liquidity_model.get_hasbrouck_lambda(
        hasbrouck_prices, volume, "monthly", scale=1
    )
    # |r_t| = λ * DV_t: the dollar volume that gives each absolute return.
    amihud_volume = pd.Series(np.abs(log_returns) / 3e-9, index=dates)
    amihud = market_liquidity_model.get_amihud_lambda(
        hasbrouck_prices, amihud_volume, "monthly", scale=1
    )

    np.testing.assert_allclose(kyle.to_numpy(), 2e-6, rtol=1e-9)
    np.testing.assert_allclose(hasbrouck.to_numpy(), 1e-4, rtol=1e-9)
    np.testing.assert_allclose(amihud.to_numpy(), 3e-9, rtol=1e-9)


def test_get_kyle_lambda_matches_least_squares(recorder):
    bars = daily_bars()
    result = market_liquidity_model.get_kyle_lambda(
        bars["Close"], bars["Volume"], "monthly"
    )
    rolling = market_liquidity_model.get_kyle_lambda(
        bars["Close"], bars["Volume"], "monthly", rolling=2
    )

    change = bars["Close"].diff()
    # The tick rule: a day without a change keeps the sign of the last change.
    sign = np.sign(change).replace(0, np.nan).ffill()
    months = bars.index.asfreq("M")
    for month in result.index:
        rows = (months == month) & sign.notna()
        regressor = (sign * bars["Volume"])[rows].to_numpy()[:, None]
        slope = np.linalg.lstsq(regressor, change[rows].to_numpy(), rcond=None)[0][0]
        assert result[month] == pytest.approx(slope * 1_000_000, rel=1e-10)

    rows = (months >= result.index[0]) & (months <= result.index[1]) & sign.notna()
    regressor, target = (sign * bars["Volume"])[rows], change[rows]
    assert rolling.iloc[1] == pytest.approx(
        (regressor * target).sum() / (regressor**2).sum() * 1_000_000, rel=1e-10
    )
    assert np.isnan(rolling.iloc[0])
    recorder.capture(result.round(6))
    recorder.capture(rolling.round(6))


def test_get_amihud_and_hasbrouck_lambda(recorder):
    bars = daily_bars()
    dollar_volume = bars["Close"] * bars["Volume"]
    frame = pd.DataFrame({"A": bars["Close"], "B": bars["Close"]})
    volumes = pd.DataFrame({"A": dollar_volume, "B": dollar_volume * 4})

    amihud = market_liquidity_model.get_amihud_lambda(frame, volumes, "quarterly")
    hasbrouck = market_liquidity_model.get_hasbrouck_lambda(frame, volumes, "quarterly")

    log_return = np.log(bars["Close"]).diff()
    quarters = bars.index.asfreq("Q")
    first = (quarters == quarters[0]) & log_return.notna()
    expected = (log_return.abs() * dollar_volume)[first].sum() / (
        dollar_volume[first] ** 2
    ).sum()
    assert amihud["A"].iloc[0] == pytest.approx(expected * 1_000_000, rel=1e-10)
    # Four times the dollar volume: a quarter of Amihud's lambda and half of Hasbrouck's.
    np.testing.assert_allclose(amihud["B"], amihud["A"] / 4, rtol=1e-10)
    np.testing.assert_allclose(hasbrouck["B"], hasbrouck["A"] / 2, rtol=1e-10)
    with pytest.raises(TypeError):
        market_liquidity_model.get_hasbrouck_lambda(1, 2)  # type: ignore
    recorder.capture(amihud.round(6))
    recorder.capture(hasbrouck.round(6))


def test_get_roll_impact(recorder):
    bars = daily_bars()
    dollar_volume = bars["Close"] * bars["Volume"]

    result = market_liquidity_model.get_roll_impact(
        bars["Close"], dollar_volume, "monthly", scale=1
    )

    months = bars.index.asfreq("M")
    for month in result.index:
        rows = months == month
        changes = bars["Close"][rows].diff().dropna().to_numpy()
        autocovariance = np.cov(changes[:-1], changes[1:], ddof=1)[0, 1]
        if autocovariance >= 0:
            assert np.isnan(result[month])
            continue
        expected = 2 * np.sqrt(-autocovariance) / dollar_volume[rows].mean()
        assert result[month] == pytest.approx(expected, rel=1e-10)

    assert result.notna().any()
    with pytest.raises(TypeError):
        market_liquidity_model.get_roll_impact(1, 2)  # type: ignore
    recorder.capture(result.round(12))


def test_get_bulk_volume_classification(recorder):
    bars = daily_bars()

    normal = market_liquidity_model.get_bulk_volume_classification(
        bars["Close"], bars["Volume"], window=20
    )
    student = market_liquidity_model.get_bulk_volume_classification(
        bars["Close"], bars["Volume"], window=20, degrees_of_freedom=0.25
    )

    change = bars["Close"].diff()
    standardized = change / change.rolling(20).std()
    pd.testing.assert_series_equal(
        normal, bars["Volume"] * stats.norm.cdf(standardized), check_names=False
    )
    pd.testing.assert_series_equal(
        student,
        bars["Volume"] * stats.t.cdf(standardized, 0.25),
        check_names=False,
    )
    assert normal.iloc[:20].isna().all()
    recorder.capture(normal.round(2))


def test_get_vpin_equal_buckets_and_split_days(recorder, monkeypatch):
    # Three days with a buy share of 0.9, 0.2 and 0.6 and volumes of 1.5, 0.5 and 1 bucket.
    dates = pd.period_range("2024-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 101.0, 100.5, 101.0], index=dates)
    volume = pd.Series([1.0, 150.0, 50.0, 100.0], index=dates)
    shares = pd.Series([np.nan, 0.9, 0.2, 0.6], index=dates)

    monkeypatch.setattr(
        market_liquidity_model,
        "get_bulk_volume_classification",
        lambda prices, volume, window, degrees_of_freedom: volume
        * shares,  # noqa: ARG005
    )
    result = market_liquidity_model.get_vpin(
        prices, volume, buckets=2, bucket_volume=100
    )

    # Bucket 1 is 100 of day 1 (90 bought), bucket 2 is 50 of day 1 and 50 of day 2 (45 + 10
    # bought) and bucket 3 is day 3 (60 bought).
    first = (abs(100 - 2 * 90) + abs(100 - 2 * 55)) / 200
    second = (abs(100 - 2 * 55) + abs(100 - 2 * 60)) / 200
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(first)
    assert result.iloc[3] == pytest.approx(second)
    recorder.capture(result)


def test_get_vpin(recorder):
    bars = daily_bars(days=200)
    frame = pd.DataFrame({"A": bars["Close"], "B": bars["Close"]})
    volume = pd.DataFrame({"A": bars["Volume"], "B": bars["Volume"].mean()})

    result = market_liquidity_model.get_vpin(frame, volume, buckets=20, window=20)
    equal = market_liquidity_model.get_bulk_volume_classification(
        bars["Close"], volume["B"], window=20
    )
    imbalance = (volume["B"] - 2 * equal).abs()

    # With the same volume every day, each day is a bucket and VPIN is a rolling mean.
    pd.testing.assert_series_equal(
        result["B"],
        imbalance.rolling(20).sum() / (20 * volume["B"]),
        check_names=False,
        rtol=1e-10,
    )
    assert result["A"].dropna().between(0, 1).all()
    assert result["A"].notna().any()
    with pytest.raises(TypeError):
        market_liquidity_model.get_vpin(1, 2)  # type: ignore
    recorder.capture(result.round(6))


def test_get_vpin_bucket_volume_has_no_look_ahead():
    bars = daily_bars(days=200)
    later = bars["Volume"].copy()
    later.iloc[100:] *= 5

    # The default bucket is the mean volume of the first `window` days, so volume after
    # them changes neither the bucket nor the VPIN of the days before.
    base = market_liquidity_model.get_vpin(bars["Close"], bars["Volume"], 20, None, 20)
    changed = market_liquidity_model.get_vpin(bars["Close"], later, 20, None, 20)
    pd.testing.assert_series_equal(base.iloc[:100], changed.iloc[:100])
    pd.testing.assert_series_equal(
        base,
        market_liquidity_model.get_vpin(
            bars["Close"], bars["Volume"], 20, bars["Volume"].iloc[:20].mean(), 20
        ),
    )
