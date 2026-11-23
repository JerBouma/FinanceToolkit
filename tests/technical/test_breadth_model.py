"""Breadth Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.breadth import breadth_model
from tests.technical.sample_prices import sample_prices


def test_get_mcclellan_oscillator(recorder):
    prices = sample_prices()
    result = breadth_model.get_mcclellan_oscillator(prices["Close"], 10, 20).round(2)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_advancers_decliners(recorder):
    recorder.capture(
        breadth_model.get_advancers_decliners(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02])
        )
    )


def test_get_on_balance_volume(recorder):
    recorder.capture(
        breadth_model.get_on_balance_volume(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
        )
    )


def test_get_accumulation_distribution_line(recorder):
    recorder.capture(
        breadth_model.get_accumulation_distribution_line(
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([100, 200, 300, 10, 20]),
            pd.Series([0.01, -0.03, 0.05, 0.01, -0.02]),
            pd.Series([0.1, 103, 0.05, 0.01, -0.02]),
        )
    )


def test_get_chaikin_oscillator(recorder):
    prices = sample_prices()
    result = breadth_model.get_chaikin_oscillator(
        prices["High"], prices["Low"], prices["Close"], prices["Volume"], 10, 20
    ).round(3)

    # The window fills within the sample, so there is something to compare.
    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_get_chaikin_money_flow(recorder):
    recorder.capture(
        breadth_model.get_chaikin_money_flow(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            4,
        ).round(3)
    )


def test_get_ease_of_movement(recorder):
    recorder.capture(
        breadth_model.get_ease_of_movement(
            pd.Series([10, 11, 12, 11, 13, 14, 13, 15]),
            pd.Series([9, 10, 10, 9, 11, 12, 11, 13]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
            3,
        ).round(3)
    )


def test_get_negative_volume_index(recorder):
    recorder.capture(
        breadth_model.get_negative_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )


def test_get_positive_volume_index(recorder):
    recorder.capture(
        breadth_model.get_positive_volume_index(
            pd.Series([9.5, 10.5, 11, 10, 12, 13, 12, 14.5]),
            pd.Series([1000, 1200, 900, 1100, 1500, 800, 1300, 1600]),
        ).round(3)
    )


def test_mcclellan_oscillator_seeds_on_actual_advances_and_declines():
    prices = pd.Series([10.0, 11, 10, 12, 13, 12, 14, 15, 14, 16])

    oscillator = breadth_model.get_mcclellan_oscillator(prices, 2, 4)

    # The first signal is on the second price, so the 4-period EMA is seeded on periods
    # 1 to 4 and the 2-period EMA on periods 1 to 2 (then updated up to period 4).
    signals = [1, -1, 1, 1]
    short = sum(signals[:2]) / 2
    for signal in signals[2:]:
        short = short + 2 / 3 * (signal - short)

    assert oscillator.first_valid_index() == 4
    assert oscillator.iloc[4] == pytest.approx(short - sum(signals) / 4)


def test_volume_indicators_are_nan_before_listing_and_on_missing_days():
    index = pd.date_range("2024-01-01", periods=6)
    close = pd.Series([float("nan"), 10.0, 11.0, float("nan"), 12.0, 11.0], index=index)
    volume = pd.Series(
        [float("nan"), 100.0, 200.0, float("nan"), 150.0, 120.0], index=index
    )

    for result in [
        breadth_model.get_on_balance_volume(close, volume),
        breadth_model.get_negative_volume_index(close, volume),
        breadth_model.get_positive_volume_index(close, volume),
        breadth_model.get_advancers_decliners(close),
    ]:
        assert result.iloc[[0, 3]].isna().all()
        assert result.iloc[[1, 2, 4, 5]].notna().all()

    # The day after the gap compares with the last close before it (12 against 11).
    assert (
        breadth_model.get_on_balance_volume(close, volume).tolist()[4] == 0 + 200 + 150
    )
    assert breadth_model.get_advancers_decliners(close).iloc[4] == 1
    # The negative volume index moves only on lower volume: 150 < 200 on 12 against 11.
    assert breadth_model.get_negative_volume_index(close, volume).iloc[
        4
    ] == pytest.approx(1000 * 12 / 11)


def test_accumulation_distribution_line_adds_nothing_when_high_equals_low():
    index = pd.date_range("2024-01-01", periods=3)
    high = pd.Series([11.0, 10.0, 12.0], index=index)
    low = pd.Series([9.0, 10.0, 10.0], index=index)
    close = pd.Series([11.0, 10.0, 10.0], index=index)
    volume = pd.Series([100.0, 500.0, 50.0], index=index)

    adl = breadth_model.get_accumulation_distribution_line(high, low, close, volume)

    # As in TA-Lib: a bar without range has a multiplier of 0 and the line runs on.
    assert adl.tolist() == [100.0, 100.0, 50.0]


def test_new_highs_new_lows_use_full_windows_and_strict_comparisons():
    index = pd.date_range("2024-01-01", periods=5)
    closes = pd.DataFrame(
        {
            "A": [10.0, 11.0, 11.0, 12.0, 9.0],
            "B": [float("nan"), float("nan"), 5.0, 5.0, 6.0],
        },
        index=index,
    )

    result = breadth_model.get_new_highs_new_lows(closes, window=3)

    # No ticker has three closes until the third day; A's 11 equals its previous high, so it
    # is no new high.
    assert result.index.equals(index)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 0
    assert result.iloc[3] == 1
    # A makes a new low and B, with its first full window, makes a new high.
    assert result.iloc[4] == 0


def test_volume_indicators_skip_gaps_per_ticker_in_a_dataframe():
    index = pd.date_range("2024-01-01", periods=8)
    close = pd.DataFrame(
        {
            "A": [10.0, 11, float("nan"), 12, 11, 13, 14, 15],
            "B": [float("nan"), 20.0, 21, 20, 22, 23, 22, 24],
        },
        index=index,
    )
    volume = pd.DataFrame(
        {
            "A": [100.0, 200, float("nan"), 150, 50, 80, 90, 70],
            "B": [float("nan"), 300.0, 100, 200, 400, 250, 150, 350],
        },
        index=index,
    )

    obv = breadth_model.get_on_balance_volume(close, volume)

    # Each column equals the Series result, so the day after A's gap compares 12 with 11.
    for ticker in close.columns:
        pd.testing.assert_series_equal(
            obv[ticker],
            breadth_model.get_on_balance_volume(close[ticker], volume[ticker]),
            check_names=False,
        )
    assert obv.loc[index[3], "A"] == 200 + 150


def test_mcclellan_oscillator_continues_after_a_gap():
    prices = pd.Series([10.0, 11, 10, float("nan"), 12, 13, 12, 14, 15, 14, 16])

    oscillator = breadth_model.get_mcclellan_oscillator(prices, 2, 4)
    without_gap = breadth_model.get_mcclellan_oscillator(prices.dropna(), 2, 4)

    # The close after the gap is an advance against the last close before it, so the
    # oscillator matches the one computed without the missing day.
    pd.testing.assert_series_equal(
        oscillator.dropna().reset_index(drop=True),
        without_gap.dropna().reset_index(drop=True),
    )


def test_price_volume_trend_weights_volume_with_the_change_in_price():
    prices_close = pd.Series([100.0, 110.0, None, 99.0, 99.0])
    volumes = pd.Series([1000.0, 500.0, 800.0, 200.0, 300.0])

    result = breadth_model.get_price_volume_trend(prices_close, volumes)

    # 0 on the first observation; the observation after the gap compares with 110.
    expected = pd.Series([0.0, 50.0, None, 50.0 - 200.0 * 0.1, 30.0])
    pd.testing.assert_series_equal(result, expected)


def ohlcv_walk() -> pd.DataFrame:
    """200 bars of a seeded random walk with a range and volume on every bar, long enough
    for the exponential averages of the reference libraries to converge."""
    rng = np.random.default_rng(11)
    close = 100 + rng.normal(0, 1, 200).cumsum()
    return pd.DataFrame(
        {
            "High": close + rng.uniform(0.1, 1.5, 200),
            "Low": close - rng.uniform(0.1, 1.5, 200),
            "Close": close,
            "Volume": rng.integers(1_000, 5_000, 200).astype(float),
        }
    )


def test_get_klinger_volume_oscillator(recorder):
    prices = sample_prices()
    result = breadth_model.get_klinger_volume_oscillator(
        prices["High"], prices["Low"], prices["Close"], prices["Volume"], 5, 10, 4
    ).round(2)

    assert result.notna().to_numpy().any()
    recorder.capture(result)


def test_klinger_volume_oscillator_matches_talib():
    walk = ohlcv_walk()
    result = breadth_model.get_klinger_volume_oscillator(
        walk["High"], walk["Low"], walk["Close"], walk["Volume"], 5, 10, 4
    )

    # TA-Lib 0.8.3 KVO(5, 10, 4): its averages start from the first volume force and ours
    # from a simple average, which no longer matters once they have converged.
    assert result["Klinger Volume Oscillator"].iloc[-1] == pytest.approx(
        5131.5042643398465, rel=1e-9
    )
    assert result["Signal Line"].iloc[-1] == pytest.approx(4645.243270029379, rel=1e-9)
    # The first value is on period long_window + 1, as TA-Lib's.
    assert result["Klinger Volume Oscillator"].first_valid_index() == 10


def test_klinger_volume_oscillator_restarts_the_cumulative_range_on_a_new_trend():
    high = pd.Series([11.0, 11.5, 13.0, 13.0])
    low = pd.Series([9.0, 10.5, 11.0, 9.0])
    close = pd.Series([10.0, 11.0, 12.0, 10.0])
    volume = pd.Series([100.0, 100.0, 100.0, 100.0])

    result = breadth_model.get_klinger_volume_oscillator(
        high, low, close, volume, 1, 2, 1
    )["Klinger Volume Oscillator"]

    # High + Low + Close rises on bars 1 and 2 and falls on bar 3, with ranges 2, 1, 2, 4:
    # cm = 2 + 1 on bar 1 and 3 + 2 on bar 2, and restarts at 2 + 4 when the trend turns.
    force = [
        100 * abs(2 * 1 / 3 - 1) * 100,
        100 * abs(2 * 2 / 5 - 1) * 100,
        -100 * abs(2 * 4 / 6 - 1) * 100,
    ]
    # An EMA of one period is the force itself; the two-period EMA is seeded on bars 1-2.
    long_average = (force[0] + force[1]) / 2
    assert result.iloc[2] == pytest.approx(force[1] - long_average)
    long_average += 2 / 3 * (force[2] - long_average)
    assert result.iloc[3] == pytest.approx(force[2] - long_average)


def test_market_facilitation_index_matches_talib():
    walk = ohlcv_walk()
    result = breadth_model.get_market_facilitation_index(
        walk["High"], walk["Low"], walk["Volume"]
    )

    # TA-Lib 0.8.3 MARKETFI, pandas-ta-classic marketfi.
    assert result.iloc[-1] == pytest.approx(0.000370699507557265, rel=1e-12)
    assert breadth_model.get_market_facilitation_index(
        walk["High"], walk["Low"], walk["Volume"], 1e3
    ).iloc[-1] == pytest.approx(0.370699507557265, rel=1e-12)
    assert np.isnan(
        breadth_model.get_market_facilitation_index(
            pd.Series([2.0]), pd.Series([1.0]), pd.Series([0.0])
        ).iloc[0]
    )


def test_williams_accumulation_distribution_matches_talib():
    walk = ohlcv_walk()
    result = breadth_model.get_williams_accumulation_distribution(
        walk["High"], walk["Low"], walk["Close"]
    )

    # TA-Lib 0.8.3 WAD, which also starts at 0.
    assert result.iloc[0] == 0
    assert result.iloc[-1] == pytest.approx(8.09187345886727, rel=1e-12)


def test_williams_accumulation_distribution_uses_the_true_range():
    high = pd.Series([11.0, 12.0, 10.5, 10.5])
    low = pd.Series([9.0, 10.5, 9.0, 9.5])
    close = pd.Series([10.0, 11.0, 9.5, 9.5])

    result = breadth_model.get_williams_accumulation_distribution(high, low, close)

    # Up close: 11 - min(10.5, 10) = 1; down close: 9.5 - max(10.5, 11) = -1.5; unchanged: 0.
    assert result.tolist() == [0.0, 1.0, -0.5, -0.5]


def test_relative_volume_matches_talib():
    walk = ohlcv_walk()
    result = breadth_model.get_relative_volume(walk["Volume"], 20)

    # TA-Lib 0.8.3 RVOL(20): the average covers the 20 periods before the current one.
    assert result.iloc[-1] == pytest.approx(0.9082362123711706, rel=1e-12)
    assert result.first_valid_index() == 20
    assert result.iloc[20] == walk["Volume"].iloc[20] / walk["Volume"].iloc[:20].mean()


def test_volume_flow_indicator_matches_pandas_ta():
    walk = ohlcv_walk()
    result = breadth_model.get_volume_flow_indicator(
        walk["High"], walk["Low"], walk["Close"], walk["Volume"], 10, 0.2, 2.5, 3
    )

    # pandas-ta-classic 0.9.0 vfi(length=10), which also uses the population deviation (finta
    # 1.3 VFI(period=10) happens to agree on this bar).
    assert result.iloc[-1] == pytest.approx(-2.294125273726582, rel=1e-9)
    # The cutoff needs 30 changes of the typical price (and the average volume 10 earlier
    # periods), the sum 10 signed volumes and the EMA 3 sums.
    assert result.first_valid_index() == 30 + 10 + 3 - 2


def test_finite_volume_elements_matches_finta():
    walk = ohlcv_walk()
    result = breadth_model.get_finite_volume_elements(
        walk["High"], walk["Low"], walk["Close"], walk["Volume"], 10, 0.003
    )

    # finta 1.3 FVE(period=10, factor=0.3), in percent.
    assert result.iloc[-1] == pytest.approx(-0.28855371636735473, rel=1e-12)
    assert result.dropna().between(-1, 1).all()


def test_twiggs_money_flow_follows_twiggs():
    walk = ohlcv_walk()
    high, low, close, volume = (walk[column] for column in walk.columns)
    result = breadth_model.get_twiggs_money_flow(high, low, close, volume, 21)

    # Twiggs' definition written out: the close within the true range, both sums smoothed
    # with Wilder's average (seeded with the simple average of the first 21 values).
    true_high = np.maximum(high, close.shift(1).fillna(high))
    true_low = np.minimum(low, close.shift(1).fillna(low))
    flow = volume * ((close - true_low) - (true_high - close)) / (true_high - true_low)

    def wilder(values):
        average = values.iloc[:21].mean()
        for value in values.iloc[21:]:
            average += (value - average) / 21
        return average

    assert result.iloc[-1] == pytest.approx(wilder(flow) / wilder(volume), rel=1e-12)
    assert result.first_valid_index() == 20
    assert result.dropna().between(-1, 1).all()


def test_volume_zone_oscillator_matches_finta():
    walk = ohlcv_walk()
    result = breadth_model.get_volume_zone_oscillator(walk["Close"], walk["Volume"], 5)

    # finta 1.3 VZO(period=5), in percent, after its averages have converged.
    assert result.iloc[-1] == pytest.approx(0.1712412122623073, rel=1e-9)
    assert result.first_valid_index() == 5
    assert result.dropna().between(-1, 1).all()


def test_cumulative_force_index_matches_finta():
    walk = ohlcv_walk()

    # finta 1.3 WOBV: the running total of the unsmoothed force.
    assert breadth_model.get_cumulative_force_index(
        walk["Close"], walk["Volume"], 1
    ).iloc[-1] == pytest.approx(22200.752678070137, rel=1e-12)
    # finta 1.3 CFI: the same change once the EMAs have converged (the level keeps an offset).
    assert breadth_model.get_cumulative_force_index(
        walk["Close"], walk["Volume"], 10
    ).diff().iloc[-1] == pytest.approx(-548.9318179936017, rel=1e-9)


def test_price_volume_and_rank_match_pandas_ta():
    walk = ohlcv_walk()

    # pandas-ta-classic 0.9.0 pvol(signed=True) and pvr.
    assert breadth_model.get_price_volume(
        walk["Close"], walk["Volume"], signed=True
    ).iloc[-1] == pytest.approx(264376.51213396544, rel=1e-12)
    assert breadth_model.get_price_volume_rank(walk["Close"], walk["Volume"]).iloc[
        -6:
    ].tolist() == [4.0, 1.0, 4.0, 1.0, 3.0, 2.0]
    assert np.isnan(
        breadth_model.get_price_volume_rank(walk["Close"], walk["Volume"]).iloc[0]
    )


def test_get_volume_profile(recorder):
    walk = ohlcv_walk()
    result = breadth_model.get_volume_profile(walk["Close"], walk["Volume"], 4)

    # pandas-ta-classic 0.9.0 vp(width=4, sort_close=True): the up and down volume per bin.
    assert result["Up Volume"].tolist() == [64463.0, 45898.0, 119754.0, 64837.0]
    assert result["Down Volume"].tolist() == [59057.0, 59810.0, 113881.0, 43617.0]
    # The total also holds the first bar, which has no change.
    assert result["Total Volume"].sum() == walk["Volume"].sum()
    assert result["Low Price"].iloc[0] == walk["Close"].min()
    assert result["High Price"].iloc[-1] == walk["Close"].max()
    recorder.capture(result.round(4))


def test_volume_profile_keeps_bins_without_volume():
    close = pd.Series([10.0, 10.5, 20.0])
    volume = pd.Series([100.0, 200.0, 300.0])

    result = breadth_model.get_volume_profile(close, volume, 4)

    assert result["Total Volume"].tolist() == [300.0, 0.0, 0.0, 300.0]
    assert result["Up Volume"].tolist() == [200.0, 0.0, 0.0, 300.0]


def test_buying_selling_pressure_matches_finta():
    walk = ohlcv_walk()
    high, low, close, volume = (walk[column] for column in walk.columns)

    # finta 1.3 BASP(period=5) and BASPN(period=5), after their averages have converged.
    raw = breadth_model.get_buying_selling_pressure(high, low, close, volume, 5, 1)
    assert raw.iloc[-1].tolist() == pytest.approx(
        [0.6295505746191441, 0.6333180702137301], rel=1e-9
    )
    smoothed = breadth_model.get_buying_selling_pressure(
        high, low, close, volume, 5, 20
    )
    assert smoothed.iloc[-1].tolist() == pytest.approx(
        [0.8577031617856733, 0.9429558422113259], rel=1e-6
    )


def test_archer_on_balance_volume_matches_pandas_ta():
    walk = ohlcv_walk()
    result = breadth_model.get_archer_on_balance_volume(
        walk["Close"], walk["Volume"], 4, 12, 2, 2
    )

    # pandas-ta-classic 0.9.0 aobv, whose OBV starts at the first volume (3562) instead of 0.
    first_volume = walk["Volume"].iloc[0]
    assert (result.iloc[-1, :5] + first_volume).tolist() == pytest.approx(
        [22149.0, 19589.0, 22149.0, 21243.133545919194, 21103.21240503417], rel=1e-12
    )
    assert result.iloc[-1, 5:].tolist() == [0.0, 1.0]


def test_volume_profile_of_a_constant_close_puts_the_volume_in_the_first_bin():
    close = pd.Series([10.0, 10.0, 10.0])
    volume = pd.Series([100.0, 200.0, 300.0])

    result = breadth_model.get_volume_profile(close, volume, 4)

    assert result["Total Volume"].tolist() == [600.0, 0.0, 0.0, 0.0]
    assert result["Up Volume"].sum() == result["Down Volume"].sum() == 0
    assert (result[["Low Price", "High Price"]] == 10.0).all().all()


def test_volume_profile_of_a_close_constant_up_to_rounding_fills_the_first_bin():
    close = pd.Series([1.0, 1.0000000000000002] * 2)
    profile = breadth_model.get_volume_profile(close, pd.Series([1.0] * 4), 10)
    assert profile["Total Volume"].tolist() == [4.0] + [0.0] * 9
