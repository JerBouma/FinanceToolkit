"""Bars Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.utilities import bars_model


def daily_prices() -> pd.DataFrame:
    dates = pd.period_range("2024-01-01", periods=6, freq="D")

    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 11.0, 10.0, 9.0],
            "High": [11.0, 12.5, 12.0, 11.5, 10.5, 9.5],
            "Low": [9.5, 10.5, 11.0, 10.0, 9.0, 8.5],
            "Close": [10.5, 12.0, 11.5, 10.5, 9.5, 9.0],
            "Volume": [100.0, 250.0, 300.0, 50.0, 400.0, 100.0],
        },
        index=dates,
    )


def test_get_volume_bars(recorder):
    prices = daily_prices()

    result = bars_model.get_bars(prices, threshold=300)

    # 100 + 250 reaches 300 on day 2, 300 on day 3 and 50 + 400 on day 5; day 6 forms no bar.
    assert list(result.index.astype(str)) == ["2024-01-02", "2024-01-03", "2024-01-05"]
    assert list(result["Days"]) == [2, 1, 2]
    first = result.iloc[0]
    assert (first["Open"], first["High"], first["Low"], first["Close"]) == (
        10.0,
        12.5,
        9.5,
        12.0,
    )
    assert first["Volume"] == 350
    assert first["Dollar Volume"] == pytest.approx(100 * 10.5 + 250 * 12.0)
    assert first["VWAP"] == pytest.approx(first["Dollar Volume"] / 350)
    recorder.capture(result)


def test_get_dollar_bars(recorder):
    prices = daily_prices()
    prices.iloc[2, prices.columns.get_loc("Volume")] = np.nan

    result = bars_model.get_bars(
        prices.drop(columns=["Open", "High", "Low"]), threshold=4000, bar_type="dollar"
    )

    # A day without Volume is left out; without an Open, High and Low the Close stands in.
    assert list(result["Dollar Volume"]) == pytest.approx(
        [100 * 10.5 + 250 * 12.0, 50 * 10.5 + 400 * 9.5]
    )
    assert list(result["Start Date"].astype(str)) == ["2024-01-01", "2024-01-04"]
    assert result["High"].iloc[0] == 12.0  # noqa: PLR2004
    recorder.capture(result)


def test_get_bars_invalid_input():
    with pytest.raises(ValueError, match="Bar type"):
        bars_model.get_bars(daily_prices(), threshold=100, bar_type="tick")
    with pytest.raises(ValueError, match="threshold"):
        bars_model.get_bars(daily_prices(), threshold=0)
