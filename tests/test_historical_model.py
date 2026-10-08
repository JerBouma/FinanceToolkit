"""Historical Model Tests"""

# ruff: noqa

# pylint: disable=missing-function-docstring

import pandas as pd

from financetoolkit import historical_model


def _frame(start: str, end: str) -> pd.DataFrame:
    """A daily historical frame spanning the given dates, as the providers return."""
    index = pd.PeriodIndex(pd.date_range(start=start, end=end, freq="D"), freq="D")

    return pd.DataFrame({"Adj Close": 1.0}, index=index)


def test_covers_requested_range_accepts_a_full_response():
    data = _frame("2005-01-01", "2025-01-01")

    assert historical_model._covers_requested_range(data, "2005-01-01", "2025-01-01")


def test_covers_requested_range_rejects_a_drastically_short_span():
    # Three months returned where twenty years were asked for.
    data = _frame("2024-10-01", "2025-01-01")

    assert not historical_model._covers_requested_range(
        data, "2005-01-01", "2025-01-01"
    )


def test_covers_requested_range_rejects_a_row_capped_response():
    """The response runs to the requested end but is missing history at the start.

    FinancialModelingPrep caps at a fixed row count rather than a fixed span, so a
    capped response spans ~99% of the request and is only short at the beginning.
    Comparing span lengths alone does not catch it.
    """
    data = _frame("2006-02-16", "2025-01-01")

    assert not historical_model._covers_requested_range(
        data, "2005-01-01", "2025-01-01"
    )


def test_covers_requested_range_tolerates_a_start_on_a_non_trading_day():
    # 2022-01-01 was a Saturday; the first bar is the following Monday.
    data = _frame("2022-01-03", "2025-01-01")

    assert historical_model._covers_requested_range(data, "2022-01-01", "2025-01-01")


def test_covers_requested_range_skips_short_requests():
    data = _frame("2024-06-10", "2024-06-14")

    assert historical_model._covers_requested_range(data, "2024-06-01", "2024-06-14")


def test_covers_requested_range_handles_missing_dates_and_empty_data():
    data = _frame("2024-01-01", "2024-06-01")

    assert historical_model._covers_requested_range(data, None, "2025-01-01")
    assert historical_model._covers_requested_range(data, "2005-01-01", None)
    assert historical_model._covers_requested_range(
        pd.DataFrame(), "2005-01-01", "2025-01-01"
    )


def test_reaches_further_back_prefers_the_longer_history():
    yahoo = _frame("2005-01-03", "2025-01-01")
    truncated_fmp = _frame("2006-02-16", "2025-01-01")

    assert historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_reaches_further_back_rejects_an_equally_short_history():
    """A young ticker yields the same short history from both providers.

    In that case there is nothing to gain by switching provider, so the original
    response is kept rather than replaced.
    """
    yahoo = _frame("2024-03-21", "2025-01-01")
    truncated_fmp = _frame("2024-03-21", "2025-01-01")

    assert not historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_reaches_further_back_rejects_an_empty_candidate():
    truncated_fmp = _frame("2006-02-16", "2025-01-01")

    assert not historical_model._reaches_further_back(pd.DataFrame(), truncated_fmp)


def test_reaches_further_back_accepts_anything_over_nothing():
    yahoo = _frame("2024-03-21", "2025-01-01")

    assert historical_model._reaches_further_back(yahoo, pd.DataFrame())


def test_reaches_further_back_compares_across_index_types():
    """PeriodIndex and DatetimeIndex frames must stay comparable."""
    yahoo = _frame("2005-01-03", "2025-01-01")
    truncated_fmp = _frame("2006-02-16", "2025-01-01")
    truncated_fmp.index = truncated_fmp.index.to_timestamp()

    assert historical_model._reaches_further_back(yahoo, truncated_fmp)


def test_fill_non_trading_days_keeps_returns_consistent_with_prices():
    """A holiday on one exchange gets a price, no volume and no invented return."""
    import numpy as np

    index = pd.period_range("2024-01-01", periods=5, freq="D")
    frames = {
        # A has no row for 3 January, B traded that day.
        "A": pd.DataFrame(
            {"Adj Close": [100.0, 101.0, 103.0, 104.0], "Volume": 10.0},
            index=index.delete(2),
        ),
        "B": pd.DataFrame(
            {"Adj Close": [50.0, 51, 52, 53, 54], "Volume": 5.0}, index=index
        ),
    }

    for frame in frames.values():
        frame["Return"] = frame["Adj Close"].pct_change()

    historical_data = pd.concat(frames).unstack(level=0)

    filled = historical_model.fill_non_trading_days(historical_data)

    assert filled.loc[index[2], ("Adj Close", "A")] == 102.0
    assert filled.loc[index[2], ("Volume", "A")] == 0
    # The returns compound to the actual move, 100 to 104, and B is untouched.
    assert np.isclose((1 + filled[("Return", "A")].fillna(0)).prod(), 1.04)
    pd.testing.assert_series_equal(
        filled[("Return", "B")], historical_data[("Return", "B")]
    )
