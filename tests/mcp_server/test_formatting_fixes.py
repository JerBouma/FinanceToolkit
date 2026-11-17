"""Formatting and Column Selection Tests"""

import pandas as pd

from financetoolkit.mcp_server.registry.formatting_model import _df_to_records


def test_levels_with_the_same_names_say_which_level_a_key_is():
    responses = pd.concat(
        {
            "AAPL": pd.DataFrame({"AAPL": [1.0], "MSFT": [2.0]}),
            "MSFT": pd.DataFrame({"AAPL": [3.0], "MSFT": [4.0]}),
        },
        axis=1,
        names=["Shock", "Response"],
    )
    responses.index.name = "Horizon"

    records = _df_to_records(responses)

    assert records[0] == {
        "Horizon": 0,
        "Response": "AAPL",
        "Shock AAPL": 1.0,
        "Shock MSFT": 3.0,
    }


def test_skipped_periods_are_named_in_a_note():
    from financetoolkit.mcp_server.registry.formatting_model import skipped_periods_note

    result = pd.DataFrame(
        {"United States": [1.0, None, 2.0, None], "Germany": [1.0, 1.5, 2.0, 2.5]},
        index=pd.period_range("2025-09", periods=4, freq="M"),
    )

    note = skipped_periods_note(result)

    # Only the gap between values counts, not a month not published yet.
    assert note.startswith("No value for United States (2025-10)")
    assert skipped_periods_note(result[["Germany"]]) is None
    assert skipped_periods_note(result.reset_index(drop=True)) is None


def test_a_series_without_a_period_parameter_is_converted_to_the_period():
    from financetoolkit.mcp_server.registry.formatting_model import convert_to_period

    weeks = pd.date_range("2023-01-05", "2024-12-26", freq="W-THU")
    rates = pd.DataFrame({"United States": range(len(weeks))}, index=weeks)

    yearly = convert_to_period(rates, "yearly")

    # The last week of each year.
    assert yearly["United States"].tolist() == [51, len(weeks) - 1]
    assert yearly.index.tolist() == [pd.Period("2023", "Y"), pd.Period("2024", "Y")]

    # Already coarser, or not a time series: nothing to convert.
    assert convert_to_period(yearly, "monthly") is None
    assert convert_to_period(pd.DataFrame({"a": [1, 2]}), "yearly") is None


def test_days_a_ticker_did_not_trade_are_named():
    from financetoolkit.mcp_server.registry.formatting_model import filled_day_notes

    days = pd.period_range("2026-10-01", periods=4, freq="D")
    prices = pd.concat(
        {
            "Close": pd.DataFrame(
                {"^KS11": [6971.35, 7003.74, 7003.74, 6941.39], "SPY": [1, 2, 3, 4]},
                index=days,
            ),
            "Volume": pd.DataFrame(
                {"^KS11": [292300, 239000, 0, 269800], "SPY": [5, 5, 5, 5]}, index=days
            ),
            "Return": pd.DataFrame(
                {"^KS11": [None, 0.0046, None, -0.0089], "SPY": [None, 1.0, 0.5, 0.33]},
                index=days,
            ),
        },
        axis=1,
    )

    notes = filled_day_notes(prices)

    assert len(notes) == 1
    assert notes[0].startswith("^KS11 did not trade on 2026-10-03 (")
    assert filled_day_notes(prices["Close"]) == []


def test_an_answer_without_data_keeps_its_notes_and_sources():
    import json

    from financetoolkit.mcp_server.registry.formatting_model import format_result

    answer = json.loads(
        format_result(
            pd.DataFrame({"a": [None]}),
            reason="no key",
            notes=["`perid` is not a parameter"],
            sources=["Eurostat"],
        )
    )

    assert answer == {
        "error": "No data available.",
        "reason": "no key",
        "_notes": ["`perid` is not a parameter"],
        "_sources": ["Eurostat"],
    }


def test_rows_of_two_levels_with_columns_of_two_levels_become_records():
    rows = pd.MultiIndex.from_product(
        [pd.period_range("2022", periods=2, freq="Y"), ["Mean", "Std"]]
    )
    columns = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Close", "MSFT")])
    table = pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], index=rows, columns=columns
    )

    records = _df_to_records(table)

    assert records[:2] == [
        {"date": "2022", "metric": "Mean", "ticker": "AAPL", "Close": 0.0},
        {"date": "2022", "metric": "Mean", "ticker": "MSFT", "Close": 1.0},
    ]
    assert len(records) == 8  # noqa: PLR2004


def test_a_week_belongs_to_the_month_it_ends_in():
    from financetoolkit.mcp_server.registry.formatting_model import convert_to_period

    weeks = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.PeriodIndex(["2024-01-05", "2024-01-12", "2024-02-02"], freq="W-FRI"),
    )

    monthly = convert_to_period(weeks, "monthly")

    # The week of 27 January to 2 February is February's, the one ending 5 January
    # January's.
    assert monthly.index.tolist() == [
        pd.Period("2024-01", "M"),
        pd.Period("2024-02", "M"),
    ]
    assert monthly["x"].tolist() == [2.0, 3.0]


def test_a_week_to_sunday_belongs_to_the_month_of_its_last_weekday():
    from financetoolkit.mcp_server.registry.formatting_model import convert_to_period

    weeks = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.period_range("2026-01-12", periods=4, freq="W-SUN"),
    )

    monthly = convert_to_period(weeks, "monthly")

    # The week of 26 January to Sunday 1 February closes on Friday 30 January.
    assert monthly.to_dict() == {
        pd.Period("2026-01", "M"): 3.0,
        pd.Period("2026-02", "M"): 4.0,
    }
