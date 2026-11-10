"""Formatting and Column Selection Tests"""

import pandas as pd

from financetoolkit.mcp_server.formatting_model import _df_to_records


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
    from financetoolkit.mcp_server.formatting_model import skipped_periods_note

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
    from financetoolkit.mcp_server.formatting_model import convert_to_period

    weeks = pd.date_range("2023-01-05", "2024-12-26", freq="W-THU")
    rates = pd.DataFrame({"United States": range(len(weeks))}, index=weeks)

    yearly = convert_to_period(rates, "yearly")

    # The last week of each year.
    assert yearly["United States"].tolist() == [51, len(weeks) - 1]
    assert yearly.index.tolist() == [pd.Period("2023", "Y"), pd.Period("2024", "Y")]

    # Already coarser, or not a time series: nothing to convert.
    assert convert_to_period(yearly, "monthly") is None
    assert convert_to_period(pd.DataFrame({"a": [1, 2]}), "yearly") is None
