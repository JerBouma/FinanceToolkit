"""Formatting and Column Selection Tests"""

import pandas as pd

from financetoolkit.mcp_server.formatting_model import _df_to_records

# pylint: disable=missing-function-docstring


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
