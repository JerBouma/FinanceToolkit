"""Registry Description Tests"""

# pylint: disable=missing-function-docstring

import asyncio

import yaml

from financetoolkit.mcp_server import mcp_controller


def test_long_tool_descriptions_are_complete():
    """The guidance at the end of a long description reaches the client."""
    with open("financetoolkit/mcp_server/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    configured = {
        group["tool_name"]: group["description"].strip()
        for group in config["tool_groups"]
        if group.get("description")
    }
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("macroeconomics", "rates"):
        assert configured[name][-200:] in tools[name].description


def test_the_server_reports_the_finance_toolkit_version():
    """Clients show the server's version, which is the package's, not the MCP SDK's."""
    from importlib import metadata

    options = mcp_controller.mcp._mcp_server.create_initialization_options()

    assert options.server_version == metadata.version("financetoolkit")


def test_tickers_without_data_are_named_in_the_answer():
    from financetoolkit.mcp_server.diagnostics_model import missing_data_notes

    assert missing_data_notes(
        [
            "No quote data is available for EURUSD=X, BTC-USD.",
            "No quote data is available for EURUSD=X, BTC-USD.",
            "Something unrelated",
        ]
    ) == ["No quote data is available for EURUSD=X, BTC-USD."]


def test_the_search_tools_say_they_belong_to_the_finance_toolkit():
    tools = {tool.name: tool for tool in asyncio.run(mcp_controller.mcp.list_tools())}

    for name in ("search_metric_categories", "search_by_category", "search_metrics"):
        assert tools[name].description.startswith("Finance Toolkit:")

    # Looking up instruments is left to the Finance Database MCP server, whose tools do
    # not share a name with these.
    assert "search_instruments" not in tools
    assert "search_categories" not in tools


def test_a_category_of_one_tool_is_named_after_it():
    """An agent can go from a category straight to its tool."""
    from collections import defaultdict

    with open("financetoolkit/mcp_server/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tools_per_category = defaultdict(list)

    for group in config["tool_groups"]:
        category = group.get("index_category") or group["module_name"]
        tools_per_category[category].append(group["tool_name"])

    for category, tools in tools_per_category.items():
        if len(tools) == 1:
            assert tools == [category]
        assert category in config["category_descriptions"]


def test_an_unnamed_result_is_named_after_its_method():
    from financetoolkit.mcp_server.provider_model import readable_name

    assert readable_name("get_value_at_risk") == "Value at Risk"
    assert readable_name("get_return_on_equity") == "Return on Equity"


def test_a_large_answer_is_cut_to_its_most_recent_rows():
    import json

    import numpy as np
    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    prices = pd.DataFrame(
        {"Close": np.arange(5000.0)},
        index=pd.period_range("2006-01-01", periods=5000, freq="D"),
    )

    answer = json.loads(format_result(prices, max_characters=5000))

    assert len(json.dumps(answer["data"])) <= 5000  # noqa: PLR2004
    assert answer["data"][-1]["Close"] == 4999  # noqa: PLR2004
    assert "last" in answer["_notes"][0] and "start_date" in answer["_notes"][0]

    # A small answer is left as it is.
    assert "_notes" not in json.loads(
        format_result(prices.tail(3), max_characters=5000)
    )


def test_a_very_wide_answer_keeps_its_first_columns():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    curve = pd.DataFrame(
        [[0.02] * 3000] * 3, columns=[f"{year}Y" for year in range(1, 3001)]
    )

    answer = json.loads(format_result(curve, max_characters=4000))

    assert len(json.dumps(answer["data"])) <= 4000  # noqa: PLR2004
    assert "columns" in answer["_notes"][0]


def test_a_long_text_is_shortened_rather_than_dropped():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    transcripts = pd.DataFrame(
        {"Date": ["2026-07-30", "2026-07-29"], "Transcript": ["word " * 20000] * 2},
        index=["AAPL", "MSFT"],
    )

    answer = json.loads(format_result(transcripts, max_characters=10000))

    assert len(json.dumps(answer["data"])) <= 10000  # noqa: PLR2004
    assert all(record["Transcript"].endswith("...") for record in answer["data"])
    assert "texts shortened" in answer["_notes"][0]


def test_columns_of_three_levels_are_joined_into_one_name():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    draws = pd.DataFrame(
        [[0.01, 0.02]],
        columns=pd.MultiIndex.from_tuples(
            [("AAPL", "MSFT", "AAPL"), ("AAPL", "MSFT", "MSFT")]
        ),
    )

    record = json.loads(format_result(draws))[0]

    assert record["AAPL | MSFT | AAPL"] == 0.01  # noqa: PLR2004
    assert record["AAPL | MSFT | MSFT"] == 0.02  # noqa: PLR2004


def test_limit_and_the_cut_keep_the_latest_periods_of_a_sideways_table():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result, select_rows

    ratios = pd.DataFrame(
        [list(range(30))],
        index=["AAPL"],
        columns=pd.period_range("1997", periods=30, freq="Y"),
    )

    assert select_rows(ratios, -2).columns.astype(str).tolist() == ["2025", "2026"]
    assert select_rows(ratios, 1).columns.astype(str).tolist() == ["1997"]

    answer = json.loads(format_result(ratios, max_characters=60))

    assert "2026" in answer["data"][0] and "1997" not in answer["data"][0]
    assert "last" in answer["_notes"][0] and "periods" in answer["_notes"][0]


def test_floats_lose_their_binary_artefacts_and_infinity_becomes_null():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    weights = pd.DataFrame({"Weight": [0.08317952249999999, float("inf")]})

    assert json.loads(format_result(weights)) == [
        {"index": 0, "Weight": 0.0831795225},
        {"index": 1, "Weight": None},
    ]
