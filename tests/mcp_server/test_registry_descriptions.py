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


def test_a_factor_model_has_a_row_per_ticker_and_the_coefficients_as_keys():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    model = pd.DataFrame(
        [[0.1, 1.2, 0.2, 0.9]],
        index=pd.period_range("2026", periods=1, freq="Y"),
        columns=pd.MultiIndex.from_product(
            [["ASML.AS", "MSFT"], ["Intercept", "Mkt-RF Slope"]]
        ),
    )

    records = json.loads(format_result(model))

    assert records[0]["ticker"] == "ASML.AS"
    assert records[0]["Mkt-RF Slope"] == 1.2  # noqa: PLR2004


def test_an_unnamed_level_of_countries_is_called_country():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    money = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=pd.period_range("2025", periods=1, freq="Y"),
        columns=pd.MultiIndex.from_product([["Japan", "United States"], ["M0", "M1"]]),
    )

    records = json.loads(format_result(money))

    assert records[0]["country"] == "Japan" and records[0]["M1"] == 2.0  # noqa: PLR2004


def test_the_last_periods_skip_one_most_rows_have_not_reported():
    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import select_rows

    caps = pd.DataFrame(
        {"2024": [1.0, 2.0, 3.0], "2025": [1.5, 2.5, 3.5], "2026": [None, None, 4.0]},
        index=["AAPL", "KO", "MSFT"],
    )
    caps.columns = pd.PeriodIndex(caps.columns, freq="Y")

    assert select_rows(caps, -1).columns.astype(str).tolist() == ["2025"]


def test_every_parameter_has_a_real_description():
    import asyncio
    import importlib
    import sys

    # Imported afresh, as another test leaves the server modules built on a stub.
    for module in (
        "financetoolkit.mcp_server.mcp_controller",
        "financetoolkit.mcp_server.registry_controller",
        "financetoolkit.mcp_server.tools_model",
    ):
        sys.modules.pop(module, None)
    mcp_controller = importlib.import_module("financetoolkit.mcp_server.mcp_controller")

    placeholders = [
        f"{tool.name}.{name}"
        for tool in asyncio.run(mcp_controller.mcp.list_tools())
        for name, schema in tool.inputSchema.get("properties", {}).items()
        if schema.get("description", "Value for").startswith("Value for")
    ]

    assert placeholders == []


def test_docstring_arguments_are_read_without_their_defaults():
    from financetoolkit.mcp_server.registry_controller import _docstring_arguments

    docstring = """Calculate something.

Args:
    window (int, optional): The number of periods. Defaults to 14.
    put_option (bool): Whether to use puts
        instead of calls.

Returns:
    pd.Series: The result.
"""

    assert _docstring_arguments(docstring) == {
        "window": "The number of periods.",
        "put_option": "Whether to use puts instead of calls.",
    }


def test_the_last_rows_of_a_series_are_the_last_ones_with_data():
    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import select_rows

    poverty = pd.Series(
        [0.18, 0.18, None, None],
        index=pd.period_range("2022", periods=4, freq="Y"),
    )

    assert select_rows(poverty, -1).index.astype(str).tolist() == ["2023"]


def test_symbols_with_an_exchange_suffix_are_called_tickers():
    import json

    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import format_result

    ratios = pd.DataFrame({"2025": [37.1, 33.1]}, index=["ASML.AS", "SAP.DE"])
    ratios.columns = pd.PeriodIndex(ratios.columns, freq="Y")
    risk = pd.DataFrame(
        {"Value at Risk": [-0.18, -0.06]}, index=["BTCUSD", "Benchmark"]
    )

    assert json.loads(format_result(ratios))[0]["ticker"] == "ASML.AS"
    assert json.loads(format_result(risk))[0]["ticker"] == "BTCUSD"


def test_the_server_tells_a_client_how_to_use_it():
    import importlib
    import sys

    for module in (
        "financetoolkit.mcp_server.mcp_controller",
        "financetoolkit.mcp_server.registry_controller",
        "financetoolkit.mcp_server.tools_model",
    ):
        sys.modules.pop(module, None)
    mcp_controller = importlib.import_module("financetoolkit.mcp_server.mcp_controller")

    assert "search_metrics" in mcp_controller.mcp.instructions
    assert "decimal fractions" in mcp_controller.mcp.instructions


def test_limit_keeps_whole_periods_of_rows_per_date_and_item():
    import pandas as pd

    from financetoolkit.mcp_server.formatting_model import select_rows

    correlations = pd.DataFrame(
        {"SMB": [1.0, 0.2, 1.0, 0.3], "HML": [0.2, 1.0, 0.3, 1.0]},
        index=pd.MultiIndex.from_product(
            [pd.period_range("2025", periods=2, freq="Y"), ["SMB", "HML"]]
        ),
    )

    latest = select_rows(correlations, -1)

    assert latest.index.get_level_values(1).tolist() == ["SMB", "HML"]
    assert latest.index.get_level_values(0).astype(str).unique().tolist() == ["2026"]


def test_the_slow_egarch_fits_are_not_offered():
    import asyncio
    import importlib
    import sys

    for module in (
        "financetoolkit.mcp_server.mcp_controller",
        "financetoolkit.mcp_server.registry_controller",
        "financetoolkit.mcp_server.tools_model",
    ):
        sys.modules.pop(module, None)
    mcp_controller = importlib.import_module("financetoolkit.mcp_server.mcp_controller")

    risk = next(
        tool
        for tool in asyncio.run(mcp_controller.mcp.list_tools())
        if tool.name == "risk"
    )
    indicators = risk.inputSchema["properties"]["indicator"]["enum"]

    assert "get_garch" in indicators
    assert not [name for name in indicators if "egarch" in name]
