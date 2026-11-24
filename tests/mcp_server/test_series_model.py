"""Tests of the indicator references an econometrics call can analyze (the series parameter)."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.registry import series_model
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider


def test_references_are_parsed():
    references = series_model.parse_series_references(
        "valuation:get_price_to_earnings_ratio:AAPL, MSFT; get_inflation_rate:Germany\n"
        "rates:government_bond_yield_curve:United States:10Y"
    )

    assert [reference.tool for reference in references] == ["valuation", None, "rates"]
    assert references[0].entities == ["AAPL", "MSFT"]
    assert references[1].indicator == "get_inflation_rate"
    assert references[2].indicator == "get_government_bond_yield_curve"
    assert references[2].column == "10Y"

    with pytest.raises(ValueError, match="tool:indicator:entities"):
        series_model.parse_series_references("inflation")


def test_answers_become_series_by_date():
    years = pd.period_range("2020", periods=3, freq="Y")
    ratios = pd.DataFrame(
        [[30.0, 28.0, 25.0], [35.0, 33.0, 31.0]], index=["AAPL", "MSFT"], columns=years
    )
    reference = series_model.parse_series_references("valuation:get_pe:AAPL")[0]

    # The tickers are rows and the years columns; only the requested ticker is kept.
    assert list(series_model.to_series_columns(ratios, reference).columns) == ["AAPL"]

    days = pd.period_range("2026-01-01", periods=3, freq="D")
    curve = pd.DataFrame(
        np.full((3, 2), 0.04),
        index=days,
        columns=pd.MultiIndex.from_tuples(
            [("United States", "2Y"), ("United States", "10Y")]
        ),
    )
    ten_year = series_model.parse_series_references(
        "rates:get_government_bond_yield_curve:United States:10Y"
    )[0]
    assert list(series_model.to_series_columns(curve, ten_year).columns) == [
        "United States 10Y"
    ]

    missing = ten_year._replace(column="99Y")
    with pytest.raises(ValueError, match="Choose from: 10Y, 2Y"):
        series_model.to_series_columns(curve, missing)


def test_shared_names_get_the_indicator_added():
    days = pd.period_range("2026-01-01", periods=2, freq="D")
    first, second = series_model.parse_series_references(
        "valuation:get_price_to_earnings_ratio:AAPL; valuation:get_price_to_book_ratio:AAPL"
    )
    combined = series_model.combine_series(
        [
            (first, pd.DataFrame({"AAPL": [30.0, 31.0]}, index=days)),
            (second, pd.DataFrame({"AAPL": [40.0, 42.0]}, index=days)),
        ]
    )

    assert list(combined.columns) == [
        "AAPL Price To Earnings Ratio",
        "AAPL Price To Book Ratio",
    ]


@pytest.fixture
def indicator_answers(monkeypatch):
    """Answer every indicator with a monthly series per requested country."""
    calls = []
    rng = np.random.default_rng(0)
    months = pd.period_range("2010-01", periods=180, freq="M")

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append((module_name, method_name, category, kwargs))
        countries = kwargs.get("countries") or []
        return pd.DataFrame(
            {
                country: 0.03 + rng.normal(0, 0.002, len(months)).cumsum() / 10
                for country in countries
            },
            index=months,
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call_econometrics(**arguments):
    result = asyncio.run(mcp_controller.mcp.call_tool("econometrics", arguments))
    return str(result)


def test_an_econometrics_call_analyzes_other_indicators(indicator_answers):
    answer = call_econometrics(
        indicator="get_unit_root_test",
        series="macroeconomics:get_inflation_rate:USA,Germany",
        start_date="2010-01-01",
    )

    # The indicator was retrieved through its own tool, with the country names resolved.
    module_name, method_name, category, kwargs = indicator_answers[-1]
    assert (module_name, method_name, category) == (
        "economics",
        "get_inflation_rate",
        "standalone",
    )
    assert kwargs["countries"] == ["United States", "Germany"]
    assert kwargs["start_date"] == "2010-01-01"
    assert "ADF Statistic" in answer and "Germany" in answer


def test_series_errors_are_explained(indicator_answers):
    assert "names no known indicator" in call_econometrics(
        indicator="get_ols", series="get_imaginary_metric:Germany"
    )
    assert "is not an indicator of `rates`" in call_econometrics(
        indicator="get_ols", series="rates:get_inflation_rate:Germany"
    )
    # Without series, the tickers are still required.
    assert "requires a `tickers` parameter" in call_econometrics(indicator="get_ols")


def test_changes_none_reaches_econometrics_as_none(indicator_answers, monkeypatch):
    received = []

    def fake_with_data(
        method_name, data, changes="percentage", **kwargs
    ):  # noqa: ARG001
        received.append(changes)
        return pd.DataFrame({"Value": [1.0]})

    monkeypatch.setattr(
        ToolkitProvider, "call_econometrics_with_data", staticmethod(fake_with_data)
    )
    series = "macroeconomics:get_inflation_rate:Germany,France"

    call_econometrics(indicator="get_ols", series=series, changes="none")
    call_econometrics(indicator="get_ols", series=series)

    # A tool argument is text: 'none' is changes=None, and no argument is the default.
    assert received == [None, "percentage"]


def test_a_reference_needs_tickers_or_countries_and_a_call_a_few_references():
    with pytest.raises(ValueError, match="names no tickers or countries"):
        series_model.parse_series_references("get_inflation_rate: , ")

    many = ";".join(["get_inflation_rate:Germany"] * 21)
    with pytest.raises(ValueError, match="at most 20"):
        series_model.parse_series_references(many)


def test_the_period_reaches_the_indicators_that_take_one(indicator_answers):
    answer = call_econometrics(
        indicator="get_level_significance",
        series="rates:get_long_term_interest_rate:United States; get_government_bond_yield:Germany",
        start_date="2010-01-01",
        period="monthly",
    )

    # Both indicators take a period, so both are asked for the monthly series.
    assert [kwargs.get("period") for *_, kwargs in indicator_answers[-2:]] == [
        "monthly",
        "monthly",
    ]
    assert "Highest Since" in answer and "Germany" in answer

    # Without a period every indicator keeps its own default.
    call_econometrics(
        indicator="get_level_significance",
        series="rates:get_long_term_interest_rate:United States",
        start_date="2010-01-01",
    )
    assert "period" not in indicator_answers[-1][3]


def test_an_indicator_not_published_at_the_period_is_read_at_its_own(
    indicator_answers, monkeypatch
):
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs.get("period"))
        if kwargs.get("period") == "weekly":
            return pd.Series(dtype=object)
        months = pd.period_range("2010-01", periods=60, freq="M")
        return pd.DataFrame({"Germany": np.linspace(0.01, 0.03, 60)}, index=months)

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)

    answer = call_econometrics(
        indicator="get_level_significance",
        series="macroeconomics:get_inflation_rate:Germany",
        start_date="2010-01-01",
        period="weekly",
    )

    assert calls == ["weekly", None]
    assert "Highest Since" in answer


def test_the_economic_calendar_is_read_as_series(monkeypatch):
    calls = []
    months = pd.period_range("2021-05", periods=66, freq="M")

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append(kwargs)
        return pd.DataFrame(
            {
                ("China", "Inflation Rate YoY"): np.linspace(0.013, 0.007, 66),
                ("China", "Producer Price Index YoY"): np.linspace(0.09, 0.057, 66),
                ("China", "Producer Price Index YoY Estimate"): np.linspace(
                    0.08, 0.053, 66
                ),
            },
            index=months,
        )

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)

    answer = call_econometrics(
        indicator="get_streaks",
        series=(
            "macroeconomics:get_economic_calendar:China:Producer Price Index YoY; "
            "macroeconomics:get_economic_calendar:China:Inflation Rate YoY"
        ),
        spread_over="China Inflation Rate YoY",
        start_date="2021-06-01",
        period="monthly",
    )

    # The calendar is asked for its series, and the event is picked by its name only.
    assert all(kwargs.get("as_series") is True for kwargs in calls)
    assert "China Producer Price Index YoY - China Inflation Rate YoY" in answer
    assert "Estimate" not in answer
    assert "Run Length" in answer


def test_a_daily_analysis_asks_statement_ratios_for_their_daily_values(
    indicator_answers,
):
    call_econometrics(
        indicator="get_level_significance",
        series="valuation:get_price_to_earnings_ratio:NVDA",
        period="daily",
        start_date="2016-01-01",
    )

    # The P/E has no period; its daily values are asked for with show_daily.
    _, method_name, _, kwargs = indicator_answers[0]
    assert method_name == "get_price_to_earnings_ratio"
    assert kwargs["show_daily"] is True

    indicator_answers.clear()
    call_econometrics(
        indicator="get_level_significance",
        series="valuation:get_price_to_earnings_ratio:NVDA",
        period="yearly",
        start_date="2016-01-01",
    )

    assert "show_daily" not in indicator_answers[0][3]


def test_a_reference_passes_arguments_to_its_indicator(indicator_answers):
    reference = series_model.parse_series_references(
        "rates:get_outstanding_mortgage_rates:United States:Below 4%:cumulative=true"
    )[0]
    assert reference.entities == ["United States"]
    assert reference.column == "Below 4%"
    assert reference.arguments == {"cumulative": "true"}

    # The argument reaches the indicator typed as it takes it.
    call_econometrics(
        indicator="get_level_significance",
        series="rates:get_outstanding_mortgage_rates:United States:cumulative=true",
        start_date="2013-01-01",
    )
    module_name, method_name, _, kwargs = indicator_answers[-1]
    assert (module_name, method_name) == (
        "economics",
        "get_outstanding_mortgage_rates",
    )
    assert kwargs["cumulative"] is True

    # An argument the indicator does not take is named.
    answer = call_econometrics(
        indicator="get_level_significance",
        series="rates:get_outstanding_mortgage_rates:United States:cumulativ=true",
    )
    assert "takes no argument cumulativ" in answer


def test_a_reference_takes_only_the_arguments_its_tool_exposes(indicator_answers):
    series = "rates:get_outstanding_mortgage_rates:United States"

    # Parameters of the method the tool does not offer, or that are not parameters at all.
    for argument in ("self=1", "rounding=1"):
        answer = call_econometrics(
            indicator="get_level_significance", series=f"{series}:{argument}"
        )
        assert f"takes no argument {argument.split('=')[0]}" in answer
    assert "multiple values" not in answer

    # Parameters the econometrics call sets for every reference.
    answer = call_econometrics(
        indicator="get_level_significance", series=f"{series}:start_date=2020-01-01"
    )
    assert "start_date is set by the tool, pass it as a tool parameter" in answer

    # A name given twice, and one written with a capital.
    answer = call_econometrics(
        indicator="get_level_significance",
        series=f"{series}:cumulative=true:cumulative=false",
    )
    assert "gives cumulative more than once" in answer
    answer = call_econometrics(
        indicator="get_level_significance", series=f"{series}:Cumulative=true"
    )
    assert "Write it as 'cumulative='" in answer
    assert not indicator_answers


def test_the_reason_a_referenced_indicator_has_no_values_is_given(monkeypatch):
    from financetoolkit.utilities.logger_model import get_logger

    def failing(self, module_name, method_name, category, **kwargs):  # noqa: ARG001
        get_logger().error(
            "get_relative_strength_index could not be calculated. ValueError: Method "
            "must be one of standard, connors, laguerre, dynamic, intraday, not bogus."
        )
        return pd.DataFrame()

    monkeypatch.setattr(ToolkitProvider, "call_method", failing)

    answer = call_econometrics(
        indicator="get_percentile",
        series="momentum:get_relative_strength_index:AAPL:method=bogus",
    )

    assert "Method must be one of standard, connors" in answer
    assert "returned no values over time" not in answer


def test_a_requested_ticker_that_is_the_benchmark_is_kept():
    days = pd.period_range("2026-01-01", periods=3, freq="D")
    # The Toolkit holds SPY, the benchmark, under "Benchmark" rather than as a ticker.
    historical = pd.DataFrame(
        np.full((3, 6), 0.01),
        index=days,
        columns=pd.MultiIndex.from_product(
            [["Close", "Return"], ["AAPL", "MSFT", "Benchmark"]]
        ),
    )
    reference = series_model.parse_series_references(
        "market_data:get_historical_data:AAPL,MSFT,SPY:Return"
    )[0]

    assert list(
        series_model.to_series_columns(historical, reference, benchmark="SPY").columns
    ) == ["Return AAPL", "Return MSFT", "Return SPY"]
    # A benchmark that was not asked for stays out.
    without_spy = reference._replace(entities=["AAPL", "MSFT"])
    assert list(
        series_model.to_series_columns(historical, without_spy, benchmark="SPY").columns
    ) == ["Return AAPL", "Return MSFT"]


def test_a_requested_ticker_without_data_is_named_in_a_warning(caplog):
    days = pd.period_range("2026-01-01", periods=3, freq="D")
    historical = pd.DataFrame(
        np.full((3, 2), 0.01),
        index=days,
        columns=pd.MultiIndex.from_product([["Return"], ["AAPL", "MSFT"]]),
    )
    reference = series_model.parse_series_references(
        "market_data:get_historical_data:AAPL,MSFT,XYZ:Return"
    )[0]

    with caplog.at_level("WARNING"):
        columns = series_model.to_series_columns(historical, reference).columns

    assert list(columns) == ["Return AAPL", "Return MSFT"]
    assert (
        "No data is available for XYZ in get_historical_data, so it is left out of the "
        "series." in caplog.text
    )


def test_the_answer_of_a_single_ticker_is_named_after_it():
    years = pd.period_range("2020", periods=3, freq="Y")
    statements = {
        ticker: pd.DataFrame(
            [[100.0, 110.0, 120.0], [10.0, 11.0, 12.0]],
            index=["Revenue", "Net Income"],
            columns=years,
        )
        for ticker in ("AAPL", "MSFT")
    }
    frames = [
        (reference, series_model.to_series_columns(statements[ticker], reference))
        for ticker, reference in zip(
            ("AAPL", "MSFT"),
            series_model.parse_series_references(
                "market_data:get_income_statement:AAPL:Revenue;"
                "market_data:get_income_statement:MSFT:Revenue"
            ),
            strict=True,
        )
    ]

    # Two references of one ticker each are two series, as AAPL,MSFT in one reference is.
    assert list(series_model.combine_series(frames).columns) == [
        "AAPL Revenue",
        "MSFT Revenue",
    ]


@pytest.fixture
def ticker_answers(monkeypatch):
    """Answer historical data and statements as the Toolkit holds them, and record the
    series the regression receives."""
    days = pd.period_range("2024-01-01", periods=120, freq="D")
    years = pd.period_range("2015", periods=10, freq="Y")
    rng = np.random.default_rng(1)
    received = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        tickers = kwargs["tickers"]
        if method_name == "get_income_statement":
            return pd.DataFrame(
                rng.normal(100, 10, (2, len(years))),
                index=["Revenue", "Net Income"],
                columns=years,
            )
        held = [
            "Benchmark" if ticker == kwargs["benchmark_ticker"] else ticker
            for ticker in tickers
        ]
        return pd.DataFrame(
            rng.normal(0, 0.01, (len(days), len(held))),
            index=days,
            columns=pd.MultiIndex.from_product([["Return"], held]),
        )

    def fake_with_data(
        method_name, data, changes="percentage", **kwargs
    ):  # noqa: ARG001
        received.append(list(data.columns))
        return pd.DataFrame({"Coefficient": [1.0]}, index=["Intercept"])

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    monkeypatch.setattr(
        ToolkitProvider, "call_econometrics_with_data", staticmethod(fake_with_data)
    )
    return received


def test_the_econometrics_tool_keeps_the_benchmark_and_names_single_tickers(
    ticker_answers,
):
    call_econometrics(
        indicator="get_ols",
        series="market_data:get_historical_data:AAPL,MSFT,SPY:Return",
        dependent_ticker="Return AAPL",
        changes="none",
    )
    answer = call_econometrics(
        indicator="get_ols",
        series="market_data:get_income_statement:AAPL:Revenue;"
        "market_data:get_income_statement:MSFT:Revenue",
    )

    assert ticker_answers == [
        ["Return AAPL", "Return MSFT", "Return SPY"],
        ["AAPL Revenue", "MSFT Revenue"],
    ]
    assert "appear more than once" not in answer
