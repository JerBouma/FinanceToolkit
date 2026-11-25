"""Tests for the limits that keep one MCP call to a size a shared server can hold."""

import asyncio
import json
import threading
import time
import types

import numpy as np
import pandas as pd
import pytest
import requests

from financetoolkit.mcp_server import mcp_controller
from financetoolkit.mcp_server.analytics import analytics_model
from financetoolkit.mcp_server.registry.coercion_model import (
    refused_value_problem,
    size_limit_problem,
)
from financetoolkit.mcp_server.registry.formatting_model import format_result
from financetoolkit.mcp_server.registry.provider_model import ToolkitProvider
from financetoolkit.utilities import requests_model
from tests.mcp_server.answers import read_answer


@pytest.fixture
def captured(monkeypatch):
    """Record what the tool wrapper passes to the provider instead of fetching data."""
    calls = []

    def fake_call_method(
        self, module_name, method_name, category, **kwargs
    ):  # noqa: ARG001
        calls.append((method_name, kwargs))
        return pd.DataFrame({"Value": [1.0]})

    monkeypatch.setattr(ToolkitProvider, "call_method", fake_call_method)
    return calls


def call(tool, **arguments):
    return str(asyncio.run(mcp_controller.mcp.call_tool(tool, arguments)))


def test_a_simulation_larger_than_the_server_allows_is_refused(captured):
    answer = call(
        "risk",
        indicator="get_copula_simulation",
        tickers="AAPL,MSFT",
        simulations=50_000,
    )

    assert "`simulations` may be at most 10,000" in answer
    assert not captured

    call(
        "risk",
        indicator="get_copula_simulation",
        tickers="AAPL,MSFT",
        simulations=500,
    )
    assert captured[-1][1]["simulations"] == 500  # noqa: PLR2004


def test_parameters_computed_together_are_limited_with_the_indicator_defaults(captured):
    # 100,000 paths of the indicator's default 100 steps is over the product limit,
    # although each value on its own is within its maximum.
    answer = call(
        "options",
        indicator="get_monte_carlo_option_price",
        tickers="AAPL",
        simulations=100_000,
    )

    assert "`simulations` times `time_steps` may be at most 5,000,000" in answer
    assert not captured


def test_the_backtest_lookback_is_limited(captured):
    assert "`lookback` may be at most 504" in call(
        "portfolio",
        indicator="get_optimization_backtest",
        tickers="AAPL,MSFT",
        lookback=756,
    )
    assert not captured


def test_the_resamples_are_limited(captured):
    assert "`resamples` may be at most 1,000" in call(
        "portfolio",
        indicator="get_resampled_efficient_weights",
        tickers="AAPL,MSFT",
        resamples=5000,
    )
    assert not captured


def test_the_short_volume_history_is_limited_and_never_unbounded(captured):
    assert "`history_days` may be at most 366" in call(
        "market_data", indicator="get_short_volume", tickers="AAPL", history_days=6000
    )
    assert not captured

    # Left out (or null) it is the MCP default, rather than the whole date range.
    call("market_data", indicator="get_short_volume", tickers="AAPL", history_days=None)
    assert captured[-1][1]["history_days"] == 92  # noqa: PLR2004


def test_the_slow_egarch_fits_are_refused_and_the_other_garch_methods_offered(captured):
    for indicator in ("get_garch", "get_garch_forecast", "get_garch_parameters"):
        answer = call("risk", indicator=indicator, tickers="AAPL", method="EGARCH")
        assert "`method='EGARCH'` is not available on this server" in answer
        assert "method='gjr'" in answer
    assert not captured

    call("risk", indicator="get_garch", tickers="AAPL", method="gjr")
    assert captured[-1][0] == "get_garch"
    assert captured[-1][1]["method"] == "gjr"


def test_refused_values_only_apply_to_their_indicators_and_parameter():
    limits = {
        "refused_values": [
            {
                "methods": ["get_a"],
                "parameter": "method",
                "values": ["slow"],
                "reason": "x",
            }
        ]
    }

    assert refused_value_problem("get_a", {"method": " Slow "}, limits)
    assert refused_value_problem("get_a", {"method": "fast"}, limits) is None
    assert refused_value_problem("get_b", {"method": "slow"}, limits) is None
    assert refused_value_problem("get_a", {"variant": "slow"}, limits) is None
    assert refused_value_problem("get_a", {"method": "slow"}, {}) is None


def test_size_limits_check_every_value_of_a_list():
    limits = {"maximums": {"lags": 100}, "products": []}

    assert size_limit_problem({"lags": [1, 5, 500]}, {}, limits)
    assert size_limit_problem({"lags": [1, 5, 50]}, {}, limits) is None
    # A product leaves out an indicator that does not take every parameter of it.
    product = {"maximums": {}, "products": [{"parameters": ["a", "b"], "maximum": 10}]}
    assert size_limit_problem({"a": 100}, {}, product) is None
    assert size_limit_problem({"a": 100}, {"b": 1}, product)


def test_tickers_and_benchmarks_are_validated_before_any_request(captured):
    for tickers in ("AAPL&apikey=X", "../v7/finance", "A" * 30):
        assert "invalid ticker" in call(
            "valuation", indicator="get_price_to_earnings_ratio", tickers=tickers
        )
    assert "Invalid `benchmark_ticker`" in call(
        "performance",
        indicator="get_sharpe_ratio",
        tickers="AAPL",
        benchmark_ticker="SPY?x=1",
    )
    assert not captured

    call(
        "performance",
        indicator="get_sharpe_ratio",
        tickers="ASML.AS,BTC-USD,EURUSD=X",
        benchmark_ticker="^GSPC",
    )
    assert captured


def test_series_references_are_held_to_the_entity_limit(captured):
    tickers = ",".join(f"T{number}" for number in range(30))
    answer = call(
        "econometrics",
        indicator="get_ols",
        series=(
            f"valuation:get_price_to_earnings_ratio:{tickers};valuation:get_price_to_book_ratio:{tickers}"
        ),
    )

    assert "60 tickers and countries together; at most 50" in answer
    assert "invalid ticker" in call(
        "econometrics",
        indicator="get_ols",
        series="valuation:get_price_to_earnings_ratio:AAPL&x=1",
    )
    assert not captured


def test_series_references_are_held_to_the_limits_of_their_own_tool(captured):
    """An indicator analyzed through series= is refused what its own tool refuses."""
    # The slow EGARCH fit, which the risk tool refuses by value.
    answer = call(
        "econometrics",
        indicator="get_percentile",
        series="risk:get_garch:AAPL:method=egarch",
    )
    assert "`method='egarch'` is not available on this server" in answer
    assert "`risk` (`get_garch`) in the series" in answer

    # A forecast far over the maximum horizon.
    answer = call(
        "econometrics",
        indicator="get_percentile",
        series="risk:get_garch_forecast:AAPL:forecast_steps=999999",
    )
    assert "`forecast_steps` may be at most 1,000 on this server" in answer

    # Within the limits of one reference but not the other: neither is collected.
    answer = call(
        "econometrics",
        indicator="get_ols",
        series="risk:get_garch:AAPL; risk:get_garch:MSFT:method=EGARCH",
    )
    assert "is not available on this server" in answer
    assert not captured

    call(
        "econometrics",
        indicator="get_percentile",
        series="risk:get_garch_forecast:AAPL:method=gjr:forecast_steps=10",
    )
    assert captured[-1][1]["method"] == "gjr"
    assert captured[-1][1]["forecast_steps"] == 10  # noqa: PLR2004


def test_a_large_table_is_cut_before_its_records_are_built(monkeypatch):
    """The records of a table of millions of rows would not fit in memory."""
    from financetoolkit.mcp_server.registry import formatting_model

    built = []
    to_records = formatting_model._df_to_records
    monkeypatch.setattr(
        formatting_model,
        "_df_to_records",
        lambda frame: built.append(len(frame)) or to_records(frame),
    )
    table = pd.DataFrame(
        np.random.default_rng(0).normal(size=(1_000_000, 2)), columns=["A", "B"]
    )

    answer = read_answer(format_result(table, max_characters=60_000))

    assert max(built) <= 60_000 // 4  # noqa: PLR2004
    assert "first" in answer["_notes"][-1] and "1,000,000 rows" in answer["_notes"][-1]
    assert len(json.dumps(answer["data"])) <= 60_000  # noqa: PLR2004


@pytest.fixture
def hosted_limits(monkeypatch):
    """A hosted server's limits: a short time limit and two calls per user."""
    user = {"key": "first"}
    monkeypatch.setattr(analytics_model, "_running_calls", {})
    analytics_model.limit_calls(0.5, 2, lambda: user["key"])
    yield user
    analytics_model.limit_calls(None, None, str)


def test_a_call_over_the_time_limit_is_answered_and_the_user_limit_holds(
    hosted_limits,
):
    release = threading.Event()

    def slow() -> str:
        release.wait(5)
        return "done"

    tool = analytics_model.run_in_thread(slow)

    async def run_calls():
        first = await asyncio.gather(tool(), tool())
        # Both timed out, but their work still runs, so the user has no call left...
        refused = await tool()
        # ...while another user does.
        hosted_limits["key"] = "second"
        other = await tool()
        return first, refused, other

    first, refused, other = asyncio.run(run_calls())
    release.set()

    assert all("longer than the 0.5 seconds" in answer for answer in first)
    assert "You have 2 calls running" in refused
    assert "longer than the 0.5 seconds" in other

    # Once the work ends, the user's calls are counted as finished.
    deadline = time.time() + 5
    while analytics_model._running_calls and time.time() < deadline:
        time.sleep(0.05)
    assert analytics_model._running_calls == {}


def test_a_hosted_server_never_retries_without_verifying_the_certificate(monkeypatch):
    verified = []

    def fake_get(url, headers, timeout, verify):  # noqa: ARG001
        verified.append(verify)
        raise requests.exceptions.SSLError("certificate verify failed")

    no_system_certificates = types.SimpleNamespace(cafile=None)
    monkeypatch.setattr(requests_model.SESSION, "get", fake_get)
    monkeypatch.setattr(
        requests_model.ssl,
        "get_default_verify_paths",
        lambda: no_system_certificates,
    )
    monkeypatch.setenv(requests_model.UNVERIFIED_SSL_ENVIRONMENT_VARIABLE, "0")

    with pytest.raises(requests.exceptions.SSLError):
        requests_model.get_request("https://example.org/data.csv")

    assert False not in verified


def test_size_limits_of_one_indicator():
    limits = {"maximums": {"get_a.window": 100}, "products": []}

    assert size_limit_problem({"window": 500}, {}, limits, "get_a")
    assert size_limit_problem({"window": 50}, {}, limits, "get_a") is None
    assert size_limit_problem({"window": 500}, {}, limits, "get_b") is None
    assert size_limit_problem({"window": 500}, {}, limits) is None


def test_a_switch_is_refused_by_its_value():
    limits = {
        "refused_values": [
            {
                "methods": ["get_a"],
                "parameter": "american_option",
                "values": ["true"],
                "reason": "x",
            }
        ]
    }

    assert "`american_option='true'`" in refused_value_problem(
        "get_a", {"american_option": True}, limits
    )
    assert refused_value_problem("get_a", {"american_option": False}, limits) is None


def test_option_grids_are_limited_in_strikes(captured):
    # A strike step below the 0.5 of listed options multiplies the strikes of every option
    # grid; the option values of a grid are capped in the package (see the test below).
    answer = call(
        "options",
        indicator="get_black_scholes_model",
        tickers="AAPL",
        strike_step_size=0.01,
    )
    assert "`strike_step_size` must be at least 0.5" in answer
    assert not captured

    call(
        "options",
        indicator="get_american_option",
        tickers="AAPL",
        expiration_time_range=366,
        strike_step_size=0.5,
    )
    call(
        "options", indicator="get_cev_model", tickers="AAPL", expiration_time_range=366
    )
    assert [method for method, _ in captured] == [
        "get_american_option",
        "get_cev_model",
    ]


def test_a_product_applies_only_to_its_indicators_and_values():
    limits = {
        "products": [
            {
                "methods": ["get_asian_option"],
                "when": {"method": ["monte_carlo"]},
                "parameters": ["simulations", "fixings|time_steps"],
                "maximum": 1000,
            }
        ]
    }
    defaults = {"simulations": 100, "time_steps": 100, "method": "geometric"}

    # The closed forms take the same simulations and steps without simulating them.
    assert size_limit_problem({}, defaults, limits, "get_asian_option") is None
    answer = size_limit_problem(
        {"method": "Monte_Carlo"}, defaults, limits, "get_asian_option"
    )
    assert "`simulations` times `time_steps` may be at most 1,000" in answer
    assert (
        size_limit_problem({"method": "monte_carlo"}, defaults, limits, "get_b") is None
    )


def test_a_product_takes_the_first_given_of_its_alternatives():
    limits = {
        "products": [
            {"parameters": ["simulations", "fixings|time_steps"], "maximum": 1000}
        ]
    }
    defaults = {"simulations": 100, "time_steps": 100, "fixings": None}

    # Left out (None), the fixings fall back to the steps: 100 × 100.
    assert "`time_steps`" in size_limit_problem({}, defaults, limits)
    assert size_limit_problem({"fixings": 5}, defaults, limits) is None
    assert "`fixings`" in size_limit_problem({"fixings": 50}, defaults, limits)


def test_a_product_divides_by_a_divisor_and_names_it():
    limits = {
        "products": [{"parameters": ["range_years", "1/refit_every"], "maximum": 1}]
    }

    answer = size_limit_problem({"range_years": 10, "refit_every": 5}, {}, limits)
    assert "the years from start_date to end_date times 1 / `refit_every`" in answer
    assert "or a larger `refit_every`" in answer
    assert (
        size_limit_problem({"range_years": 10, "refit_every": 21}, {}, limits) is None
    )
    # A refit every 0 returns is never within the limit.
    assert size_limit_problem({"range_years": 1, "refit_every": 0}, {}, limits)


def test_the_years_of_the_range_count_in_the_limits_only(captured):
    # Twenty-five years of daily returns refitted every 21 returns is 300 refits per ticker.
    answer = call(
        "risk",
        indicator="get_var_backtest",
        tickers="AAPL",
        method="garch",
        start_date="2000-01-01",
        end_date="2025-01-01",
    )
    assert "the years from start_date to end_date times 1 / `refit_every`" in answer
    assert "a larger refit_every" in answer
    assert not captured
    # The historical VaR does not refit a model.
    call(
        "risk",
        indicator="get_var_backtest",
        tickers="AAPL",
        start_date="2000-01-01",
        end_date="2025-01-01",
    )
    assert captured

    call(
        "risk",
        indicator="get_var_backtest",
        tickers="AAPL",
        start_date="2000-01-01",
        end_date="2025-01-01",
        method="garch",
        refit_every=63,
    )
    assert "range_years" not in captured[-1][1]


def test_the_number_of_tickers_counts_in_the_limits_only(captured):
    tickers = ",".join(f"T{number}" for number in range(20))

    answer = call(
        "portfolio",
        indicator="get_estimation_uncertainty",
        tickers=tickers,
        n_bootstrap=100_000,
    )
    assert (
        "`n_bootstrap` times the number of tickers may be at most 1,000,000" in answer
    )
    assert not captured

    call(
        "portfolio",
        indicator="get_estimation_uncertainty",
        tickers=tickers,
        n_bootstrap=10_000,
    )
    assert captured[-1][1]["n_bootstrap"] == 10_000  # noqa: PLR2004
    # The count is not passed on: the provider receives the tickers themselves.
    assert captured[-1][1]["tickers"] == tickers.split(",")


def test_a_list_is_limited_in_length():
    limits = {"lengths": {"get_a.expiration_dates": 2}}

    answer = size_limit_problem(
        {"expiration_dates": "2026-01-01,2026-02-01,2026-03-01"}, {}, limits, "get_a"
    )
    assert "`expiration_dates` may list at most 2 values" in answer
    assert (
        size_limit_problem(
            {"expiration_dates": ["2026-01-01", "2026-02-01"]}, {}, limits, "get_a"
        )
        is None
    )
    assert (
        size_limit_problem(
            {"expiration_dates": "2026-01-01,2026-02-01,2026-03-01"},
            {},
            limits,
            "get_b",
        )
        is None
    )


def test_the_server_caps_the_option_values_of_a_grid():
    import yaml

    from financetoolkit.options import helpers

    with open(
        mcp_controller.pathlib.Path(mcp_controller.__file__).parent / "config.yaml",
        encoding="utf-8",
    ) as file:
        configured = yaml.safe_load(file)["parameter_limits"]["option_values"]

    assert configured == helpers.MAX_OPTION_VALUES
