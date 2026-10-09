"""Historical Caching Tests"""

import pandas as pd
import pytest

from financetoolkit import historical_model
from financetoolkit.cache import cache_controller


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Provide a cache backed by an isolated temporary database."""
    cache_controller.reset_cache_registry()

    yield cache_controller.Cache(location=tmp_path / "cache.db")

    cache_controller.reset_cache_registry()


@pytest.fixture(name="recorded_requests")
def fixture_recorded_requests(monkeypatch):
    """Replace the external price call with a recorder returning synthetic bars."""
    requests: list[dict] = []

    def fake_get_historical_data(ticker, api_key, start, end, **kwargs):
        requests.append({"ticker": ticker, "start": start, "end": end})

        index = pd.period_range(start=start, end=end, freq="D")

        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Adj Close": 1.0,
                "Volume": 100.0,
                "Dividends": 0.0,
            },
            index=index,
        )

    monkeypatch.setattr(
        historical_model.fmp_model, "get_historical_data", fake_get_historical_data
    )

    return requests


def collect(tickers, start, end, cache):
    """Run the historical collection against the stubbed source."""
    return historical_model.get_historical_data(
        tickers=tickers,
        api_key="test-key",
        enforce_source="FinancialModelingPrep",
        start=start,
        end=end,
        interval="1d",
        show_errors=False,
        cache=cache,
    )


def test_second_identical_run_makes_no_request(cache, recorded_requests):
    """Test that repeating the same request is served entirely from the cache."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    assert len(recorded_requests) == 1

    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    assert len(recorded_requests) == 1


def test_widening_the_period_only_requests_the_new_years(cache, recorded_requests):
    """Test that an earlier start date requests only the period not yet held."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)
    recorded_requests.clear()

    collect(["AAPL"], "2018-01-01", "2020-12-31", cache)

    assert len(recorded_requests) == 1
    assert recorded_requests[0]["start"] == "2018-01-01"
    assert recorded_requests[0]["end"] == "2019-12-31"


def test_adding_a_ticker_only_requests_that_ticker(cache, recorded_requests):
    """Test that extending the ticker list leaves the existing tickers untouched."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)
    recorded_requests.clear()

    collect(["AAPL", "MSFT"], "2020-01-01", "2020-12-31", cache)

    assert [request["ticker"] for request in recorded_requests] == ["MSFT"]


def test_widened_period_returns_the_full_range(cache, recorded_requests):
    """Test that the cached and freshly fetched halves are returned as one frame."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    historical_data, _ = collect(["AAPL"], "2018-01-01", "2020-12-31", cache)

    assert historical_data.index.min() == pd.Period("2018-01-01", freq="D")
    assert historical_data.index.max() == pd.Period("2020-12-31", freq="D")


def test_cumulative_return_is_recomputed_over_the_merged_window(
    cache, recorded_requests
):
    """Test that the window dependent columns reflect the merged range, not the fetch."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    historical_data, _ = collect(["AAPL"], "2018-01-01", "2020-12-31", cache)

    cumulative_return = historical_data["Cumulative Return"]["AAPL"]

    # The series is flat, so what matters is the column spanning the merged window.
    assert cumulative_return.notna().all()
    assert cumulative_return.loc[pd.Period("2018-06-01", freq="D")] == 1.0


def test_without_a_cache_every_run_requests_again(recorded_requests):
    """Test that behaviour is unchanged when no cache is supplied."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", None)
    collect(["AAPL"], "2020-01-01", "2020-12-31", None)

    assert len(recorded_requests) == 2


def test_price_history_is_cached_under_the_serving_provider(cache, recorded_requests):
    """Test that the cache source names the provider that actually served the data."""
    collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    contents = {(entry["source"], entry["dataset"]) for entry in cache.get_contents()}

    assert ("FinancialModelingPrep", "historical") in contents


def test_a_yahoo_served_ticker_is_cached_under_yahoo_finance(cache, monkeypatch):
    """Test that a ticker falling through to Yahoo Finance is stored as such."""
    monkeypatch.setattr(
        historical_model.fmp_model,
        "get_historical_data",
        lambda **kwargs: pd.DataFrame(),  # noqa: ARG005
    )

    def fake_yfinance(ticker, start, end, **kwargs):  # noqa: ARG001
        index = pd.period_range(start=start, end=end, freq="D")

        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Adj Close": 1.0,
                "Volume": 100.0,
            },
            index=index,
        )

    monkeypatch.setattr(
        historical_model.yfinance_model, "get_historical_data", fake_yfinance
    )

    historical_model.get_historical_data(
        tickers=["AAPL"],
        api_key="test-key",
        enforce_source=None,
        start="2020-01-01",
        end="2020-12-31",
        interval="1d",
        show_errors=False,
        cache=cache,
    )

    contents = {(entry["source"], entry["dataset"]) for entry in cache.get_contents()}

    assert ("YahooFinance", "historical") in contents
    assert ("FinancialModelingPrep", "historical") not in contents


def test_a_truncated_fmp_response_falls_back_to_yahoo_finance(cache, monkeypatch):
    """A non-empty but drastically truncated FMP response (observed with FMP's
    lower-tier commodities data: a few weeks back instead of the requested
    years) must still trigger the Yahoo fallback, not be silently accepted as
    "the data" just because it isn't literally empty."""

    def fake_fmp_thin_response(ticker, api_key, start, end, **kwargs):  # noqa: ARG001
        # Only the last 30 days of a 20-year requested window.
        index = pd.period_range(end="2020-12-31", periods=30, freq="D")
        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Adj Close": 1.0,
                "Volume": 100.0,
                "Dividends": 0.0,
            },
            index=index,
        )

    monkeypatch.setattr(
        historical_model.fmp_model, "get_historical_data", fake_fmp_thin_response
    )

    def fake_yfinance(ticker, start, end, **kwargs):  # noqa: ARG001
        index = pd.period_range(start=start, end=end, freq="D")
        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Adj Close": 1.0,
                "Volume": 100.0,
            },
            index=index,
        )

    monkeypatch.setattr(
        historical_model.yfinance_model, "get_historical_data", fake_yfinance
    )

    historical_data, _ = historical_model.get_historical_data(
        tickers=["GC=F"],
        api_key="test-key",
        enforce_source=None,
        start="2000-01-01",
        end="2020-12-31",
        interval="1d",
        show_errors=False,
        cache=cache,
    )

    assert historical_data.index.min() == pd.Period("2000-01-01", freq="D")

    contents = {(entry["source"], entry["dataset"]) for entry in cache.get_contents()}
    assert ("YahooFinance", "historical") in contents
    assert ("FinancialModelingPrep", "historical") not in contents


def test_a_response_covering_the_requested_span_is_not_treated_as_truncated(
    cache, recorded_requests
):
    """A full response over a genuinely long requested window must not be
    discarded -- only a response that falls short of what was asked for
    should ever trigger the Yahoo fallback."""
    historical_data, _ = collect(["AAPL"], "2020-01-01", "2020-12-31", cache)

    assert not historical_data.empty
    contents = {(entry["source"], entry["dataset"]) for entry in cache.get_contents()}
    assert ("FinancialModelingPrep", "historical") in contents


def test_intraday_data_is_cached(cache, monkeypatch):
    """Test that intraday bars are cached rather than silently refetched."""
    calls: list[str] = []

    def fake_intraday(ticker, api_key, start, end, **kwargs):  # noqa: ARG001
        calls.append(ticker)
        index = pd.period_range(start=start, end=end, freq="h")

        return pd.DataFrame(
            {
                "Open": 1.0,
                "High": 1.0,
                "Low": 1.0,
                "Close": 1.0,
                "Adj Close": 1.0,
                "Volume": 100.0,
            },
            index=index,
        )

    monkeypatch.setattr(historical_model.fmp_model, "get_intraday_data", fake_intraday)

    def run():
        return historical_model.get_historical_data(
            tickers=["AAPL"],
            api_key="test-key",
            start="2024-01-01",
            end="2024-01-02",
            interval="1hour",
            show_errors=False,
            cache=cache,
        )

    run()

    assert len(calls) == 1
    assert ("FinancialModelingPrep", "intraday") in {
        (entry["source"], entry["dataset"]) for entry in cache.get_contents()
    }

    run()

    assert len(calls) == 1


def bars(start, end):
    """Synthetic daily bars for the requested range."""
    return pd.DataFrame(
        {
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "Adj Close": 1.0,
            "Volume": 100.0,
        },
        index=pd.period_range(start=start, end=end, freq="D"),
    )


def collect_from_any_provider(start, end, cache):
    """Run the collection with both providers allowed."""
    data, _ = historical_model.get_historical_data(
        tickers=["AAPL"],
        api_key="test-key",
        enforce_source=None,
        start=start,
        end=end,
        interval="1d",
        show_errors=False,
        cache=cache,
    )

    return data


def test_a_gap_served_by_the_other_provider_returns_the_full_range(cache, monkeypatch):
    """Cached FMP prices extended by Yahoo Finance are not cut down to the gap."""
    fmp_available = {"value": True}
    yahoo_requests: list[tuple[str, str]] = []

    monkeypatch.setattr(
        historical_model.fmp_model,
        "get_historical_data",
        lambda ticker, api_key, start, end, **kwargs: (  # noqa: ARG005
            bars(start, end) if fmp_available["value"] else pd.DataFrame()
        ),
    )

    def fake_yfinance(ticker, start, end, **kwargs):  # noqa: ARG001
        yahoo_requests.append((start, end))
        return bars(start, end)

    monkeypatch.setattr(
        historical_model.yfinance_model, "get_historical_data", fake_yfinance
    )

    collect_from_any_provider("2018-01-01", "2020-12-31", cache)
    fmp_available["value"] = False

    data = collect_from_any_provider("2018-01-01", "2021-12-31", cache)

    assert str(data.index.min()) == "2018-01-01"
    assert str(data.index.max()) == "2021-12-31"
    # The gap first, then the whole range from the provider that has it now.
    assert yahoo_requests[-1] == ("2018-01-01", "2021-12-31")


def test_a_gap_that_cannot_be_retrieved_returns_the_cached_part(cache, monkeypatch):
    """A failed extension returns what is cached and is retried on the next run."""
    fmp_requests: list[tuple[str, str]] = []
    fmp_available = {"value": True}

    def fake_fmp(ticker, api_key, start, end, **kwargs):  # noqa: ARG001
        fmp_requests.append((start, end))
        return bars(start, end) if fmp_available["value"] else pd.DataFrame()

    monkeypatch.setattr(historical_model.fmp_model, "get_historical_data", fake_fmp)
    monkeypatch.setattr(
        historical_model.yfinance_model,
        "get_historical_data",
        lambda **kwargs: pd.DataFrame(),  # noqa: ARG005
    )

    collect_from_any_provider("2018-01-01", "2020-12-31", cache)
    fmp_available["value"] = False

    data = collect_from_any_provider("2018-01-01", "2021-12-31", cache)

    assert str(data.index.min()) == "2018-01-01"
    assert str(data.index.max()) == "2020-12-31"

    fmp_available["value"] = True
    requests_before = len(fmp_requests)
    data = collect_from_any_provider("2018-01-01", "2021-12-31", cache)

    assert len(fmp_requests) == requests_before + 1
    assert str(data.index.max()) == "2021-12-31"


def test_a_key_prefers_financial_modeling_prep_over_cached_yahoo_finance_data(
    cache, monkeypatch
):
    """Prices Yahoo Finance served before the key was set do not stand in for FMP's."""
    requests_made = {"fmp": 0, "yahoo": 0}
    fmp_has_data = {"value": True}

    def fake_fmp(ticker, api_key, start, end, **kwargs):  # noqa: ARG001
        requests_made["fmp"] += 1
        return bars(start, end) if fmp_has_data["value"] else pd.DataFrame()

    def fake_yfinance(ticker, start, end, **kwargs):  # noqa: ARG001
        requests_made["yahoo"] += 1
        return bars(start, end)

    monkeypatch.setattr(historical_model.fmp_model, "get_historical_data", fake_fmp)
    monkeypatch.setattr(
        historical_model.yfinance_model, "get_historical_data", fake_yfinance
    )

    def collect_with(api_key):
        data, _ = historical_model.get_historical_data(
            tickers=["AAPL"],
            api_key=api_key,
            start="2020-01-01",
            end="2020-12-31",
            interval="1d",
            show_errors=False,
            cache=cache,
        )
        return data

    collect_with("")
    assert requests_made == {"fmp": 0, "yahoo": 1}

    # A ticker FinancialModelingPrep has no data for falls back to the cached copy.
    fmp_has_data["value"] = False
    assert not collect_with("key").empty
    assert requests_made == {"fmp": 1, "yahoo": 1}

    # Otherwise FinancialModelingPrep's own prices are retrieved and cached.
    fmp_has_data["value"] = True
    collect_with("key")
    collect_with("key")
    assert requests_made == {"fmp": 2, "yahoo": 1}
