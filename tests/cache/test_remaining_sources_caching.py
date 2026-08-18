"""Discovery, Plan, Treasury And Statistics Caching Tests"""

import pandas as pd
import pytest

from financetoolkit import fmp_model, yfinance_model
from financetoolkit.cache import cache_controller, request_model
from financetoolkit.discovery import discovery_model
from financetoolkit.fixedincome import fmp_model as fixedincome_fmp_model


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Publish an isolated cache as the active one for the duration of a test."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache_controller.set_active_cache(cache)

    yield cache

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


def test_redact_credentials_removes_keys_but_keeps_parameters():
    """Test that a URL keeps what identifies the request and loses the credential."""
    redacted = request_model.redact_credentials(
        "https://example.invalid/data?symbol=AAPL&apikey=secret&limit=10"
    )

    assert "secret" not in redacted
    assert "symbol=AAPL" in redacted
    assert "limit=10" in redacted


def test_redact_credentials_handles_a_trailing_key():
    """Test that stripping the last parameter does not leave a dangling separator."""
    assert (
        request_model.redact_credentials(
            "https://example.invalid/data?symbol=AAPL&apikey=secret"
        )
        == "https://example.invalid/data?symbol=AAPL"
    )


def test_two_api_keys_share_one_discovery_entry(cache, monkeypatch):
    """Test that the credential is not part of what identifies a discovery response."""
    calls: list[str] = []

    def fake_fetch(url, sleep_timer=True, user_subscription="Free"):  # noqa: ARG001
        calls.append(url)

        return pd.DataFrame({"symbol": ["AAPL"], "changesPercentage": [5.0]})

    monkeypatch.setattr(discovery_model, "get_financial_data", fake_fetch)

    base = "https://financialmodelingprep.com/stable/biggest-gainers?apikey="
    discovery_model.get_cached_financial_data(url=base + "key-one")
    discovery_model.get_cached_financial_data(url=base + "key-two")

    assert len(calls) == 1


def test_discovery_error_response_is_not_cached(cache, monkeypatch):
    """Test that a rate limited response is not remembered as though it were data."""
    responses = [
        pd.DataFrame(columns=["LIMIT REACH"]),
        pd.DataFrame({"symbol": ["AAPL"]}),
    ]

    def fake_fetch(url, sleep_timer=True, user_subscription="Free"):  # noqa: ARG001
        return responses.pop(0)

    monkeypatch.setattr(discovery_model, "get_financial_data", fake_fetch)

    url = "https://financialmodelingprep.com/stable/stock-list?apikey=key"

    assert discovery_model.get_cached_financial_data(url=url).empty
    assert not discovery_model.get_cached_financial_data(url=url).empty


def test_subscription_plan_is_probed_once(cache, monkeypatch):
    """Test that the plan probe does not run on every construction."""
    calls: list[str] = []

    def fake_fetch(url, sleep_timer=True, user_subscription="Free"):  # noqa: ARG001
        calls.append(url)

        return pd.DataFrame({"revenue": [1]})

    monkeypatch.setattr(fmp_model, "get_financial_data", fake_fetch)

    first = fmp_model.determine_subscription_plan("secret-key")
    second = fmp_model.determine_subscription_plan("secret-key")

    assert first == second == ("Premium", False)
    assert len(calls) == 1


def test_subscription_plan_never_stores_the_api_key(cache, monkeypatch):
    """Test that the key itself is not written into the cache."""
    monkeypatch.setattr(
        fmp_model,
        "get_financial_data",
        lambda url, sleep_timer=True, user_subscription="Free": pd.DataFrame(  # noqa: ARG005
            {"revenue": [1]}
        ),
    )

    fmp_model.determine_subscription_plan("super-secret-key")

    entities = [
        entity
        for entry in cache.get_contents()
        if entry["dataset"] == "subscription_plan"
        for entity in entry["entities"]
    ]

    assert entities
    assert all("super-secret-key" not in entity for entity in entities)


def test_rate_limited_plan_probe_is_not_cached(cache, monkeypatch):
    """Test that a rate limited probe is retried rather than remembered as Free."""
    responses = [
        pd.DataFrame(columns=["LIMIT REACH"]),
        pd.DataFrame({"revenue": [1]}),
    ]

    monkeypatch.setattr(
        fmp_model,
        "get_financial_data",
        lambda url, sleep_timer=True, user_subscription="Free": responses.pop(
            0
        ),  # noqa: ARG005
    )

    assert fmp_model.determine_subscription_plan("key")[0] == "Free"
    assert fmp_model.determine_subscription_plan("key")[0] == "Premium"


def test_treasury_rates_only_paginate_over_the_missing_range(cache, monkeypatch):
    """Test that extending the treasury range does not repaginate the whole history."""
    windows: list[tuple[str, str]] = []

    def fake_fetch(url, sleep_timer=True, user_subscription="Free"):  # noqa: ARG001
        start = url.split("from=")[1].split("&")[0]
        end = url.split("to=")[1].split("&")[0]
        windows.append((start, end))

        return pd.DataFrame(
            {
                "date": pd.date_range(start=start, end=end, freq="D").strftime(
                    "%Y-%m-%d"
                ),
                "month1": 5.0,
            }
        )

    monkeypatch.setattr(fixedincome_fmp_model, "get_financial_data", fake_fetch)

    fixedincome_fmp_model.get_treasury_rates(
        api_key="key", start_date="2023-01-01", end_date="2023-12-31"
    )
    cold_windows = len(windows)
    windows.clear()

    rates = fixedincome_fmp_model.get_treasury_rates(
        api_key="key", start_date="2022-01-01", end_date="2023-12-31"
    )

    assert cold_windows > 1
    assert windows
    assert all(end < "2023-01-02" for _, end in windows)
    assert str(rates.index.min()) == "2022-01-01"
    assert str(rates.index.max()) == "2023-12-31"


def test_historical_statistics_are_cached_per_ticker(cache, monkeypatch):
    """Test that the instrument descriptors are only looked up once per ticker."""
    calls: list[str] = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "currency": "USD",
                                "symbol": "AAPL",
                                "exchangeName": "NMS",
                            }
                        }
                    ]
                }
            }

    def fake_get_request(url, timeout=60):  # noqa: ARG001
        calls.append(url)

        return FakeResponse()

    monkeypatch.setattr(yfinance_model, "get_request", fake_get_request)

    first = yfinance_model.get_historical_statistics("AAPL")
    second = yfinance_model.get_historical_statistics("AAPL")

    assert len(calls) == 1
    pd.testing.assert_series_equal(first, second)


def test_store_claims_coverage_for_surplus_data(cache):
    """Test that data returned beyond the requested range counts as covered.

    The price endpoints deliberately over-fetch a year on either side so that
    returns at the edges of the window are correct. Recording only the requested
    range would throw that surplus away on the next widening.
    """
    cache.store(
        source="FinancialModelingPrep",
        dataset="historical",
        entity="AAPL",
        data=pd.DataFrame(
            {"Close": 1.0},
            index=pd.period_range("2019-01-01", "2022-12-31", freq="D"),
        ),
        start="2020-01-01",
        end="2021-12-31",
    )

    plan = cache.plan(
        source="FinancialModelingPrep",
        dataset="historical",
        entities=["AAPL"],
        start="2019-06-01",
        end="2021-12-31",
    )

    assert plan.fully_cached
