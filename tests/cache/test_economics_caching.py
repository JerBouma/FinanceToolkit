"""Economics And Fixed Income Caching Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache import cache_controller
from financetoolkit.economics import gmdb_model, oecd_model
from financetoolkit.fixedincome import fed_model, fred_model, helpers


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Publish an isolated cache as the active one for the duration of a test."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache_controller.set_active_cache(cache)

    yield cache

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


class FakeResponse:
    """Minimal stand-in for a requests Response carrying fixed text."""

    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        """Accept the response as successful."""


def test_fred_second_run_makes_no_request(cache, monkeypatch):
    """Test that a repeated FRED request within the TTL is served from cache."""
    requests: list[tuple[str, str, str]] = []

    def fake_request(series_id, start_date, end_date, api_key):  # noqa: ARG001
        requests.append((series_id, start_date, end_date))
        index = pd.period_range(start=start_date, end=end_date, freq="D")

        return pd.DataFrame(
            {series_id: np.arange(len(index), dtype=float)}, index=index
        )

    monkeypatch.setattr(fred_model, "_request_series", fake_request)

    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key")
    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key")

    assert len(requests) == 1


def test_fred_widening_the_range_only_requests_the_gap(cache, monkeypatch):
    """Test that an earlier start date only requests the years not yet stored."""
    requests: list[tuple[str, str, str]] = []

    def fake_request(series_id, start_date, end_date, api_key):  # noqa: ARG001
        requests.append((series_id, start_date, end_date))
        index = pd.period_range(start=start_date, end=end_date, freq="D")

        return pd.DataFrame({series_id: 1.0}, index=index)

    monkeypatch.setattr(fred_model, "_request_series", fake_request)

    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key")
    requests.clear()

    data = fred_model.fetch_single_series("DGS10", "2018-01-01", "2020-12-31", "key")

    assert requests == [("DGS10", "2018-01-01", "2019-12-31")]
    assert data.index.min() == pd.Period("2018-01-01", freq="D")
    assert data.index.max() == pd.Period("2020-12-31", freq="D")


def test_fred_api_key_is_not_part_of_the_cache_key(cache, monkeypatch):
    """Test that a different API key still hits the same cached observations."""
    calls: list[str] = []

    def fake_request(series_id, start_date, end_date, api_key):
        calls.append(api_key)
        index = pd.period_range(start=start_date, end=end_date, freq="D")

        return pd.DataFrame({series_id: 1.0}, index=index)

    monkeypatch.setattr(fred_model, "_request_series", fake_request)

    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key-one")
    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key-two")

    assert calls == ["key-one"]


def test_fred_without_a_cache_requests_every_time(monkeypatch):
    """Test that FRED behaviour is unchanged when no cache is active."""
    cache_controller.clear_active_cache()
    calls: list[str] = []

    def fake_request(series_id, start_date, end_date, api_key):  # noqa: ARG001
        calls.append(series_id)

        return pd.DataFrame(
            {series_id: 1.0},
            index=pd.period_range(start=start_date, end=end_date, freq="D"),
        )

    monkeypatch.setattr(fred_model, "_request_series", fake_request)

    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key")
    fred_model.fetch_single_series("DGS10", "2020-01-01", "2020-12-31", "key")

    assert len(calls) == 2


def test_oecd_second_run_makes_no_request(cache, monkeypatch):
    """Test that a repeated OECD query within the TTL is served from cache."""
    csv_text = (
        "REF_AREA,TIME_PERIOD,OBS_VALUE\n"
        "NLD,2020,100.0\n"
        "NLD,2021,102.0\n"
        "USA,2020,100.0\n"
        "USA,2021,104.0\n"
    )
    calls: list[str] = []

    def fake_get_request(url, timeout=300):  # noqa: ARG001
        calls.append(url)

        return FakeResponse(csv_text)

    monkeypatch.setattr(oecd_model, "get_request", fake_get_request)

    first = oecd_model.collect_oecd_data("SOME/QUERY", "Y", "2020-01-01", "2021-12-31")
    second = oecd_model.collect_oecd_data("SOME/QUERY", "Y", "2020-01-01", "2021-12-31")

    assert len(calls) == 1
    assert list(first.columns) == ["Netherlands", "United States"]
    pd.testing.assert_frame_equal(first, second)


def test_oecd_rate_limit_falls_back_on_the_cache(cache, monkeypatch):
    """Test that a 429 serves the stored response when that is opted in to."""
    import requests as requests_library

    csv_text = "REF_AREA,TIME_PERIOD,OBS_VALUE\nNLD,2020,100.0\n"

    monkeypatch.setattr(
        oecd_model,
        "get_request",
        lambda url, timeout=300: FakeResponse(csv_text),  # noqa: ARG005
    )
    oecd_model.collect_oecd_data("SOME/QUERY", "Y", "2020-01-01", "2020-12-31")

    def rate_limited(url, timeout=300):  # noqa: ARG001
        response = requests_library.Response()
        response.status_code = 429

        raise requests_library.exceptions.HTTPError(response=response)

    monkeypatch.setattr(oecd_model, "get_request", rate_limited)
    oecd_model.configure_oecd_cache(allow_stale_on_rate_limit=True)

    # A different period forces a live call, which is then rate limited.
    data = oecd_model.collect_oecd_data("SOME/QUERY", "Y", "1990-01-01", "2020-12-31")

    assert not data.empty
    assert list(data.columns) == ["Netherlands"]


def test_oecd_rate_limit_returns_empty_when_not_opted_in(cache, monkeypatch):
    """Test that a 429 returns empty data when the stale fallback is disabled."""
    import requests as requests_library

    def rate_limited(url, timeout=300):  # noqa: ARG001
        response = requests_library.Response()
        response.status_code = 429

        raise requests_library.exceptions.HTTPError(response=response)

    monkeypatch.setattr(oecd_model, "get_request", rate_limited)
    oecd_model.configure_oecd_cache(allow_stale_on_rate_limit=False)

    assert oecd_model.collect_oecd_data(
        "OTHER/QUERY", "Y", "2020-01-01", "2020-12-31"
    ).empty

    oecd_model.configure_oecd_cache(allow_stale_on_rate_limit=True)


def test_ecb_response_is_cached_whole(cache, monkeypatch):
    """Test that the ECB endpoint, which has no date range, is cached as a whole."""
    csv_text = "TIME_PERIOD,OBS_VALUE\n2020-01-01,1.0\n2020-01-02,1.5\n"
    calls: list[str] = []

    def fake_get_request(url, timeout=60):  # noqa: ARG001
        calls.append(url)

        return FakeResponse(csv_text)

    monkeypatch.setattr(helpers, "get_request", fake_get_request)

    first = helpers.collect_ecb_data("SOME.KEY", "FM")
    second = helpers.collect_ecb_data("SOME.KEY", "FM")

    assert len(calls) == 1
    pd.testing.assert_series_equal(first, second)


def test_fed_response_is_cached_whole(cache, monkeypatch):
    """Test that the Federal Reserve endpoint is cached as a whole."""
    csv_text = (
        "Effective Date,Rate (%),1st Percentile (%),25th Percentile (%),"
        "75th Percentile (%),99th Percentile (%),Volume ($Billions),"
        "Target Rate From (%),Target Rate To (%)\n"
        "2020-01-01,1.5,1.4,1.45,1.55,1.6,100,1.5,1.75\n"
    )
    calls: list[str] = []

    def fake_get_request(url, timeout=60):  # noqa: ARG001
        calls.append(url)

        return FakeResponse(csv_text)

    monkeypatch.setattr(fed_model, "get_request", fake_get_request)

    fed_model.collect_fed_data("500")
    fed_model.collect_fed_data("500")

    assert len(calls) == 1


def test_global_macro_database_is_cached_whole(cache, monkeypatch):
    """Test that a cached Global Macro Database skips the download entirely."""
    calls: list[str] = []

    def fail_on_request(*args, **kwargs):
        calls.append("requested")

        raise AssertionError("The dataset should have been served from the cache.")

    monkeypatch.setattr(gmdb_model, "get_request", fail_on_request)

    dataset = pd.DataFrame(
        {"gdp": [1.0, 2.0]},
        index=pd.PeriodIndex(["2020", "2021"], freq="Y"),
    )
    cache.set(
        source="GlobalMacroDatabase", dataset="dataset", entity="global", data=dataset
    )

    result = gmdb_model.collect_global_macro_database_dataset(cache=cache)

    assert calls == []
    pd.testing.assert_frame_equal(result, dataset)
