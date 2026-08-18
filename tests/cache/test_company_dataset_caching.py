"""Company Dataset Caching Tests"""

import pandas as pd
import pytest

from financetoolkit import toolkit_controller
from financetoolkit.cache import cache_controller


@pytest.fixture(name="cache_location")
def fixture_cache_location(tmp_path):
    """Point every Toolkit created in a test at an isolated cache database."""
    cache_controller.reset_cache_registry()

    yield str(tmp_path)

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


def create_toolkit(tickers, cache_location):
    """Build a Toolkit that caches but never contacts the plan-detection endpoint."""
    return toolkit_controller.Toolkit(
        tickers,
        api_key="test-key",
        sleep_timer=False,
        use_cached_data=cache_location,
    )


def test_profile_only_requests_the_newly_added_ticker(cache_location, monkeypatch):
    """Test that extending the ticker list reuses the profiles already stored."""
    requested: list[list[str]] = []

    def fake_get_profile(tickers, api_key, user_subscription):  # noqa: ARG001
        requested.append(list(tickers))

        return (
            pd.DataFrame(
                {ticker: [f"{ticker} Inc", "Technology"] for ticker in tickers},
                index=["Company Name", "Sector"],
            ),
            [],
        )

    monkeypatch.setattr(toolkit_controller, "_get_profile", fake_get_profile)

    create_toolkit(["AAPL"], cache_location).get_profile()
    requested.clear()

    profile = create_toolkit(["AAPL", "MSFT"], cache_location).get_profile()

    assert requested == [["MSFT"]]
    assert sorted(profile.columns) == ["AAPL", "MSFT"]
    assert profile.loc["Company Name", "AAPL"] == "AAPL Inc"


def test_profile_repeated_request_makes_no_call(cache_location, monkeypatch):
    """Test that asking for the same tickers again contacts nothing."""
    calls: list[list[str]] = []

    def fake_get_profile(tickers, api_key, user_subscription):  # noqa: ARG001
        calls.append(list(tickers))

        return (
            pd.DataFrame(
                {ticker: ["Technology"] for ticker in tickers}, index=["Sector"]
            ),
            [],
        )

    monkeypatch.setattr(toolkit_controller, "_get_profile", fake_get_profile)

    create_toolkit(["AAPL"], cache_location).get_profile()
    create_toolkit(["AAPL"], cache_location).get_profile()

    assert len(calls) == 1


def test_earnings_calendar_only_requests_the_new_ticker(cache_location, monkeypatch):
    """Test the same behaviour for a dataset that keeps tickers on the index."""
    requested: list[list[str]] = []

    def fake_get_earnings_calendar(tickers, **kwargs):  # noqa: ARG001
        requested.append(list(tickers))

        return (
            pd.concat(
                {
                    ticker: pd.DataFrame(
                        {"EPS": [1.0, 2.0]}, index=["2020-01-01", "2020-04-01"]
                    )
                    for ticker in tickers
                },
                axis=0,
            ),
            [],
        )

    monkeypatch.setattr(
        toolkit_controller, "_get_earnings_calendar", fake_get_earnings_calendar
    )

    create_toolkit(["AAPL"], cache_location).get_earnings_calendar()
    requested.clear()

    calendar = create_toolkit(["AAPL", "MSFT"], cache_location).get_earnings_calendar()

    assert requested == [["MSFT"]]
    assert set(calendar.index.get_level_values(0)) == {"AAPL", "MSFT"}


def test_partially_cached_result_matches_a_single_fetch(cache_location, monkeypatch):
    """Test that a half-cached frame is identical to one fetched in one call."""

    def fake_get_profile(tickers, api_key, user_subscription):  # noqa: ARG001
        return (
            pd.DataFrame(
                {ticker: [f"{ticker} Inc", "Technology"] for ticker in tickers},
                index=["Company Name", "Sector"],
            ),
            [],
        )

    monkeypatch.setattr(toolkit_controller, "_get_profile", fake_get_profile)

    create_toolkit(["AAPL"], cache_location).get_profile()
    partially_cached = create_toolkit(["AAPL", "MSFT"], cache_location).get_profile()

    single_fetch, _ = fake_get_profile(["AAPL", "MSFT"], "", "")

    pd.testing.assert_frame_equal(partially_cached, single_fetch)


def test_caching_disabled_requests_every_time(tmp_path, monkeypatch):
    """Test that behaviour is unchanged when the Toolkit caches nothing."""
    cache_controller.reset_cache_registry()
    calls: list[list[str]] = []

    def fake_get_profile(tickers, api_key, user_subscription):  # noqa: ARG001
        calls.append(list(tickers))

        return (
            pd.DataFrame(
                {ticker: ["Technology"] for ticker in tickers}, index=["Sector"]
            ),
            [],
        )

    monkeypatch.setattr(toolkit_controller, "_get_profile", fake_get_profile)

    for _ in range(2):
        toolkit_controller.Toolkit(
            ["AAPL"], api_key="test-key", sleep_timer=False, use_cached_data=False
        ).get_profile()

    assert calls == [["AAPL"], ["AAPL"]]

    cache_controller.reset_cache_registry()
