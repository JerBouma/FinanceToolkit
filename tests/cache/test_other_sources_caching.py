"""Options, Factor And Futures Caching Tests"""

import pandas as pd
import pytest

from financetoolkit.cache import cache_controller
from financetoolkit.economics import yfinance_model as economics_yfinance_model
from financetoolkit.options import options_model
from financetoolkit.performance import performance_model


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Publish an isolated cache as the active one for the duration of a test."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache_controller.set_active_cache(cache)

    yield cache

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


class FakeTicker:
    """Stand-in for yfinance.Ticker that records how often it is asked for data."""

    calls: list[tuple[str, str]] = []

    def __init__(self, ticker: str):
        self._ticker = ticker

    @property
    def options(self) -> tuple[str, ...]:
        """Return a fixed set of expiry dates and record the lookup."""
        FakeTicker.calls.append((self._ticker, "options"))

        return ("2026-01-16", "2026-02-20")

    def history(self, start=None, end=None):
        """Return a fixed daily price history and record the lookup."""
        FakeTicker.calls.append((self._ticker, "history"))
        index = pd.date_range(start=start, end=end, freq="D", tz="UTC")

        return pd.DataFrame({"Close": 1.0}, index=index)


def test_option_expiry_dates_are_cached(cache, monkeypatch):
    """Test that the expiry date lookup happens once per ticker."""
    FakeTicker.calls = []
    monkeypatch.setattr(options_model.yf, "Ticker", FakeTicker)

    first = options_model.get_option_expiry_dates("AAPL")
    second = options_model.get_option_expiry_dates("AAPL")

    assert first == second
    assert FakeTicker.calls == [("AAPL", "options")]


def test_option_expiry_dates_without_a_cache_look_up_every_time(monkeypatch):
    """Test that behaviour is unchanged when no cache is active."""
    cache_controller.clear_active_cache()
    FakeTicker.calls = []
    monkeypatch.setattr(options_model.yf, "Ticker", FakeTicker)

    options_model.get_option_expiry_dates("AAPL")
    options_model.get_option_expiry_dates("AAPL")

    assert len(FakeTicker.calls) == 2


def test_ken_french_factor_dataset_is_cached(cache, monkeypatch):
    """Test that a cached factor dataset skips the zip download entirely."""
    dataset = pd.DataFrame(
        {"Mkt-RF": [0.1, 0.2]},
        index=pd.PeriodIndex(["2020-01-01", "2020-01-02"], freq="D"),
    )
    url = "https://example.invalid/factors.zip"

    cache.set(source="KenFrench", dataset="factors_decimal", entity=url, data=dataset)

    def fail_on_request(*args, **kwargs):
        raise AssertionError("The dataset should have been served from the cache.")

    monkeypatch.setattr(performance_model, "get_request", fail_on_request)

    result = performance_model.obtain_fama_and_french_dataset(url)

    pd.testing.assert_frame_equal(result, dataset)


def test_carhart_momentum_dataset_is_cached(cache, monkeypatch):
    """Test that the momentum factor file is served from the cache as well."""
    dataset = pd.DataFrame(
        {"Mom": [0.3]}, index=pd.PeriodIndex(["2020-01-01"], freq="D")
    )
    url = "https://example.invalid/momentum.zip"

    cache.set(source="KenFrench", dataset="factors_decimal", entity=url, data=dataset)

    def fail_on_request(*args, **kwargs):
        raise AssertionError("The dataset should have been served from the cache.")

    monkeypatch.setattr(performance_model, "get_request", fail_on_request)

    pd.testing.assert_frame_equal(
        performance_model.obtain_carhart_momentum_dataset(url), dataset
    )


def test_forward_curve_caches_each_contract_separately(cache, monkeypatch):
    """Test that rebuilding a forward curve does not refetch every delivery month."""
    FakeTicker.calls = []
    monkeypatch.setattr(economics_yfinance_model.yf, "Ticker", FakeTicker)

    first = economics_yfinance_model.get_commodity_forward_curve(
        "Gold", "2026-01-01", "2026-01-10", contracts=3
    )
    first_call_count = len(FakeTicker.calls)
    FakeTicker.calls = []

    second = economics_yfinance_model.get_commodity_forward_curve(
        "Gold", "2026-01-01", "2026-01-10", contracts=3
    )

    assert first_call_count == 3
    assert FakeTicker.calls == []
    pd.testing.assert_frame_equal(first, second)
