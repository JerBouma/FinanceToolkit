"""Multi User Caching Tests"""

import pandas as pd
import pytest

from financetoolkit import fundamentals_model
from financetoolkit.cache import cache_controller


@pytest.fixture(name="cache")
def fixture_cache(tmp_path):
    """Publish an isolated cache as the active one for the duration of a test."""
    cache_controller.reset_cache_registry()

    cache = cache_controller.Cache(location=tmp_path / "cache.db")
    cache_controller.set_active_cache(cache)

    yield cache

    cache_controller.clear_active_cache()
    cache_controller.reset_cache_registry()


def test_a_disabled_cache_does_not_unpublish_an_enabled_one(tmp_path):
    """Test that opting out in one place does not switch caching off process-wide.

    A long running server publishes its cache once at startup. Controllers are
    constructed freely afterwards, and one of them defaulting to no caching must
    not silently disable caching for everything else.
    """
    cache_controller.reset_cache_registry()

    enabled = cache_controller.Cache(location=tmp_path / "cache.db")
    cache_controller.set_active_cache(enabled)

    assert cache_controller.get_active_cache() is enabled

    cache_controller.set_active_cache(
        cache_controller.Cache(location=tmp_path / "other.db", enabled=False)
    )

    assert cache_controller.get_active_cache() is enabled

    cache_controller.set_active_cache(None)

    assert cache_controller.get_active_cache() is enabled

    cache_controller.clear_active_cache()

    assert cache_controller.get_active_cache() is None

    cache_controller.reset_cache_registry()


def test_statements_are_not_shared_across_subscription_plans(cache, monkeypatch):
    """Test that a Free-plan response is never served to a Premium caller.

    The endpoints return a different amount of history per plan, so entries from
    the two are not interchangeable. On a shared cache, serving a Free-plan frame
    to a Premium caller would silently truncate their data.
    """
    requested_plans: list[str] = []

    def fake_statement(ticker, user_subscription, **kwargs):  # noqa: ARG001
        requested_plans.append(user_subscription)
        periods = 2 if user_subscription == "Free" else 6

        return pd.DataFrame(
            {f"20{20 + index}": [1.0] for index in range(periods)},
            index=["Revenue"],
        )

    monkeypatch.setattr(
        fundamentals_model.fmp_model, "get_financial_statement", fake_statement
    )

    def collect(plan):
        return fundamentals_model.collect_financial_statements(
            tickers=["AAPL"],
            statement="balance",
            api_key="key",
            start_date="2020-01-01",
            end_date="2025-12-31",
            user_subscription=plan,
            enforce_source="FinancialModelingPrep",
            cache=cache,
        )

    collect("Free")
    collect("Premium")

    assert requested_plans == ["Free", "Premium"]

    # And each plan is served from its own entry on a repeat call.
    requested_plans.clear()
    collect("Free")
    collect("Premium")

    assert requested_plans == []
