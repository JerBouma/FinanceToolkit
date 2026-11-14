"""Tests for the sources an answer's data comes from (sources_model)."""

import pandas as pd

from financetoolkit.economics import helpers as economics_helpers
from financetoolkit.helpers import run_in_parallel
from financetoolkit.mcp_server.formatting_model import format_result
from financetoolkit.utilities import sources_model
from financetoolkit.utilities.sources_model import (
    collect_sources,
    readable_sources,
    record_source,
    record_url,
    track_sources,
)


def test_a_url_is_named_by_its_publisher_and_an_unknown_host_by_itself():
    assert (
        sources_model.source_of_url("https://financialmodelingprep.com/stable/quote")
        == "FinancialModelingPrep"
    )
    assert sources_model.source_of_url("https://data-api.ecb.europa.eu/x") == (
        "EuropeanCentralBank"
    )
    assert sources_model.source_of_url("https://ec.europa.eu/eurostat/api") == (
        "Eurostat"
    )
    assert sources_model.source_of_url("https://registers.esma.europa.eu/x") == "ESMA"
    assert sources_model.source_of_url("https://example.org/data.csv") == (
        "example.org"
    )
    assert readable_sources({"BIS", "FinancialModelingPrep", "example.org"}) == [
        "Bank for International Settlements",
        "FinancialModelingPrep",
        "example.org",
    ]


def test_sources_are_only_recorded_inside_a_collection_and_reach_worker_threads():
    record_source("FRED")

    with collect_sources() as sources:
        record_url("https://api.bls.gov/publicAPI/v2/timeseries/data/")
        run_in_parallel(record_source, [("BIS",), ("OECD",)])

    assert sources == {"BLS", "BIS", "OECD"}


@track_sources
class _Holder:
    """Keeps retrieved data between calls, as the Toolkit does."""

    def __init__(self):
        self._prices = pd.DataFrame()
        self._rates = pd.DataFrame()

    def get_prices(self):
        if self._prices.empty:
            record_source("FinancialModelingPrep")
            self._prices = pd.DataFrame({"Close": [1.0]})
        return self._prices

    def get_rates(self):
        if self._rates.empty:
            record_source("USTreasury")
            self._rates = pd.DataFrame({"Rate": [0.04]})
        return self._rates

    def get_returns(self):
        # Reads the kept prices directly rather than through get_prices.
        return self._prices.pct_change()

    @property
    def summary(self):
        return self.get_prices()


def test_data_kept_on_an_instance_still_names_its_source_when_reused():
    holder = _Holder()

    with collect_sources() as first:
        holder.get_prices()
        holder.get_rates()
    assert first == {"FinancialModelingPrep", "USTreasury"}

    # Nothing is retrieved again, yet each answer names where its data came from.
    with collect_sources() as through_method:
        holder.get_prices()
    with collect_sources() as through_attribute:
        holder.get_returns()
    with collect_sources() as through_property:
        _ = holder.summary

    assert through_method == {"FinancialModelingPrep"}
    assert through_attribute == {"FinancialModelingPrep"}
    assert through_property == {"FinancialModelingPrep"}


def test_a_tracked_method_runs_unchanged_outside_a_collection():
    holder = _Holder()

    assert holder.get_prices()["Close"].iloc[0] == 1.0
    assert sources_model.PROVENANCE_ATTRIBUTE not in vars(holder)


def test_a_fresher_publisher_is_named_only_when_it_extends_the_history():
    months = pd.period_range("2026-01", periods=3, freq="M")
    history = pd.DataFrame({"United States": [0.04, 0.041, 0.042]}, index=months)
    agreeing = pd.DataFrame(
        {"United States": [0.042, 0.043]},
        index=pd.period_range("2026-03", periods=2, freq="M"),
    )
    disagreeing = agreeing + 0.01

    with collect_sources() as extended:
        economics_helpers.extend_with_recent(
            history, agreeing, "United States", {"BLS"}
        )
    with collect_sources() as unchanged:
        economics_helpers.extend_with_recent(
            history, disagreeing, "United States", {"BLS"}
        )

    assert extended == {"BLS"}
    assert unchanged == set()


def test_choose_sources_takes_each_country_from_its_most_recent_source():
    months = pd.period_range("2026-01", periods=3, freq="M")
    eurostat = pd.DataFrame({"Germany": [1.0, 2.0, 3.0]}, index=months)
    bis = pd.DataFrame(
        {"Germany": [1.0, 2.0, None], "United States": [1.0, 2.0, 3.0]}, index=months
    )

    assert economics_helpers.choose_sources([eurostat, bis]) == {
        "Germany": 0,
        "United States": 1,
    }


def test_an_answer_lists_its_sources():
    frame = pd.DataFrame({"AAPL": [1.0]}, index=["2026"])

    answer = format_result(frame, sources=["FinancialModelingPrep"])

    assert '"_sources": ["FinancialModelingPrep"]' in answer
    assert "_sources" not in format_result(frame)


def test_combined_sources_name_only_the_source_of_each_requested_country():
    from financetoolkit import Economics

    months = pd.period_range("2026-01", periods=3, freq="M")

    def eurostat():
        record_source("Eurostat")
        return pd.DataFrame({"Germany": [1.0, 2.0, 3.0]}, index=months)

    def bis():
        record_source("BIS")
        return pd.DataFrame(
            {"Germany": [1.0, 2.0, None], "United States": [1.0, 2.0, 3.0]},
            index=months,
        )

    economics = Economics(start_date="2026-01-01", end_date="2026-03-31")

    for countries, expected in (
        (["United States"], {"BIS"}),
        (["Germany"], {"Eurostat"}),
        (None, {"BIS", "Eurostat"}),
    ):
        with collect_sources() as sources:
            economics._combine_sources_for([(eurostat, None), (bis, None)], countries)
        assert sources == expected
