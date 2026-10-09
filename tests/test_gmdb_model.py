"""Offline tests for the Global Macro Database loader and the options it adds."""

import io

import pandas as pd
import pytest
import requests

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics import gmdb_model


@pytest.fixture(autouse=True)
def isolate_from_the_active_cache():
    """Start every test without an active cache another test may have left behind."""
    clear_active_cache()
    yield
    clear_active_cache()


class FakeResponse:
    def __init__(self, text="", content=b""):
        self.text = text
        self.content = content
        self.headers = {}

    def raise_for_status(self):
        return None


def _release() -> bytes:
    """A two-country, three-year release with a forecast year and a level split."""
    data = pd.DataFrame(
        {
            "countryname": ["Germany"] * 3 + ["Japan"] * 3,
            "ISO3": ["DEU"] * 3 + ["JPN"] * 3,
            "id": ["DEU"] * 3 + ["JPN"] * 3,
            "year": [2024.0, 2025.0, 2026.0] * 2,
            "income_group": ["High income"] * 6,
            "cons": [200.0, 210.0, 220.0, 300.0, 310.0, 320.0],
            "hcons": [150.0, 160.0, 170.0, 220.0, 230.0, 240.0],
            "deflator": [100.0, 105.0, 110.0, 100.0, 100.0, 100.0],
            "govdebt_GDP": [62.0, 63.0, 64.0, 240.0, 236.0, 230.0],
            "cgovdebt_GDP": [40.0, 41.0, 42.0, 180.0, 178.0, 175.0],
            "forecast_cons": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "forecast_govdebt_GDP": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    buffer = io.BytesIO()
    data.to_stata(buffer, write_index=False)
    return buffer.getvalue()


def _serve(
    monkeypatch,
    release: bytes,
    versions="versions,version_package\n2026_06,2.0.0\n2026_09,2.0.0\n",
):
    requested = []

    def fake(url, timeout):
        requested.append(url)
        if url == gmdb_model.VERSIONS_URL:
            return FakeResponse(text=versions)
        return FakeResponse(content=release)

    monkeypatch.setattr(gmdb_model, "get_request", fake)
    return requested


def test_the_latest_release_is_read_without_its_forecast_years(monkeypatch):
    requested = _serve(monkeypatch, _release())

    dataset = gmdb_model.collect_global_macro_database_dataset()

    assert requested[-1] == gmdb_model.RELEASE_URL.format(version="2026_09")
    consumption = gmdb_model.get_series(dataset, "cons")
    # 2026 is flagged as a forecast, so it is left out; variables without a flag stay.
    assert consumption["Germany"].dropna().index[-1] == pd.Period("2025", "Y")
    assert (
        gmdb_model.get_series(dataset, "hcons").loc[pd.Period("2026", "Y"), "Japan"]
        == 240
    )
    assert not any(
        column.startswith("forecast_") for column in dataset.columns.get_level_values(0)
    )

    with_forecasts = gmdb_model.collect_global_macro_database_dataset(
        include_forecasts=True
    )
    assert with_forecasts["cons"].loc[pd.Period("2026", "Y"), "Germany"] == 220


def test_the_january_2025_file_is_used_when_the_releases_cannot_be_reached(monkeypatch):
    requested = []

    def fake(url, timeout):
        requested.append(url)
        if url == gmdb_model.VERSIONS_URL:
            raise requests.exceptions.ConnectionError("unreachable")
        return FakeResponse(content=_release())

    monkeypatch.setattr(gmdb_model, "get_request", fake)

    dataset = gmdb_model.collect_global_macro_database_dataset()

    assert requested[-1] == gmdb_model.GMD_LOCATION
    assert not dataset.empty


def test_real_consumption_is_derived_when_the_release_has_none(monkeypatch):
    _serve(monkeypatch, _release())
    dataset = gmdb_model.collect_global_macro_database_dataset(include_forecasts=True)

    real = gmdb_model.get_real_total_consumption(dataset)

    assert real.loc[pd.Period("2025", "Y"), "Germany"] == pytest.approx(200)
    assert real.loc[pd.Period("2026", "Y"), "Japan"] == pytest.approx(320)

    with pytest.raises(ValueError, match="no variable 'gcons'"):
        gmdb_model.get_series(dataset, "gcons")


def test_economics_reads_the_database_lazily_with_the_new_options(monkeypatch):
    from financetoolkit import Economics

    requested = _serve(monkeypatch, _release())
    economics = Economics(start_date="2024-01-01", end_date="2026-12-31")

    # Nothing is downloaded until a method needs the database.
    assert requested == []

    household = economics.get_total_consumption(component="household")
    real = economics.get_total_consumption(
        component="household", inflation_adjusted=True
    )
    central = economics.get_government_debt_to_gdp_ratio(level="central")
    consolidated = economics.get_government_debt_to_gdp_ratio()

    assert household.loc[pd.Period("2025", "Y"), "Germany"] == 160
    assert real.loc[pd.Period("2025", "Y"), "Germany"] == pytest.approx(
        152.381, abs=1e-3
    )
    assert central.loc[pd.Period("2025", "Y"), "Japan"] == pytest.approx(1.78)
    assert consolidated["Japan"].dropna().index[-1] == pd.Period("2025", "Y")
    # The database is downloaded once for all four methods.
    assert requested.count(gmdb_model.RELEASE_URL.format(version="2026_09")) == 1
    assert economics.get_government_debt(level="state").empty


def test_projected_years_are_a_choice_per_method(monkeypatch):
    from financetoolkit import Economics

    requested = _serve(monkeypatch, _release())
    economics = Economics(start_date="2024-01-01", end_date="2026-12-31")
    projected_year = pd.Period("2026", "Y")

    observed = economics.get_government_debt_to_gdp_ratio(countries="Germany")
    projected = economics.get_government_debt_to_gdp_ratio(
        countries="Germany", gmdb_forecasts=True
    )
    observed_again = economics.get_government_debt_to_gdp_ratio(countries="Germany")

    assert projected_year not in observed.dropna(how="all").index
    assert projected.loc[projected_year, "Germany"] == pytest.approx(0.64)
    assert observed_again.equals(observed)
    # Each version of the database is read once per instance.
    releases = [url for url in requested if url != gmdb_model.VERSIONS_URL]
    assert len(releases) == 2  # noqa: PLR2004
