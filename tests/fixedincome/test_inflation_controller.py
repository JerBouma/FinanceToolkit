"""Offline tests of the inflation swap rates and the forward CPI, on recorded-shape Treasury,
Bundesbank and BLS data of November 2026."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics.sources import bls_model, treasury_model
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.inflation import inflation_swap_model
from financetoolkit.fixedincome.sources import bundesbank_model
from financetoolkit.utilities.error_model import InvalidArgumentError

DAYS = pd.bdate_range("2026-11-02", "2026-11-20").to_period("D")

NOMINAL_COLUMNS = {
    "1 Mo": 1 / 12,
    "3 Mo": 0.25,
    "6 Mo": 0.5,
    "1 Yr": 1,
    "2 Yr": 2,
    "3 Yr": 3,
    "5 Yr": 5,
    "7 Yr": 7,
    "10 Yr": 10,
    "20 Yr": 20,
    "30 Yr": 30,
}
REAL_COLUMNS = {"5 YR": 5, "7 YR": 7, "10 YR": 10, "20 YR": 20, "30 YR": 30}


def yield_curve(curve, start_date, end_date):
    columns = NOMINAL_COLUMNS if curve == "nominal" else REAL_COLUMNS
    years = np.array(list(columns.values()))
    level, slope, scale = (
        (0.039, 0.016, 8) if curve == "nominal" else (0.018, 0.012, 10)
    )
    shape = level + slope * (1 - np.exp(-years / scale))
    return pd.DataFrame(
        np.tile(shape, (len(DAYS), 1)), index=DAYS, columns=list(columns)
    )


def german_curve(start_date, end_date):
    years = np.arange(1, 31)
    shape = 0.022 + 0.012 * (1 - np.exp(-years / 6))
    return pd.DataFrame(
        np.tile(shape, (len(DAYS), 1)),
        index=DAYS,
        columns=[f"{year}Y" for year in years],
    )


def german_real_curve(start_date, end_date):
    years = [3, 5, 7, 10, 15, 20, 25]
    shape = 0.002 + 0.008 * (1 - np.exp(-np.array(years) / 10))
    return pd.DataFrame(
        np.tile(shape, (len(DAYS), 1)),
        index=DAYS,
        columns=[f"{year}Y" for year in years],
    )


# The CPI-U, not seasonally adjusted, of the last three years with a seasonal pattern.
CPI_MONTHS = pd.period_range("2023-11", "2026-10", freq="M")
CPI = pd.Series(
    305.0
    * 1.0022 ** np.arange(len(CPI_MONTHS))
    * (1 + 0.004 * np.sin(2 * np.pi * (CPI_MONTHS.month - 2) / 12)),
    index=CPI_MONTHS,
).round(3)


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    monkeypatch.setattr(treasury_model, "get_yield_curve", yield_curve)
    monkeypatch.setattr(bundesbank_model, "get_yield_curve", german_curve)
    monkeypatch.setattr(bundesbank_model, "get_real_yield_curve", german_real_curve)
    monkeypatch.setattr(
        bls_model,
        "get_consumer_price_index",
        lambda: pd.DataFrame({"all": CPI}),
    )
    return FixedIncome(
        start_date="2026-10-01",
        end_date="2026-11-20",
        fred_api_key="",
        use_cached_data=False,
    )


def test_get_inflation_swap_rate(recorder, fixedincome):
    recorder.capture(fixedincome.get_inflation_swap_rate())
    recorder.capture(
        fixedincome.get_inflation_swap_rate(
            country="Germany", maturities="2Y,5Y,10Y,20Y", date="2026-11-13"
        )
    )


def test_inflation_swap_rate_is_the_zero_coupon_breakeven(fixedincome):
    rates = fixedincome.get_inflation_swap_rate(maturities=[3, 5, 10], rounding=12)

    assert rates["Zero-Coupon Inflation Swap Rate"].to_numpy() == pytest.approx(
        (
            (1 + rates["Nominal Zero Rate"]) / (1 + rates["Real Zero Rate"]) - 1
        ).to_numpy(),
        rel=1e-9,
    )
    # Below five years the real par yield is held at the five-year yield.
    assert rates.loc[3, "Real Zero Rate"] == pytest.approx(
        rates.loc[5, "Real Zero Rate"], abs=1e-12
    )
    assert (rates["Date"] == "2026-11-20").all()


def test_inflation_swap_rate_of_an_unknown_country_raises(fixedincome):
    with pytest.raises(InvalidArgumentError, match="country must be one of"):
        fixedincome.get_inflation_swap_rate(country="Japan")


def test_get_forward_inflation_index(recorder, fixedincome):
    recorder.capture(fixedincome.get_forward_inflation_index(years=2))
    recorder.capture(
        fixedincome.get_forward_inflation_index(
            date="2026-11-10", years=1, seasonality=False
        )
    )


def test_forward_inflation_index_starts_from_the_reference_month(fixedincome):
    forward = fixedincome.get_forward_inflation_index(years=2, rounding=12)
    rates = fixedincome.get_inflation_swap_rate(maturities=[1, 2], rounding=12)

    # The base is August 2026, three months before November, the month of the curves.
    assert forward.index[0] == pd.Period("2026-09", "M")
    # Whole years keep the swap curve, as the seasonal factors cancel out.
    assert forward.loc[pd.Period("2027-08", "M"), "Forward CPI"] == pytest.approx(
        CPI[pd.Period("2026-08", "M")]
        * (1 + rates.loc[1, "Zero-Coupon Inflation Swap Rate"]),
        rel=1e-9,
    )
    factors = inflation_swap_model.get_inflation_seasonality(CPI.loc[:"2026-08"])
    assert forward.loc[pd.Period("2027-02", "M"), "Seasonal Factor"] == pytest.approx(
        factors[2] / factors[8], rel=1e-9
    )


def test_inflation_swap_rate_needs_positive_maturities(fixedincome, monkeypatch):
    monkeypatch.setenv("FINANCETOOLKIT_STRICT_ERRORS", "1")
    with pytest.raises(ValueError, match="must be positive"):
        fixedincome.get_inflation_swap_rate(maturities="0")


def test_a_real_curve_without_maturities_says_why(fixedincome, monkeypatch, caplog):
    # A real curve whose columns are no maturities has nothing to bootstrap from.
    def without_real_maturities(curve, start_date, end_date):
        curves = yield_curve(curve, start_date, end_date)
        return curves.rename(columns=str.lower) if curve == "real" else curves

    monkeypatch.setattr(treasury_model, "get_yield_curve", without_real_maturities)

    with caplog.at_level("WARNING"):
        assert fixedincome.get_inflation_swap_rate().empty
    assert "no maturity with a yield to bootstrap from" in caplog.text
    assert "zero-size array" not in caplog.text
