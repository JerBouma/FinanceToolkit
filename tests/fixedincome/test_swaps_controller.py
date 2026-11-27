"""Offline tests of the FixedIncome swaps and forwards, on simulated curves, exchange rates and bonds."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics.sources import bis_model
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.sources import eiopa_model
from financetoolkit.fixedincome.swaps import swap_model
from financetoolkit.utilities.error_model import InvalidArgumentError

GOVERNMENT_LABELS = [
    "1M",
    "3M",
    "6M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "20Y",
    "30Y",
]
GOVERNMENT_YEARS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

# A level and a slope per EIOPA currency, for curves of 1 to 150 years.
EIOPA_CURVES = {
    "United States": (0.040, 0.008),
    "Euro Area": (0.022, 0.012),
    "Japan": (0.006, 0.020),
    "United Kingdom": (0.038, 0.010),
}


def government_curves() -> pd.DataFrame:
    """Daily par curves of November 2024 that rise a basis point a day."""
    dates = pd.bdate_range("2024-11-01", "2024-11-29").to_period("D")
    shape = 0.042 + 0.006 * (1 - np.exp(-GOVERNMENT_YEARS / 4))
    curves = shape[None, :] + 0.0001 * np.arange(len(dates))[:, None]
    return pd.DataFrame(curves, index=dates, columns=GOVERNMENT_LABELS)


def eiopa_term_structures(curve, start_date, end_date) -> pd.DataFrame:
    """Monthly term structures from September to November 2024, 1 to 150 years."""
    months = pd.period_range("2024-09", "2024-11", freq="M")
    years = np.arange(1, 151)
    columns, values = [], []
    for country, (level, slope) in EIOPA_CURVES.items():
        for maturity in years:
            columns.append((country, f"{maturity}Y"))
            values.append(
                [
                    level + slope * (1 - np.exp(-maturity / 8)) + 0.001 * month
                    for month in range(len(months))
                ]
            )
    structures = pd.DataFrame(
        np.array(values).T,
        index=months,
        columns=pd.MultiIndex.from_tuples(columns, names=["Country", "Maturity"]),
    )
    return structures.loc[: pd.Period(end_date, "M")]


def exchange_rates(frequency, start_date, end_date) -> pd.DataFrame:
    """Daily units of each currency per US dollar, without weekend rates."""
    dates = pd.period_range("2024-10-25", "2024-11-29", freq="D")
    rates = pd.DataFrame(
        {
            "United States": 1.0,
            "Euro Area": 0.92 + 0.0005 * np.arange(len(dates)),
            "Japan": 150 + 0.1 * np.arange(len(dates)),
            "United Kingdom": 0.78,
        },
        index=dates,
    )
    rates[rates.index.dayofweek >= 5] = np.nan
    return rates.loc[pd.Period(start_date, "D") : pd.Period(end_date, "D")]


def bond_universe(self, **kwargs) -> pd.DataFrame:
    """Three bonds priced on two days."""
    return pd.DataFrame(
        {
            "Issuer": ["APPLE INC", "APPLE INC", "BOEING CO"],
            "CUSIP": ["037833AA0", "037833AB0", "097023AA0"],
            "Coupon": [0.03, 0.045, 0.055],
            "Maturity": ["2027-05-15", "2034-02-15", "2031-08-01"],
            "Rule 144A": [False, False, False],
            "Price": [97.5, 99.2, 101.3],
            "Yield to Maturity": [0.0412, 0.0461, 0.0532],
            "Modified Duration": [2.3, 7.4, 5.6],
            "G-Spread": [0.0012, 0.0041, 0.0098],
            "Price Date": ["2024-11-20", "2024-11-20", "2024-11-21"],
            "Funds": ["LQD", "LQD", "LQD, IGSB"],
        },
        index=pd.Index(["US037833AA00", "US037833AB00", "US097023AA00"], name="ISIN"),
    )


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    curves = government_curves()
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {
            "United States": lambda: curves,
            "Japan": lambda: curves - 0.035,
            "Euro Area": lambda: curves - 0.02,
        },
    )
    monkeypatch.setattr(
        eiopa_model, "get_risk_free_rate_term_structures", eiopa_term_structures
    )
    monkeypatch.setattr(bis_model, "get_exchange_rates", exchange_rates)
    monkeypatch.setattr(FixedIncome, "get_bond_universe", bond_universe)
    yield FixedIncome(start_date="2024-11-01", end_date="2024-11-22", rounding=6)
    clear_active_cache()


def test_get_swap_rate(recorder, fixedincome):
    recorder.capture(fixedincome.get_swap_rate())
    recorder.capture(
        fixedincome.get_swap_rate(
            maturities="2,5,10", country="Euro Area", fixed_frequency=2
        )
    )
    recorder.capture(
        fixedincome.get_swap_rate(
            maturities=[1, 5], curve="government", forward_start="1Y"
        )
    )


def test_get_swap_rate_takes_the_last_month_that_ended(fixedincome):
    # On 2024-11-22 November has not ended, so the October curve applies.
    october = fixedincome._swap_zero_curve("eiopa", "United States", "2024-11-22")
    november = fixedincome._swap_zero_curve("eiopa", "United States", "2024-11-30")

    assert november[10] - october[10] == pytest.approx(0.001)


def test_get_swap_rate_unknown_curve_or_country(fixedincome):
    with pytest.raises(InvalidArgumentError, match="curve must be one of"):
        fixedincome.get_swap_rate(curve="libor")
    with pytest.raises(InvalidArgumentError, match="country must be one of"):
        fixedincome.get_swap_rate(country="Atlantis")


def test_get_forward_rate_agreement(recorder, fixedincome):
    recorder.capture(fixedincome.get_forward_rate_agreement())
    recorder.capture(
        fixedincome.get_forward_rate_agreement(
            fras="3x6,12x24", day_count="act/365", date="2024-11-08"
        )
    )
    recorder.capture(
        fixedincome.get_forward_rate_agreement(curve="eiopa", country="Euro Area")
    )


def test_get_forward_rate_agreement_invalid(fixedincome):
    assert fixedincome.get_forward_rate_agreement(fras="3-6").empty
    with pytest.raises(InvalidArgumentError, match="day_count must be one of"):
        fixedincome.get_forward_rate_agreement(day_count="30/360")


def test_get_asset_swap_spread(recorder, fixedincome):
    recorder.capture(fixedincome.get_asset_swap_spread())
    recorder.capture(
        fixedincome.get_asset_swap_spread(curve="government", float_frequency=2)
    )


def test_get_i_spread(recorder, fixedincome):
    spreads = fixedincome.get_i_spread(rounding=10)
    recorder.capture(fixedincome.get_i_spread())
    recorder.capture(fixedincome.get_i_spread(curve="government"))

    # The swap rate is that of a semi-annual swap to the bond's maturity on the curve.
    zero_curve = fixedincome._swap_zero_curve("eiopa", "United States", "2024-11-21")
    years = 6 + 250 / 360  # 2024-11-21 to 2031-08-01 on the 30/360 Bond Basis
    assert spreads.loc["US097023AA00", "Swap Rate"] == pytest.approx(
        swap_model.get_swap_rate(zero_curve, years, 2)[0]
    )


def test_get_fx_forward_rates(recorder, fixedincome, caplog):
    recorder.capture(fixedincome.get_fx_forward_rates())
    recorder.capture(
        fixedincome.get_fx_forward_rates(
            countries="Euro Area,Japan", base_country="United Kingdom", points=True
        )
    )
    recorder.capture(
        fixedincome.get_fx_forward_rates(
            countries=["Euro Area", "Japan", "United Kingdom"],
            curve="government",
            tenors="3M,1Y",
        )
    )
    # The United Kingdom has no government curve here.
    assert "No FX forward rates for United Kingdom" in caplog.text


def test_get_fx_forward_rates_follow_covered_interest_parity(fixedincome):
    forwards = fixedincome.get_fx_forward_rates(
        countries="Euro Area", tenors="2Y", rounding=10
    )
    euro = fixedincome._swap_zero_curve("eiopa", "Euro Area")
    dollar = fixedincome._swap_zero_curve("eiopa", "United States")

    # The last BIS rate on or before Friday 2024-11-22.
    spot = 0.92 + 0.0005 * 28
    assert forwards.loc["Euro Area", "Spot Rate"] == pytest.approx(spot)
    assert forwards.loc["Euro Area", "2Y"] == pytest.approx(
        spot * ((1 + euro[2]) / (1 + dollar[2])) ** 2
    )


def test_swaps_without_source_data_say_why(fixedincome, monkeypatch, caplog):
    # EIOPA and the BIS could not be reached, so their sources returned no data.
    monkeypatch.setattr(
        eiopa_model,
        "get_risk_free_rate_term_structures",
        lambda curve, start_date, end_date: pd.DataFrame(),
    )

    with caplog.at_level("WARNING"):
        assert fixedincome.get_swap_rate().empty
    assert "EIOPA returned no risk-free rate term structures" in caplog.text
    assert "IndexError" not in caplog.text

    monkeypatch.setattr(
        bis_model,
        "get_exchange_rates",
        lambda frequency, start_date, end_date: pd.DataFrame(),
    )
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert fixedincome.get_fx_forward_rates(curve="government").empty
    assert "The BIS returned no exchange rates" in caplog.text
