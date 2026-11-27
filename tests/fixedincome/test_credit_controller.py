"""Offline tests of the FixedIncome survival curves and CDS-equivalent spreads, on simulated
ICE BofA option-adjusted spreads and Treasury curves."""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.fixedincome.credit import (
    cds_model,
    credit_controller,
    credit_curve_model,
)
from financetoolkit.fixedincome.fixedincome_controller import FixedIncome
from financetoolkit.fixedincome.sources import fred_model

TREASURY_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
TREASURY_YEARS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

DATES = pd.bdate_range("2024-11-01", "2024-11-29").to_period("D")

MATURITY_SPREADS = {
    "1-3 Years": 0.0062,
    "3-5 Years": 0.0084,
    "5-7 Years": 0.0098,
    "7-10 Years": 0.0113,
    "10-15 Years": 0.011,
    "15+ Years": 0.0111,
}
RATING_SPREADS = {
    "AAA": 0.0048,
    "AA": 0.0069,
    "A": 0.0077,
    "BBB": 0.0116,
    "BB": 0.0213,
    "B": 0.0367,
    "CCC": 0.132,
}


def treasury_curves() -> pd.DataFrame:
    """Daily par curves of November 2024 that rise a basis point a day."""
    shape = 0.042 + 0.006 * (1 - np.exp(-TREASURY_YEARS / 4))
    curves = shape[None, :] + 0.0001 * np.arange(len(DATES))[:, None]
    return pd.DataFrame(curves, index=DATES, columns=TREASURY_LABELS)


def spreads(levels: dict):
    """Daily spreads that widen half a basis point a day, without weekend rows."""

    def fetch(start_date, end_date, api_key) -> pd.DataFrame:
        data = pd.DataFrame(
            {
                name: level + 0.00005 * np.arange(len(DATES))
                for name, level in levels.items()
            },
            index=DATES,
        )
        return data.loc[pd.Period(start_date, "D") : pd.Period(end_date, "D")]

    return fetch


@pytest.fixture
def fixedincome(monkeypatch):
    clear_active_cache()
    curves = treasury_curves()
    monkeypatch.setattr(
        FixedIncome,
        "_get_country_yield_curves",
        lambda self, start_date: {"United States": lambda: curves},
    )
    monkeypatch.setattr(
        fred_model,
        "get_maturity_option_adjusted_spread",
        spreads(MATURITY_SPREADS),
    )
    monkeypatch.setattr(
        fred_model, "get_rating_option_adjusted_spread", spreads(RATING_SPREADS)
    )
    yield FixedIncome(
        start_date="2024-11-01",
        end_date="2024-11-22",
        rounding=6,
        fred_api_key="test",
    )
    clear_active_cache()


def test_get_survival_curve(recorder, fixedincome):
    recorder.capture(fixedincome.get_survival_curve())
    recorder.capture(fixedincome.get_survival_curve(maturity=False))
    recorder.capture(
        fixedincome.get_survival_curve(
            ratings="BBB,CCC", recovery_rate=0.25, horizons="6M,1,4"
        )
    )
    recorder.capture(fixedincome.get_survival_curve(hazard_rates=True))


def test_get_credit_default_swap_spread(recorder, fixedincome):
    recorder.capture(fixedincome.get_credit_default_swap_spread())
    recorder.capture(fixedincome.get_credit_default_swap_spread(maturity=False))
    recorder.capture(
        fixedincome.get_credit_default_swap_spread(
            ratings=["AA", "B"], tenors="2,5", recovery_rate=0.3
        )
    )


def test_get_survival_curve_uses_the_spreads_and_curve_of_the_end_date(fixedincome):
    survival = fixedincome.get_survival_curve(horizons="5", rounding=12)
    zero_curve = fixedincome._latest_zero_curve("United States", "2024-11-22")
    # 2024-11-22 is the 16th business day of the month.
    bucket_spreads = np.array(list(MATURITY_SPREADS.values())) + 0.00005 * 15
    hazard = credit_curve_model.get_bond_implied_hazard_rates(
        list(credit_controller.ICE_BOFA_BUCKET_MATURITIES.values()),
        bucket_spreads,
        zero_curve,
    )

    assert survival.loc["5Y", "Corporate"] == pytest.approx(
        credit_curve_model.get_survival_probability(hazard, hazard.index, 5)
    )


def test_get_credit_default_swap_spread_by_rating_is_near_the_bond_spread(
    fixedincome,
):
    cds_spreads = fixedincome.get_credit_default_swap_spread(
        maturity=False, tenors="7", rounding=12
    )
    zero_curve = fixedincome._latest_zero_curve("United States", "2024-11-22")
    hazard = credit_curve_model.get_bond_implied_hazard_rates(
        [7], [RATING_SPREADS["BBB"] + 0.00005 * 15], zero_curve
    )

    assert cds_spreads.loc["7Y", "BBB"] == pytest.approx(
        cds_model.get_credit_default_swap(7, zero_curve, hazard, hazard.index)[
            "Par Spread"
        ]
    )
    # The CDS-equivalent spread of a flat hazard rate is close to the bond spread, apart
    # from CCC, where the basis between a bond and a CDS grows with the default risk.
    assert cds_spreads.loc["7Y"].to_numpy()[:-1] == pytest.approx(
        np.array(list(RATING_SPREADS.values()))[:-1] + 0.00005 * 15, rel=0.05
    )


def test_credit_curves_invalid(fixedincome):
    assert fixedincome.get_survival_curve(ratings="ZZZ").empty
    assert fixedincome.get_credit_default_swap_spread(recovery_rate=1.0).empty
