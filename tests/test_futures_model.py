"""Offline tests for the commodity futures curve: last trading days, the curve as it stood
on each day and the term structure, with crude oil and gold settlements from Yahoo Finance.
"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit import Economics
from financetoolkit.cache.cache_controller import clear_active_cache
from financetoolkit.economics import futures_model, yfinance_model
from financetoolkit.economics.helpers import get_exchange_business_days

# The NYMEX WTI contracts December 2026 to November 2027 as Yahoo Finance listed them on
# 2026-11-08, after the October (last trading day 2026-09-22) and November 2026 contracts
# (2026-10-20) had expired and were no longer published.
CRUDE_OIL = {
    "2026-09-15": [
        95.51,
        91.02,
        87.36,
        84.51,
        82.13,
        80.11,
        78.41,
        76.95,
        75.72,
        74.71,
        73.88,
        73.2,
    ],
    "2026-10-20": [
        97.82,
        96.02,
        94.05,
        92.18,
        90.5,
        89.03,
        87.67,
        86.35,
        85.09,
        83.9,
        82.81,
        81.83,
    ],
    "2026-10-21": [
        97.15,
        95.46,
        93.71,
        91.99,
        90.38,
        88.93,
        87.59,
        86.3,
        85.07,
        83.94,
        82.89,
        81.95,
    ],
    "2026-10-30": [
        104.12,
        102.21,
        99.94,
        97.49,
        95.12,
        92.96,
        90.96,
        89.08,
        87.31,
        85.7,
        84.27,
        83.01,
    ],
    "2026-11-02": [
        96.46,
        94.8,
        92.99,
        91.13,
        89.4,
        87.84,
        86.36,
        84.93,
        83.55,
        82.26,
        81.1,
        80.05,
    ],
    "2026-11-03": [
        96.09,
        94.77,
        93.0,
        91.14,
        89.43,
        87.9,
        86.45,
        85.08,
        83.75,
        82.5,
        81.35,
        80.31,
    ],
    "2026-11-04": [
        88.12,
        87.1,
        85.76,
        84.27,
        82.85,
        81.57,
        80.35,
        79.2,
        78.1,
        77.08,
        76.15,
        75.32,
    ],
    "2026-11-05": [
        90.52,
        89.2,
        87.39,
        85.44,
        83.63,
        82.05,
        80.63,
        79.33,
        78.12,
        77.01,
        76.0,
        75.11,
    ],
    "2026-11-06": [
        90.83,
        89.59,
        88.0,
        86.31,
        84.73,
        83.29,
        81.95,
        80.69,
        79.49,
        78.38,
        77.34,
        76.42,
    ],
}


def crude_oil_curve() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        CRUDE_OIL,
        orient="index",
        columns=[
            str(month) for month in pd.period_range("2026-12", "2027-11", freq="M")
        ],
    ).set_axis(pd.PeriodIndex(list(CRUDE_OIL), freq="D"))


@pytest.fixture
def economics(monkeypatch):
    clear_active_cache()
    monkeypatch.setattr(
        yfinance_model,
        "get_commodity_forward_curve",
        lambda commodity, start_date, end_date, contracts: crude_oil_curve(),
    )
    yield Economics(start_date="2026-09-01", end_date="2026-11-06")
    clear_active_cache()


@pytest.mark.parametrize(
    ("rule", "delivery_month", "last_trading_day"),
    [
        # The 25th is a Sunday, so four business days before it; Yahoo's CL=F rolled to
        # the December contract on 2026-10-21.
        ("crude_oil", "2026-11", "2026-10-20"),
        ("crude_oil", "2026-10", "2026-09-22"),
        ("crude_oil", "2025-05", "2025-04-22"),
        # The third last business day of October; Yahoo's NG=F rolled on 2026-10-29.
        ("natural_gas", "2026-11", "2026-10-28"),
        # Thanksgiving (2026-11-26) is not a business day.
        ("natural_gas", "2026-12", "2026-11-25"),
        ("metals", "2026-11", "2026-11-25"),
        ("grains", "2026-12", "2026-12-14"),
    ],
)
def test_the_last_trading_day_follows_the_contract_specification(
    rule, delivery_month, last_trading_day
):
    business_days = get_exchange_business_days("2025-01-01", "2027-12-31")

    assert futures_model.get_last_trading_day(
        rule, pd.Period(delivery_month, freq="M"), business_days
    ) == pd.Timestamp(last_trading_day)


def test_the_curve_of_a_day_starts_with_its_own_front_and_leaves_expired_ones_empty():
    curve = crude_oil_curve()
    curve["2026-10"] = np.nan
    curve["2026-11"] = np.nan
    business_days = get_exchange_business_days("2026-08-01", "2027-12-31")
    last_trading_days = pd.Series(
        {
            label: futures_model.get_last_trading_day(
                "crude_oil", pd.Period(label, freq="M"), business_days
            )
            for label in curve.columns
        }
    )

    prices, labels = futures_model.get_constant_maturity_curve(
        curve, last_trading_days, contracts=12
    )

    # On 2026-09-15 the October and November contracts, published no more, were the front
    # and +1: they are empty rather than filled with the December contract.
    september = pd.Period("2026-09-15", freq="D")
    assert labels.loc[september, "Front"] == "2026-10"
    assert np.isnan(prices.loc[september, "Front"])
    assert labels.loc[september, "+1"] == "2026-11"
    assert labels.loc[september, "+2"] == "2026-12"
    assert prices.loc[september, "+2"] == pytest.approx(95.51)
    assert labels.loc[september, "+11"] == "2027-09"

    # The November contract traded up to 2026-10-20 and December is the front a day later.
    assert labels.loc[pd.Period("2026-10-20", freq="D"), "Front"] == "2026-11"
    assert labels.loc[pd.Period("2026-10-21", freq="D"), "Front"] == "2026-12"
    assert prices.loc[pd.Period("2026-10-21", freq="D"), "Front"] == pytest.approx(
        97.15
    )


def test_the_term_structure_matches_the_spreads_worked_out_by_hand(economics):
    term_structure = economics.get_commodity_term_structure("Crude Oil", rounding=6)
    november = term_structure.loc[pd.Period("2026-11-06", freq="D")]

    assert november["Front Contract"] == "2026-12"
    assert november["Far Contract"] == "2027-11"
    assert november["Months"] == 11
    assert november["Spread"] == pytest.approx(14.41)
    assert november["Slope"] == pytest.approx(76.42 / 90.83 - 1, abs=1e-6)
    assert november["Slope"] == pytest.approx(-0.1586, abs=1e-4)
    assert november["Roll Yield"] == pytest.approx(
        np.log(90.83 / 76.42) * 12 / 11, abs=1e-6
    )
    assert november["Structure"] == "Backwardation"

    october = term_structure.loc[pd.Period("2026-10-30", freq="D")]
    assert october["Spread"] == pytest.approx(21.11)
    assert october["Slope"] == pytest.approx(-0.2027, abs=1e-4)

    # The October 2026 front of 2026-09-15 expired and is no longer published.
    september = term_structure.loc[pd.Period("2026-09-15", freq="D")]
    assert september["Front Contract"] == "2026-10"
    assert np.isnan(september["Spread"])


def test_a_fixed_calendar_spread_keeps_its_full_history(economics):
    term_structure = economics.get_commodity_term_structure(
        "Crude Oil", constant_maturity=False
    )

    assert term_structure["Spread"].max() == pytest.approx(22.31)
    assert term_structure["Spread"].idxmax() == pd.Period("2026-09-15", freq="D")
    assert term_structure.loc[pd.Period("2026-09-15", freq="D"), "Front"] == 95.51
    assert term_structure.loc[pd.Period("2026-09-15", freq="D"), "Far"] == 73.2
    assert set(term_structure["Front Contract"]) == {"2026-12"}


def test_the_change_of_each_contract(economics):
    growth = economics.get_commodity_forward_curve("Crude Oil", growth=True, lag=5)
    november = growth.loc[pd.Period("2026-11-06", freq="D")]

    assert november["2026-12"] == pytest.approx(-0.1276, abs=1e-4)
    assert november["2027-11"] == pytest.approx(-0.0794, abs=1e-4)


def test_several_commodities_come_as_commodity_and_field(economics):
    term_structure = economics.get_commodity_term_structure("Crude Oil, Gold")

    assert list(term_structure.columns.get_level_values(0).unique()) == [
        "Crude Oil",
        "Gold",
    ]
    assert term_structure[("Crude Oil", "Structure")].dropna().eq("Backwardation").all()


def test_gold_in_contango_has_a_positive_slope():
    labels = pd.DataFrame(
        [["2026-11", "2026-12", "2027-10", "2027-11"]],
        index=pd.PeriodIndex(["2026-11-06"], freq="D"),
        columns=["Front", "+1", "+2", "+3"],
    )
    prices = pd.DataFrame(
        [[4064.5, 4074.9, 4241.5, 4258.3]], index=labels.index, columns=labels.columns
    )

    gold = futures_model.get_term_structure(prices, labels).iloc[0]

    assert gold["Far Contract"] == "2027-10"
    assert gold["Slope"] == pytest.approx(0.0435, abs=1e-4)
    assert gold["Spread"] == pytest.approx(-177.0)
    assert gold["Roll Yield"] < 0
    assert gold["Structure"] == "Contango"
