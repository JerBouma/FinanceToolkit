"""
Live checks of the keyless Economics sources, run weekly by the Live Sources workflow.

Each source is asked for recent data and the answer is checked for being current and
plausible, so a source that changed its format or stopped updating is noticed before
users do. Skipped unless FINANCETOOLKIT_LIVE_SOURCES=1, since it needs network access.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from financetoolkit.economics import (
    bis_model,
    boe_model,
    boj_model,
    ecb_model,
    eurostat_model,
    mof_model,
    ons_model,
    sbj_model,
    treasury_model,
)
from financetoolkit.fixedincome import fed_model

pytestmark = pytest.mark.skipif(
    os.environ.get("FINANCETOOLKIT_LIVE_SOURCES") != "1",
    reason="Live source checks run only with FINANCETOOLKIT_LIVE_SOURCES=1.",
)

TODAY = datetime.now()
START = (TODAY - timedelta(days=200)).strftime("%Y-%m-%d")
END = TODAY.strftime("%Y-%m-%d")

# How old the latest observation may be: monthly figures are published one to three
# months after the month (the UK labour market survey the latest), daily rates within days.
MONTHLY_MAX_AGE_DAYS = 130
DAILY_MAX_AGE_DAYS = 10

SOURCES = {
    "Eurostat inflation rate": (
        lambda: eurostat_model.get_inflation_rate(START, END),
        "Euro Area",
        "monthly",
    ),
    "Eurostat consumer price index": (
        lambda: eurostat_model.get_consumer_price_index(START, END),
        "Germany",
        "monthly",
    ),
    "Eurostat unemployment rate": (
        lambda: eurostat_model.get_unemployment_rate(START, END),
        "Euro Area",
        "monthly",
    ),
    "ONS inflation rate": (ons_model.get_inflation_rate, "United Kingdom", "monthly"),
    "ONS consumer price index": (
        ons_model.get_consumer_price_index,
        "United Kingdom",
        "monthly",
    ),
    "ONS unemployment rate": (
        ons_model.get_unemployment_rate,
        "United Kingdom",
        "monthly",
    ),
    "Statistics Bureau of Japan inflation rate": (
        sbj_model.get_inflation_rate,
        "Japan",
        "monthly",
    ),
    "BIS policy rates": (
        lambda: bis_model.get_central_bank_policy_rate(START, END),
        "United States",
        "daily",
    ),
    "ECB 10-year yield": (
        lambda: ecb_model.get_long_term_interest_rate(START, END),
        "Euro Area",
        "daily",
    ),
    "ECB euro short-term rate": (
        lambda: ecb_model.get_overnight_rate(START, END),
        "Euro Area",
        "daily",
    ),
    "Bank of England gilt yield": (
        lambda: boe_model.get_long_term_interest_rate(START, END),
        "United Kingdom",
        "daily",
    ),
    "Bank of England SONIA": (
        lambda: boe_model.get_overnight_rate(START, END),
        "United Kingdom",
        "daily",
    ),
    "Bank of Japan call rate": (
        lambda: boj_model.get_overnight_rate(START, END),
        "Japan",
        "daily",
    ),
    "Ministry of Finance JGB yield": (
        mof_model.get_long_term_interest_rate,
        "Japan",
        "daily",
    ),
    "Treasury 10-year yield": (
        lambda: treasury_model.get_long_term_interest_rate(START, END),
        "United States",
        "daily",
    ),
    "New York Fed SOFR": (
        lambda: fed_model.get_secured_overnight_financing_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
}


@pytest.mark.parametrize("name", SOURCES)
def test_source_is_current_and_plausible(name):
    fetch, country, frequency = SOURCES[name]

    data = fetch()

    assert not data.empty, f"{name} returned no data"
    assert country in data.columns, f"{name} has no {country} column"

    series = data[country].dropna()
    latest = series.index[-1].to_timestamp(how="end")
    max_age = MONTHLY_MAX_AGE_DAYS if frequency == "monthly" else DAILY_MAX_AGE_DAYS

    assert latest >= pd.Timestamp(
        TODAY - timedelta(days=max_age)
    ), f"{name} is not current: the latest observation is {series.index[-1]}"

    # Rates are decimals and indices are levels; both must stay in a plausible range.
    if "index" in name:
        assert (
            20 < series.iloc[-1] < 1000
        ), f"{name} has an implausible level {series.iloc[-1]}"
    else:
        assert (
            -0.05 < series.iloc[-1] < 0.5
        ), f"{name} has an implausible rate {series.iloc[-1]}"
