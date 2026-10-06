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
    bls_model,
    boe_model,
    boj_model,
    ecb_model,
    eurostat_model,
    frb_model,
    freddie_mac_model,
    ibge_model,
    imf_model,
    mof_model,
    nber_model,
    ons_model,
    sbj_model,
    shiller_model,
    treasury_model,
)
from financetoolkit.fixedincome import (
    boc_model,
    bundesbank_model,
    eiopa_model,
    fed_model,
    norgesbank_model,
    riksbank_model,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FINANCETOOLKIT_LIVE_SOURCES") != "1",
    reason="Live source checks run only with FINANCETOOLKIT_LIVE_SOURCES=1.",
)

TODAY = datetime.now()
START = (TODAY - timedelta(days=200)).strftime("%Y-%m-%d")
END = TODAY.strftime("%Y-%m-%d")

# How old the latest observation may be: quarterly GDP is published one to three months
# after the quarter, monthly figures one to three months after the month (the UK labour
# market survey the latest) and daily rates within days.
QUARTERLY_MAX_AGE_DAYS = 200
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
    "Eurostat GDP growth": (
        lambda: eurostat_model.get_gross_domestic_product_growth(START, END),
        "Euro Area",
        "quarterly",
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
    "ONS GDP growth": (
        ons_model.get_gross_domestic_product_growth,
        "United Kingdom",
        "quarterly",
    ),
    "BLS unemployment rate": (
        bls_model.get_unemployment_rate,
        "United States",
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
    "Treasury real yield curve": (
        lambda: treasury_model.get_real_yield_curve(START, END).rename(
            columns={"10 Year": "United States"}
        ),
        "United States",
        "daily",
    ),
    "Freddie Mac mortgage rate": (
        freddie_mac_model.get_mortgage_rate_30_year,
        "United States",
        "daily",
    ),
    "Federal Reserve industrial production index": (
        frb_model.get_industrial_production_index,
        "United States",
        "monthly",
    ),
    "BIS consumer prices": (
        lambda: bis_model.get_consumer_prices("inflation_rate", START, END),
        "United States",
        "monthly",
    ),
    "IBGE unemployment rate": (ibge_model.get_unemployment_rate, "Brazil", "monthly"),
    "ECB yield curve": (
        lambda: ecb_model.get_yield_curve(START, END).rename(
            columns={"2Y": "Euro Area"}
        ),
        "Euro Area",
        "daily",
    ),
    "Bank of England gilt curve": (
        lambda: boe_model.get_yield_curve(START, END).rename(
            columns={"5Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    "ECB long-term convergence yields": (
        lambda: ecb_model.get_long_term_convergence_yields(START, END),
        "France",
        "monthly",
    ),
    "Bundesbank yield curve": (
        lambda: bundesbank_model.get_yield_curve(START, END).rename(
            columns={"2Y": "Germany"}
        ),
        "Germany",
        "daily",
    ),
    "Bundesbank German breakeven inflation": (
        lambda: bundesbank_model.get_breakeven_inflation(START, END)
        .iloc[:, -1:]
        .set_axis(["Germany"], axis=1),
        "Germany",
        "daily",
    ),
    "Bank of Canada yield curve": (
        lambda: boc_model.get_yield_curve(START, END).rename(columns={"10Y": "Canada"}),
        "Canada",
        "daily",
    ),
    "Riksbank yield curve": (
        lambda: riksbank_model.get_yield_curve(START, END).rename(
            columns={"10Y": "Sweden"}
        ),
        "Sweden",
        "daily",
    ),
    "Norges Bank yield curve": (
        lambda: norgesbank_model.get_yield_curve(START, END).rename(
            columns={"10Y": "Norway"}
        ),
        "Norway",
        "daily",
    ),
    "Federal Reserve excess bond premium": (
        lambda: frb_model.get_excess_bond_premium().rename(
            columns={"GZ Credit Spread": "United States"}
        ),
        "United States",
        "monthly",
    ),
    "EIOPA risk-free rates": (
        lambda: eiopa_model.get_risk_free_rate_term_structures(
            "spot_no_va", START, END
        ).xs("10Y", axis=1, level=1),
        "Euro Area",
        "monthly",
    ),
    "Bank of England implied inflation curve": (
        lambda: boe_model.get_spot_curve("inflation", START, END).rename(
            columns={"10Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    "BIS daily exchange rates": (
        lambda: bis_model.get_exchange_rates("daily", START, END),
        "Japan",
        "daily",
    ),
    "BIS commercial property price index": (
        lambda: bis_model.get_commercial_property_prices(START, END),
        "United States",
        "quarterly",
    ),
    "IMF consumer prices": (
        lambda: imf_model.get_inflation_rate(START, END),
        "Pakistan",
        "monthly",
    ),
    "Shiller stock market data": (
        lambda: shiller_model.get_stock_market_data().rename(
            columns={"Long Interest Rate": "United States"}
        ),
        "United States",
        "monthly",
    ),
    "ECB corporate borrowing cost": (
        lambda: ecb_model.get_corporate_borrowing_cost(START, END),
        "Euro Area",
        "monthly",
    ),
    "ECB financial stress level": (
        lambda: ecb_model.get_financial_stress_index(START, END).clip(upper=0.49),
        "Euro Area",
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
    max_age = {
        "quarterly": QUARTERLY_MAX_AGE_DAYS,
        "monthly": MONTHLY_MAX_AGE_DAYS,
        "daily": DAILY_MAX_AGE_DAYS,
    }[frequency]

    assert latest >= pd.Timestamp(
        TODAY - timedelta(days=max_age)
    ), f"{name} is not current: the latest observation is {series.index[-1]}"

    # Rates are decimals and indices are levels; both must stay in a plausible range.
    if "exchange" in name:
        assert (
            0 < series.iloc[-1] < 1_000_000
        ), f"{name} has an implausible rate {series.iloc[-1]}"
    elif "index" in name:
        assert (
            20 < series.iloc[-1] < 1000
        ), f"{name} has an implausible level {series.iloc[-1]}"
    else:
        assert (
            -0.05 < series.iloc[-1] < 0.5
        ), f"{name} has an implausible rate {series.iloc[-1]}"


def test_recession_indicator_covers_the_current_month():
    indicator = nber_model.get_recession_indicator()["United States"]

    assert indicator.index[-1].strftime("%Y-%m") == TODAY.strftime("%Y-%m")
    assert set(indicator.unique()) <= {0, 1}
    # The 2020 recession, as the NBER dated it.
    assert indicator.loc["2020-03-01":"2020-04-01"].tolist() == [1, 1]
