"""
Live checks of every data source the Finance Toolkit reads without an API key, run
weekly by the Live Sources workflow.

The checks cover the keyless Economics sources (Eurostat, ONS, BLS, IBGE, the Statistics
Bureau of Japan, the OECD, the IMF, BIS, ECB, Bank of England, Bank of Japan, Japan's
Ministry of Finance, the U.S. Treasury, the Federal Reserve Board, Freddie Mac, the NBER,
Shiller, Cboe, STOXX, EEX, the Global Macro Database, the Jordà-Schularick-Taylor
Macrohistory Database, the ESRB, the NGFS and the DNB scenario set), the keyless Fixed
Income sources (the ECB policy rates and Euribor, the Bundesbank, Bank of Canada,
Riksbank, Norges Bank, the Reserve Bank of Australia, the New York Fed, EIOPA and ESMA's
CEREP), Kenneth French's factor library, and Yahoo Finance (prices, exchange rates,
financial statements, statistics, option chains, the asset class proxies and the
commodity futures curves) together with the Cboe option chains it falls back on.

Each source is asked for a small, recent piece of data. Sources that update daily,
weekly, monthly or quarterly must return current data within a plausible range; sources
that publish a dataset once a year or less (the long-run histories and the stress test
scenarios) must be reachable and parse into the expected columns. A failing check names
its source, so the workflow opens one issue per broken source.

Sources that need a key (FinancialModelingPrep, FRED) are not checked here. Skipped
unless FINANCETOOLKIT_LIVE_SOURCES=1, since it needs network access.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from financetoolkit import historical_model, yfinance_model
from financetoolkit.economics import (
    bis_model,
    bls_model,
    boe_model,
    boj_model,
    cboe_model,
    dnb_model,
    ecb_model,
    eex_model,
    esrb_model,
    eurostat_model,
    frb_model,
    freddie_mac_model,
    gmdb_model,
    ibge_model,
    imf_model,
    macrohistory_model,
    mof_model,
    nber_model,
    ngfs_model,
    oecd_model,
    ons_model,
    sbj_model,
    shiller_model,
    stoxx_model,
    treasury_model,
    yfinance_model as futures_model,
)
from financetoolkit.economics.economics_controller import ASSET_CLASS_PROXIES
from financetoolkit.fixedincome import (
    boc_model,
    bundesbank_model,
    ecb_model as ecb_policy_model,
    eiopa_model,
    esma_model,
    euribor_model,
    fed_model,
    norgesbank_model,
    rba_model,
    riksbank_model,
)
from financetoolkit.options import options_model
from financetoolkit.performance import performance_model

pytestmark = pytest.mark.skipif(
    os.environ.get("FINANCETOOLKIT_LIVE_SOURCES") != "1",
    reason="Live source checks run only with FINANCETOOLKIT_LIVE_SOURCES=1.",
)

TODAY = datetime.now()
START = (TODAY - timedelta(days=200)).strftime("%Y-%m-%d")
END = TODAY.strftime("%Y-%m-%d")

# The yearly datasets are asked for a few years, since their latest year can be two or
# three years back (life tables, national accounts).
YEARLY_START = f"{TODAY.year - 6}-01-01"

# How old the latest observation may be: quarterly GDP is published one to three months
# after the quarter, monthly figures one to three months after the month (the UK labour
# market survey the latest) and daily rates within days. Kenneth French's daily factors
# are updated once a month, a month or two behind, so they count as monthly.
QUARTERLY_MAX_AGE_DAYS = 200
MONTHLY_MAX_AGE_DAYS = 130
DAILY_MAX_AGE_DAYS = 10

# The ticker the Yahoo Finance checks ask for: liquid, listed options, a long history.
YAHOO_TICKER = "AAPL"

# Each source: a request, the column to check, how often it updates and, optionally, the
# range its latest value must fall in. Without a range, rates must be decimals between
# -5% and 50%, indices levels between 20 and 1000 and exchange rates positive.
SOURCES = {
    # Eurostat, ONS, BLS, IBGE and the Statistics Bureau of Japan
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
    "IBGE unemployment rate": (ibge_model.get_unemployment_rate, "Brazil", "monthly"),
    "Statistics Bureau of Japan inflation rate": (
        sbj_model.get_inflation_rate,
        "Japan",
        "monthly",
    ),
    # OECD, one request per dataflow
    "OECD composite leading indicator": (
        lambda: oecd_model.get_composite_leading_indicator(START, END),
        "United States",
        "monthly",
        (80, 120),
    ),
    "OECD unemployment rate": (
        lambda: oecd_model.get_unemployment_rate("monthly", START, END),
        "United States",
        "monthly",
    ),
    "OECD consumer price index": (
        lambda: oecd_model.get_consumer_price_index("monthly", START, END),
        "United States",
        "monthly",
    ),
    "OECD long-term interest rate": (
        lambda: oecd_model.get_long_term_interest_rate("monthly", START, END),
        "United States",
        "monthly",
    ),
    "OECD short-term interest rate": (
        lambda: oecd_model.get_short_term_interest_rate("monthly", START, END),
        "United States",
        "monthly",
    ),
    "OECD GDP growth": (
        lambda: oecd_model.get_gross_domestic_product_growth(False, START, END),
        "United States",
        "quarterly",
        (-0.1, 0.1),
    ),
    "OECD house price index": (
        lambda: oecd_model.get_house_prices(True, True, START, END),
        "United States",
        "quarterly",
    ),
    # IMF and BIS
    "IMF consumer prices": (
        lambda: imf_model.get_inflation_rate(START, END),
        "Pakistan",
        "monthly",
    ),
    "BIS policy rates": (
        lambda: bis_model.get_central_bank_policy_rate(START, END),
        "United States",
        "daily",
    ),
    "BIS consumer prices": (
        lambda: bis_model.get_consumer_prices("inflation_rate", START, END),
        "United States",
        "monthly",
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
    # European Central Bank
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
    "ECB yield curve": (
        lambda: ecb_model.get_yield_curve(START, END).rename(
            columns={"2Y": "Euro Area"}
        ),
        "Euro Area",
        "daily",
    ),
    "ECB long-term convergence yields": (
        lambda: ecb_model.get_long_term_convergence_yields(START, END),
        "France",
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
    "ECB survey inflation expectations": (
        lambda: ecb_model.get_survey_inflation_expectations(START, END),
        "Euro Area",
        "quarterly",
    ),
    "ECB main refinancing rate": (
        lambda: ecb_policy_model.get_main_refinancing_operations().to_frame(
            "Euro Area"
        ),
        "Euro Area",
        "daily",
    ),
    "ECB deposit facility rate": (
        lambda: ecb_policy_model.get_deposit_facility().to_frame("Euro Area"),
        "Euro Area",
        "daily",
    ),
    "ECB marginal lending facility rate": (
        lambda: ecb_policy_model.get_marginal_lending_facility().to_frame("Euro Area"),
        "Euro Area",
        "daily",
    ),
    "ECB Euribor": (
        lambda: euribor_model.get_euribor_rate("3M").to_frame("Euro Area"),
        "Euro Area",
        "monthly",
    ),
    "ECB real Euribor": (
        lambda: euribor_model.get_euribor_rate("3M", nominal=False).to_frame(
            "Euro Area"
        ),
        "Euro Area",
        "monthly",
        (-0.15, 0.2),
    ),
    # Bank of England
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
    "Bank of England gilt curve": (
        lambda: boe_model.get_yield_curve(START, END).rename(
            columns={"5Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    "Bank of England nominal spot curve": (
        lambda: boe_model.get_spot_curve("nominal", START, END).rename(
            columns={"10Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    "Bank of England real spot curve": (
        lambda: boe_model.get_spot_curve("real", START, END).rename(
            columns={"10Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    "Bank of England implied inflation curve": (
        lambda: boe_model.get_spot_curve("inflation", START, END).rename(
            columns={"10Y": "United Kingdom"}
        ),
        "United Kingdom",
        "daily",
    ),
    # Japan
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
    # United States
    "Treasury 10-year yield": (
        lambda: treasury_model.get_long_term_interest_rate(START, END),
        "United States",
        "daily",
    ),
    "Treasury nominal yield curve": (
        lambda: treasury_model.get_yield_curve("nominal", START, END).rename(
            columns={"2 Yr": "United States"}
        ),
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
    "Treasury breakeven inflation": (
        lambda: treasury_model.get_breakeven_inflation_expectations(START, END).rename(
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
    "Federal Reserve excess bond premium": (
        lambda: frb_model.get_excess_bond_premium().rename(
            columns={"GZ Credit Spread": "United States"}
        ),
        "United States",
        "monthly",
    ),
    "New York Fed SOFR": (
        lambda: fed_model.get_secured_overnight_financing_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
    "New York Fed effective federal funds rate": (
        lambda: fed_model.get_effective_federal_funds_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
    "New York Fed overnight bank funding rate": (
        lambda: fed_model.get_overnight_banking_funding_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
    "New York Fed tri-party general collateral rate": (
        lambda: fed_model.get_tri_party_general_collateral_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
    "New York Fed broad general collateral rate": (
        lambda: fed_model.get_broad_general_collateral_rate()[["Rate"]].rename(
            columns={"Rate": "United States"}
        ),
        "United States",
        "daily",
    ),
    "Shiller stock market data": (
        lambda: shiller_model.get_stock_market_data().rename(
            columns={"Long Interest Rate": "United States"}
        ),
        "United States",
        "monthly",
    ),
    # Bundesbank and the other central banks' yield curves
    "Bundesbank yield curve": (
        lambda: bundesbank_model.get_yield_curve(START, END).rename(
            columns={"2Y": "Germany"}
        ),
        "Germany",
        "daily",
    ),
    "Bundesbank German breakeven inflation": (
        lambda: bundesbank_model.get_breakeven_inflation(START, END).rename(
            columns={"10Y": "Germany"}
        ),
        "Germany",
        "daily",
    ),
    "Bundesbank survey inflation expectations": (
        lambda: bundesbank_model.get_survey_inflation_expectations(START, END).rename(
            columns={"10Y": "Germany"}
        ),
        "Germany",
        "monthly",
    ),
    "Bundesbank corporate bond yield": (
        lambda: bundesbank_model.get_corporate_bond_yield(START, END)[
            ["Corporate"]
        ].rename(columns={"Corporate": "Germany"}),
        "Germany",
        "monthly",
    ),
    "Bundesbank real yield curve": (
        lambda: bundesbank_model.get_real_yield_curve(START, END).rename(
            columns={"10Y": "Germany"}
        ),
        "Germany",
        "daily",
    ),
    "Bundesbank inflation-linked bond yields": (
        # The bonds come and go as they are issued and mature, so their average is checked.
        lambda: bundesbank_model.get_inflation_linked_yields(START, END)
        .mean(axis=1)
        .to_frame("Germany"),
        "Germany",
        "daily",
        (-0.1, 0.2),
    ),
    "Bundesbank expected real rates": (
        lambda: bundesbank_model.get_expected_real_rates(START, END).rename(
            columns={"10Y": "Germany"}
        ),
        "Germany",
        "monthly",
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
    "RBA corporate bond yields": (
        lambda: rba_model.get_corporate_bond_yields()[["BBB 10Y"]].rename(
            columns={"BBB 10Y": "Australia"}
        ),
        "Australia",
        "monthly",
    ),
    "RBA government bond yields": (
        lambda: rba_model.get_yield_curve()[["10Y"]].rename(
            columns={"10Y": "Australia"}
        ),
        "Australia",
        "daily",
    ),
    # EIOPA
    "EIOPA risk-free rates": (
        lambda: eiopa_model.get_risk_free_rate_term_structures(
            "spot_no_va", START, END
        ).xs("10Y", axis=1, level=1),
        "Euro Area",
        "monthly",
    ),
    "EIOPA symmetric adjustment": (
        lambda: eiopa_model.get_symmetric_adjustment()[["Symmetric Adjustment"]].rename(
            columns={"Symmetric Adjustment": "Euro Area"}
        ),
        "Euro Area",
        "monthly",
    ),
    # Market data: STOXX, EEX and Yahoo Finance (Cboe follows below)
    "STOXX VSTOXX implied volatility": (
        lambda: stoxx_model.get_vstoxx_term_structure()[["1M"]].rename(
            columns={"1M": "Euro Area"}
        )
        / 100,
        "Euro Area",
        "daily",
        (0.03, 3),
    ),
    "EEX carbon price index level": (
        lambda: eex_model.get_carbon_price(START, END),
        "European Union",
        "daily",
    ),
    "Yahoo Finance daily prices": (
        lambda: yfinance_model.get_historical_data(YAHOO_TICKER, START, END)[
            ["Adj Close"]
        ].rename(columns={"Adj Close": YAHOO_TICKER}),
        YAHOO_TICKER,
        "daily",
        (1, 100_000),
    ),
    "Yahoo Finance exchange rate": (
        lambda: yfinance_model.get_historical_data("EURUSD=X", START, END)[
            ["Close"]
        ].rename(columns={"Close": "EURUSD"}),
        "EURUSD",
        "daily",
        (0.5, 2.5),
    ),
    # Kenneth French's data library: daily returns, published monthly
    "Kenneth French five factors": (
        performance_model.obtain_fama_and_french_dataset,
        "Mkt-RF",
        "monthly",
        (-0.25, 0.25),
    ),
    "Kenneth French three factors": (
        performance_model.obtain_fama_and_french_three_factor_dataset,
        "Mkt-RF",
        "monthly",
        (-0.25, 0.25),
    ),
    "Kenneth French momentum factor": (
        performance_model.obtain_carhart_momentum_dataset,
        "Mom",
        "monthly",
        (-0.25, 0.25),
    ),
}

# The Cboe volatility indices behind the implied volatility term structures, in percent.
CBOE_VOLATILITY_INDICES = [
    "VIX",
    "VIX9D",
    "VIX3M",
    "VIX6M",
    "VIX1Y",
    "VXEEM",
    "VVIX",
    "VXTLT",
    "OVX",
    "GVZ",
]

for _symbol in CBOE_VOLATILITY_INDICES:
    SOURCES[f"Cboe {_symbol} implied volatility"] = (
        lambda symbol=_symbol: (cboe_model.get_index(symbol) / 100).to_frame(
            "United States"
        ),
        "United States",
        "daily",
        (0.03, 3),
    )

# Datasets published once a year or less: each must be reachable and parse into the
# expected column, whose latest value must lie in the given range; their age is not
# checked. Each source: a request, the column to check and the range.
STATIC_SOURCES = {
    "Eurostat life tables": (
        lambda: eurostat_model.get_life_table(
            "life_expectancy", "total", YEARLY_START, END, ["DE"]
        ),
        ("Germany", 0),
        (60, 100),
    ),
    "OECD household savings rate": (
        lambda: oecd_model.get_household_savings_rate(False, YEARLY_START, END),
        "United States",
        (-0.2, 0.5),
    ),
    "OECD annual GDP": (
        lambda: oecd_model.get_annual_gross_domestic_product(YEARLY_START, END),
        "United States",
        (1_000, 1_000_000),
    ),
    "OECD output gap": (
        lambda: oecd_model.get_output_gap(YEARLY_START, END),
        "United States",
        (-0.2, 0.2),
    ),
    "OECD population": (
        lambda: oecd_model.get_population(None, YEARLY_START, END),
        "United States",
        (100_000_000, 1_000_000_000),
    ),
    "Bank of England millennium of macroeconomic data": (
        boe_model.get_millennium_data,
        "Bank Rate",
        (0, 0.25),
    ),
    "JST Macrohistory Database": (
        lambda: macrohistory_model.get_asset_returns("bills", False),
        "United States",
        (-0.05, 0.25),
    ),
    "Global Macro Database": (
        lambda: gmdb_model.get_inflation_rate(
            gmdb_model.collect_global_macro_database_dataset()
        ),
        "United States",
        (-0.05, 0.5),
    ),
    "ESRB macro-financial scenario": (
        lambda: esrb_model.get_macro_financial_scenario("adverse"),
        ("United States", "Unemployment rate"),
        (0, 0.3),
    ),
    "Federal Reserve supervisory scenario": (
        lambda: frb_model.get_supervisory_scenario("adverse"),
        ("United States", "Unemployment rate"),
        (0, 0.3),
    ),
    "NGFS carbon price scenario": (
        lambda: ngfs_model.get_carbon_price("Net Zero 2050", "REMIND-MAgPIE"),
        "World",
        (1, 100_000),
    ),
}


@pytest.mark.parametrize("name", SOURCES)
def test_source_is_current_and_plausible(name):
    fetch, country, frequency, *bounds = SOURCES[name]

    data = fetch()

    assert not data.empty, f"{name} returned no data"
    assert country in data.columns, f"{name} has no {country} column"

    series = data[country].dropna()
    assert not series.empty, f"{name} has no values for {country}"

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
    if bounds:
        low, high = bounds[0]
        assert (
            low < series.iloc[-1] < high
        ), f"{name} has an implausible value {series.iloc[-1]}"
    elif "exchange" in name:
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


@pytest.mark.parametrize("name", STATIC_SOURCES)
def test_source_is_reachable_and_parses(name):
    fetch, column, (low, high) = STATIC_SOURCES[name]

    data = fetch()

    assert not data.empty, f"{name} returned no data"
    assert column in data.columns, f"{name} has no {column} column"

    series = data[column].dropna()
    assert not series.empty, f"{name} has no values for {column}"
    assert (
        low <= series.iloc[-1] <= high
    ), f"{name} has an implausible value {series.iloc[-1]}"


# The checks below need more than one column or request. Each is parametrized by the
# name of its source, which the workflow uses as the title of the issue it opens.


@pytest.mark.parametrize("name", ["NBER recession indicator"])
def test_recession_indicator_covers_the_current_month(name):
    indicator = nber_model.get_recession_indicator()["United States"]

    assert indicator.index[-1].strftime("%Y-%m") == TODAY.strftime("%Y-%m")
    assert set(indicator.unique()) <= {0, 1}
    # The 2020 recession, as the NBER dated it.
    assert indicator.loc["2020-03-01":"2020-04-01"].tolist() == [1, 1]


@pytest.mark.parametrize("name", ["IMF country list"])
def test_imf_country_list(name):
    countries = imf_model.get_countries()

    assert len(countries) > 150, f"{name} lists only {len(countries)} countries"
    assert "United States" in countries["Country"].tolist()


# The DNB scenario set is a workbook of 20,000 scenarios per variable, which takes a
# minute or two to download and read.
@pytest.mark.timeout(900)
@pytest.mark.parametrize("name", ["DNB scenario set"])
def test_dnb_scenario_set(name):
    link, quarter = dnb_model.get_scenario_set_link("market_consistent")
    assert link.startswith("https://"), f"{name} has no market-consistent set"
    pd.Period(quarter, freq="Q")

    scenarios = dnb_model.get_scenario_set("real_world")

    assert "Nominal rate 1Y" in scenarios.columns.get_level_values(0)
    one_year_rates = scenarios["Nominal rate 1Y"].loc[1]
    assert (
        -0.1 < one_year_rates.median() < 0.2
    ), f"{name} has an implausible median one-year rate {one_year_rates.median()}"


@pytest.mark.parametrize("name", ["ESMA CEREP rating statistics"])
def test_esma_rating_statistics(name):
    # CEREP publishes a year in the first half of the next, so two years back is final.
    year = TODAY.year - 2
    agency = esma_model.resolve_agency("S&P")
    start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")

    default_rates = esma_model.get_default_rates(agency, "corporate", start, end)

    assert not default_rates.empty, f"{name} returned no default rates for {year}"
    assert {"AAA", "BBB", "CCC"} <= set(default_rates.index)
    assert default_rates["Default Rate"].dropna().between(0, 1).all()

    transition_matrix = esma_model.get_transition_matrix(
        agency, "corporate", start, end
    )

    assert not transition_matrix.empty, f"{name} returned no transitions for {year}"


# Each statement with an item it must report.
YAHOO_STATEMENTS = {
    "Yahoo Finance income statement": ("income", "TotalRevenue"),
    "Yahoo Finance balance sheet": ("balance", "TotalAssets"),
    "Yahoo Finance cash flow statement": ("cashflow", "OperatingCashFlow"),
}


@pytest.mark.parametrize("name", YAHOO_STATEMENTS)
def test_yahoo_finance_financial_statement(name):
    statement, item = YAHOO_STATEMENTS[name]

    # Annual statements are filed within months of the fiscal year, quarterly ones within
    # weeks of the quarter.
    for quarter, max_age in [(False, 550), (True, QUARTERLY_MAX_AGE_DAYS)]:
        data = yfinance_model.get_financial_statement(
            YAHOO_TICKER, statement, quarter=quarter
        )

        assert not data.empty, f"{name} returned no data (quarter={quarter})"
        assert item in data.index, f"{name} has no {item} (quarter={quarter})"

        latest = max(data.columns)
        assert latest.to_timestamp(how="end") >= pd.Timestamp(
            TODAY - timedelta(days=max_age)
        ), f"{name} is not current: the latest period is {latest} (quarter={quarter})"
        assert (
            data.loc[item, latest] > 0
        ), f"{name} has an implausible {item} {data.loc[item, latest]}"


@pytest.mark.parametrize("name", ["Yahoo Finance statistics"])
def test_yahoo_finance_statistics(name):
    statistics = yfinance_model.get_historical_statistics(YAHOO_TICKER)

    assert statistics.get("Currency") == "USD", f"{name} has no USD trading currency"
    assert yfinance_model.get_reported_currency(YAHOO_TICKER) == "USD"

    periods = pd.PeriodIndex([str(TODAY.year - 1)], freq="Y")
    statement = yfinance_model.get_statistics_statement(YAHOO_TICKER, periods)

    assert statement.loc["financialCurrency"].eq("USD").all()


@pytest.mark.parametrize("name", ["Yahoo Finance option chain"])
def test_yahoo_finance_option_chain(name, monkeypatch):
    # Without Yahoo Finance the option functions fall back on Cboe, which is checked on its
    # own below; switching that off makes this check fail when Yahoo Finance does.
    monkeypatch.setattr(
        options_model.cboe_model, "get_option_chain", lambda ticker: pd.DataFrame()
    )

    expiry_dates = options_model.get_option_expiry_dates(YAHOO_TICKER)

    assert expiry_dates, f"{name} has no expiry dates"
    assert pd.Timestamp(expiry_dates[-1]) > pd.Timestamp(TODAY)

    chain = options_model.get_option_chains([YAHOO_TICKER], expiry_dates[0])

    assert not chain.empty, f"{name} returned no options for {expiry_dates[0]}"
    assert {"Bid", "Ask", "Implied Volatility"} <= set(chain.columns)
    assert chain.index.get_level_values("Strike Price").min() > 0


@pytest.mark.parametrize("name", ["Cboe option chain"])
def test_cboe_option_chain(name):
    chain = cboe_model.get_option_chain(YAHOO_TICKER)

    assert not chain.empty, f"{name} returned no options"
    assert {"Expiration", "Put", "Strike", "Bid", "Ask"} <= set(chain.columns)
    assert chain["Put"].any() and not chain["Put"].all()
    assert pd.to_datetime(chain["Expiration"]).max() > pd.Timestamp(TODAY)
    assert (chain["Strike"] > 0).all()


@pytest.mark.parametrize("name", ["Yahoo Finance asset class proxies"])
def test_asset_class_proxies(name):
    tickers = list(ASSET_CLASS_PROXIES.values())

    # Without an API key, the historical data comes from Yahoo Finance.
    historical_data, _ = historical_model.get_historical_data(
        tickers,
        enforce_source="YahooFinance",
        start=START,
        end=END,
        fill_nan=False,
        show_ticker_seperation=False,
    )

    assert not historical_data.empty, f"{name} returned no data"
    prices = historical_data["Adj Close"]

    for asset_class, ticker in ASSET_CLASS_PROXIES.items():
        assert ticker in prices.columns, f"{name} has no {ticker} ({asset_class})"
        series = prices[ticker].dropna()
        assert not series.empty, f"{name} has no prices for {ticker} ({asset_class})"
        assert series.index[-1].to_timestamp(how="end") >= pd.Timestamp(
            TODAY - timedelta(days=DAILY_MAX_AGE_DAYS)
        ), f"{name} is not current for {ticker}: the latest is {series.index[-1]}"
        assert series.iloc[-1] > 0


@pytest.mark.parametrize(
    "name",
    [
        f"Yahoo Finance {commodity} forward curve"
        for commodity in futures_model.COMMODITY_TICKERS
    ],
)
def test_commodity_forward_curve(name):
    commodity = name.removeprefix("Yahoo Finance ").removesuffix(" forward curve")

    # A year of contracts: the front month may have expired, and grains trade only five
    # delivery months a year.
    curve = futures_model.get_commodity_forward_curve(
        commodity, START, END, contracts=12
    )

    assert not curve.empty, f"{name} returned no contracts"
    latest = curve.iloc[-1].dropna()
    assert (
        len(latest) >= 3
    ), f"{name} has only {len(latest)} contracts with a current price"
    assert curve.index[-1].to_timestamp(how="end") >= pd.Timestamp(
        TODAY - timedelta(days=DAILY_MAX_AGE_DAYS)
    ), f"{name} is not current: the latest observation is {curve.index[-1]}"
    assert (latest > 0).all(), f"{name} has non-positive prices"
