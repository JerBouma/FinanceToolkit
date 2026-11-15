"""
Live checks of the data sources the Finance Toolkit reads with an API key,
FinancialModelingPrep and FRED, run weekly by the Live Sources workflow next to the
checks of test_live_sources.py.

Every public method that reads FinancialModelingPrep is called once, the Toolkit's with
enforce_source="FinancialModelingPrep" so that the Yahoo Finance fallback cannot hide a
broken endpoint, and must return data; the price history and the quote must also be
current. Every FRED series group is asked for its recent observations, which must be
current for its frequency.

The FinancialModelingPrep checks run when FINANCIAL_MODELING_PREP_API_KEY is set and the
FRED checks when FRED_API_KEY is set; without a key they are skipped. Like the other live
checks, they run only with FINANCETOOLKIT_LIVE_SOURCES=1.
"""

import os
from datetime import timedelta

import pandas as pd
import pytest

from financetoolkit import Discovery, Toolkit
from financetoolkit.economics.sources import fred_model as economics_fred_model
from financetoolkit.fixedincome import fred_model
from tests.live_sources.test_live_sources import (
    DAILY_MAX_AGE_DAYS,
    END,
    MONTHLY_MAX_AGE_DAYS,
    START,
    TODAY,
    fetch_or_explain,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FINANCETOOLKIT_LIVE_SOURCES") != "1",
    reason="Live source checks run only with FINANCETOOLKIT_LIVE_SOURCES=1.",
)

FMP_API_KEY = os.environ.get("FINANCIAL_MODELING_PREP_API_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

needs_fmp = pytest.mark.skipif(
    not FMP_API_KEY, reason="FINANCIAL_MODELING_PREP_API_KEY is not set."
)
needs_fred = pytest.mark.skipif(not FRED_API_KEY, reason="FRED_API_KEY is not set.")

# The company and fund the FinancialModelingPrep checks ask for: liquid, covered by
# analysts, paying dividends, with segments and a long history.
TICKER = "AAPL"
FUND = "SPY"

# The quarter before the last, so the statements and transcripts of the latest quarter
# need not have been published.
STATEMENT_START = (TODAY - timedelta(days=800)).strftime("%Y-%m-%d")


def toolkit(ticker: str = TICKER, start_date: str = STATEMENT_START) -> Toolkit:
    """A Toolkit that reads FinancialModelingPrep only."""
    return Toolkit(
        ticker,
        api_key=FMP_API_KEY,
        start_date=start_date,
        end_date=END,
        enforce_source="FinancialModelingPrep",
        progress_bar=False,
        use_cached_data=False,
    )


def discovery() -> Discovery:
    """A Discovery instance that retrieves everything afresh."""
    return Discovery(api_key=FMP_API_KEY, use_cached_data=False)


def fund() -> Toolkit:
    """A Toolkit of one fund, for the ETF module."""
    return Toolkit(
        FUND,
        api_key=FMP_API_KEY,
        start_date=START,
        end_date=END,
        benchmark_ticker=None,
        progress_bar=False,
        use_cached_data=False,
    )


# Each check: the call that reads FinancialModelingPrep.
FMP_SOURCES = {
    # Company data of the Toolkit
    "FMP historical data": lambda: toolkit().get_historical_data(),
    "FMP intraday data": lambda: toolkit().get_intraday_data(period="1hour"),
    "FMP quote": lambda: toolkit().get_quote(),
    "FMP profile": lambda: toolkit().get_profile(),
    "FMP rating": lambda: toolkit().get_rating(),
    "FMP balance sheet statement": lambda: toolkit().get_balance_sheet_statement(),
    "FMP income statement": lambda: toolkit().get_income_statement(),
    "FMP cash flow statement": lambda: toolkit().get_cash_flow_statement(),
    "FMP statistics statement": lambda: toolkit().get_statistics_statement(),
    "FMP analyst estimates": lambda: toolkit().get_analyst_estimates(),
    "FMP earnings calendar": lambda: toolkit().get_earnings_calendar(),
    "FMP earnings surprise": lambda: toolkit().get_earnings_surprise(),
    "FMP dividend calendar": lambda: toolkit().get_dividend_calendar(),
    "FMP revenue product segmentation": (
        lambda: toolkit().get_revenue_product_segmentation()
    ),
    "FMP revenue geographic segmentation": (
        lambda: toolkit().get_revenue_geographic_segmentation()
    ),
    "FMP executives": lambda: toolkit().get_executives(),
    "FMP executive compensation": lambda: toolkit().get_executive_compensation(),
    "FMP company notes": lambda: toolkit().get_company_notes(),
    "FMP employee count": lambda: toolkit().get_employee_count(),
    "FMP shares float": lambda: toolkit().get_shares_float(),
    # A company with a split, mergers or a futures market in the range.
    "FMP stock splits": lambda: toolkit("NVDA", "2020-01-01").get_stock_splits(),
    "FMP insider trade statistics": lambda: toolkit().get_insider_trade_statistics(),
    "FMP institutional ownership": lambda: toolkit().get_institutional_ownership(),
    "FMP stock grades": lambda: toolkit().get_stock_grades(),
    "FMP price targets": lambda: toolkit().get_price_targets(),
    "FMP price target history": lambda: toolkit().get_price_target_history(),
    "FMP analyst recommendations": lambda: toolkit().get_analyst_recommendations(),
    "FMP earnings call transcripts": lambda: toolkit().get_earnings_call_transcripts(),
    "FMP mergers and acquisitions": (
        lambda: toolkit("CVX", "2021-01-01").get_mergers_acquisitions()
    ),
    "FMP commitment of traders": (
        lambda: toolkit("GC", START).get_commitment_of_traders()
    ),
    "FMP market risk premium": lambda: toolkit().get_market_risk_premium(),
    "FMP ESG scores": lambda: toolkit().get_esg_scores(),
    # Market discovery
    "FMP instrument search": lambda: discovery().search_instruments("Apple"),
    "FMP stock screener": lambda: discovery().get_stock_screener(
        market_cap_higher=1_000_000_000_000
    ),
    "FMP stock list": lambda: discovery().get_stock_list(),
    "FMP shares float list": lambda: discovery().get_stock_shares_float(),
    "FMP sectors performance": lambda: discovery().get_sectors_performance(),
    "FMP biggest gainers": lambda: discovery().get_biggest_gainers(),
    "FMP biggest losers": lambda: discovery().get_biggest_losers(),
    "FMP most active stocks": lambda: discovery().get_most_active_stocks(),
    "FMP delisted stocks": lambda: discovery().get_delisted_stocks(),
    "FMP crypto list": lambda: discovery().get_crypto_list(),
    "FMP forex list": lambda: discovery().get_forex_list(),
    "FMP commodity list": lambda: discovery().get_commodity_list(),
    "FMP ETF list": lambda: discovery().get_etf_list(),
    "FMP index list": lambda: discovery().get_index_list(),
    "FMP index constituents": lambda: discovery().get_index_constituents(),
    "FMP index constituent changes": (
        lambda: discovery().get_index_constituent_changes()
    ),
    "FMP stock news": lambda: discovery().get_stock_news(),
    "FMP general news": lambda: discovery().get_general_news(),
    "FMP press releases": lambda: discovery().get_press_releases(),
    "FMP crypto news": lambda: discovery().get_crypto_news(),
    "FMP forex news": lambda: discovery().get_forex_news(),
    "FMP stock news search": lambda: discovery().search_stock_news(TICKER),
    "FMP press release search": lambda: discovery().search_press_releases(TICKER),
    "FMP crypto news search": lambda: discovery().search_crypto_news("BTCUSD"),
    "FMP forex news search": lambda: discovery().search_forex_news("EURUSD"),
    "FMP IPO calendar": lambda: discovery().get_ipo_calendar(),
    "FMP IPO disclosures": lambda: discovery().get_ipo_disclosures(),
    "FMP IPO prospectuses": lambda: discovery().get_ipo_prospectuses(),
    "FMP stock splits calendar": lambda: discovery().get_stock_splits_calendar(),
    "FMP sector performance history": (
        lambda: discovery().get_sector_performance(sector="Technology")
    ),
    "FMP industry performance history": (
        lambda: discovery().get_industry_performance(industry="Semiconductors")
    ),
    "FMP sector P/E history": lambda: discovery().get_sector_pe(sector="Technology"),
    "FMP industry P/E history": (
        lambda: discovery().get_industry_pe(industry="Semiconductors")
    ),
    "FMP latest mergers and acquisitions": (
        lambda: discovery().get_mergers_acquisitions_latest()
    ),
    "FMP market earnings calendar": lambda: discovery().get_earnings_calendar(),
    "FMP 8-K filings": lambda: discovery().get_sec_filings_8k(),
    "FMP latest insider trading": lambda: discovery().get_insider_trading_latest(),
    # Funds. The lambdas keep the Toolkit from being created when the module is imported.
    "FMP ETF holdings": lambda: fund().etf.get_holdings(),  # noqa: PLW0108
    "FMP ETF information": lambda: fund().etf.get_information(),  # noqa: PLW0108
    "FMP ETF country weightings": (
        lambda: fund().etf.get_country_weightings()  # noqa: PLW0108
    ),
    "FMP ETF sector weightings": (
        lambda: fund().etf.get_sector_weightings()  # noqa: PLW0108
    ),
}

# The FinancialModelingPrep data that must also be current: the latest close, and the
# quote of the latest trading day.
CURRENT_FMP_SOURCES = {"FMP historical data", "FMP quote"}

# Each FRED series group: a request and how often the series update.
FRED_SOURCES = {
    "FRED ICE BofA spreads by rating": (
        lambda: fred_model.get_rating_option_adjusted_spread(START, END, FRED_API_KEY),
        "daily",
    ),
    "FRED ICE BofA yields by maturity": (
        lambda: fred_model.get_maturity_effective_yield(START, END, FRED_API_KEY),
        "daily",
    ),
    "FRED ICE BofA total returns by rating": (
        lambda: fred_model.get_rating_total_return(START, END, FRED_API_KEY),
        "daily",
    ),
    "FRED ICE BofA yields to worst by maturity": (
        lambda: fred_model.get_maturity_yield_to_worst(START, END, FRED_API_KEY),
        "daily",
    ),
    "FRED Moody's corporate bond yields": (
        lambda: fred_model.get_moodys_corporate_bond_yields(
            "daily", START, END, FRED_API_KEY
        ),
        "daily",
    ),
    "FRED HQM corporate bond yield curve": (
        lambda: fred_model.get_hqm_corporate_bond_yield_curve(
            "par", START, END, FRED_API_KEY
        ),
        "monthly",
    ),
    "FRED core consumer price index": (
        lambda: economics_fred_model.get_price_index(
            "consumer_core", START, END, FRED_API_KEY
        ),
        "monthly",
    ),
    "FRED nonfarm payrolls": (
        lambda: economics_fred_model.get_nonfarm_payrolls(START, END, FRED_API_KEY),
        "monthly",
    ),
    "FRED retail sales": (
        lambda: economics_fred_model.get_retail_sales(START, END, FRED_API_KEY),
        "monthly",
    ),
    "FRED 30-year mortgage rate": (
        lambda: economics_fred_model.get_mortgage_rate_30_year(
            START, END, FRED_API_KEY
        ),
        "daily",
    ),
}


def is_empty(result: object) -> bool:
    """Whether a result holds no data, whatever its type."""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame | pd.Series):
        return result.dropna(how="all").empty
    if isinstance(result, dict | list | tuple):
        return len(result) == 0
    return False


@needs_fmp
@pytest.mark.parametrize("name", FMP_SOURCES)
def test_fmp_source_returns_data(name, caplog):
    result = fetch_or_explain(name, FMP_SOURCES[name], caplog, is_empty=is_empty)

    if name == "FMP historical data":
        latest = result.dropna(how="all").index[-1].to_timestamp()
        assert latest >= pd.Timestamp(
            TODAY - timedelta(days=DAILY_MAX_AGE_DAYS)
        ), f"{name} is not current: the latest close is of {latest.date()}"
    elif name in CURRENT_FMP_SOURCES:
        assert result.notna().to_numpy().any(), f"{name} has no values"


@needs_fred
@pytest.mark.parametrize("name", FRED_SOURCES)
def test_fred_source_is_current(name, caplog):
    fetch, frequency = FRED_SOURCES[name]

    data = fetch_or_explain(name, fetch, caplog, is_empty=is_empty)

    latest = pd.Period(data.dropna(how="all").index[-1], "D").to_timestamp(how="end")
    max_age = {"daily": DAILY_MAX_AGE_DAYS, "monthly": MONTHLY_MAX_AGE_DAYS}[frequency]

    assert latest >= pd.Timestamp(
        TODAY - timedelta(days=max_age)
    ), f"{name} is not current: the latest observation is of {latest.date()}"
