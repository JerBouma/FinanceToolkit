"""Policy Module"""

__docformat__ = "google"

from dataclasses import dataclass

DAY = 86400

# Spelled exactly as `enforce_source` spells them, so one word works in both.
FINANCIAL_MODELING_PREP = "FinancialModelingPrep"
YAHOO_FINANCE = "YahooFinance"
FRED = "FRED"
OECD = "OECD"
GLOBAL_MACRO_DATABASE = "GlobalMacroDatabase"
EUROPEAN_CENTRAL_BANK = "EuropeanCentralBank"
FEDERAL_RESERVE = "FederalReserve"
KEN_FRENCH = "KenFrench"
MCP = "MCP"
EUROSTAT = "Eurostat"
BANK_FOR_INTERNATIONAL_SETTLEMENTS = "BIS"
BANK_OF_ENGLAND = "BankOfEngland"
BANK_OF_JAPAN = "BankOfJapan"
JAPAN_MINISTRY_OF_FINANCE = "JapanMinistryOfFinance"
OFFICE_FOR_NATIONAL_STATISTICS = "ONS"
STATISTICS_BUREAU_OF_JAPAN = "StatisticsBureauOfJapan"
US_TREASURY = "USTreasury"
BUREAU_OF_LABOR_STATISTICS = "BLS"
FREDDIE_MAC = "FreddieMac"
FEDERAL_RESERVE_BOARD = "FederalReserveBoard"
NATIONAL_BUREAU_OF_ECONOMIC_RESEARCH = "NBER"
IBGE = "IBGE"
BUNDESBANK = "Bundesbank"
BANK_OF_CANADA = "BankOfCanada"
RIKSBANK = "Riksbank"
NORGES_BANK = "NorgesBank"

# The market risk premium is published per country, so it is one cache entry. Its name
# changed when the premiums became decimals (v2.2.2), so an entry cached in percent by an
# earlier version is never read back as decimals.
MARKET_RISK_PREMIUM_ENTITY = "global_decimals"


@dataclass(frozen=True)
class CachePolicy:
    """
    Freshness rules for a single dataset.

    Two independent knobs are needed because financial time series age in two
    different ways. Observations far in the past are effectively immutable, so
    re-requesting them wastes API credits; the observations near the end of the
    series are not, because the current bar is unfinished, statistical agencies
    revise their releases for months, and issuers restate their filings.

    The two settings answer different questions. The time-to-live decides *whether*
    a stored range is refreshed at all, so repeated runs inside that window make no
    external calls whatsoever. The revision window decides *how much* is re-requested
    once it is refreshed, so a daily rerun asks for the volatile tail rather than
    the entire history again.

    Attributes:
        ttl_seconds (int): How long a stored range is served without contacting the
            source at all.
        revision_days (int): How many days back from the end of the request are
            re-requested once the stored range has gone stale. Zero means the whole
            stale range is requested again, which is the right behaviour for sources
            that cannot be queried by date range anyway.
    """

    ttl_seconds: int
    revision_days: int = 0


# Point-in-time data has no date range, so only the TTL applies.
DEFAULT_POLICY = CachePolicy(ttl_seconds=DAY)

POLICIES: dict[str, CachePolicy] = {
    # Splits and dividends restate recent bars, so only the last week is re-requested.
    f"{FINANCIAL_MODELING_PREP}.historical": CachePolicy(
        ttl_seconds=DAY, revision_days=7
    ),
    f"{YAHOO_FINANCE}.historical": CachePolicy(ttl_seconds=DAY, revision_days=7),
    # Intraday bars move continuously; only FinancialModelingPrep publishes them.
    f"{FINANCIAL_MODELING_PREP}.intraday": CachePolicy(
        ttl_seconds=900, revision_days=2
    ),
    # Filings are restated and take no date range, so a refresh asks for everything.
    f"{FINANCIAL_MODELING_PREP}.statements": CachePolicy(ttl_seconds=DAY),
    f"{YAHOO_FINANCE}.statements": CachePolicy(ttl_seconds=DAY),
    # Company descriptors barely move.
    f"{FINANCIAL_MODELING_PREP}.profile": CachePolicy(ttl_seconds=30 * DAY),
    f"{FINANCIAL_MODELING_PREP}.rating": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.quote": CachePolicy(ttl_seconds=60),
    f"{FINANCIAL_MODELING_PREP}.analyst_estimates": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.earnings_calendar": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.dividend_calendar": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.esg_scores": CachePolicy(ttl_seconds=7 * DAY),
    f"{FINANCIAL_MODELING_PREP}.revenue_geographic_segmentation": CachePolicy(
        ttl_seconds=7 * DAY
    ),
    f"{FINANCIAL_MODELING_PREP}.revenue_product_segmentation": CachePolicy(
        ttl_seconds=7 * DAY
    ),
    f"{FINANCIAL_MODELING_PREP}.market_risk_premium": CachePolicy(ttl_seconds=7 * DAY),
    f"{FINANCIAL_MODELING_PREP}.commitment_of_traders": CachePolicy(ttl_seconds=DAY),
    # Leadership, listed notes and filings-based figures change a few times a year at most.
    f"{FINANCIAL_MODELING_PREP}.executives": CachePolicy(ttl_seconds=7 * DAY),
    f"{FINANCIAL_MODELING_PREP}.executive_compensation": CachePolicy(
        ttl_seconds=7 * DAY
    ),
    f"{FINANCIAL_MODELING_PREP}.company_notes": CachePolicy(ttl_seconds=7 * DAY),
    f"{FINANCIAL_MODELING_PREP}.employee_count": CachePolicy(ttl_seconds=7 * DAY),
    f"{FINANCIAL_MODELING_PREP}.mergers_acquisitions": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.stock_splits": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.shares_float": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.insider_trade_statistics": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.stock_grades": CachePolicy(ttl_seconds=DAY),
    # Fund holdings and allocations are republished daily by most issuers.
    f"{FINANCIAL_MODELING_PREP}.etf_holdings": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.etf_information": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.etf_country_weightings": CachePolicy(ttl_seconds=DAY),
    f"{FINANCIAL_MODELING_PREP}.etf_sector_weightings": CachePolicy(ttl_seconds=DAY),
    # A published transcript never changes, so each one is kept for a year; which
    # transcripts make up the latest selection does change with every new call.
    f"{FINANCIAL_MODELING_PREP}.earnings_call_transcripts": CachePolicy(
        ttl_seconds=365 * DAY
    ),
    f"{FINANCIAL_MODELING_PREP}.earnings_call_transcripts_selection": CachePolicy(
        ttl_seconds=DAY
    ),
    f"{FINANCIAL_MODELING_PREP}.treasury_rates": CachePolicy(
        ttl_seconds=DAY, revision_days=7
    ),
    # Short lived because several are intraday movers rather than static lists.
    f"{FINANCIAL_MODELING_PREP}.discovery": CachePolicy(ttl_seconds=3600),
    # Probed on every construction, so a short lifetime removes almost all the calls.
    f"{FINANCIAL_MODELING_PREP}.subscription_plan": CachePolicy(ttl_seconds=3600),
    # Describes the instrument rather than its price, so it barely changes.
    f"{YAHOO_FINANCE}.historical_statistics": CachePolicy(ttl_seconds=30 * DAY),
    # Chains move continuously while open; the expiry list is stable far longer.
    f"{YAHOO_FINANCE}.option_chain": CachePolicy(ttl_seconds=900),
    f"{YAHOO_FINANCE}.option_expiries": CachePolicy(ttl_seconds=DAY),
    # A forward curve fetches one contract per delivery month, so cache per contract.
    f"{YAHOO_FINANCE}.futures": CachePolicy(ttl_seconds=DAY, revision_days=7),
    # Macro sources revise heavily and publish on a lag, so the tail stays long.
    f"{FRED}.series": CachePolicy(ttl_seconds=DAY, revision_days=365),
    f"{OECD}.query": CachePolicy(ttl_seconds=DAY, revision_days=1095),
    f"{GLOBAL_MACRO_DATABASE}.dataset": CachePolicy(ttl_seconds=7 * DAY),
    # Full history per series and no date range accepted, so only a TTL applies.
    f"{EUROPEAN_CENTRAL_BANK}.series": CachePolicy(ttl_seconds=DAY),
    f"{FEDERAL_RESERVE}.rate": CachePolicy(ttl_seconds=DAY),
    # Statistical offices and central banks that publish a series whole and revise its
    # recent past; most release daily or monthly, so a day keeps the data current.
    # A rerun only asks for the revision window: a year of monthly releases for Eurostat,
    # a month of daily rates for the central banks.
    f"{EUROSTAT}.dataset": CachePolicy(ttl_seconds=DAY, revision_days=365),
    f"{BANK_FOR_INTERNATIONAL_SETTLEMENTS}.dataset": CachePolicy(
        ttl_seconds=DAY, revision_days=31
    ),
    f"{BANK_FOR_INTERNATIONAL_SETTLEMENTS}.consumer_prices": CachePolicy(
        ttl_seconds=DAY, revision_days=365
    ),
    f"{BANK_OF_ENGLAND}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{BANK_OF_JAPAN}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{EUROPEAN_CENTRAL_BANK}.economics_series": CachePolicy(
        ttl_seconds=DAY, revision_days=31
    ),
    f"{JAPAN_MINISTRY_OF_FINANCE}.yields": CachePolicy(ttl_seconds=DAY),
    f"{OFFICE_FOR_NATIONAL_STATISTICS}.series": CachePolicy(ttl_seconds=DAY),
    f"{STATISTICS_BUREAU_OF_JAPAN}.series": CachePolicy(ttl_seconds=DAY),
    # The keyless BLS API allows 25 requests a day, so the latest releases are fetched once.
    f"{BUREAU_OF_LABOR_STATISTICS}.series": CachePolicy(ttl_seconds=DAY),
    # A past year of the Treasury yield curve is final; only the current year changes.
    f"{US_TREASURY}.par_yield_curve": CachePolicy(ttl_seconds=DAY),
    f"{US_TREASURY}.par_yield_curve_year": CachePolicy(ttl_seconds=30 * DAY),
    f"{US_TREASURY}.par_yield_curve_real": CachePolicy(ttl_seconds=DAY),
    f"{US_TREASURY}.par_yield_curve_real_year": CachePolicy(ttl_seconds=30 * DAY),
    f"{FREDDIE_MAC}.series": CachePolicy(ttl_seconds=DAY),
    f"{IBGE}.series": CachePolicy(ttl_seconds=DAY),
    # Daily bond yields; a rerun only asks for the last month, since yields are not revised
    # but a holiday or late publication can leave the latest days open.
    f"{BUNDESBANK}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{BANK_OF_CANADA}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{RIKSBANK}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{NORGES_BANK}.series": CachePolicy(ttl_seconds=DAY, revision_days=31),
    f"{EUROPEAN_CENTRAL_BANK}.convergence_yields": CachePolicy(
        ttl_seconds=DAY, revision_days=93
    ),
    f"{FEDERAL_RESERVE_BOARD}.series": CachePolicy(ttl_seconds=DAY),
    # The NBER dates a turning point months after the fact, a few times a decade.
    f"{NATIONAL_BUREAU_OF_ECONOMIC_RESEARCH}.business_cycle_dates": CachePolicy(
        ttl_seconds=7 * DAY
    ),
    # The Ken French factor files are published monthly as a single zip archive; named "factors_decimal" because the loaders were corrected to divide the published percentages by 100, and the rename is what stops a cache warmed by an older release from serving percent-scaled factors against decimal returns, so the policy has to follow that rename or the archive falls back to the one day default and is re-downloaded every day.  # noqa: E501
    f"{KEN_FRENCH}.factors_decimal": CachePolicy(ttl_seconds=7 * DAY),
    # Computed MCP tool responses layered on top of the source caches.
    f"{MCP}.tool": CachePolicy(ttl_seconds=DAY),
}


def get_policy(source: str, dataset: str) -> CachePolicy:
    """
    Look up the freshness policy for a source and dataset combination.

    Args:
        source (str): The external data source, e.g. "fmp".
        dataset (str): The dataset within that source, e.g. "historical".

    Returns:
        CachePolicy: The registered policy, or a conservative default when the
            combination is not registered.
    """
    return POLICIES.get(f"{source}.{dataset}", DEFAULT_POLICY)


def register_policy(source: str, dataset: str, policy: CachePolicy) -> None:
    """
    Register or override the freshness policy for a source and dataset.

    Args:
        source (str): The external data source.
        dataset (str): The dataset within that source.
        policy (CachePolicy): The policy to apply.
    """
    POLICIES[f"{source}.{dataset}"] = policy
