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
    # The Ken French factor files are published monthly as a single zip archive.
    # Named "factors_decimal" because the loaders were corrected to divide the
    # published percentages by 100; the rename is what stops a cache warmed by an
    # older release from serving percent-scaled factors against decimal returns.
    # The policy has to follow that rename or the archive falls back to the
    # one day default and is re-downloaded every day.
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
