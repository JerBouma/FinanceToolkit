"""Sources Model"""

__docformat__ = "google"

import functools
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from types import CodeType
from typing import Any, TypeVar
from urllib.parse import urlsplit

from financetoolkit.cache import policy_model

# The publishers the Finance Toolkit retrieves data from, by the name the cache knows them
# by, as they are named in an answer.
SOURCE_NAMES = {
    policy_model.FINANCIAL_MODELING_PREP: "FinancialModelingPrep",
    policy_model.YAHOO_FINANCE: "Yahoo Finance",
    policy_model.FRED: "FRED (Federal Reserve Bank of St. Louis)",
    policy_model.OECD: "OECD",
    policy_model.GLOBAL_MACRO_DATABASE: "Global Macro Database",
    policy_model.EUROPEAN_CENTRAL_BANK: "European Central Bank",
    policy_model.FEDERAL_RESERVE: "Federal Reserve Bank of New York",
    policy_model.KEN_FRENCH: "Kenneth R. French Data Library",
    policy_model.EUROSTAT: "Eurostat",
    policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS: "Bank for International Settlements",
    policy_model.BANK_OF_ENGLAND: "Bank of England",
    policy_model.BANK_OF_JAPAN: "Bank of Japan",
    policy_model.JAPAN_MINISTRY_OF_FINANCE: "Japan Ministry of Finance",
    policy_model.OFFICE_FOR_NATIONAL_STATISTICS: "Office for National Statistics",
    policy_model.STATISTICS_BUREAU_OF_JAPAN: "Statistics Bureau of Japan",
    policy_model.US_TREASURY: "U.S. Department of the Treasury",
    policy_model.BUREAU_OF_LABOR_STATISTICS: "U.S. Bureau of Labor Statistics",
    policy_model.FREDDIE_MAC: "Freddie Mac",
    policy_model.DEPARTMENT_OF_LABOR: "U.S. Department of Labor",
    policy_model.CENSUS_BUREAU: "U.S. Census Bureau",
    policy_model.BUREAU_OF_ECONOMIC_ANALYSIS: "U.S. Bureau of Economic Analysis",
    policy_model.FEDERAL_RESERVE_BOARD: "Federal Reserve Board",
    policy_model.NATIONAL_BUREAU_OF_ECONOMIC_RESEARCH: "National Bureau of Economic Research",
    policy_model.IBGE: "IBGE (Brazilian Institute of Geography and Statistics)",
    policy_model.BUNDESBANK: "Deutsche Bundesbank",
    policy_model.BANK_OF_CANADA: "Bank of Canada",
    policy_model.RIKSBANK: "Sveriges Riksbank",
    policy_model.NORGES_BANK: "Norges Bank",
    policy_model.EIOPA: "EIOPA",
    policy_model.INTERNATIONAL_MONETARY_FUND: "International Monetary Fund",
    policy_model.SHILLER: "Robert Shiller (Yale)",
    policy_model.WORLD_BANK: "World Bank",
    policy_model.CBOE: "Cboe",
    policy_model.STOXX: "STOXX",
    policy_model.RESERVE_BANK_OF_AUSTRALIA: "Reserve Bank of Australia",
    policy_model.EEX: "European Energy Exchange",
    policy_model.ESMA: "ESMA",
    policy_model.NGFS: "NGFS",
    policy_model.MACROHISTORY: "Jordà-Schularick-Taylor Macrohistory Database",
    policy_model.DE_NEDERLANDSCHE_BANK: "De Nederlandsche Bank",
    policy_model.EUROPEAN_SYSTEMIC_RISK_BOARD: "European Systemic Risk Board",
    policy_model.DAMODARAN: "Aswath Damodaran (NYU Stern)",
    policy_model.FINRA: "FINRA",
}

# The publisher behind each host the Finance Toolkit requests data from, matched on the end
# of the host name. A host not listed is named by itself, so no source goes unmentioned.
HOST_SOURCES = {
    "financialmodelingprep.com": policy_model.FINANCIAL_MODELING_PREP,
    "finance.yahoo.com": policy_model.YAHOO_FINANCE,
    "stlouisfed.org": policy_model.FRED,
    "oecd.org": policy_model.OECD,
    "globalmacrodata.com": policy_model.GLOBAL_MACRO_DATABASE,
    "gmd-releases.s3.ap-southeast-2.amazonaws.com": policy_model.GLOBAL_MACRO_DATABASE,
    "ecb.europa.eu": policy_model.EUROPEAN_CENTRAL_BANK,
    "newyorkfed.org": policy_model.FEDERAL_RESERVE,
    "mba.tuck.dartmouth.edu": policy_model.KEN_FRENCH,
    "ec.europa.eu": policy_model.EUROSTAT,
    "bis.org": policy_model.BANK_FOR_INTERNATIONAL_SETTLEMENTS,
    "bankofengland.co.uk": policy_model.BANK_OF_ENGLAND,
    "boj.or.jp": policy_model.BANK_OF_JAPAN,
    "mof.go.jp": policy_model.JAPAN_MINISTRY_OF_FINANCE,
    "ons.gov.uk": policy_model.OFFICE_FOR_NATIONAL_STATISTICS,
    "stat.go.jp": policy_model.STATISTICS_BUREAU_OF_JAPAN,
    "treasury.gov": policy_model.US_TREASURY,
    "treasurydirect.gov": policy_model.US_TREASURY,
    "bls.gov": policy_model.BUREAU_OF_LABOR_STATISTICS,
    "freddiemac.com": policy_model.FREDDIE_MAC,
    "doleta.gov": policy_model.DEPARTMENT_OF_LABOR,
    "census.gov": policy_model.CENSUS_BUREAU,
    "bea.gov": policy_model.BUREAU_OF_ECONOMIC_ANALYSIS,
    "federalreserve.gov": policy_model.FEDERAL_RESERVE_BOARD,
    "nber.org": policy_model.NATIONAL_BUREAU_OF_ECONOMIC_RESEARCH,
    "ibge.gov.br": policy_model.IBGE,
    "bundesbank.de": policy_model.BUNDESBANK,
    "bankofcanada.ca": policy_model.BANK_OF_CANADA,
    "riksbank.se": policy_model.RIKSBANK,
    "norges-bank.no": policy_model.NORGES_BANK,
    "eiopa.europa.eu": policy_model.EIOPA,
    "imf.org": policy_model.INTERNATIONAL_MONETARY_FUND,
    "shillerdata.com": policy_model.SHILLER,
    "worldbank.org": policy_model.WORLD_BANK,
    "cboe.com": policy_model.CBOE,
    "stoxx.com": policy_model.STOXX,
    "rba.gov.au": policy_model.RESERVE_BANK_OF_AUSTRALIA,
    "eex.com": policy_model.EEX,
    "eex-group.com": policy_model.EEX,
    "esma.europa.eu": policy_model.ESMA,
    # The NGFS climate scenarios are served by IIASA's Scenario Explorer.
    "iiasa.ac.at": policy_model.NGFS,
    "ngfs.net": policy_model.NGFS,
    "macrohistory.net": policy_model.MACROHISTORY,
    "dnb.nl": policy_model.DE_NEDERLANDSCHE_BANK,
    "esrb.europa.eu": policy_model.EUROPEAN_SYSTEMIC_RISK_BOARD,
    "stern.nyu.edu": policy_model.DAMODARAN,
    "finra.org": policy_model.FINRA,
}

TrackedClass = TypeVar("TrackedClass", bound=type)

# The sources of the data used while a block runs (see collect_sources), shared with the
# worker threads it starts, as helpers.run_in_parallel runs each in a copy of the context.
_SOURCES: ContextVar[set[str] | None] = ContextVar(
    "financetoolkit_sources", default=None
)

# The instance attribute that holds which sources the data in each attribute came from.
PROVENANCE_ATTRIBUTE = "_financetoolkit_sources"

# Settings rather than data: an attribute holding one of these is not data that came from
# a source, such as a start date a retrieval adjusted.
SETTING_TYPES = (type(None), bool, int, float, str, bytes, date, datetime)


def record_source(source: str) -> None:
    """
    Records that data of a source is used, inside collect_sources; does nothing outside.

    Args:
        source (str): The source, as the cache names it (see policy_model), or a host.
    """
    sources = _SOURCES.get()

    if sources is not None:
        sources.add(source)


def source_of_url(url: str) -> str:
    """
    The source a URL belongs to, by its host.

    Args:
        url (str): The URL requested.

    Returns:
        str: The source as the cache names it, or the host when it is not a known one.
    """
    host = (urlsplit(url).hostname or url).lower()

    for suffix, source in HOST_SOURCES.items():
        if host == suffix or host.endswith("." + suffix):
            return source

    return host


def record_url(url: str) -> None:
    """
    Records the source of a URL that is requested, inside collect_sources.

    Args:
        url (str): The URL requested.
    """
    if _SOURCES.get() is not None:
        record_source(source_of_url(url))


@contextmanager
def collect_sources() -> Generator[set[str]]:
    """
    Collects the sources of the data used while the block runs, whether it is retrieved,
    read from the cache, or reused from an instance that retrieved it before (see
    track_sources), including in the worker threads the block starts.

    Yields:
        set[str]: The sources, filled while the block runs.
    """
    sources: set[str] = set()
    token = _SOURCES.set(sources)

    try:
        yield sources
    finally:
        _SOURCES.reset(token)


def readable_sources(sources: set[str] | list[str]) -> list[str]:
    """
    Names the sources as publishers, sorted.

    Args:
        sources (set[str] | list[str]): The sources, as the cache names them or as hosts.

    Returns:
        list[str]: The names of the publishers, each once.
    """
    return sorted({SOURCE_NAMES.get(source, source) for source in sources})


@functools.cache
def _names_in_code(code: CodeType) -> frozenset[str]:
    """
    The attribute and global names a code object refers to, including those of the
    functions defined inside it.

    Args:
        code (CodeType): The code object.

    Returns:
        frozenset[str]: The names.
    """
    names = set(code.co_names)

    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names |= _names_in_code(constant)

    return frozenset(names)


@functools.cache
def _referenced_names(function: Callable) -> frozenset[str]:
    """
    The names a method refers to, through the decorators that wrap it.

    Args:
        function (Callable): The method, as defined in its class.

    Returns:
        frozenset[str]: The names, such as the attributes it reads.
    """
    names: set[str] = set()
    pending = [function]
    seen: set[int] = set()

    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        code = getattr(current, "__code__", None)
        if code is not None:
            names |= _names_in_code(code)

        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)

        # Decorators such as handle_errors do not set __wrapped__; the function they wrap
        # is in their closure.
        for cell in getattr(current, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            if callable(content) and hasattr(content, "__code__"):
                pending.append(content)

    return frozenset(names)


def _traced(function: Callable) -> Callable:
    """
    Wraps a method so that, inside collect_sources, the sources of the data it uses are
    also recorded when that data was retrieved by an earlier call and kept on the instance.

    The method records what it retrieves itself. Each attribute it assigns that holds data
    keeps the sources recorded while it ran, and a later call of any method that reads that
    attribute records them again. Outside collect_sources the method runs as it is.

    Args:
        function (Callable): The method.

    Returns:
        Callable: The wrapped method.
    """

    @functools.wraps(function)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        outer = _SOURCES.get()
        state = getattr(self, "__dict__", None)

        if outer is None or state is None:
            return function(self, *args, **kwargs)

        provenance: dict[str, frozenset[str]] = state.setdefault(
            PROVENANCE_ATTRIBUTE, {}
        )
        sources: set[str] = set()

        # The data this method reads that an earlier call retrieved.
        for name in _referenced_names(function) & provenance.keys():
            sources |= provenance.get(name, frozenset())

        # Copied rather than iterated, as worker threads can assign attributes meanwhile.
        before = {name: id(value) for name, value in state.copy().items()}
        token = _SOURCES.set(sources)

        try:
            return function(self, *args, **kwargs)
        finally:
            _SOURCES.reset(token)
            outer |= sources

            if sources:
                for name, value in state.copy().items():
                    if (
                        name != PROVENANCE_ATTRIBUTE
                        and before.get(name) != id(value)
                        and not isinstance(value, SETTING_TYPES)
                    ):
                        provenance[name] = provenance.get(name, frozenset()) | sources

    return wrapper


def track_sources(cls: TrackedClass) -> TrackedClass:
    """
    Class decorator that traces the sources of the data an instance keeps between calls,
    so that an answer reusing that data still names where it came from (see _traced). It
    wraps the methods and property getters defined in the class, and __init__.

    Args:
        cls (TrackedClass): The class.

    Returns:
        TrackedClass: The same class.
    """
    for name, member in list(vars(cls).items()):
        if name.startswith("__") and name != "__init__":
            continue

        if isinstance(member, property) and member.fget is not None:
            setattr(
                cls,
                name,
                property(
                    _traced(member.fget), member.fset, member.fdel, member.__doc__
                ),
            )
        elif callable(member) and not isinstance(
            member, staticmethod | classmethod | type
        ):
            setattr(cls, name, _traced(member))

    return cls
