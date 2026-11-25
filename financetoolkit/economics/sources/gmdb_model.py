"""Global Macro Database (GMDB) Model"""

import io
import struct
import threading
import time

import numpy as np
import pandas as pd
import requests
from pandas.io.stata import StataReader

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import Cache
from financetoolkit.utilities.dataframe_model import to_dataframe
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request
from financetoolkit.utilities.sources_model import collect_sources, record_source

logger = get_logger()

# The Global Macro Database publishes a release every quarter, named after its year and
# month (e.g. 2026_09), on the storage its own Python, R and Stata packages read from. The
# list of releases tells which is the latest.
RELEASES_URL = "https://gmd-releases.s3.ap-southeast-2.amazonaws.com/data/"
VERSIONS_URL = f"{RELEASES_URL}helpers/versions.csv"
RELEASE_URL = f"{RELEASES_URL}distribute/GMD_{{version}}.dta"

# The file in the GitHub repository, which stopped being updated with the January 2025
# release. It is only used when the release storage cannot be reached.
GMD_LOCATION = "https://github.com/KMueller-Lab/Global-Macro-Database/blob/main/data/final/data_final.dta?raw=True"

# The identifying columns of the file, and the prefix of the column that flags, per
# variable, the years that are forecasts (IMF World Economic Outlook projections) rather
# than observations.
IDENTIFIERS = ["countryname", "ISO3", "id", "year", "income_group"]
FORECAST_PREFIX = "forecast_"

# The countries the Global Macro Database names differently from the OECD, the BIS and
# the other sources, renamed so that a country has one name throughout the Finance Toolkit
# and a GMDB series lines up with an OECD one column by column.
COUNTRY_NAMES = {
    "Russian Federation": "Russia",
    "Eswatini": "Swaziland",
    "Congo DR": "Democratic Republic of the Congo",
}


def get_latest_version() -> str:
    """
    Looks up the latest release of the Global Macro Database.

    Returns:
        str: The release, e.g. "2026_09".

    Raises:
        ValueError: When the list of releases is empty or unreadable.
    """
    versions = pd.read_csv(io.StringIO(get_request(VERSIONS_URL, timeout=30).text))

    if "versions" not in versions.columns or versions.empty:
        raise ValueError(
            "The list of Global Macro Database releases is empty or unreadable."
        )

    # Releases are named YYYY_MM, so the latest sorts last.
    return str(sorted(versions["versions"].astype(str))[-1])


def _parse_dataset(content: bytes, include_forecasts: bool) -> pd.DataFrame:
    """
    Reads the Stata file into one column per variable and country, indexed by year.

    Args:
        content (bytes): The Stata file.
        include_forecasts (bool): Whether to keep the years the file flags as forecasts.

    Returns:
        pd.DataFrame: The variables, indexed by year with a (variable, country) column.
    """
    # Read through an explicit StataReader: the whole file is consumed in one go,
    # while `.read()` is typed as returning a DataFrame (pd.read_stata itself is
    # typed as a DataFrame | StataReader union that trips up the type checker).
    with StataReader(io.BytesIO(content)) as reader:
        gmd_dataset = reader.read()

    flags = [
        column for column in gmd_dataset.columns if column.startswith(FORECAST_PREFIX)
    ]

    if not include_forecasts:
        for flag in flags:
            variable = flag.removeprefix(FORECAST_PREFIX)

            if variable in gmd_dataset.columns:
                gmd_dataset.loc[gmd_dataset[flag] == 1, variable] = np.nan

    gmd_dataset = gmd_dataset.drop(
        columns=flags
        + [column for column in IDENTIFIERS if column not in ("year", "countryname")],
        errors="ignore",
    )
    gmd_dataset["year"] = pd.PeriodIndex(gmd_dataset["year"].astype(int), freq="Y")
    gmd_dataset["countryname"] = gmd_dataset["countryname"].replace(COUNTRY_NAMES)
    gmd_dataset = gmd_dataset.set_index(["year", "countryname"])
    gmd_dataset.index.names = [None] * gmd_dataset.index.nlevels
    gmd_dataset = to_dataframe(gmd_dataset.unstack(level=1))

    return gmd_dataset.dropna(axis="columns", how="all").sort_index(axis=1)


# One copy of the dataset per process, shared by every Economics instance: it is over
# 100 MB in memory and takes seconds to read, so a server that meets many date ranges
# (one instance each) would otherwise hold and read it once per range. Kept for an hour,
# after which the next request reads the (cached or newly released) file again. The
# sources it was read from are kept with it, so an answer served from this copy names
# them as well.
SHARED_DATASET_SECONDS = 3600
_SHARED_DATASETS: dict[bool, tuple[float, pd.DataFrame, frozenset[str]]] = {}
_SHARED_DATASETS_LOCK = threading.Lock()


def get_shared_dataset(
    cache: Cache | None = None, include_forecasts: bool = False
) -> pd.DataFrame:
    """
    Get the Global Macro Database dataset, read once per hour per process and shared between the indicators that use
    it.

    Args:
        cache (Cache | None): An optional cache to serve the dataset from and store it in.
        include_forecasts (bool): Whether to keep the years that are projections. Defaults
            to False.

    Returns:
        pd.DataFrame: The dataset (see collect_global_macro_database_dataset). It is shared,
            so it must not be modified in place.
    """
    with _SHARED_DATASETS_LOCK:
        stored = _SHARED_DATASETS.get(include_forecasts)
        if stored is not None and time.monotonic() - stored[0] < SHARED_DATASET_SECONDS:
            for source in stored[2]:
                record_source(source)
            return stored[1]

        with collect_sources() as sources:
            dataset = collect_global_macro_database_dataset(
                cache=cache, include_forecasts=include_forecasts
            )

        # Collected apart to keep them with the copy, and passed on to the caller's answer.
        for source in sources:
            record_source(source)

        if not dataset.empty:
            _SHARED_DATASETS[include_forecasts] = (
                time.monotonic(),
                dataset,
                frozenset(sources),
            )

        return dataset


def collect_global_macro_database_dataset(
    gmd_location: str | None = None,
    cache: Cache | None = None,
    include_forecasts: bool = False,
) -> pd.DataFrame:
    """
    Collect and transform the Global Macro Database dataset: the yearly macroeconomic series of every country in one
    table.

    The file is reshaped into one column per variable and country, indexed by year.

    The Global Macro Database is published as one annual Stata file covering every
    country at once, so there is no per-country request to make and nothing to append
    to incrementally. It is therefore cached whole, which removes a download of around
    60 MB from every run that happens within the cache's freshness window.

    The file extends most variables with the IMF's World Economic Outlook projections up
    to five years ahead and flags those years per variable. They are left out unless
    forecasts are included, so a series ends with its last observation.

    Note on units: the Global Macro Database quotes every rate, share and ratio variable in
    percentage points (3.625 for a 3.625% policy rate, 81.0 for consumption worth 81% of
    GDP). Each getter below that returns such a series divides it by 100, so the Finance
    Toolkit returns decimal fractions throughout and the GMDB series line up with the OECD
    and FRED ones. Levels (currency amounts, head counts), price indices (CPI, HPI, REER,
    the GDP deflator), exchange rates and the binary crisis dummies are returned unchanged.

    Args:
        gmd_location (str | None): A Stata file to read instead of the latest release.
            Defaults to None.
        cache (Cache | None): An optional cache to serve the dataset from and store it in.
        include_forecasts (bool): Whether to keep the years that are projections. Defaults
            to False.

    Returns:
        pd.DataFrame: A transformed DataFrame indexed by 'year' with country-wise columns, or
            an empty DataFrame when the Global Macro Database cannot be reached or its file
            cannot be read (both logged).
    """
    # The quarterly releases are cached apart from the January 2025 file read before,
    # since they have variables that file lacks.
    entity = "release_with_forecasts" if include_forecasts else "release"

    if cache is not None and cache.enabled:
        cached_dataset = cache.get(
            source=policy_model.GLOBAL_MACRO_DATABASE,
            dataset="dataset",
            entity=entity,
        )

        if cached_dataset is not None:
            return cached_dataset

    # The January 2025 file is not cached when it stands in for an unreachable release,
    # so the next run tries the release again.
    cacheable = True

    try:
        if gmd_location is None:
            try:
                version = get_latest_version()
                response = get_request(RELEASE_URL.format(version=version), timeout=300)
                response.raise_for_status()
            except (requests.exceptions.RequestException, ValueError) as error:
                logger.warning(
                    "Could not retrieve the latest Global Macro Database release (%s), so "
                    "the January 2025 release is used instead.",
                    error,
                )
                response = get_request(GMD_LOCATION, timeout=300)
                response.raise_for_status()
                cacheable = False
        else:
            response = get_request(gmd_location, timeout=300)
            response.raise_for_status()
    except requests.exceptions.RequestException as error:
        # Offline or the host is down: reported like the other sources, with no data
        # rather than an exception, so the metrics built on it report it as missing.
        logger.error(
            "Could not reach the Global Macro Database (%s). No data is returned for it.",
            error,
        )
        return pd.DataFrame()

    try:
        gmd_dataset = _parse_dataset(response.content, include_forecasts)
    except (struct.error, EOFError, ValueError) as error:
        # An empty or cut-off download, or an error page instead of the Stata file, which
        # the Stata reader fails on with an unpacking error, an end of file or a ValueError.
        logger.error(
            "The Global Macro Database file could not be read (%s: %s). No data is "
            "returned for it.",
            type(error).__name__,
            error,
        )
        return pd.DataFrame()

    if cache is not None and cache.enabled and cacheable:
        cache.set(
            source=policy_model.GLOBAL_MACRO_DATABASE,
            dataset="dataset",
            entity=entity,
            data=gmd_dataset,
        )

    return gmd_dataset


def get_series(
    gmd_dataset: pd.DataFrame, variable: str, in_percent: bool = False
) -> pd.DataFrame:
    """
    Retrieve one variable of the Global Macro Database with a column per country.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
        variable (str): The variable, e.g. "hcons_GDP".
        in_percent (bool): Whether the variable is quoted in percentage points, which is
            divided by 100 to return a decimal fraction. Defaults to False.

    Returns:
        pd.DataFrame: The variable, indexed by year with a column per country, or an empty
        DataFrame when the dataset is empty (the database could not be reached).

    Raises:
        ValueError: When the dataset has no such variable, which is the case for the
            variables added after the January 2025 release when that is used instead.
    """
    # Already reported where the database could not be reached.
    if gmd_dataset.empty:
        return pd.DataFrame()

    if variable not in gmd_dataset.columns.get_level_values(0):
        raise ValueError(
            f"The Global Macro Database release in use has no variable '{variable}'."
        )

    series = gmd_dataset[variable].dropna(axis="rows", how="all")

    # The GMDB quotes rates, shares and ratios in percentage points.
    return series / 100 if in_percent else series


def get_nominal_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the nominal GDP of every country from the Global Macro Database ('nGDP'): the value of everything an
    economy produces in a year at current prices.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "nGDP")


def get_real_gross_domestic_product(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the real GDP of every country from the Global Macro Database ('rGDP'): the value of everything an economy
    produces in a year at constant prices.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "rGDP")


def get_real_gross_domestic_product_per_capita(
    gmd_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Retrieve the real GDP per capita of every country from the Global Macro Database ('rGDP_pc'): the
    inflation-adjusted output per person.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "rGDP_pc")


def get_gross_domestic_product_deflator(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the GDP deflator of every country from the Global Macro Database ('deflator'): the price level of
    everything an economy produces.

    Rows without a value for any country are left out.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "deflator")


def get_population(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the population of every country from the Global Macro Database ('pop'): the number of people living in
    each country.

    Rows without a value for any country are left out.

    This is a head count in millions rather than a rate, so it is returned on the scale
    the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "pop")


def get_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the total consumption of every country from the Global Macro Database ('cons'): what households and the
    government spend on goods and services.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "cons")


def get_total_consumption_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the total consumption to GDP ratio of every country from the Global Macro Database ('cons_GDP'): the
    share of the economy that is consumed.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.81 for consumption worth 81% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    total_consumption_to_gdp_ratio = get_series(gmd_dataset, "cons_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return total_consumption_to_gdp_ratio / 100


def get_real_total_consumption(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve the real total consumption of every country from the Global Macro Database in 2015 prices: what
    households and the government spend on goods and services, adjusted for inflation.

    The January 2025 release published it ('rcons'); later releases do not, so it is total consumption divided by the
    GDP deflator (2015 = 100), the way the Global Macro Database itself derives real GDP from nominal GDP.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    # Already reported where the database could not be reached.
    if gmd_dataset.empty:
        return pd.DataFrame()

    if "rcons" in gmd_dataset.columns.get_level_values(0):
        return get_series(gmd_dataset, "rcons")

    real_total_consumption = gmd_dataset["cons"] / gmd_dataset["deflator"] * 100

    return real_total_consumption.dropna(axis="rows", how="all").dropna(
        axis="columns", how="all"
    )


def get_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the investment of every country from the Global Macro Database ('inv'): gross capital formation, what is
    spent on fixed assets plus the change in inventories.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "inv")


def get_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the investment to GDP ratio of every country from the Global Macro Database ('inv_GDP'): the share of the
    economy that is invested.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.2248 for investment worth 22.48% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    investment_to_gdp_ratio = get_series(gmd_dataset, "inv_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return investment_to_gdp_ratio / 100


def get_fixed_investment(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the fixed investment of every country from the Global Macro Database ('finv'): gross fixed capital
    formation, what is spent on machinery, equipment and buildings.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "finv")


def get_fixed_investment_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the fixed investment to GDP ratio of every country from the Global Macro Database ('finv_GDP'): the share
    of the economy invested in fixed assets.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.2179 for fixed investment worth 21.79% of
    GDP), matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    fixed_investment_to_gdp_ratio = get_series(gmd_dataset, "finv_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return fixed_investment_to_gdp_ratio / 100


def get_exports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the exports of every country from the Global Macro Database ('exports'): the goods and services a country
    sells abroad.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "exports")


def get_exports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the exports to GDP ratio of every country from the Global Macro Database ('exports_GDP'): the share of
    output sold abroad.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1016 for exports worth 10.16% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    exports_to_gdp_ratio = get_series(gmd_dataset, "exports_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return exports_to_gdp_ratio / 100


def get_imports(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the imports of every country from the Global Macro Database ('imports'): the goods and services a country
    buys from abroad.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "imports")


def get_imports_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the imports to GDP ratio of every country from the Global Macro Database ('imports_GDP'): what a country
    buys from abroad against the size of its economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1213 for imports worth 12.13% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    imports_to_gdp_ratio = get_series(gmd_dataset, "imports_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return imports_to_gdp_ratio / 100


def get_current_account_balance(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the current account balance of every country from the Global Macro Database ('CA'): the trade balance
    plus net income and transfers from abroad.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "CA")


def get_current_account_balance_to_gdp(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the current account balance to GDP ratio of every country from the Global Macro Database ('CA_GDP'): the
    current account as a share of the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (-0.0211 for a deficit worth 2.11% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    current_account_balance_to_gdp = get_series(gmd_dataset, "CA_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return current_account_balance_to_gdp / 100


def get_usd_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the US dollar exchange rate of every country from the Global Macro Database ('USDfx'): the units of
    national currency per US dollar.

    Rows without a value for any country are left out.

    This is a price (units of local currency per US dollar) rather than a rate, so it is
    returned on the scale the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "USDfx")


def get_real_effective_exchange_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the real effective exchange rate of every country from the Global Macro Database ('REER'): the
    inflation-adjusted value of a currency against its trading partners.

    Rows without a value for any country are left out.

    This is an index (100 in the base year) rather than a rate, so it is returned on the
    scale the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "REER")


def get_government_debt(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government debt of every country from the Global Macro Database ('govdebt'): what a government owes
    to its creditors.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "govdebt")


def get_government_debt_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government debt to GDP ratio of every country from the Global Macro Database ('govdebt_GDP'): what a
    government owes against the size of its economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (1.3173 for debt worth 131.73% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    government_debt_to_gdp_ratio = get_series(gmd_dataset, "govdebt_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_debt_to_gdp_ratio / 100


def get_government_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government revenue of every country from the Global Macro Database ('govrev'): what a government
    collects from taxes and other sources.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "govrev")


def get_government_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government revenue to GDP ratio of every country from the Global Macro Database ('govrev_GDP'): the
    revenue as a share of the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.3104 for revenue worth 31.04% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    government_revenue_to_gdp_ratio = get_series(gmd_dataset, "govrev_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_revenue_to_gdp_ratio / 100


def get_government_tax_revenue(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government tax revenue of every country from the Global Macro Database ('govtax'): what a government
    collects from taxes.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "govtax")


def get_government_tax_revenue_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government tax revenue to GDP ratio of every country from the Global Macro Database ('govtax_GDP'):
    the tax burden as a share of the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.1022 for tax revenue worth 10.22% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    government_tax_revenue_to_gdp_ratio = get_series(gmd_dataset, "govtax_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_tax_revenue_to_gdp_ratio / 100


def get_government_expenditure(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government expenditure of every country from the Global Macro Database ('govexp'): what a government
    spends.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "govexp")


def get_government_expenditure_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government expenditure to GDP ratio of every country from the Global Macro Database ('govexp_GDP'):
    the spending as a share of the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.3708 for spending worth 37.08% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    government_expenditure_to_gdp_ratio = get_series(gmd_dataset, "govexp_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_expenditure_to_gdp_ratio / 100


def get_government_deficit(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government deficit of every country from the Global Macro Database ('govdef'): how much more a
    government spends than it collects.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "govdef")


def get_government_deficit_to_gdp_ratio(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the government deficit to GDP ratio of every country from the Global Macro Database ('govdef_GDP'): the
    shortfall as a share of the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (-0.0604 for a deficit worth 6.04% of GDP),
    matching the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    government_deficit_to_gdp_ratio = get_series(gmd_dataset, "govdef_GDP")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return government_deficit_to_gdp_ratio / 100


def get_money_supply(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the money supply of every country from the Global Macro Database ('M0', 'M1', 'M2', 'M3', 'M4'): the
    money in circulation, from cash to the broad aggregates.

    Rows without a value for any country are left out.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    # Already reported where the database could not be reached.
    if gmd_dataset.empty:
        return pd.DataFrame()

    return gmd_dataset[["M0", "M1", "M2", "M3", "M4"]].dropna(axis="rows", how="all")


def get_central_bank_policy_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the central bank policy rate of every country from the Global Macro Database ('cbrate'): the rate a
    central bank sets to steer the economy.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0538 for a 5.375% policy rate), matching
    the convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    central_bank_policy_rate = get_series(gmd_dataset, "cbrate")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return central_bank_policy_rate / 100


def get_short_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the short-term interest rate of every country from the Global Macro Database ('strate'): the rate on
    short-term borrowing.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0408 for 4.08%), putting it on exactly
    the same scale as the OECD short-term interest rate.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    short_term_interest_rate = get_series(gmd_dataset, "strate")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return short_term_interest_rate / 100


def get_long_term_interest_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the long-term interest rate of every country from the Global Macro Database ('ltrate'): the yield on
    long-term government bonds.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0416 for 4.16%), putting it on exactly
    the same scale as the OECD long-term interest rate.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    long_term_interest_rate = get_series(gmd_dataset, "ltrate")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return long_term_interest_rate / 100


def get_consumer_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the consumer price index of every country from the Global Macro Database ('CPI'): the level of the prices
    consumers pay.

    Rows without a value for any country are left out.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "CPI")


def get_house_price_index(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the house price index of every country from the Global Macro Database ('HPI'): the level of residential
    property prices.

    Rows without a value for any country are left out.

    This is a price index (100 in the base year) rather than a rate, so it is returned
    on the scale the Global Macro Database publishes it on.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "HPI")


def get_inflation_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the inflation rate of every country from the Global Macro Database ('infl'): how fast consumer prices
    rise.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0214 for 2.14% inflation), matching the
    convention used everywhere else in the Finance Toolkit.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    inflation_rate = get_series(gmd_dataset, "infl")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return inflation_rate / 100


def get_unemployment_rate(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the unemployment rate of every country from the Global Macro Database ('unemp'): the share of the labour
    force without work.

    Rows without a value for any country are left out.

    The Global Macro Database quotes this series in percentage points, so it is divided
    by 100 here to return a decimal fraction (0.0397 for 3.97% unemployment), putting it
    on exactly the same scale as the OECD unemployment rate.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    unemployment_rate = get_series(gmd_dataset, "unemp")

    # The GMDB quotes this in percentage points, so divide by 100 for the decimal.
    return unemployment_rate / 100


def get_sovereign_debt_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the sovereign debt crisis dummy of every country from the Global Macro Database ('SovDebtCrisis'): 1 in a
    year of a sovereign debt crisis.

    Rows without a value for any country are left out.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "SovDebtCrisis")


def get_currency_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the currency crisis dummy of every country from the Global Macro Database ('CurrencyCrisis'): 1 in a year
    of a currency crisis.

    Rows without a value for any country are left out.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "CurrencyCrisis")


def get_banking_crisis(gmd_dataset: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the banking crisis dummy of every country from the Global Macro Database ('BankingCrisis'): 1 in a year
    of a systemic banking crisis.

    Rows without a value for any country are left out.

    The values are binary (0 = no crisis, 1 = crisis) Reinhart & Rogoff style crisis-dating
    indicators, not a continuous economic series.

    Args:
        gmd_dataset (pd.DataFrame): The dataset of collect_global_macro_database_dataset.
    """
    return get_series(gmd_dataset, "BankingCrisis")
