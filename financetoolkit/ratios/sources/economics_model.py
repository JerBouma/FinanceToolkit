"""Economics Model: the macroeconomic data some ratios compare company figures with."""

import pandas as pd

from financetoolkit.cache.cache_controller import Cache
from financetoolkit.utilities.statistics_model import normalize_country_name


def get_gross_domestic_product(
    allow_stale_oecd_cache: bool, cache: Cache | None, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the yearly gross domestic product of every country in US dollars, which the
    market cap to GDP ratio divides by.

    Args:
        allow_stale_oecd_cache (bool): Whether an expired OECD cache may answer when the OECD
            cannot be reached.
        cache (Cache | None): The cache the Economics module reads and writes.
        api_key (str): The Financial Modeling Prep API key.

    Returns:
        pd.DataFrame: The gross domestic product per year (rows) and country (columns).
    """
    # Imported when first used, as only the market cap to GDP ratio needs it.
    from financetoolkit.economics.economics_controller import (  # noqa: PLC0415
        Economics,
    )

    # Every year, as each date is compared with the GDP of the year before it.
    return Economics(
        rounding=None,
        allow_stale_oecd_cache=allow_stale_oecd_cache,
        cache=cache,
        api_key=api_key,
    ).get_gross_domestic_product(usd=True)


def get_consumer_price_index(
    country: str, allow_stale_oecd_cache: bool, cache: Cache | None, api_key: str
) -> pd.Series:
    """
    Retrieve the monthly consumer price index of a country, which the cyclically adjusted
    price-to-earnings ratio restates the earnings with.

    Args:
        country (str): The country, in any name Economics recognises ("USA" or "United States").
        allow_stale_oecd_cache (bool): Whether an expired OECD cache may answer when the OECD
            cannot be reached.
        cache (Cache | None): The cache the Economics module reads and writes.
        api_key (str): The Financial Modeling Prep API key.

    Returns:
        pd.Series: The consumer price index per month, empty when the country has none.
    """
    # Imported when first used, as only the cyclically adjusted P/E needs it.
    from financetoolkit.economics.economics_controller import (  # noqa: PLC0415
        Economics,
    )

    # Every month of the Economics history, as the earnings go back up to decades.
    consumer_price_index = Economics(
        rounding=None,
        allow_stale_oecd_cache=allow_stale_oecd_cache,
        cache=cache,
        api_key=api_key,
    ).get_consumer_price_index(countries=country, period="monthly")

    # Economics returns the name its sources use ("USA" gives "United States").
    country = normalize_country_name(country, consumer_price_index.columns)

    if country not in consumer_price_index.columns:
        return pd.Series(dtype=float)

    return consumer_price_index[country]
