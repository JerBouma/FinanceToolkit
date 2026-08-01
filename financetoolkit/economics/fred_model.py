"""FRED Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.fred_model import fetch_single_series

FRED_SERIES_COLUMN = "United States"


def _get_fred_series(
    series_id: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Fetch a single FRED series and label its column "United States", matching the
    country-name column convention used by the rest of the Economics module (whose
    other data comes from the OECD and Global Macro Database, both multi-country).
    Shared by every US-specific indicator function in this module.

    Args:
        series_id (str): The FRED series identifier (e.g. "PAYEMS").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame indexed by date.
    """
    data = fetch_single_series(series_id, start_date, end_date, api_key)

    if not data.empty:
        data.columns = [FRED_SERIES_COLUMN]

    return data


def get_nonfarm_payrolls(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieves Total Nonfarm Payroll Employment from FRED (series PAYEMS).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the level of
            nonfarm payroll employment, in thousands of persons.
    """
    return _get_fred_series("PAYEMS", start_date, end_date, api_key)


def get_initial_jobless_claims(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves Initial Claims for Unemployment Insurance from FRED (series ICSA).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of weekly initial
            jobless claims, seasonally adjusted.
    """
    return _get_fred_series("ICSA", start_date, end_date, api_key)


def get_retail_sales(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieves Advance Retail Sales: Retail and Food Services from FRED (series RSAFS).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of total retail and
            food services sales, in millions of dollars.
    """
    return _get_fred_series("RSAFS", start_date, end_date, api_key)


def get_industrial_production_index(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the Industrial Production Index from FRED (series INDPRO).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the Industrial
            Production Index (2017 = 100).
    """
    return _get_fred_series("INDPRO", start_date, end_date, api_key)


def get_housing_starts(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieves Housing Starts: Total New Privately-Owned Housing Units from FRED
    (series HOUST).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of new housing
            starts, in thousands of units, seasonally adjusted annual rate.
    """
    return _get_fred_series("HOUST", start_date, end_date, api_key)


def get_real_personal_income(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves Real Personal Income Excluding Current Transfer Receipts from FRED
    (series W875RX1).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of real personal
            income excluding current transfer receipts, in billions of chained 2017
            dollars.
    """
    return _get_fred_series("W875RX1", start_date, end_date, api_key)


def get_mortgage_rate_30_year(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the 30-Year Fixed Rate Mortgage Average from FRED (series
    MORTGAGE30US, Freddie Mac's Primary Mortgage Market Survey).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the weekly
            average 30-year fixed mortgage rate, in percent.
    """
    return _get_fred_series("MORTGAGE30US", start_date, end_date, api_key)


def get_recession_indicator(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the NBER-based US Recession Indicator from FRED (series USREC).

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame, 1 during NBER-dated
            recession months (peak through trough) and 0 otherwise.
    """
    return _get_fred_series("USREC", start_date, end_date, api_key)
