"""FRED Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.sources.fred_model import fetch_single_series

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


# FRED's copies of the BLS price indices, not seasonally adjusted: the Producer Price
# Index for final demand (from November 2009), the export and import price indices of
# all commodities (from 1983 and 1982) and the CPI-U components (from 1913 to 1957).
PRICE_INDEX_SERIES = {
    "producer": "PPIFID",
    "export": "IQ",
    "import": "IR",
    "consumer_core": "CPILFENS",
    "consumer_energy": "CPIENGNS",
    "consumer_food": "CPIUFDNS",
    "consumer_goods": "CUUR0000SAC",
    "consumer_services": "CUUR0000SAS",
    "consumer_shelter": "CUUR0000SAH1",
}


def get_price_index(
    measure: str, start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the full history of a US price index from FRED, which republishes the BLS series: the producer, export or
    import price index or a component of the CPI.

    The series are the Producer Price Index for final demand (PPIFID), the export or import price index of all
    commodities (IQ and IR), or a component of the Consumer Price Index (CPILFENS for the core index, CPIENGNS,
    CPIUFDNS, CUUR0000SAC, CUUR0000SAS and CUUR0000SAH1), not seasonally adjusted.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        measure (str): The price index. Options are: "producer" (producer price index), "export" (export price index),
            "import" (import price index), "consumer_core" (consumer prices less food and energy), "consumer_energy"
            (energy), "consumer_food" (food), "consumer_goods" (goods), "consumer_services" (services),
            "consumer_shelter" (shelter).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single "United States" column indexed by month.
    """
    data = _get_fred_series(PRICE_INDEX_SERIES[measure], start_date, end_date, api_key)

    if not data.empty:
        # FRED dates a month on its first day; the other sources index it by month.
        data.index = data.index.asfreq("M")
        data.index.name = None

    return data


def get_nonfarm_payrolls(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieve total nonfarm payroll employment from FRED (series PAYEMS): the number of US jobs outside farming.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the monthly level of
            nonfarm payroll employment, in thousands of persons, seasonally adjusted.
    """
    return _get_fred_series("PAYEMS", start_date, end_date, api_key)


def get_job_openings_rate(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieves the job openings rate of total nonfarm from FRED (series JTSJOR), which republishes the BLS Job Openings
    and Labor Turnover Survey, seasonally adjusted, from December 2000.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: The job openings rate as a decimal (0.045 for 4.5%), a single
        "United States" column indexed by month.
    """
    data = _get_fred_series("JTSJOR", start_date, end_date, api_key)

    if not data.empty:
        # FRED dates a month on its first day; the other sources index it by month.
        data.index = data.index.asfreq("M")
        data.index.name = None

    # FRED publishes the rate in percent.
    return data / 100


def get_initial_jobless_claims(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the initial claims for unemployment insurance from FRED (series ICSA): how many people newly file for
    unemployment benefits.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of initial jobless
            claims, as a count of claims, weekly (week ending Saturday) and seasonally
            adjusted.
    """
    return _get_fred_series("ICSA", start_date, end_date, api_key)


def get_continuing_jobless_claims(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the continued claims (insured unemployment) from FRED (series CCSA): how many people keep receiving
    unemployment benefits.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of continued jobless
            claims, as a count of claims, weekly (the Saturday ending the week of
            unemployment) and seasonally adjusted.
    """
    return _get_fred_series("CCSA", start_date, end_date, api_key)


def get_insured_unemployment_rate(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the insured unemployment rate from FRED (series IURSA): the continued claims as a share of covered
    employment.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the insured
            unemployment rate as a decimal (0.012 for 1.2%), weekly (the Saturday ending
            the week of unemployment) and seasonally adjusted.
    """
    # FRED publishes the rate in percent.
    return _get_fred_series("IURSA", start_date, end_date, api_key) / 100


def get_retail_sales(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieve the advance retail sales of retail trade and food services from FRED (series RSAFS): what US consumers
    spend in shops and restaurants.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of monthly total retail
            trade and food services sales, in millions of dollars, seasonally adjusted.
    """
    return _get_fred_series("RSAFS", start_date, end_date, api_key)


def get_industrial_production_index(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the Industrial Production Index from FRED (series INDPRO): the real output of US manufacturing, mining
    and utilities.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the monthly
            Industrial Production Index (2017 = 100), seasonally adjusted.
    """
    return _get_fred_series("INDPRO", start_date, end_date, api_key)


def get_housing_starts(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieve the housing starts (total new privately-owned housing units) from FRED (series HOUST): how many new US
    homes are started.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of monthly new housing
            starts, in thousands of units, at a seasonally adjusted annual rate -- so the
            value is the annualised pace implied by that month, not the count of units
            started in it.
    """
    return _get_fred_series("HOUST", start_date, end_date, api_key)


def get_real_personal_income(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the real personal income excluding current transfer receipts from FRED (series W875RX1): the income US
    households earn, adjusted for inflation.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of monthly real personal
            income excluding current transfer receipts, in billions of chained 2017
            dollars, at a seasonally adjusted annual rate.
    """
    return _get_fred_series("W875RX1", start_date, end_date, api_key)


def get_mortgage_rate_30_year(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the 30-year fixed rate mortgage average from FRED (series MORTGAGE30US, Freddie Mac's Primary Mortgage
    Market Survey): what new US home buyers pay.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the 30-year fixed
            mortgage rate as a decimal fraction per annum (0.065 for 6.5%), weekly (week
            ending Thursday) and not seasonally adjusted.

    Notes:
        FRED publishes this series in percentage points (6.5 for 6.5%); it is divided by
        100 here so that every rate the Finance Toolkit returns is a decimal fraction.
    """
    mortgage_rate_30_year = _get_fred_series(
        "MORTGAGE30US", start_date, end_date, api_key
    )

    # FRED quotes the rate in percentage points, so divide by 100 for the decimal.
    return mortgage_rate_30_year / 100


def get_recession_indicator(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the NBER-based US recession indicator from FRED (series USREC): 1 in the months of a recession and 0
    otherwise.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame, monthly and not
            seasonally adjusted, 1 during NBER-dated recession months and 0 otherwise.
            FRED's midpoint convention means the series switches to 1 in the month
            *following* the business cycle peak and back to 0 in the month following
            the trough, so the peak month itself is still marked as an expansion.
    """
    return _get_fred_series("USREC", start_date, end_date, api_key)


def get_commercial_real_estate_prices(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieve the commercial real estate price index of the United States from FRED (series COMREPUSQ159N, from the
    IMF's Financial Soundness Indicators).

    This tracks commercial (office, retail, industrial, apartment) property prices,
    as distinct from residential house prices (see `get_house_prices`). It is a
    transaction-based index rather than the appraisal-smoothed methodology used by
    institutional benchmarks like the NCREIF Property Index, so it will show more
    volatility and less autocorrelation than an appraisal-based series would.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Args:
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): FRED API key.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of quarterly commercial
            real estate prices expressed as a year-over-year percent change, as a decimal
            fraction (0.05 for 5%), not seasonally adjusted. FRED publishes this series
            only as that percent change, not as an index level.

    Notes:
        FRED publishes this series in percentage points (5.0 for 5%); it is divided by
        100 here so that every rate the Finance Toolkit returns is a decimal fraction.
    """
    commercial_real_estate_prices = _get_fred_series(
        "COMREPUSQ159N", start_date, end_date, api_key
    )

    # FRED quotes this series in percentage points, so divide by 100 for the decimal.
    return commercial_real_estate_prices / 100


REAL_YIELD_SERIES: dict[str, str] = {
    "DFII5": "5 Year",
    "DFII7": "7 Year",
    "DFII10": "10 Year",
    "DFII20": "20 Year",
    "DFII30": "30 Year",
}
