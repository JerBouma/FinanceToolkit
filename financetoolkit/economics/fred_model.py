"""FRED Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.fixedincome.fred_model import fetch_single_series, get_fred_data

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


def get_commercial_real_estate_prices(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves the Commercial Real Estate Price Index for the United States from FRED
    (series COMREPUSQ159N, sourced from the IMF's Financial Soundness Indicators).

    This tracks commercial (office, retail, industrial, apartment) property prices,
    as distinct from residential house prices (see `get_house_prices`). It is a
    transaction-based index rather than the appraisal-smoothed methodology used by
    institutional benchmarks like the NCREIF Property Index, so it will show more
    volatility and less autocorrelation than an appraisal-based series would.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A single-column ("United States") DataFrame of the quarterly
            Commercial Real Estate Price Index, as a year-over-year percent change.
    """
    return _get_fred_series("COMREPUSQ159N", start_date, end_date, api_key)


REAL_YIELD_SERIES: dict[str, str] = {
    "DFII5": "5 Year",
    "DFII7": "7 Year",
    "DFII10": "10 Year",
    "DFII20": "20 Year",
    "DFII30": "30 Year",
}


def get_real_yield_curve(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Retrieves the daily real (TIPS-implied) Treasury yield curve from FRED -- the
    Market Yield on U.S. Treasury Inflation-Protected Securities at Constant
    Maturity, for the 5, 7, 10, 20 and 30-Year maturities (series DFII5, DFII7,
    DFII10, DFII20, DFII30).

    This is genuine market-observed data, as distinct from
    `fixedincome.get_breakeven_inflation_rate`, which is a pure formula applied to a
    hand-specified sample curve rather than real TIPS market data.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A DataFrame indexed by date with one column per maturity (5, 7,
            10, 20, 30 Year), in percent.
    """
    real_yield_curve = get_fred_data(
        list(REAL_YIELD_SERIES), start_date, end_date, api_key
    )
    real_yield_curve = real_yield_curve.rename(columns=REAL_YIELD_SERIES)

    return real_yield_curve


def get_breakeven_inflation_expectations(
    start_date: str, end_date: str, api_key: str
) -> pd.DataFrame:
    """
    Retrieves market-implied (Q-measure) breakeven inflation expectations from FRED
    -- nominal Treasury yield minus real TIPS yield -- at the 5, 7, 10, 20 and
    30-Year maturities, plus the 5-Year, 5-Year Forward Inflation Expectation Rate
    (the market's implied average inflation rate for the five years starting five
    years from now).

    FRED only publishes ready-made daily breakeven series for the 5 and 10-Year
    maturities (T5YIE, T10YIE); its 7, 20 and 30-Year breakeven series (T7YIEM,
    T20YIEM, T30YIEM) only exist at monthly frequency, so those three points are
    instead computed here as nominal minus real (e.g. DGS7 - DFII7) from FRED's own
    daily Treasury and TIPS series, keeping every maturity on a daily frequency.

    Requires a free FRED API key. Register at https://fred.stlouisfed.org/docs/api/api_key.html.

    Returns:
        pd.DataFrame: A DataFrame indexed by date with one column per maturity (5, 7,
            10, 20, 30 Year) plus the 5-Year, 5-Year Forward Rate, in percent.
    """
    series_ids = [
        "T5YIE",
        "DGS7",
        "DFII7",
        "T10YIE",
        "DGS20",
        "DFII20",
        "DGS30",
        "DFII30",
        "T5YIFR",
    ]
    data = get_fred_data(series_ids, start_date, end_date, api_key)

    breakeven_inflation = pd.DataFrame(index=data.index)
    breakeven_inflation["5 Year"] = data["T5YIE"]
    breakeven_inflation["7 Year"] = data["DGS7"] - data["DFII7"]
    breakeven_inflation["10 Year"] = data["T10YIE"]
    breakeven_inflation["20 Year"] = data["DGS20"] - data["DFII20"]
    breakeven_inflation["30 Year"] = data["DGS30"] - data["DFII30"]
    breakeven_inflation["5 Year, 5 Year Forward"] = data["T5YIFR"]

    return breakeven_inflation
