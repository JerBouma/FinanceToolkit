"""OECD Model"""

__docformat__ = "google"

from io import StringIO

import pandas as pd
import requests

from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

# pylint: disable=too-many-lines


BASE_URL = "https://sdmx.oecd.org/public/rest/data/"
EXTENSIONS = "?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"

CODE_TO_COUNTRY = {
    "AGO": "Angola",
    "ALB": "Albania",
    "ARE": "United Arab Emirates",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BEL": "Belgium",
    "BEN": "Benin",
    "BFA": "Burkina Faso",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BHR": "Bahrain",
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "BRN": "Brunei",
    "BWA": "Botswana",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "CHN": "China",
    "CIV": "Ivory Coast",
    "CMR": "Cameroon",
    "COD": "Democratic Republic of the Congo",
    "COG": "Republic of the Congo",
    "COL": "Colombia",
    "CRI": "Costa Rica",
    "CUB": "Cuba",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DEW": "Former Federal Republic of Germany",
    "DNK": "Denmark",
    "DOM": "Dominican Republic",
    "DZA": "Algeria",
    "EA": "Euro Area",
    "EA17": "Euro Area 17",
    "EA19": "Euro Area 19",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ERI": "Eritrea",
    "ESP": "Spain",
    "EST": "Estonia",
    "ETH": "Ethiopia",
    "EU": "European Union",
    "EU27_2020": "European Union 27",
    "EU28": "European Union 28",
    "FIN": "Finland",
    "FJI": "Fiji",
    "FRA": "France",
    "G-20": "G20",
    "G-7": "G7",
    "G7M": "G7M",
    "GAB": "Gabon",
    "GBR": "United Kingdom",
    "GEO": "Georgia",
    "GHA": "Ghana",
    "GNQ": "Equatorial Guinea",
    "GRC": "Greece",
    "GTM": "Guatemala",
    "HKG": "Hong Kong",
    "HND": "Honduras",
    "HRV": "Croatia",
    "HTI": "Haiti",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KGZ": "Kyrgyzstan",
    "KHM": "Cambodia",
    "KOR": "South Korea",
    "KWT": "Kuwait",
    "LAO": "Laos",
    "LBN": "Lebanon",
    "LBY": "Libya",
    "LIE": "Liechtenstein",
    "LKA": "Sri Lanka",
    "LSO": "Lesotho",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "MAR": "Morocco",
    "MDA": "Moldova",
    "MDG": "Madagascar",
    "MEX": "Mexico",
    "MKD": "Macedonia",
    "MLI": "Mali",
    "MLT": "Malta",
    "MMR": "Myanmar",
    "MNE": "Montenegro",
    "MNG": "Mongolia",
    "MOZ": "Mozambique",
    "MRT": "Mauritania",
    "MUS": "Mauritius",
    "MWI": "Malawi",
    "MYS": "Malaysia",
    "NAM": "Namibia",
    "NER": "Niger",
    "NGA": "Nigeria",
    "NIC": "Nicaragua",
    "NLD": "Netherlands",
    "NMEC": "Non-OECD Economies",
    "NOR": "Norway",
    "NPL": "Nepal",
    "NZL": "New Zealand",
    "OAVG": "OECD - Average",
    "OECD": "OECD - Total",
    "OECDE": "OECD - Europe",
    "OEU": "OECD - Europe",
    "OMN": "Oman",
    "PAK": "Pakistan",
    "PAN": "Panama",
    "PER": "Peru",
    "PHL": "Philippines",
    "PNG": "Papua New Guinea",
    "POL": "Poland",
    "PRK": "North Korea",
    "PRT": "Portugal",
    "PRY": "Paraguay",
    "PSE": "Palestine",
    "QAT": "Qatar",
    "ROU": "Romania",
    "RUS": "Russia",
    "RWA": "Rwanda",
    "SAU": "Saudi Arabia",
    "SDN": "Sudan",
    "SEN": "Senegal",
    "SGP": "Singapore",
    "SLE": "Sierra Leone",
    "SLV": "El Salvador",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "SWZ": "Swaziland",
    "SYR": "Syria",
    "TCD": "Chad",
    "TGO": "Togo",
    "THA": "Thailand",
    "TJK": "Tajikistan",
    "TKM": "Turkmenistan",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "TWN": "Taiwan",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "URY": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "WLD": "World",
    "YEM": "Yemen",
    "ZAF": "South Africa",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
}


# Number of periods subtracted from `start_date` before it is sent to the OECD API as
# `startPeriod`, per frequency. This exists solely so that `finalize_dataset`'s
# rolling/trailing smoothing and growth/lag calculations (which need history from
# before the display window to compute the first rows in it correctly) still have
# enough prior observations once the underlying fetch is date-scoped. There is no
# equivalent buffer needed for `endPeriod`, since future data has no bearing on
# growth/rolling calculations.
START_BUFFER_PERIODS = {"Y": 10, "Q": 16, "M": 24}


def _format_oecd_period(date: str, period_code: str, buffer_periods: int = 0) -> str:
    """
    Convert a YYYY-MM-DD date string into the OECD API's period format for the given
    frequency, optionally shifted back by `buffer_periods` periods first.

    Args:
        date (str): A date in YYYY-MM-DD format.
        period_code (str): The period code of the data. Can be 'M' for monthly,
            'Q' for quarterly or 'Y' for yearly.
        buffer_periods (int): Number of periods to subtract before formatting.
            Defaults to 0.

    Returns:
        str: The OECD-formatted period string (e.g. "2020", "2020-Q2" or "2020-05").
    """
    period = pd.Period(date, freq=period_code) - buffer_periods

    if period_code == "Y":
        return str(period.year)
    if period_code == "Q":
        return f"{period.year}-Q{period.quarter}"

    return f"{period.year}-{period.month:02d}"


def collect_oecd_data(
    oecd_data_string: str,
    period_code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Collect the data from the OECD API and return it as a DataFrame. This is
    a helper function for the other functions in this module.

    Args:
        oecd_data_string (str): The string that is appended to the base URL to
            get the data from the OECD API.
        period_code (str): The period code of the data. Can be 'M' for monthly,
            'Q' for quarterly or 'Y' for yearly.
        start_date (str | None): Restrict the query to observations at or after this
            date (YYYY-MM-DD), via the API's `startPeriod` parameter. A buffer of
            several periods is subtracted first (see `START_BUFFER_PERIODS`) so that
            downstream rolling/trailing/growth calculations still have enough history
            before the display window. The exact requested range is still enforced
            afterwards by the caller (`finalize_dataset`). Defaults to None (fetch
            full history).
        end_date (str | None): Restrict the query to observations at or before this
            date (YYYY-MM-DD), via the API's `endPeriod` parameter. Defaults to None
            (fetch up to the latest available observation).

    Returns:
       pd.DataFrame: A DataFrame containing the data from the OECD API.
    """
    extensions = EXTENSIONS

    if start_date:
        buffer_periods = START_BUFFER_PERIODS.get(period_code, 0)
        extensions += (
            f"&startPeriod="
            f"{_format_oecd_period(start_date, period_code, buffer_periods)}"
        )
    if end_date:
        extensions += f"&endPeriod={_format_oecd_period(end_date, period_code)}"

    try:
        response = get_request(f"{BASE_URL}{oecd_data_string}{extensions}", timeout=300)
    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code == 429:  # noqa
            logger.warning(
                "OECD API rate limit reached (429 Too Many Requests). "
                "Please wait a moment before retrying."
            )
            return pd.DataFrame()
        raise

    oecd_data = pd.read_csv(StringIO(response.text))

    oecd_data["REF_AREA"] = oecd_data["REF_AREA"].replace(CODE_TO_COUNTRY)

    oecd_data = oecd_data[["TIME_PERIOD", "REF_AREA", "OBS_VALUE"]]

    oecd_data = oecd_data.drop_duplicates(subset=["TIME_PERIOD", "REF_AREA"])

    oecd_data = oecd_data.pivot(
        index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE"
    )

    oecd_data.index.name = None
    oecd_data.columns.name = None

    oecd_data.index = pd.PeriodIndex(oecd_data.index, freq=period_code)

    # Only remove data if all of it is NaN or it is NaT
    oecd_data = oecd_data.dropna(axis=1, how="all")
    oecd_data = oecd_data[~oecd_data.index.isna()]

    return oecd_data


def get_annual_gross_domestic_product(
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    """
    Get the Gross Domestic Product for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data is displayed as per capita which is the GDP divided by the
    population of the country.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = (
        "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE_HCPC,/A....B1GQ_POP......."
    )

    gross_domestic_product = collect_oecd_data(
        oecd_data_string, "Y", start_date, end_date
    )

    return gross_domestic_product


def get_consumer_confidence_index(
    start_date: str | None = None, end_date: str | None = None
):
    """
    Get the Consumer Confidence Index for a variety of countries over
    time from the OECD. The Consumer Confidence Index is an indicator
    that measures the degree of optimism that consumers feel about
    the overall state of the economy and their personal financial
    situation. Data is defined in months.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Consumer Confidence Index for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.CCICP...AA...H"

    consumer_confidence_index = collect_oecd_data(
        oecd_data_string, "M", start_date, end_date
    )

    return consumer_confidence_index


def get_business_confidence_index(
    start_date: str | None = None, end_date: str | None = None
):
    """
    Get the Business Confidence Index for a variety of countries over
    time from the OECD. The Business Confidence Index is an indicator
    that measures the degree of optimism that businesses feel about
    the overall state of the economy and their personal financial
    situation. Data is defined in months.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Business Confidence Index for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.BCICP...AA...H"

    business_confidence_index = collect_oecd_data(
        oecd_data_string, "M", start_date, end_date
    )

    return business_confidence_index


def get_composite_leading_indicator(
    start_date: str | None = None, end_date: str | None = None
):
    """
    Get the Composite Leading Indicator for a variety of countries over
    time from the OECD. The Composite Leading Indicator is an indicator
    that tries to determine the turning points in business cycles.
    Data is defined in months.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Composite Leading Indicator for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.LI...AA...H"

    composite_leading_indicator = collect_oecd_data(
        oecd_data_string, "M", start_date, end_date
    )

    return composite_leading_indicator


def get_house_prices(
    quarterly: bool | None = False,
    inflation_adjusted: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Residential Property Prices Indices (RPPIs), also named House price indices (HPIs),
    are index numbers that measure the prices of residential properties over time. RPPIs
    are key statistics not only for citizens and households across the world, but also
    for economic and monetary policy makers. They can help, for example, to monitor
    potential macroeconomic imbalances and the risk exposure of the household
    and financial sectors.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the house prices for a variety
        of countries over time.
    """
    if inflation_adjusted:
        # RHP = Real house price indices (inflation-adjusted)
        oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.RHP."
    else:
        # HPI = Nominal house price indices
        oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.HPI."

    house_prices = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y", start_date, end_date
    )

    return house_prices


def get_rent_prices(
    quarterly: bool | None = False,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Covers the Rent Prices as an index number that measures the prices of residential
    properties over time. Rent Prices are key statistics not only for citizens and
    households across the world, but also for economic and monetary policy makers.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the rent prices for a variety
        of countries over time.
    """
    oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.RPI."

    rent_prices = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y", start_date, end_date
    )

    return rent_prices


def get_unemployment_rate(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the unemployment rate for a variety of countries over time from the OECD.
    The unemployment rate is the percentage of the total labor force that is unemployed.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the unemployment rate for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f"OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,/..._Z.Y._T.Y_GE15..{period_data}"
    )

    unemployment_rate = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data,
        start_date,
        end_date,
    )

    # Divide by 100 to get percentage
    unemployment_rate = unemployment_rate / 100

    return unemployment_rate


def get_long_term_interest_rate(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the long term interest rate for a variety of countries over time from the OECD.
    The long term interest rate is defined as the yield on government bonds with a
    maturity of 10 years.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the long term interest rate for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f"OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.{period_data}.IRLT.PA....."
    )

    long_term_interest_rate = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data if period_data == "Q" else "M",
        start_date,
        end_date,
    )

    # Divide by 100 to get percentage
    long_term_interest_rate = long_term_interest_rate / 100

    return long_term_interest_rate


def get_short_term_interest_rate(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the short term interest rate for a variety of countries over time from the OECD.
    The short term interest rate is defined as the yield on government bonds with a
    maturity of 3 months.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the short term interest rate for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f"OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.{period_data}.IR3TIB.PA....."
    )

    short_term_interest_rate = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data if period_data == "Q" else "M",
        start_date,
        end_date,
    )

    # Divide by 100 to get percentage
    short_term_interest_rate = short_term_interest_rate / 100

    return short_term_interest_rate


def get_consumer_price_index(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the Consumer Price Index (CPI) for a variety of countries over time from the OECD.
    The CPI measures the average change over time in the prices paid by consumers for a
    basket of goods and services, and is the standard measure of inflation.

    Unlike the Global Macro Database CPI (see `economics_controller.get_consumer_price_index`
    without `oecd_source=True`), which is annual-only, this OECD source additionally supports
    monthly and quarterly frequency, useful for tracking inflation more closely in real time.

    The index is set to 100 in the base year, which can vary per country.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Consumer Price Index for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f"OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/.{period_data}.CP.IX._Z._Z._Z"

    consumer_price_index = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data if period_data == "Q" else "M",
        start_date,
        end_date,
    )

    return consumer_price_index


def get_household_savings_rate(
    quarterly: bool | None = False,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Get the Gross Household Savings Rate for a variety of countries over time from the
    OECD's Household Dashboard. The household savings rate is the share of household
    gross disposable income (adjusted for the net change in pension entitlements) that
    is saved rather than spent on final consumption.

    It is a key input to consumption-smoothing and life-cycle/permanent-income theories
    of household behaviour, and a closely watched signal of both near-term consumption
    momentum (a falling savings rate can temporarily prop up spending even as income
    growth slows) and a household sector's buffer against future income shocks.

    See definition: https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_HHDASH%40DF_HHDASH_INDIC

    Also known as: household savings ratio, personal savings rate.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the household savings rate for a variety
        of countries over time.
    """
    oecd_data_string = (
        f"OECD.SDD.NAD,DSD_HHDASH@DF_HHDASH_INDIC,/"
        f"{'Q' if quarterly else 'A'}..B8GS1M_B6GA.PT_B6GA_S1M"
    )

    household_savings_rate = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y", start_date, end_date
    )

    # Divide by 100 to get percentage
    household_savings_rate = household_savings_rate / 100

    return household_savings_rate


def get_household_debt_to_income_ratio(
    quarterly: bool | None = False,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Get the Household Debt to Disposable Income Ratio for a variety of countries over
    time from the OECD's Household Dashboard. This expresses total household gross debt
    (loans and debt securities) as a percentage of household gross disposable income.

    It is a standard household-leverage indicator used in financial-stability analysis:
    a high or rapidly rising ratio signals households are more exposed to income shocks
    or interest rate increases (debt-servicing costs rise directly with rates on
    variable-rate or refinanced debt), and has historically preceded credit-cycle
    downturns (e.g. in the lead-up to the 2008 financial crisis).

    See definition: https://data-explorer.oecd.org/vis?df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_HHDASH%40DF_HHDASH_INDIC

    Also known as: household leverage ratio, debt-to-income ratio.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the household debt to income ratio for a
        variety of countries over time.
    """
    oecd_data_string = (
        f"OECD.SDD.NAD,DSD_HHDASH@DF_HHDASH_INDIC,/"
        f"{'Q' if quarterly else 'A'}..LES1M_FD4.PT_B6G_S1M"
    )

    household_debt_to_income_ratio = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y", start_date, end_date
    )

    # Divide by 100 to get percentage
    household_debt_to_income_ratio = household_debt_to_income_ratio / 100

    return household_debt_to_income_ratio


def get_producer_price_index(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the Producer Price Index (PPI) for a variety of countries over time from the OECD.
    The PPI measures the average change over time in the prices received by domestic
    producers (manufacturing) for their output. Because producers pass rising input costs
    on to their customers with a lag, the PPI is generally seen as a leading indicator of
    upstream cost pressure that later shows up in the Consumer Price Index (CPI).

    The index is set to 100 in the base year, which can vary per country.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the Producer Price Index for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f"OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/.{period_data}.PP.IX.C._Z._Z"

    producer_price_index = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data if period_data == "Q" else "M",
        start_date,
        end_date,
    )

    return producer_price_index


def get_exchange_rates(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Get the exchange rates for a variety of countries over time from the OECD.
    The exchange rate is defined as the number of units of the national currency
    that are needed to purchase one unit of US dollars.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the exchange rates for a variety
        of countries over time.
    """
    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f"OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.{period_data}.CC......"

    exchange_rates = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data if period_data == "Q" else "M",
        start_date,
        end_date,
    )

    return exchange_rates


def get_share_prices(
    period: str, start_date: str | None = None, end_date: str | None = None
):
    """
    Share price indices are calculated from the prices of common shares of
    companies traded on national or foreign stock exchanges. They are usually
    determined by the stock exchange, using the closing daily values for the
    monthly data, and normally expressed as simple arithmetic averages of
    the daily data.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the share prices for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f"OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/.{period_data}.SHARE......"
    )

    share_prices = collect_oecd_data(
        oecd_data_string,
        "Y" if period_data == "A" else period_data,
        start_date,
        end_date,
    )

    return share_prices


def get_labour_productivity(start_date: str | None = None, end_date: str | None = None):
    """
    GDP per hour worked is a measure of labour productivity. It measures
    how efficiently labour input is combined with other factors of production
    and used in the production process. Labour input is defined as total hours
    worked of all persons engaged in production. Labour productivity only partially
    reflects the productivity of labour in terms of the personal capacities of
    workers or the intensity of their effort.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the labour productivity for a
         variety of countries over time.
    """
    oecd_data_string = "OECD.SDD.TPS,DSD_PDB@DF_PDB,/.A.GDPHRS._T.USD_PPP_H.L.GY._Z.PPP"

    labour_productivity = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return labour_productivity


def get_output_gap(start_date: str | None = None, end_date: str | None = None):
    """
    Get the Output Gap for a variety of countries over time from the OECD Economic Outlook.
    The output gap is the difference between actual Gross Domestic Product (GDP) and estimated
    potential GDP, expressed as a percentage of potential GDP. Potential GDP is the level of
    output an economy can sustain over the long term without generating excess inflationary or
    disinflationary pressure, based on the full, non-inflationary use of its productive resources
    (labour, capital and technology).

    A positive output gap indicates the economy is running above its long-run potential (a
    "boom", typically associated with rising inflationary pressure), while a negative output gap
    indicates the economy is running below potential (a "slack", typically associated with rising
    unemployment and disinflationary pressure).

    This data is only available on a yearly basis, since the OECD Economic Outlook is published
    as a set of annual projections and estimates.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the output gap for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.ECO.MAD,DSD_EO@DF_EO,1.5/.GAP.A"

    output_gap = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return output_gap


def get_population(
    gender: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Population is defined as all nationals present in, or temporarily absent
    from a country, and aliens permanently settled in a country.

    The number is presented in millions of people.

    Args:
        gender (str): specify the population based on gender.
            Defaults to None which returns the total population.
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the population for a
            variety of countries over time.
    """
    if gender is not None and gender not in ["men", "women"]:
        raise ValueError("Please choose either 'men' or 'women'.")
    gender_parameter = "M" if gender == "men" else "F" if gender == "women" else "_T"
    oecd_data_string = (
        f"OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,/..PS.{gender_parameter}.."
    )

    population = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return population


def get_income_inequality(start_date: str | None = None, end_date: str | None = None):
    """
    Income is defined as household disposable income in a particular year.
    It consists of earnings, self-employment and capital income and public
    cash transfers; income taxes and social security contributions paid by
    households are deducted.

    This uses the GINI coefficient which is a measure of statistical dispersion
    intended to represent the income inequality or wealth inequality within
    a nation or any other group of people.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.INE,DSD_WISE_IDD@DF_IDD,/.A.INC_DISP_GINI..._T..."

    income_inequality = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return income_inequality


def get_poverty_rate(start_date: str | None = None, end_date: str | None = None):
    """
    The poverty rate is the ratio of the number of people whose income falls
    below the poverty line.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.INE,DSD_WISE_IDD@DF_IDD,/.A.PR_INC_DISP..._T..."

    poverty_rate = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return poverty_rate


def get_trust_in_goverment(start_date: str | None = None, end_date: str | None = None):
    """
    Trust in government refers to the share of people who report
    having confidence in the national government.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the trust in government
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.14_3.._T..."

    trust_in_government = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    # Divide by 100 to get percentage
    trust_in_government = trust_in_government / 100

    return trust_in_government


def get_renewable_energy(start_date: str | None = None, end_date: str | None = None):
    """
    Renewable energy is defined as the contribution of renewables
    to total primary energy supply (TPES).

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the renewable energy
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.12_10.._T..."

    renewable_energy = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    # Divide by 100 to get percentage
    renewable_energy = renewable_energy / 100

    return renewable_energy


def get_carbon_footprint(start_date: str | None = None, end_date: str | None = None):
    """
    Environmentally related taxes are an important instrument for
    governments to shape relative prices of goods and services.

    Args:
        start_date (str | None): Restrict the query to this start date (YYYY-MM-DD).
        end_date (str | None): Restrict the query to this end date (YYYY-MM-DD).

    Returns:
       pd.DataFrame: A DataFrame containing the environmental tax
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.12_9.._T..."

    carbon_footprint = collect_oecd_data(oecd_data_string, "Y", start_date, end_date)

    return carbon_footprint
