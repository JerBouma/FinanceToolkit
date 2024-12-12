"""OECD Model"""

__docformat__ = "google"

from io import StringIO

import pandas as pd
import requests

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


def collect_oecd_data(oecd_data_string: str, period_code: str) -> pd.DataFrame:
    """
    Collect the data from the OECD API and return it as a DataFrame. This is
    a helper function for the other functions in this module.

    Args:
        oece_data_string (str): The string that is appended to the base URL to
            get the data from the OECD API.
        period_code (str): The period code of the data. Can be 'M' for monthly,
            'Q' for quarterly or 'Y' for yearly.

    Returns:
       pd.DataFrame: A DataFrame containing the data from the OECD API.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(
        f"{BASE_URL}{oecd_data_string}{EXTENSIONS}", headers=headers, timeout=300
    )

    response.raise_for_status()

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


def get_annual_gross_domestic_product() -> pd.DataFrame:
    """
    Get the Gross Domestic Product for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data is displayed as per capita which is the GDP divided by the
    population of the country.

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = (
        "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE_HCPC,/A....B1GQ_POP......."
    )

    gross_domestic_product = collect_oecd_data(oecd_data_string, "Y")

    return gross_domestic_product


def get_quarterly_gross_domestic_product(year_on_year: bool = True) -> pd.DataFrame:
    """
    Get the Quarterly Gross Domestic Product for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data is available in two forms: compared to the previous year's value or
    compared to the previous period. The year on year data is the GDP compared to
    the same quarter in the previous year. The quarter on quarter data is the GDP
    compared to the previous quarter.

    Args:
        year_on_year (bool): Whether to return the year on year data or the quarter
            on quarter data.

    Returns:
       pd.DataFrame: A DataFrame containing the Quarterly Gross Domestic Product
       for a variety of countries over time.
    """
    if year_on_year:
        oecd_data_string = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.0/Q.....B1GQ......GY+G1."
    else:
        oecd_data_string = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.0/Q.....B1GQ......G1."

    gross_domestic_product = collect_oecd_data(oecd_data_string, "Q")

    # Division by 100 due to both metrics are in percentages
    gross_domestic_product = gross_domestic_product / 100

    return gross_domestic_product


def get_gross_domestic_product_forecast_short_term(
    quarterly: bool | None = False,
) -> pd.DataFrame:
    """
    Get the Gross Domestic Product Forecast for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data represents the annual growth rate which is the percentage change
    in GDP from one year to the next. When using quarterly data, the annual
    growth rate is the percentage change in GDP from the same quarter in the
    previous year.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = f"OECD.ECO.MAD,DSD_EO@DF_EO,/.CBGDPR.{'Q' if quarterly else 'A'}"

    gross_domestic_product = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y"
    )

    # Division by 100 due to both metrics are in percentages
    gross_domestic_product = gross_domestic_product / 100

    return gross_domestic_product


def get_consumer_confidence_index():
    """
    Get the Consumer Confidence Index for a variety of countries over
    time from the OECD. The Consumer Confidence Index is an indicator
    that measures the degree of optimism that consumers feel about
    the overall state of the economy and their personal financial
    situation. Data is defined in months.

    Returns:
       pd.DataFrame: A DataFrame containing the Consumer Confidence Index for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.CCICP...AA...H"

    consumer_confidence_index = collect_oecd_data(oecd_data_string, "M")

    return consumer_confidence_index


def get_business_confidence_index():
    """
    Get the Business Confidence Index for a variety of countries over
    time from the OECD. The Business Confidence Index is an indicator
    that measures the degree of optimism that businesses feel about
    the overall state of the economy and their personal financial
    situation. Data is defined in months.

    Returns:
       pd.DataFrame: A DataFrame containing the Business Confidence Index for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.BCICP...AA...H"

    business_confidence_index = collect_oecd_data(oecd_data_string, "M")

    return business_confidence_index


def get_composite_leading_indicator():
    """
    Get the Composite Leading Indicator for a variety of countries over
    time from the OECD. The Composite Leading Indicator is an indicator
    that tries to determine the turning points in business cycles.
    Data is defined in months.

    Returns:
       pd.DataFrame: A DataFrame containing the Composite Leading Indicator for a variety
        of countries over time.
    """
    oecd_data_string = "OECD.SDD.STES,DSD_STES@DF_CLI,4.1/.M.LI...AA...H"

    composite_leading_indicator = collect_oecd_data(oecd_data_string, "M")

    return composite_leading_indicator


def get_house_prices(quarterly: bool | None = False, inflation_adjusted: bool = True):
    """
    Residential Property Prices Indices (RPPIs), also named House price indices (HPIs),
    are index numbers that measure the prices of residential properties over time. RPPIs
    are key statistics not only for citizens and households across the world, but also
    for economic and monetary policy makers. They can help, for example, to monitor
    potential macroeconomic imbalances and the risk exposure of the household
    and financial sectors.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.

    Returns:
       pd.DataFrame: A DataFrame containing the house prices for a variety
        of countries over time.
    """
    if inflation_adjusted:
        oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.RHP."
    else:
        oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.RHP."

    house_prices = collect_oecd_data(oecd_data_string, "Q" if quarterly else "Y")

    return house_prices


def get_rent_prices(quarterly: bool | None = False):
    """
    Covers the Rent Prices as an index number that measures the prices of residential
    properties over time. Rent Prices are key statistics not only for citizens and
    households across the world, but also for economic and monetary policy makers.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.

    Returns:
       pd.DataFrame: A DataFrame containing the rent prices for a variety
        of countries over time.
    """
    oecd_data_string = f"OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/.{'Q' if quarterly else 'A'}.RPI."

    rent_prices = collect_oecd_data(oecd_data_string, "Q" if quarterly else "Y")

    return rent_prices


def get_unemployment_rate(period: str):
    """
    Get the unemployment rate for a variety of countries over time from the OECD.
    The unemployment rate is the percentage of the total labor force that is unemployed.

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
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    # Divide by 100 to get percentage
    unemployment_rate = unemployment_rate / 100

    return unemployment_rate


def get_long_term_interest_rate(period: str):
    """
    Get the long term interest rate for a variety of countries over time from the OECD.
    The long term interest rate is defined as the yield on government bonds with a
    maturity of 10 years.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

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
    )

    # Divide by 100 to get percentage
    long_term_interest_rate = long_term_interest_rate / 100

    return long_term_interest_rate


def get_short_term_interest_rate(period: str):
    """
    Get the short term interest rate for a variety of countries over time from the OECD.
    The short term interest rate is defined as the yield on government bonds with a
    maturity of 3 months.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

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
    )

    # Divide by 100 to get percentage
    short_term_interest_rate = short_term_interest_rate / 100

    return short_term_interest_rate


def get_exchange_rates(period: str):
    """
    Get the exchange rates for a variety of countries over time from the OECD.
    The exchange rate is defined as the number of units of the national currency
    that are needed to purchase one unit of US dollars.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

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
    )

    return exchange_rates


def get_share_prices(period: str):
    """
    Share price indices are calculated from the prices of common shares of
    companies traded on national or foreign stock exchanges. They are usually
    determined by the stock exchange, using the closing daily values for the
    monthly data, and normally expressed as simple arithmetic averages of
    the daily data.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

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
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return share_prices


def get_labour_productivity():
    """
    GDP per hour worked is a measure of labour productivity. It measures
    how efficiently labour input is combined with other factors of production
    and used in the production process. Labour input is defined as total hours
    worked of all persons engaged in production. Labour productivity only partially
    reflects the productivity of labour in terms of the personal capacities of
    workers or the intensity of their effort.

    Returns:
       pd.DataFrame: A DataFrame containing the labour productivity for a
         variety of countries over time.
    """
    oecd_data_string = "OECD.SDD.TPS,DSD_PDB@DF_PDB_GR,/.A.GDPHRS......"

    labour_productivity = collect_oecd_data(oecd_data_string, "Y")

    return labour_productivity


def get_population(gender: str | None = None):
    """
    Population is defined as all nationals present in, or temporarily absent
    from a country, and aliens permanently settled in a country.

    The number is presented in millions of people.

    Args:
        gender (str): specify the population based on gender.
            Defaults to None which returns the total population.

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

    population = collect_oecd_data(oecd_data_string, "Y")

    return population


def get_income_inequality():
    """
    Income is defined as household disposable income in a particular year.
    It consists of earnings, self-employment and capital income and public
    cash transfers; income taxes and social security contributions paid by
    households are deducted.

    This uses the GINI coefficient which is a measure of statistical dispersion
    intended to represent the income inequality or wealth inequality within
    a nation or any other group of people.

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.INE,DSD_WISE_IDD@DF_IDD,/.A.INC_DISP_GINI..._T..."

    income_inequality = collect_oecd_data(oecd_data_string, "Y")

    return income_inequality


def get_poverty_rate():
    """
    The poverty rate is the ratio of the number of people whose income falls
    below the poverty line.

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.INE,DSD_WISE_IDD@DF_IDD,/.A.PR_INC_DISP..._T..."

    poverty_rate = collect_oecd_data(oecd_data_string, "Y")

    return poverty_rate


def get_trust_in_goverment():
    """
    Trust in government refers to the share of people who report
    having confidence in the national government.

    Returns:
       pd.DataFrame: A DataFrame containing the trust in government
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.14_3.._T..."

    trust_in_government = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    trust_in_government = trust_in_government / 100

    return trust_in_government


def get_renewable_energy():
    """
    Renewable energy is defined as the contribution of renewables
    to total primary energy supply (TPES).

    Returns:
       pd.DataFrame: A DataFrame containing the renewable energy
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.12_10.._T..."

    renewable_energy = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    renewable_energy = renewable_energy / 100

    return renewable_energy


def get_carbon_footprint():
    """
    Environmentally related taxes are an important instrument for
    governments to shape relative prices of goods and services.

    Returns:
       pd.DataFrame: A DataFrame containing the environmental tax
        for a variety of countries over time
    """
    oecd_data_string = "OECD.WISE.WDP,DSD_HSL@DF_HSL_FWB,/.12_9.._T..."

    carbon_footprint = collect_oecd_data(oecd_data_string, "Y")

    return carbon_footprint
