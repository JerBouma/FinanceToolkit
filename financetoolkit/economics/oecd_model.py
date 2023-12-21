"""OECD Model"""
__docformat__ = "google"

import pandas as pd

# pylint: disable=too-many-lines

BASE_URL = "https://stats.oecd.org/sdmx-json/data/DP_LIVE/"
EXTENSIONS = "/OECD?contentType=csv&detail=code&separator=comma&csv-lang=en"

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
    oecd_data = pd.read_csv(f"{BASE_URL}{oecd_data_string}{EXTENSIONS}")

    oecd_data["LOCATION"] = oecd_data["LOCATION"].replace(CODE_TO_COUNTRY)

    oecd_data = oecd_data.drop(
        columns=["INDICATOR", "SUBJECT", "MEASURE", "FREQUENCY", "Flag Codes"]
    )

    oecd_data = oecd_data.drop_duplicates(subset=["TIME", "LOCATION"])

    oecd_data = oecd_data.pivot(index="TIME", columns="LOCATION", values="Value")

    oecd_data.index.name = None
    oecd_data.columns.name = None

    oecd_data.index = pd.PeriodIndex(oecd_data.index, freq=period_code)

    # Only remove data if all of it is NaN
    oecd_data = oecd_data.dropna(axis=1, how="all")

    return oecd_data


def get_annual_gross_domestic_product(per_capita: bool = False) -> pd.DataFrame:
    """
    Get the Gross Domestic Product for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data is available in two forms: per capita and total. The per capita
    data is the GDP divided by the population of the country. The total data
    is the GDP in millions of dollars.

    Args:
        per_capita (bool): Whether to return the per capita data or the total data.

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = (
        f".GDP.TOT.{'USD_CAP' if per_capita else 'MLN_USD'}.A{EXTENSIONS}"
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
        per_capita (bool): Whether to return the per capita data or the total data.

    Returns:
       pd.DataFrame: A DataFrame containing the Quarterly Gross Domestic Product
       for a variety of countries over time.
    """
    oecd_data_string = f".QGDP.TOT.{'PC_CHGPY' if year_on_year else 'PC_CHGPP'}.Q"

    gross_domestic_product = collect_oecd_data(oecd_data_string, "Q")

    # Division by 100 due to both metrics are in percentages
    gross_domestic_product = gross_domestic_product / 100

    return gross_domestic_product


def get_gross_domestic_product_forecast_short_term(
    inflation_adjusted: bool = False, quarterly: bool = False
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
        per_capita (bool): Whether to return the per capita data or the total data.

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = (
        f".{'REALGDPFORECAST' if inflation_adjusted else 'NOMGDPFORECAST'}.TOT."
        f"AGRWTH.{'Q' if quarterly else 'A'}"
    )

    gross_domestic_product = collect_oecd_data(
        oecd_data_string, "Q" if quarterly else "Y"
    )

    # Division by 100 due to both metrics are in percentages
    gross_domestic_product = gross_domestic_product / 100

    return gross_domestic_product


def get_gross_domestic_product_forecast_long_term():
    """
    Get the Gross Domestic Product Forecast for a variety of countries over
    time from the OECD. The Gross Domestic Product is the total value
    of goods produced and services provided in a country during one year.

    The data represents the millions of US dollars in GDP over a long
    period of time (> 40 years) which resembles an estimation of the
    total GDP over time.

    Returns:
       pd.DataFrame: A DataFrame containing the Gross Domestic Product for a variety
       of countries over time.
    """
    oecd_data_string = ".GDPLTFORECAST.TOT.MLN_USD.A"

    gross_domestic_product = collect_oecd_data(oecd_data_string, "Y")

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
    oecd_data_string = ".CCI.AMPLITUD.LTRENDIDX.M"

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
    oecd_data_string = ".BCI.AMPLITUD.LTRENDIDX.M"

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
    oecd_data_string = ".CLI.AMPLITUD.LTRENDIDX.M"

    composite_leading_indicator = collect_oecd_data(oecd_data_string, "M")

    return composite_leading_indicator


def get_consumer_price_index(period: str, inflation_type: str):
    """
    Get the Consumer Price Index for a variety of countries over
    time from the OECD. The Consumer Price Index is an indicator
    that measures the average change over time in the prices paid
    by urban consumers for a market basket of consumer goods and
    services.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        inflation_type (str): The type of inflation. Can be 'total', 'food' or 'energy'.

    Returns:
       pd.DataFrame: A DataFrame containing the Consumer Price Index for a variety
        of countries over time.
    """
    inflation_type = inflation_type.lower()
    period = period.lower()

    if inflation_type not in ["total", "total_excl", "food", "energy"]:
        raise ValueError(
            "Inflation type must be one of 'total', 'total_excl', 'food' or 'energy'"
        )
    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    inflation_data = (
        "TOT"
        if inflation_type == "total"
        else "FOOD"
        if inflation_type == "food"
        else "TOT_FOODENRG"
        if inflation_type == "total_excl"
        else "ENRG"
    )
    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f".CPI.{inflation_data}.IDX2015.{period_data}"

    consumer_price_index = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return consumer_price_index


def get_consumer_price_index_forecast(period: str):
    """
    Get the Consumer Price Index Forecast which is a growth rate
    for a variety of countries over time from the OECD. The
    Consumer Price Index is an indicator that measures the average
    change over time in the prices paid by urban consumers for a
    market basket of consumer goods and services.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        inflation_type (str): The type of inflation. Can be 'total', 'food' or 'energy'.

    Returns:
       pd.DataFrame: A DataFrame containing the Consumer Price Index for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["quarterly", "yearly"]:
        raise ValueError("Period must be 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f".CPIFORECAST.TOT.AGRWTH.{period_data}"

    consumer_price_index = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    # Division by 100 due to being a growth rate
    consumer_price_index = consumer_price_index / 100

    return consumer_price_index


def get_producer_price_index(period: str):
    """
    Get the Producer Price Index for a variety of countries over
    time from the OECD. The Producer Price Index is an indicator
    that measures the average change over time in the selling
    prices received by domestic producers for their output.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

    Returns:
       pd.DataFrame: A DataFrame containing the Producer Price Index for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f".PPI.TOT_MKT.IDX2015.{period_data}"

    producer_price_index = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return producer_price_index


def get_house_prices(quarterly: bool | None = False, inflation_adjusted: bool = False):
    """
    Get the house prices for a variety of countries over time from the OECD.
    The house prices are the prices of residential real estate in the country.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.
        inflation_adjusted (bool): Whether to return the inflation adjusted data or the
            nominal data.

    Returns:
       pd.DataFrame: A DataFrame containing the house prices for a variety
        of countries over time.
    """
    oecd_data_string = f".HOUSECOST.{'REAL' if inflation_adjusted else 'NOMINAL'}.IDX2015.{'Q' if quarterly else 'A'}"

    house_prices = collect_oecd_data(oecd_data_string, "Q" if quarterly else "Y")

    return house_prices


def get_rent_prices(
    quarterly: bool | None = False,
):
    """
    Get the rent prices for a variety of countries over time from the OECD.
    The rent prices are the prices of residential real estate in the country.

    Args:
        quarterly (bool): Whether to return the quarterly data or the yearly data.

    Returns:
       pd.DataFrame: A DataFrame containing the rent prices for a variety
        of countries over time.
    """
    oecd_data_string = f".HOUSECOST.RENT.IDX2015.{'Q' if quarterly else 'A'}"

    house_prices = collect_oecd_data(oecd_data_string, "Q" if quarterly else "Y")

    return house_prices


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

    oecd_data_string = f".HUR.TOT.PC_LF.{period_data}"

    unemployment_rate = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    # Divide by 100 to get percentage
    unemployment_rate = unemployment_rate / 100

    return unemployment_rate


def get_long_term_interest_rate(period: str, forecast: bool = False):
    """
    Get the long term interest rate for a variety of countries over time from the OECD.
    The long term interest rate is defined as the yield on government bonds with a
    maturity of 10 years.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        forecast (bool): Whether to return the forecast data or the historical data.

    Returns:
       pd.DataFrame: A DataFrame containing the long term interest rate for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f".{'LTINTFORECAST' if forecast else 'LTINT'}.TOT.PC_PA.{period_data}"
    )

    long_term_interest_rate = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    # Divide by 100 to get percentage
    long_term_interest_rate = long_term_interest_rate / 100

    return long_term_interest_rate


def get_short_term_interest_rate(period: str, forecast: bool = False):
    """
    Get the short term interest rate for a variety of countries over time from the OECD.
    The short term interest rate is defined as the yield on government bonds with a
    maturity of 3 months.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.
        forecast (bool): Whether to return the forecast data or the historical data.

    Returns:
       pd.DataFrame: A DataFrame containing the short term interest rate for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = (
        f".{'STINTFORECAST' if forecast else 'STINT'}.TOT.PC_PA.{period_data}"
    )

    short_term_interest_rate = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    # Divide by 100 to get percentage
    short_term_interest_rate = short_term_interest_rate / 100

    return short_term_interest_rate


def get_purchasing_power_parity():
    """
    Get the purchasing power parity for a variety of countries over time from the OECD.
    Purchasing power parity (PPP) is a measure of the price of specific goods in
    different countries and is used to compare the absolute purchasing power of
    the countries' currencies.

    Returns:
       pd.DataFrame: A DataFrame containing the purchasing power parity for a variety
        of countries over time.
    """
    oecd_data_string = ".PPP.TOT.NATUSD.A"

    purchasing_power_parity = collect_oecd_data(oecd_data_string, "Y")

    return purchasing_power_parity


def get_exchange_rates():
    """
    Get the exchange rates for a variety of countries over time from the OECD.
    The exchange rate is defined as the number of units of the national currency
    that are needed to purchase one unit of US dollars.

    Returns:
       pd.DataFrame: A DataFrame containing the exchange rates for a variety
        of countries over time.
    """
    oecd_data_string = ".EXCH.TOT.NATUSD.A"

    exchange_rates = collect_oecd_data(oecd_data_string, "Y")

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

    oecd_data_string = f".SHPRICE.TOT.IDX2015.{period_data}"

    share_prices = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return share_prices


def get_labour_productivity():
    """
    Labour productivity is defined as the ratio of a volume measure of GDP
    to a volume measure of labour input. This ratio is particularly useful
    when comparing international productivity levels. Labour productivity
    is also known as just productivity.

    Returns:
       pd.DataFrame: A DataFrame containing the labour productivity for a
         variety of countries over time.
    """
    oecd_data_string = ".GDPHRWKD.TOT.IDX2015.A"

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
    gender_parameter = (
        "MEN" if gender == "men" else "WOMEN" if gender == "women" else "TOT"
    )
    oecd_data_string = f".POP.{gender_parameter}.MLN_PER.A"

    population = collect_oecd_data(oecd_data_string, "Y")

    return population


def get_young_population_percentage():
    """
    The youth population is defined as those people aged less than 15.
    The share of the dependent population is calculated as total
    elderly and youth population expressed as a ratio of the total population.

    The number is presented in a % of the total population.

    Returns:
       pd.DataFrame: A DataFrame containing the young
        population for a variety of countries over time.
    """
    oecd_data_string = ".YNGPOP.TOT.PC_POP.A"

    young_population_percentage = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    young_population_percentage = young_population_percentage / 100

    return young_population_percentage


def get_working_age_population():
    """
    The working age population is defined as those aged 15 to 64.
    This indicator measures the share of the working age
    population in total population.

    Returns:
       pd.DataFrame: A DataFrame containing the working age
        population for a variety of countries over time.
    """
    oecd_data_string = ".WKGPOP.TOT.PC_POP.A"

    working_age_population = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    working_age_population = working_age_population / 100

    return working_age_population


def get_elderly_population_percentage():
    """
    The elderly population is defined as people aged 65 and over.
    The share of the dependent population is calculated as total
    elderly and youth population expressed as a ratio of
    the total population.

    The number is presented in a % of the total population.

    Returns:
       pd.DataFrame: A DataFrame containing the elderly
        population for a variety of countries over time.
    """
    oecd_data_string = ".ELDLYPOP.TOT.PC_POP.A"

    elderly_population_percentage = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    elderly_population_percentage = elderly_population_percentage / 100

    return elderly_population_percentage


def get_fertility_rate():
    """
    The total fertility rate in a specific year is defined as the
    total number of children that would be born to each woman if
    she were to live to the end of her child-bearing years and
    give birth to children in alignment with the prevailing
    age-specific fertility rates.

    Returns:
       pd.DataFrame: A DataFrame containing the fertility
        rate for a variety of countries over time.
    """
    oecd_data_string = ".FERTILITY.TOT.CHD_WOMAN.A"

    fertility_rate = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    fertility_rate = fertility_rate / 100

    return fertility_rate


def get_old_age_dependency_ratio():
    """
    The old-age to working-age demographic ratio is defined as
    the number of individuals aged 65 and over per 100 people
    of working age defined as those at ages 20 to 64.

    Returns:
       pd.DataFrame: A DataFrame containing the old-age
        dependency ratio for a variety of countries over time
    """
    oecd_data_string = ".OLDAGEDEP.TOT.PC.A"

    old_age_dependency = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    old_age_dependency = old_age_dependency / 100

    return old_age_dependency


def get_income_inequality(parameter: str = "GINI"):
    """
    Income is defined as household disposable income in a particular year.
    It consists of earnings, self-employment and capital income and public
    cash transfers; income taxes and social security contributions paid by
    households are deducted.

    Args:
        parameter (str): The parameter of the data. Can be 'GINI', 'P50P10',
        'P90P10', 'P90P50', 'S80S20' or 'PALMA'.

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    parameter = parameter.upper()

    if parameter not in ["GINI", "P50P10", "P90P10", "P90P50", "S80S20", "PALMA"]:
        raise ValueError(
            "Please choose one of the following parameters: 'GINI', 'P50P10', "
            "'P90P10', 'P90P50', 'S80S20', 'PALMA'."
        )

    oecd_data_string = (
        f".INCOMEINEQ.{parameter}.{'INEQ' if parameter == 'GINI' else 'RT'}.A"
    )

    income_inequality = collect_oecd_data(oecd_data_string, "Y")

    return income_inequality


def get_poverty_rate(parameter: str = "TOT"):
    """
    Income is defined as household disposable income in a particular year.
    It consists of earnings, self-employment and capital income and public
    cash transfers; income taxes and social security contributions paid by
    households are deducted.

    Args:
        parameter (str): The parameter of the data. Can be 'TOT', '0_17',
        '18_65' or '66MORE'.

    Returns:
       pd.DataFrame: A DataFrame containing the income inequality
        for a variety of countries over time
    """
    parameter = parameter.upper()

    if parameter not in ["TOT", "0_17", "18_65", "66MORE"]:
        raise ValueError(
            "Please choose one of the following parameters: 'TOT', '0_17', "
            "'18_65', '66MORE'."
        )

    oecd_data_string = f".POVERTY.{parameter}.RT.A"

    poverty_rate = collect_oecd_data(oecd_data_string, "Y")

    return poverty_rate


def get_narrow_money(period: str):
    """
    M1 includes currency i.e. banknotes and coins, plus overnight deposits.
    M1 is expressed as a seasonally adjusted index based on 2015=100.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

    Returns:
       pd.DataFrame: A DataFrame containing the narrow money for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f".M1.TOT.IDX2015.{period_data}"

    narrow_money = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return narrow_money


def get_broad_money(period: str):
    """
    Broad money (M3) includes currency, deposits with an agreed maturity of up
    to two years, deposits redeemable at notice of up to three months and
    repurchase agreements, money market fund shares/units and debt securities
    up to two years. M3 is measured as a seasonally adjusted index based on 2015=100.

    Args:
        period (str): The period of the data. Can be 'monthly', 'quarterly' or 'yearly'.

    Returns:
       pd.DataFrame: A DataFrame containing the broad money for a variety
        of countries over time.
    """
    period = period.lower()

    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    period_data = "M" if period == "monthly" else "Q" if period == "quarterly" else "A"

    oecd_data_string = f".M3.TOT.IDX2015.{period_data}"

    broad_money = collect_oecd_data(
        oecd_data_string, "Y" if period_data == "A" else period_data
    )

    return broad_money


def get_general_government_deficit():
    """
    General government deficit is defined as the balance of income
    and expenditure of government, including capital income and
    capital expenditures.

    These metrics are expressed as a percentage of GDP.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        deficit for a variety of countries over time
    """
    oecd_data_string = ".GGNLEND.TOT.PC_GDP.A"

    general_government_deficit = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_deficit = general_government_deficit / 100

    return general_government_deficit


def get_general_government_revenue():
    """
    Governments collect revenues mainly for two purposes: to finance the
    goods and services they provide to citizens and businesses, and to
    fulfil their redistributive role.

    These metrics are expressed as a percentage of GDP.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        revenue for a variety of countries over time
    """
    oecd_data_string = ".GGREV.TOT.PC_GDP.A"

    general_government_revenue = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_revenue = general_government_revenue / 100

    return general_government_revenue


def get_general_government_spending(parameter: str = "TOT"):
    """
    General government spending provides an indication of the size of
    government across countries.

    General government final consumption can be broken down into two
    distinct groups. The first reflects expenditures for collective
    consumption (defence, justice, etc.) that benefit the society as
    a whole, or large parts of society, and are often known as
    public goods and services.

    These metrics are expressed as a percentage of GDP.

    Args:
        parameter (str): The parameter of the data. Can be 'TOT',
        'SOCIETY' or 'INDIVIDUAL'.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        spending for a variety of countries over time
    """
    parameter = parameter.upper()

    if parameter not in ["TOT", "SOCIETY", "INDIVIDUAL"]:
        raise ValueError(
            "Please choose one of the following parameters: 'TOT', 'SOCIETY', 'INDIVIDUAL'."
        )

    oecd_data_string = f".GGEXP.{parameter}.PC_GDP.A"

    general_government_spending = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_spending = general_government_spending / 100

    return general_government_spending


def get_general_government_debt():
    """
    General government debt-to-GDP ratio measures the
    gross debt of the general government as a percentage of GDP.

    These metrics are expressed as a percentage of GDP.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        debt for a variety of countries over time
    """
    oecd_data_string = ".GGDEBT.TOT.PC_GDP.A"

    general_government_debt = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_debt = general_government_debt / 100

    return general_government_debt


def get_general_government_financial_wealth():
    """
    The net financial worth of the general government sector is
    he total value of its financial assets minus the total value
    of its outstanding liabilities.

    These metrics are expressed as a percentage of GDP.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        financial wealth for a variety of countries over time
    """
    oecd_data_string = ".GGWEALTH.TOT.PC_GDP.A"

    general_government_financial_wealth = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_financial_wealth = general_government_financial_wealth / 100

    return general_government_financial_wealth


def get_general_government_production_costs():
    """
    General government production costs are decisions about
    the amount and type of goods and services governments produce,
    as well as on how best to produce them.

    These metrics are expressed as a percentage of GDP.

    Returns:
       pd.DataFrame: A DataFrame containing the general government
        financial wealth for a variety of countries over time
    """
    oecd_data_string = ".GGCOST.TOT.PC_GDP.A"

    general_government_production_costs = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    general_government_production_costs = general_government_production_costs / 100

    return general_government_production_costs


def get_central_government_spending(parameter: str):
    """
    Central government expenditure is defined as the central
    government budget expenditure as reported in the final
    central government accounts.

    This indicator of central government spending by function is
    measured as a percentage of total expenditures.

    Args:
        parameter (str): The parameter of the data. Can be 'GRALPUBSER', 'DEF', 'PUBORD',
            'ECOAFF', 'ENVPROT', 'HOUCOMM', 'HEALTH', 'RECULTREL', 'EDU', 'SOCPROT'.

    Returns:
       pd.DataFrame: A DataFrame containing the central government
        spending for a variety of countries over time
    """
    if parameter not in [
        "GRALPUBSER",
        "DEF",
        "PUBORD",
        "ECOAFF",
        "ENVPROT",
        "HOUCOMM",
        "HEALTH",
        "RECULTREL",
        "EDU",
        "SOCPROT",
    ]:
        raise ValueError(
            "Please choose one of the following parameters: 'GRALPUBSER', 'DEF', 'PUBORD', "
            "'ECOAFF', 'ENVPROT', 'HOUCOMM', 'HEALTH', 'RECULTREL', 'EDU', 'SOCPROT'."
        )

    oecd_data_string = f".CGEXP.{parameter}.PC.A"

    central_government_spending = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    central_government_spending = central_government_spending / 100

    return central_government_spending


def get_trust_in_goverment():
    """
    Trust in government refers to the share of people who report
    having confidence in the national government.

    Returns:
       pd.DataFrame: A DataFrame containing the trust in government
        for a variety of countries over time
    """
    oecd_data_string = ".TRUSTGOV.TOT.PC.A"

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
    oecd_data_string = ".RENEWABLE.TOT.PC_PRYENRGSUPPLY.A"

    renewable_energy = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    renewable_energy = renewable_energy / 100

    return renewable_energy


def get_crude_oil_production():
    """
    Crude oil production is defined as the quantities of oil extracted
    from the ground after the removal of inert matter or impurities.

    The number is presented in thousand tonnes of oil equivalent.

    Returns:
       pd.DataFrame: A DataFrame containing the crude oil production
        for a variety of countries over time
    """
    oecd_data_string = ".OILPROD.TOT.KTOE.A"

    crude_oil_production = collect_oecd_data(oecd_data_string, "Y")

    return crude_oil_production


def get_crude_oil_prices():
    """
    Crude oil import prices come from the IEA's Crude Oil Import Register
    and are influenced not only by traditional movements of supply and
    demand, but also by other factors such as geopolitics.

    The number is presented in US dollars per barrel.

    Returns:
       pd.DataFrame: A DataFrame containing the crude oil prices
        for a variety of countries over time
    """
    oecd_data_string = ".OILIMPPRICE.TOT.USD_BAR.A"

    crude_oil_prices = collect_oecd_data(oecd_data_string, "Y")

    return crude_oil_prices


def get_environmental_tax(parameter: str = "TOT"):
    """
    Environmentally related taxes are an important instrument for
    governments to shape relative prices of goods and services.

    Returns:
       pd.DataFrame: A DataFrame containing the environmental tax
        for a variety of countries over time
    """
    if parameter not in ["TOT", "ENRG", "TRANSPORT", "RES", "POL"]:
        raise ValueError(
            "Please choose one of the following parameters: 'TOT', 'ENRG', 'TRANSPORT', 'RES', 'POL'."
        )
    oecd_data_string = f".TAXENV.{parameter}.PC_GDP.A"

    environmental_tax = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 100 to get percentage
    environmental_tax = environmental_tax / 100

    return environmental_tax


def get_greenhouse_emissions(parameter: str = "CO2"):
    """
    Greenhouse gases refer to the sum of seven gases that have direct effects on climate change:
    carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), chlorofluorocarbons (CFCs),
    hydrofluorocarbons (HFCs), perfluorocarbons (PFCs), sulphur hexafluoride (SF6) and
    nitrogen trifluoride (NF3).

    Returns:
       pd.DataFrame: A DataFrame containing the greenhouse emissions
        for a variety of countries over time
    """
    if parameter not in ["CO2", "CO", "GHG", "NOX", "SOX", "VOC"]:
        raise ValueError(
            "Please choose one of the following parameters: 'CO2', 'CO', 'GHG', 'NOX', 'SOX', 'VOC'."
        )

    kilo_or_tonnes = (
        "KG_CAP" if parameter in ["SOX", "CO", "NOX", "SOX", "VOC"] else "TONNE_CAP"
    )

    oecd_data_string = f".AIREMISSION.{parameter}.{kilo_or_tonnes}.A"

    greenhouse_emission = collect_oecd_data(oecd_data_string, "Y")

    # Divide by 1000 to get tonnes
    if kilo_or_tonnes == "KG_CAP":
        greenhouse_emission = greenhouse_emission / 1000

    return greenhouse_emission
