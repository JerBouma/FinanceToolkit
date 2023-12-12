"""OECD Model"""
__docformat__ = "google"

import pandas as pd

BASE_URL = "https://stats.oecd.org/sdmx-json/data/DP_LIVE/"
EXTENSIONS = "/OECD?contentType=csv&detail=code&separator=comma&csv-lang=en"

CODE_TO_COUNTRY = {
    "ALB": "Albania",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGR": "Bulgaria",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "CHN": "China",
    "CMR": "Cameroon",
    "COL": "Colombia",
    "CRI": "Costa Rica",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DNK": "Denmark",
    "EA17": "Euro Area 17",
    "EA19": "Euro Area 19",
    "ESP": "Spain",
    "EST": "Estonia",
    "EU27_2020": "European Union 27",
    "FIN": "Finland",
    "FRA": "France",
    "GEO": "Georgia",
    "GRC": "Greece",
    "GBR": "United Kingdom",
    "HKG": "Hong Kong",
    "HRV": "Croatia",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "ISL": "Iceland",
    "IRL": "Ireland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "South Korea",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "MAR": "Morocco",
    "MDG": "Madagascar",
    "MEX": "Mexico",
    "MKD": "Macedonia",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "NMEC": "Non-OECD Economies",
    "OECDE": "OECD - Europe",
    "OECD": "OECD - Total",
    "PER": "Peru",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "RUS": "Russia",
    "SAU": "Saudi Arabia",
    "SEN": "Senegal",
    "SGP": "Singapore",
    "SWE": "Sweden",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "TUR": "Turkey",
    "USA": "United States",
    "WLD": "World",
    "ZAF": "South Africa",
    "ZMB": "Zambia",
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

    if inflation_type not in ["total", "food", "energy"]:
        raise ValueError("Inflation type must be one of 'total', 'food' or 'energy'")
    if period not in ["monthly", "quarterly", "yearly"]:
        raise ValueError("Period must be one of 'monthly', 'quarterly' or 'yearly'")

    inflation_data = (
        "TOT"
        if inflation_type == "total"
        else "FOOD"
        if inflation_type == "food"
        else "ENERG"
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
