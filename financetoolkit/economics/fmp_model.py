"""FinancialModelingPrep Model"""

__docformat__ = "google"

from datetime import datetime, timedelta

import pandas as pd

from financetoolkit import helpers
from financetoolkit.discovery.discovery_model import get_cached_financial_data

# The endpoint returns at most this many days per request, so longer ranges are split.
ECONOMIC_CALENDAR_WINDOW_DAYS = 90

# The country codes the economic calendar uses, named as elsewhere in the Economics
# module. They are ISO 3166 codes apart from UK (United Kingdom), EU (the euro area,
# e.g. ECB decisions) and WL (world).
ECONOMIC_CALENDAR_COUNTRIES: dict[str, str] = {
    "AE": "United Arab Emirates", "AL": "Albania", "AM": "Armenia", "AO": "Angola",
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "AZ": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BD": "Bangladesh", "BE": "Belgium", "BG": "Bulgaria",
    "BH": "Bahrain", "BR": "Brazil", "BW": "Botswana", "BY": "Belarus", "CA": "Canada",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia", "CR": "Costa Rica",
    "CV": "Cape Verde", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "DO": "Dominican Republic", "EC": "Ecuador", "EE": "Estonia",
    "EG": "Egypt", "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia", "EU": "Euro Area",
    "FI": "Finland", "FJ": "Fiji", "FR": "France", "GB": "United Kingdom", "GE": "Georgia",
    "GH": "Ghana", "GN": "Guinea", "GR": "Greece", "GT": "Guatemala", "HK": "Hong Kong",
    "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IN": "India", "IQ": "Iraq", "IS": "Iceland", "IT": "Italy", "JM": "Jamaica",
    "JO": "Jordan", "JP": "Japan", "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia",
    "KR": "South Korea", "KW": "Kuwait", "KZ": "Kazakhstan", "LB": "Lebanon",
    "LK": "Sri Lanka", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya",
    "MA": "Morocco", "MD": "Moldova", "ME": "Montenegro", "MK": "North Macedonia",
    "ML": "Mali", "MN": "Mongolia", "MO": "Macau", "MT": "Malta", "MU": "Mauritius",
    "MV": "Maldives", "MW": "Malawi", "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique",
    "NA": "Namibia", "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway", "NP": "Nepal",
    "NZ": "New Zealand", "OM": "Oman", "PA": "Panama", "PE": "Peru",
    "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PS": "Palestine", "PT": "Portugal", "PY": "Paraguay", "QA": "Qatar", "RO": "Romania",
    "RS": "Serbia", "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia",
    "SC": "Seychelles", "SE": "Sweden", "SG": "Singapore", "SI": "Slovenia",
    "SK": "Slovakia", "SL": "Sierra Leone", "SN": "Senegal", "ST": "Sao Tome and Principe",
    "SV": "El Salvador", "TH": "Thailand", "TJ": "Tajikistan", "TM": "Turkmenistan",
    "TN": "Tunisia", "TR": "Turkey", "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine",
    "UG": "Uganda", "UK": "United Kingdom", "US": "United States", "UY": "Uruguay",
    "UZ": "Uzbekistan", "VN": "Vietnam", "WL": "World", "XK": "Kosovo",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
}  # fmt: skip


def _as_list(value: str | list[str] | None) -> list[str]:
    """
    Normalises a filter argument given as one value, a comma-separated string or a list.

    Args:
        value (str | list[str] | None): The filter value(s).

    Returns:
        list[str]: The values, stripped of surrounding spaces.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")

    return [item.strip() for item in value if item and item.strip()]


def _windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    Splits a date range into consecutive windows the endpoint accepts in one request.

    Args:
        start_date (str): The start date, formatted as YYYY-MM-DD.
        end_date (str): The end date, formatted as YYYY-MM-DD.

    Returns:
        list[tuple[str, str]]: The (start, end) date of every window.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    windows = []

    while start <= end:
        window_end = min(start + timedelta(days=ECONOMIC_CALENDAR_WINDOW_DAYS - 1), end)
        windows.append((start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        start = window_end + timedelta(days=1)

    return windows


def get_economic_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    countries: str | list[str] | None = None,
    currencies: str | list[str] | None = None,
    impact: str | list[str] | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the scheduled releases of economic data (such as inflation, employment and GDP
    figures) with the previous value, the consensus estimate and, once published, the
    actual value, optionally narrowed down to specific countries, currencies and impact.

    The endpoint returns at most 90 days per request, so a longer range is retrieved in
    90-day windows. Each window is cached the same way as the discovery calendars, since
    the calendar is updated as releases come in. The filters are applied afterwards.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with. Defaults to None, which
            together with no end date returns the endpoint's default of the last 90 days.
        end_date (str, optional): The end date to filter data with. Defaults to None, which is
            today when a start date is given.
        countries (str | list[str], optional): The countries to keep, as names ("United States")
            or codes ("US"), in a list or comma-separated. Defaults to None, which keeps all.
        currencies (str | list[str], optional): The currencies to keep, e.g. "USD" or
            ["USD", "EUR"]. Defaults to None, which keeps all.
        impact (str | list[str], optional): The market impact to keep: "Low", "Medium" and/or
            "High", or "All" for every release. Defaults to None, which keeps all.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the economic data releases, indexed by date.
    """
    base_url = (
        f"https://financialmodelingprep.com/stable/economic-calendar?apikey={api_key}"
    )

    if start_date is None and end_date is None:
        urls = [base_url]
    else:
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        start_date = start_date or (
            datetime.strptime(end_date, "%Y-%m-%d")
            - timedelta(days=ECONOMIC_CALENDAR_WINDOW_DAYS - 1)
        ).strftime("%Y-%m-%d")
        urls = [
            f"{base_url}&from={window_start}&to={window_end}"
            for window_start, window_end in _windows(start_date, end_date)
        ]

    frames = helpers.run_in_parallel(
        lambda url: get_cached_financial_data(
            url=url, user_subscription=user_subscription
        ),
        [(url,) for url in urls],
    )
    frames = [frame for frame in frames if "event" in frame.columns]

    if not frames:
        return pd.DataFrame()

    economic_calendar = pd.concat(frames, ignore_index=True).drop_duplicates()

    country_codes = economic_calendar["country"].fillna("").str.upper()

    if requested_countries := _as_list(countries):
        wanted = {country.upper() for country in requested_countries}
        names = country_codes.map(ECONOMIC_CALENDAR_COUNTRIES).fillna("").str.upper()
        economic_calendar = economic_calendar[
            country_codes.isin(wanted) | names.isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    if requested_currencies := _as_list(currencies):
        wanted = {currency.upper() for currency in requested_currencies}
        economic_calendar = economic_calendar[
            economic_calendar["currency"].fillna("").str.upper().isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    # "All" asks for every release, which is how an MCP caller widens its High default.
    requested_impact = _as_list(impact)
    if requested_impact and "all" not in {level.lower() for level in requested_impact}:
        wanted = {level.lower() for level in requested_impact}
        economic_calendar = economic_calendar[
            economic_calendar["impact"].fillna("").str.lower().isin(wanted)
        ]
        country_codes = country_codes[economic_calendar.index]

    economic_calendar = economic_calendar.rename(
        columns={
            "date": "Date",
            "country": "Country Code",
            "event": "Event",
            "currency": "Currency",
            "previous": "Previous",
            "estimate": "Estimate",
            "actual": "Actual",
            "change": "Change",
            "changePercentage": "Change %",
            "impact": "Impact",
            "unit": "Unit",
        }
    )

    # Named as elsewhere in the Economics module, with the code kept next to it.
    economic_calendar.insert(
        1,
        "Country",
        country_codes.map(ECONOMIC_CALENDAR_COUNTRIES).fillna(country_codes),
    )
    economic_calendar["Date"] = pd.to_datetime(economic_calendar["Date"])

    return economic_calendar.set_index("Date").sort_index()
