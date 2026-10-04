"""FinancialModelingPrep Model"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.discovery.discovery_model import get_cached_financial_data


def get_economic_calendar(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Get the scheduled releases of economic data (such as inflation, employment and GDP
    figures) for every country, with the previous value, the consensus estimate and,
    once published, the actual value.

    The calendar has no country or ticker to split it by and is updated as releases come
    in, so it is cached the same way as the discovery calendars.

    Args:
        api_key (str): the API key from Financial Modeling Prep.
        start_date (str, optional): The start date to filter data with.
        end_date (str, optional): The end date to filter data with.
        user_subscription (str, optional): The user subscription level. Defaults to "Free".

    Returns:
        pd.DataFrame: DataFrame of the economic data releases, indexed by date.
    """
    url = f"https://financialmodelingprep.com/stable/economic-calendar?apikey={api_key}"

    if start_date:
        url += f"&from={start_date}"
    if end_date:
        url += f"&to={end_date}"

    economic_calendar = get_cached_financial_data(
        url=url, user_subscription=user_subscription
    )

    if economic_calendar.empty or "event" not in economic_calendar.columns:
        return economic_calendar

    economic_calendar = economic_calendar.rename(
        columns={
            "date": "Date",
            "country": "Country",
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

    economic_calendar["Date"] = pd.to_datetime(economic_calendar["Date"])

    return economic_calendar.set_index("Date").sort_index()
