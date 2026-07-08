"""FXMacroData Model"""

__docformat__ = "google"

import os
from urllib.parse import urlencode

import pandas as pd

from financetoolkit.helpers import get_request

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def get_release_calendar(
    currency: str = "usd",
    limit: int = 100,
    min_tier: int | None = None,
    api_key: str | None = None,
    base_url: str = FXMACRODATA_BASE_URL,
) -> pd.DataFrame:
    """
    Retrieve an FXMacroData economic release calendar for a currency.

    Args:
        currency (str, optional): Three-letter FX currency code. Defaults to "usd".
        limit (int, optional): Maximum number of events to return. Defaults to 100.
        min_tier (int | None, optional): Optional maximum market tier to keep.
        api_key (str | None, optional): FXMacroData API key. Defaults to the
            FXMACRODATA_API_KEY environment variable when available.
        base_url (str, optional): FXMacroData API base URL.

    Returns:
        pd.DataFrame: Release calendar rows indexed by release date.
    """
    limit = max(1, int(limit))
    params = {"limit": limit}
    token = api_key or os.environ.get("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}?{urlencode(params)}"
    payload = get_request(url, timeout=30).json()
    events = payload.get("data", [])

    if min_tier is not None:
        events = [
            event
            for event in events
            if int(event.get("market_tier") or 99) <= int(min_tier)
        ]

    release_calendar = pd.DataFrame(events[:limit])
    if release_calendar.empty:
        return release_calendar

    if "date" in release_calendar.columns:
        release_calendar["date"] = pd.to_datetime(
            release_calendar["date"], errors="coerce"
        )
        release_calendar = release_calendar.set_index("date").sort_index()
        release_calendar.index.name = "Date"

    if "announcement_datetime" in release_calendar.columns:
        release_calendar["Announcement DateTime"] = pd.to_datetime(
            release_calendar.pop("announcement_datetime"),
            unit="s",
            utc=True,
            errors="coerce",
        )

    return release_calendar
