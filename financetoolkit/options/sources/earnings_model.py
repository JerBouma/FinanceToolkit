"""Earnings Model: the next earnings announcements the implied earnings move is measured to."""

from financetoolkit.fmp_model import get_upcoming_earnings_date


def get_next_earnings_dates(
    tickers: list[str], api_key: str, user_subscription: str
) -> dict[str, str]:
    """
    Retrieve the first earnings announcement from today on of each ticker.

    The earnings calendar stops at the end date, so the dates are read from the announcements
    Financial Modeling Prep lists, estimates included.

    Args:
        tickers (list[str]): The tickers.
        api_key (str): The Financial Modeling Prep API key; without one there are no dates.
        user_subscription (str): The Financial Modeling Prep plan.

    Returns:
        dict[str, str]: The next announcement date per ticker, for the tickers that have one.
    """
    if not api_key:
        return {}
    dates = {}
    for ticker in tickers:
        upcoming = get_upcoming_earnings_date(ticker, api_key, user_subscription)
        if upcoming:
            dates[ticker] = upcoming
    return dates
