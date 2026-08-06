"""FMP Model"""

__docformat__ = "google"

import time
from datetime import datetime, timedelta

import pandas as pd

from financetoolkit.cache import frame_model, policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.fmp_model import get_financial_data
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# The stable treasury-rates endpoint only returns a maximum of 90 calendar days of data
# per request regardless of the from/to range requested, so longer histories are
# paginated by requesting successive 90-day windows and concatenating the results.
WINDOW_DAYS = 90

NAMING: dict[str, str] = {
    "month1": "1 Month",
    "month2": "2 Month",
    "month3": "3 Month",
    "month6": "6 Month",
    "year1": "1 Year",
    "year2": "2 Year",
    "year3": "3 Year",
    "year5": "5 Year",
    "year7": "7 Year",
    "year10": "10 Year",
    "year20": "20 Year",
    "year30": "30 Year",
}


def get_treasury_rates(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    sleep_timer: bool = True,
    user_subscription: str = "Free",
) -> pd.DataFrame:
    """
    Retrieves the daily U.S. Treasury par yield curve rates as officially published by the
    U.S. Department of the Treasury, covering every maturity from 1 Month through 30 Year in
    a single dataset. This is the official, risk-free curve widely used as the discount curve
    for bond valuation and as the benchmark for credit spreads.

    Also known as: the Treasury yield curve, the risk-free curve.

    Args:
        api_key (str): the API Key obtained from https://www.jeroenbouma.com/fmp
        start_date (str, optional): The start date to filter data with. Defaults to 10 years ago.
        end_date (str, optional): The end date to filter data with. Defaults to today.
        sleep_timer (bool): Whether to set a sleep timer when the rate limit is reached. Note that this only works
            if you have a Premium subscription (Starter or higher) from FinancialModelingPrep. Defaults to True.
        user_subscription (str): The subscription type of the user. Defaults to "Free".

    Notes:
        The underlying endpoint caps each request at 90 calendar days of data, so this function
        paginates in 90-day windows to cover the full requested range. A 10-year default lookback
        therefore issues roughly 40 requests -- be mindful of this on a Free plan's daily request
        limit and consider passing an explicit, narrower start_date where possible.

        When a cache is active only the part of the range that is not already stored is
        paginated over, which is what makes the 40-request default bearable on repeat runs.

    Returns:
        pd.DataFrame: the Treasury par yield curve rates, in percentage points, indexed by date
        with one column per maturity.
    """
    if not api_key:
        raise ValueError(
            "Please enter an API key from FinancialModelingPrep. "
            "For more information, look here: https://www.jeroenbouma.com/fmp"
        )

    end_date_value = (
        datetime.strptime(end_date, "%Y-%m-%d")
        if end_date is not None
        else datetime.today()
    )
    start_date_value = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date is not None
        else end_date_value - timedelta(days=10 * 365)
    )

    if start_date_value > end_date_value:
        raise ValueError(
            f"Start date ({start_date_value}) must be before end date ({end_date_value}))"
        )

    cache = get_active_cache()
    cached_rates = None

    if cache is not None:
        plan = cache.plan(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="treasury_rates",
            entities=["united_states"],
            start=start_date_value.strftime("%Y-%m-%d"),
            end=end_date_value.strftime("%Y-%m-%d"),
        )
        cached_rates = plan.cached.get("united_states")
        fetch_span = plan.get_fetch_span("united_states")

        if fetch_span is None:
            return cached_rates if cached_rates is not None else pd.DataFrame()

        # Only the outstanding part of the range is paginated over, so a rerun that
        # extends the window by a few days costs a request or two rather than forty.
        requested_start, requested_end = start_date_value, end_date_value
        start_date_value = datetime.combine(fetch_span[0], datetime.min.time())
        end_date_value = datetime.combine(fetch_span[1], datetime.min.time())
    else:
        requested_start, requested_end = start_date_value, end_date_value

    logger.info(
        "Obtaining Treasury rates from %s to %s",
        start_date_value.date(),
        end_date_value.date(),
    )

    treasury_rates_list = []
    window_end = end_date_value

    while window_end >= start_date_value:
        window_start = max(
            window_end - timedelta(days=WINDOW_DAYS - 1), start_date_value
        )

        url = (
            "https://financialmodelingprep.com/stable/treasury-rates?"
            f"from={window_start.strftime('%Y-%m-%d')}&to={window_end.strftime('%Y-%m-%d')}"
            f"&apikey={api_key}"
        )

        treasury_rates_list.append(
            get_financial_data(
                url=url, sleep_timer=sleep_timer, user_subscription=user_subscription
            )
        )

        window_end = window_start - timedelta(days=1)

        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

    treasury_rates = pd.concat(treasury_rates_list, axis=0)

    if "date" not in treasury_rates.columns:
        return treasury_rates

    treasury_rates = treasury_rates.set_index("date")
    treasury_rates.index = pd.PeriodIndex(
        pd.to_datetime(treasury_rates.index), freq="D"
    )
    treasury_rates.index.name = "Date"

    treasury_rates = treasury_rates[~treasury_rates.index.duplicated(keep="first")]
    treasury_rates = treasury_rates.sort_index()

    treasury_rates = treasury_rates.rename(columns=NAMING)
    treasury_rates = treasury_rates[
        [column for column in NAMING.values() if column in treasury_rates.columns]
    ]

    if cache is not None and not treasury_rates.empty:
        cache.store(
            source=policy_model.FINANCIAL_MODELING_PREP,
            dataset="treasury_rates",
            entity="united_states",
            data=treasury_rates,
            start=start_date_value.strftime("%Y-%m-%d"),
            end=end_date_value.strftime("%Y-%m-%d"),
        )

    if cached_rates is not None and not cached_rates.empty:
        treasury_rates = frame_model.merge_frames(cached_rates, treasury_rates)
        treasury_rates = frame_model.slice_frame(
            treasury_rates,
            requested_start.strftime("%Y-%m-%d"),
            requested_end.strftime("%Y-%m-%d"),
        )

    return treasury_rates
