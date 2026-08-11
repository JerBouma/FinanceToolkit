"""Historical Module"""

__docformat__ = "google"

import contextlib
import importlib.util
import threading
import time

import numpy as np
import pandas as pd

from financetoolkit import fmp_model, helpers, yfinance_model
from financetoolkit.cache import frame_model, policy_model
from financetoolkit.cache.cache_controller import Cache
from financetoolkit.utilities import error_model, logger_model
from financetoolkit.utilities.statistics_model import (
    PERIOD_TRANSLATION,
    apply_rounding,
)

logger = logger_model.get_logger()

# Check if yfinance is installed
yf_spec = importlib.util.find_spec("yfinance")
ENABLE_YFINANCE = yf_spec is not None

# pylint: disable=too-many-locals,unsubscriptable-object,too-many-lines

TREASURY_LIMIT = 90

INTERVAL_STR = {
    "1min": "min",
    "5min": "min",
    "15min": "min",
    "30min": "min",
    "1hour": "h",
    "4hour": "h",
    "1d": "D",
}

# Minimum fraction of the requested span a response must itself span to be
# treated as "available" rather than a truncated data-plan limitation -- see
# _covers_requested_range's own docstring.
MIN_COVERAGE_RATIO = 0.5
# Requested spans shorter than this are never flagged as truncated -- too
# short to tell a genuine short window apart from a truncated one.
MIN_SPAN_TO_CHECK_DAYS = 30
# How much later than the requested start a response may begin before it is
# treated as truncated. FinancialModelingPrep caps a response at a fixed number
# of rows rather than a fixed span, so a capped response runs to the requested
# end (or beyond) and is missing history at the *start* only -- a few days of
# slack absorbs a start date that lands on a weekend or a market holiday.
MAX_START_SHORTFALL_DAYS = 30


def _first_observed_date(data: pd.DataFrame) -> pd.Timestamp | None:
    """
    The earliest date present in a historical frame, or None when it is empty.

    Args:
        data (pd.DataFrame): a historical frame, indexed by either a PeriodIndex
            or a DatetimeIndex.

    Returns:
        pd.Timestamp | None: the earliest date, normalised to a Timestamp so that
            frames carrying different index types stay comparable.
    """
    if data.empty:
        return None

    index = (
        data.index.to_timestamp() if hasattr(data.index, "to_timestamp") else data.index
    )

    return index.min()


def _reaches_further_back(candidate: pd.DataFrame, incumbent: pd.DataFrame) -> bool:
    """
    Whether `candidate` starts earlier than `incumbent` and is therefore worth
    preferring over it.

    Used to decide whether a Yahoo Finance response genuinely improves on a
    truncated FinancialModelingPrep one. A ticker that simply listed recently
    yields the same short history from both providers, in which case there is
    nothing to gain by switching and the original response is kept.

    Args:
        candidate (pd.DataFrame): the replacement response being considered.
        incumbent (pd.DataFrame): the response already in hand.

    Returns:
        bool: True only when candidate holds data and starts strictly earlier
            than incumbent.
    """
    candidate_start = _first_observed_date(candidate)
    incumbent_start = _first_observed_date(incumbent)

    if candidate_start is None:
        return False

    if incumbent_start is None:
        return True

    return candidate_start < incumbent_start


def _covers_requested_range(
    data: pd.DataFrame, fetch_start: str | None, fetch_end: str | None
) -> bool:
    """
    Whether `data` actually covers the [fetch_start, fetch_end] span requested.

    Guards against a provider returning a non-empty but truncated response, which
    a plain `.empty` check cannot see, so `get_historical_data`'s FMP-then-Yahoo
    fallback never fires and the caller silently receives less history than asked
    for. FinancialModelingPrep does this in two distinct ways, and both are checked:

    1. **A drastically shorter span.** Observed with lower-tier commodities data,
       which can return three months of history when twenty years were requested.
       Caught by comparing the response's own span against MIN_COVERAGE_RATIO of
       the requested span.
    2. **A row cap.** The endpoint returns at most a fixed number of rows (5,000
       observed on a Premium key) rather than a fixed number of years. A capped
       response therefore runs all the way to the requested end -- often past it --
       and is missing history at the *start* only. Requesting 2005-2025 for AAPL
       returns 2006-02-16 onwards: a span ratio of 0.99, which sails past the first
       check while still silently dropping a year. Caught by comparing the observed
       first date against the requested start.

    Args:
        data (pd.DataFrame): the response to check, PeriodIndex-indexed.
        fetch_start (str | None): the requested start date ('YYYY-MM-DD'), or
            None for an open-ended request (always treated as covered -- no
            fixed span to compare against).
        fetch_end (str | None): the requested end date ('YYYY-MM-DD'), or None.

    Returns:
        bool: True if data is empty (the existing `.empty` check already handles
            that case separately), fetch_start/fetch_end is missing, the requested
            span is under MIN_SPAN_TO_CHECK_DAYS, or the response both spans
            MIN_COVERAGE_RATIO of the request and begins within
            MAX_START_SHORTFALL_DAYS of the requested start. False otherwise.

    Notes:
        A genuinely young ticker -- one that listed after `fetch_start` -- is
        indistinguishable from a capped response here, and is reported as not
        covered. That is deliberate: the caller resolves it by comparing against
        what Yahoo Finance returns and keeping whichever reaches further back, so
        a young ticker costs one extra request rather than the wrong answer.
    """
    if data.empty or fetch_start is None or fetch_end is None:
        return True

    requested_start = pd.Timestamp(fetch_start)
    requested_days = (pd.Timestamp(fetch_end) - requested_start).days

    if requested_days < MIN_SPAN_TO_CHECK_DAYS:
        return True

    observed_index = (
        data.index.to_timestamp() if hasattr(data.index, "to_timestamp") else data.index
    )

    covered_days = (observed_index.max() - observed_index.min()).days

    if covered_days < MIN_COVERAGE_RATIO * requested_days:
        return False

    start_shortfall_days = (observed_index.min() - requested_start).days

    return start_shortfall_days <= MAX_START_SHORTFALL_DAYS


def get_historical_data(
    tickers: list[str] | str,
    api_key: str | None = None,
    enforce_source: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    return_column: str = "Adj Close",
    include_dividends: bool = True,
    fill_nan: bool = True,
    divide_ohlc_by: int | float | None = None,
    rounding: int | None = None,
    sleep_timer: bool = True,
    show_ticker_seperation: bool = True,
    show_errors: bool = False,
    log_message: str = "Obtaining historical data",
    user_subscription: str = "Free",
    cache: Cache | None = None,
):
    """
    Retrieves historical stock data for the given ticker(s) from Financial Modeling Prep or/and Yahoo Finance
    for a specified period. If start and/or end date are not provided, it defaults to 10 years from
    the current date.

    It intelligently uses both Financial Modeling Prep and Yahoo Finance to retrieve the data. If the data
    is not available from Financial Modeling Prep, it will use Yahoo Finance instead. This is because
    opposes limits to Free plans (e.g. no tickers from outside the American exchanges) and in some cases
    Yahoo Finance has a broader universe.

    By using threading, multiple API calls can be made at the same time, which speeds up the process
    significantly. For example, collecting historical data of 100 tickers takes around 10 seconds.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        api_key (str, optional): The FinancialModelingPrep API key. Without one only Yahoo Finance
        is used. Defaults to None.
        enforce_source (str, optional): Restrict retrieval to 'FinancialModelingPrep' or
        'YahooFinance'. Defaults to None, which tries FinancialModelingPrep first.
        start (str, optional): A string representing the start date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        end (str, optional): A string representing the end date of the period to retrieve data for
            in 'YYYY-MM-DD' format. Defaults to None.
        interval (str, optional): A string representing the interval to retrieve data for.
        return_column (str, optional): A string representing the column to use for return calculations.
        include_dividends (bool, optional): A boolean representing whether to include dividends in the
        historical data. Defaults to True.
        fill_nan (bool, optional): A boolean representing whether to fill NaN values with the previous
        value. Defaults to True.
        divide_ohlc_by (int, optional): An intege   r representing the value to divide the OHLC data by.
        This is useful if the OHLC data is presented in percentages or similar. Defaults to None.
        rounding (int, optional): The number of decimal places to round the data to. Defaults to None.
        sleep_timer (bool, optional): A boolean representing whether to introduce a sleep timer to prevent
        rate limit errors. Defaults to True.
        show_ticker_seperation (bool, optional): A boolean representing whether to show which tickers
        acquired data from FinancialModelingPrep and which tickers acquired data from YahooFinance.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        log_message (str, optional): A string representing the message to show in the log output.
        user_subscription (str, optional): The FinancialModelingPrep plan, which governs the retry
        behaviour and the available history. Defaults to "Free".
        cache (Cache, optional): An incremental cache to serve already retrieved ranges from. When
        provided, each ticker only requests the part of the period that is not cached yet, so
        widening the date range or adding a ticker does not refetch what is already stored.

    Raises:
        ValueError: If the start date is after the end date.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the historical stock data for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    # NaN rather than 0 so that a ticker without data is recognisable as missing
    # instead of reading as a genuine price, volume and return of zero.
    empty_historical_data = pd.DataFrame(
        data=np.nan,
        index=pd.PeriodIndex(pd.date_range(start, end), freq=INTERVAL_STR[interval]),
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Return",
            "Cumulative Return",
        ],
    )

    # Keyed on what changes the frame; the period is tracked as coverage instead.
    cache_parameters = {
        "interval": interval,
        "return_column": return_column,
        "include_dividends": include_dividends,
        "divide_ohlc_by": divide_ohlc_by,
        # The dividend limit depends on the plan, so a Free frame is not a Premium one.
        "user_subscription": user_subscription,
    }
    cache_dataset = (
        "intraday" if interval not in ("1d", "1wk", "1mo", "1y") else "historical"
    )

    # Stored under the provider that served it, as `enforce_source` names it.
    candidate_sources = (
        # Only FinancialModelingPrep publishes intraday bars, so there is no second.
        (policy_model.FINANCIAL_MODELING_PREP,)
        if cache_dataset == "intraday"
        else (policy_model.FINANCIAL_MODELING_PREP, policy_model.YAHOO_FINANCE)
    )
    cache_sources = [
        source
        for source in candidate_sources
        if enforce_source is None or enforce_source == source
    ]

    def resolve_from_cache(ticker):
        """Find the provider holding this ticker, with whatever gap is left to fetch."""
        for source in cache_sources:
            plan = cache_plans.get(source)

            if plan is None:
                continue

            cached_data = plan.cached.get(ticker)

            if cached_data is not None and not cached_data.empty:
                return source, cached_data, plan.get_fetch_span(ticker)

        return None, None, None

    def worker(ticker, historical_data_dict, historical_data_error_dict):
        cached_source, cached_data, fetch_span = (
            resolve_from_cache(ticker) if cache_plans else (None, None, None)
        )
        cache_source = None

        if cached_data is not None and fetch_span is None:
            historical_data_dict[ticker] = helpers.enrich_historical_data(
                historical_data=cached_data,
                start=start,
                end=end,
                return_column=return_column,
            )

            return

        fetch_start = fetch_span[0].strftime("%Y-%m-%d") if fetch_span else start
        fetch_end = fetch_span[1].strftime("%Y-%m-%d") if fetch_span else end

        historical_data = pd.DataFrame()
        # A non-empty but truncated FinancialModelingPrep response, held aside while
        # Yahoo Finance is tried, so it can be restored if Yahoo is no better.
        truncated_data = pd.DataFrame()
        attempted_fmp = False

        if api_key and interval in ["1min", "5min", "15min", "30min", "1hour", "4hour"]:
            # Intraday comes only from FinancialModelingPrep, so there is no fallback.
            historical_data = fmp_model.get_intraday_data(
                ticker=ticker,
                api_key=api_key,
                start=fetch_start,
                end=fetch_end,
                interval=interval,
                return_column=return_column,
                sleep_timer=sleep_timer,
                user_subscription=user_subscription,
            )

            if not historical_data.empty:
                cache_source = policy_model.FINANCIAL_MODELING_PREP

        elif not api_key and interval in [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "4hour",
        ]:
            raise ValueError(
                "The requested data requires the api_key parameter to be set, consider "
                "obtaining a key with the following link: "
                "https://www.jeroenbouma.com/fmp"
                "\nThe free plan allows for 250 requests per day, a limit of 5 years and has no "
                "quarterly data. Consider upgrading your plan. You can get 15% off by using the "
                "above affiliate link which also supports the project."
            )
        else:
            if api_key and enforce_source in [None, "FinancialModelingPrep"]:
                historical_data = fmp_model.get_historical_data(
                    ticker=ticker,
                    api_key=api_key,
                    start=fetch_start,
                    end=fetch_end,
                    interval=interval,
                    return_column=return_column,
                    include_dividends=include_dividends,
                    divide_ohlc_by=divide_ohlc_by,
                    sleep_timer=sleep_timer,
                    user_subscription=user_subscription,
                )

                # Only worth setting aside when there is somewhere to fall back to.
                # With the source forced to FinancialModelingPrep, a truncated
                # response is the best answer available and discarding it would
                # return nothing at all.
                can_fall_back = (
                    enforce_source != "FinancialModelingPrep" and ENABLE_YFINANCE
                )

                if (
                    not historical_data.empty
                    and can_fall_back
                    and not _covers_requested_range(
                        historical_data, fetch_start, fetch_end
                    )
                ):
                    # Held rather than dropped: Yahoo is tried below and this is
                    # restored if it does not actually reach further back -- see
                    # _covers_requested_range for why a non-empty response can
                    # still be truncated.
                    logger.debug(
                        "FinancialModelingPrep returned only %s to %s for %s, short of the %s to %s "
                        "requested -- likely a row cap or data-plan limitation for this ticker/endpoint, "
                        "trying Yahoo Finance for a longer history.",
                        historical_data.index.min(),
                        historical_data.index.max(),
                        ticker,
                        fetch_start,
                        fetch_end,
                    )
                    truncated_data = historical_data
                    historical_data = pd.DataFrame()

                if not historical_data.empty:
                    fmp_tickers.append(ticker)
                    cache_source = policy_model.FINANCIAL_MODELING_PREP

                attempted_fmp = True

            if (
                enforce_source != "FinancialModelingPrep"
                and historical_data.empty
                and ENABLE_YFINANCE
            ):
                historical_data = yfinance_model.get_historical_data(
                    ticker=ticker,
                    start=fetch_start,
                    end=fetch_end,
                    interval=interval,
                    return_column=return_column,
                    divide_ohlc_by=divide_ohlc_by,
                    fallback=attempted_fmp,
                )

                if not truncated_data.empty and not _reaches_further_back(
                    historical_data, truncated_data
                ):
                    # Yahoo is no better -- a young ticker rather than a capped
                    # response -- so keep what FinancialModelingPrep returned.
                    historical_data = truncated_data
                    fmp_tickers.append(ticker)
                    cache_source = policy_model.FINANCIAL_MODELING_PREP
                elif not historical_data.empty:
                    yf_tickers.append(ticker)
                    cache_source = policy_model.YAHOO_FINANCE

        if cache is not None and cache_source and not historical_data.empty:
            # Only for a non-empty response: an empty frame may just be a rate limit.
            cache.store(
                source=cache_source,
                dataset=cache_dataset,
                entity=ticker,
                data=historical_data,
                start=fetch_start,
                end=fetch_end,
                parameters=cache_parameters,
            )

        # Merge only when one provider served both halves; adjustments differ.
        if (
            cached_data is not None
            and not cached_data.empty
            and cached_source == cache_source
        ):
            historical_data = frame_model.merge_frames(cached_data, historical_data)

        if not historical_data.empty:
            # Return and Cumulative Return depend on the window, so recompute after merging.
            historical_data = helpers.enrich_historical_data(
                historical_data=frame_model.slice_frame(historical_data, start, end),
                start=start,
                end=end,
                return_column=return_column,
            )

        if historical_data.empty:
            no_data.append(ticker)
            historical_data_error_dict[ticker] = historical_data
            historical_data_dict[ticker] = empty_historical_data
        if not historical_data.empty:
            historical_data_dict[ticker] = historical_data

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    logger.info("%s for %d ticker(s)", log_message, len(ticker_list))
    historical_data_dict: dict[str, pd.DataFrame] = {}
    historical_data_error_dict: dict[str, pd.DataFrame] = {}
    fmp_tickers: list[str] = []
    yf_tickers: list[str] = []
    no_data: list[str] = []
    threads = []

    # One plan per allowed provider, since either may have served an earlier run.
    cache_plans = (
        {
            source: cache.plan(
                source=source,
                dataset=cache_dataset,
                entities=ticker_list,
                start=start,
                end=end,
                parameters=cache_parameters,
            )
            for source in cache_sources
        }
        if cache is not None and cache.enabled
        else {}
    )

    if cache_plans and all(
        resolved[1] is not None and resolved[2] is None
        for resolved in (resolve_from_cache(ticker) for ticker in ticker_list)
    ):
        logger.info("%s from the cache for %d ticker(s)", log_message, len(ticker_list))

    for ticker in ticker_list:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, historical_data_dict, historical_data_error_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if show_errors:
        error_model.check_for_error_messages(
            dataset_dictionary=historical_data_error_dict,
            user_subscription=user_subscription,
        )

    if fmp_tickers and yf_tickers and show_ticker_seperation:
        logger.info(
            "The following tickers acquired historical data from FinancialModelingPrep: %s",
            ", ".join(fmp_tickers),
        )
        logger.info(
            "The following tickers acquired historical data from YahooFinance: %s",
            ", ".join(yf_tickers),
        )

    if (
        yf_tickers
        and not fmp_tickers
        and enforce_source == "FinancialModelingPrep"
        and show_errors
    ):
        logger.warning(
            "No data found using FinancialModelingPrep, this is usually due to Bandwidth "
            "API limits or usage of the Free plan.\n"
            "Therefore data was retrieved from YahooFinance instead for: %s",
            ", ".join(yf_tickers),
        )

    if no_data and show_errors:
        if not ENABLE_YFINANCE:
            logger.info(
                "Due to a missing optional dependency (yfinance) and your current FinancialModelingPrep plan, "
                "data for the following tickers could not be acquired: %s\n"
                "Enable this functionality by using:\033[1m pip install yfinance\033[0m",
                ", ".join(no_data),
            )
        else:
            logger.warning(
                "No data found for the following tickers: %s", ", ".join(no_data)
            )

    if len(historical_data_dict) == 0:
        # Zero-filled so a DataFrame is returned even when nothing is found.
        for ticker in tickers:
            historical_data_dict[ticker] = empty_historical_data

    reorder_tickers = [ticker for ticker in tickers if ticker in historical_data_dict]

    if historical_data_dict and len(no_data) != len(tickers):
        historical_data = pd.concat(historical_data_dict).unstack(level=0)
        historical_data = historical_data.reindex(reorder_tickers, level=1, axis=1)

        if "Dividends" in historical_data.columns:
            historical_data["Dividends"] = historical_data["Dividends"].fillna(0)

        if fill_nan:
            # Interpolation smooths NaN gaps with limited impact on any metric.
            historical_data = historical_data.interpolate(limit_area="inside")

        historical_data = apply_rounding(historical_data, rounding)

        if not isinstance(historical_data.index, pd.Period):
            historical_data = historical_data.loc[
                [isinstance(item, pd.Period) for item in historical_data.index]
            ]
            historical_data.index = pd.PeriodIndex(
                historical_data.index, freq=INTERVAL_STR[interval]
            )

        return historical_data, no_data

    return pd.DataFrame(), no_data


def convert_daily_to_other_period(
    period: str,
    daily_historical_data: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    rounding: int | None = None,
    return_column: str = "Adj Close",
):
    """
    Converts daily historical data to another period which can be:
        - Weekly
        - Monthly
        - Quarterly
        - Yearly

    Open is taken from the first day of the period, High is the highest and Low the
    lowest value within it, Close is the last day of the period and Volume and Dividends
    are summed over it. It also calculates the following:
        - Return: The return for the given period.
        - Cumulative Return: The cumulative return for the given period.

    Args:
        period (str): The period to convert to (weekly, monthly, quarterly or yearly).
        daily_historical_data (pd.DataFrame): A DataFrame containing daily historical data.
        start (str | None, optional): The start of the range the cumulative return is
            based on. Defaults to None.
        end (str | None, optional): The end of the range the cumulative return is based
            on. Defaults to None.
        rounding (int | None, optional): The number of decimals to round to, or None for
            no rounding. Defaults to None.
        return_column (str, optional): The column the period return is calculated from.
            Defaults to "Adj Close", which makes the return a total return.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the yearly historical stock data.
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the OHLC data as the second level.
    """
    if period not in ["weekly", "monthly", "quarterly", "yearly"]:
        raise ValueError(
            f"Period {period} is not valid. It should be either "
            "weekly, monthly, quarterly or yearly."
        )

    period_str = PERIOD_TRANSLATION[period]

    daily_historical_data.index.name = "Date"
    dates = daily_historical_data.index.asfreq(period_str)
    daily_historical_data = daily_historical_data.reset_index()

    # Each column has to be aggregated on its own terms. Taking the last value of the
    # period for every column would report the final day's Open, High, Low and Volume
    # as if they described the whole period.
    aggregations = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Volume": "sum",
        "Dividends": "sum",
    }

    period_historical_data = daily_historical_data.copy()

    for column in daily_historical_data.columns:
        column_name = column[0] if isinstance(column, tuple) else column

        period_historical_data[column] = (
            daily_historical_data[column]
            .groupby(dates)
            .transform(aggregations.get(column_name, "last"))
        )

    period_historical_data["Date"] = period_historical_data["Date"]
    period_historical_data = period_historical_data.drop_duplicates().set_index("Date")
    period_historical_data.index = pd.PeriodIndex(
        period_historical_data.index, freq=period_str
    )

    if "Return" in period_historical_data:
        period_historical_data["Return"] = (
            period_historical_data[return_column]
            / period_historical_data[return_column].shift()
            - 1
        ).replace([np.inf, -np.inf], np.nan)

    if "Cumulative Return" in period_historical_data:
        if start:
            start = max(
                pd.Period(start).asfreq(period_str), period_historical_data.index[0]
            )
        if end:
            end = min(
                pd.Period(end).asfreq(period_str), period_historical_data.index[-1]
            )

        adjusted_return = period_historical_data.loc[start:end, "Return"]

        # An empty selection means the requested range falls outside the data, which is
        # a legitimate outcome rather than a reason to abort the whole conversion.
        with contextlib.suppress(IndexError):
            adjusted_return.iloc[0] = 0

        period_historical_data["Cumulative Return"] = (
            1 + adjusted_return.fillna(0)
        ).cumprod()
        period_historical_data["Cumulative Return"] = period_historical_data[
            "Cumulative Return"
        ].fillna(1)

    period_historical_data = period_historical_data.sort_index()

    period_historical_data = apply_rounding(period_historical_data, rounding)

    # Deliberately not filled with zero. The first period has no preceding period to
    # compare against, so its Return is unknown rather than flat, and a fabricated zero
    # is indistinguishable from a genuinely unchanged period. It propagated into the
    # first row of every period level performance metric, which is why an asset with a
    # positive excess return could report a negative Ulcer Performance Index.
    return period_historical_data


def get_historical_statistics(
    tickers: list[str] | str,
    api_key: str | None = None,
    show_errors: bool = False,
    log_message: str = "Obtaining historical statistics",
    user_subscription: str = "Free",
):
    """
    Retrieves statistics about each ticker's historical data. This is useful to understand why certain
    tickers might fluctuate more than others as it could be due to local regulations or the currency the instrument
    is denoted in. It returns:

        - Currency: The currency the instrument is denoted in.
        - Symbol: The symbol of the instrument.
        - Exchange Name: The name of the exchange the instrument is listed on.
        - Instrument Type: The type of instrument.
        - First Trade Date: The date the instrument was first traded.
        - Regular Market Time: The time the instrument is traded.
        - GMT Offset: The GMT offset.
        - Timezone: The timezone the instrument is traded in.
        - Exchange Timezone Name: The name of the timezone the instrument is traded in.

    It attempts to collect data from Yahoo Finance first and if that fails it will attempt to collect data from
    FinancialModelingPrep. If both fail it will return an empty DataFrame for the ticker. Generally Yahoo Finance
    should be sufficient but for delisted companies, only FinancialModelingPrep will offer data.

    Args:
        tickers (list of str): A list of one or more ticker symbols to retrieve data for.
        api_key (str, optional): The API key to use to retrieve the data from FinancialModelingPrep.
        show_errors (bool, optional): A boolean representing whether to show errors. Defaults to True.
        log_message (str, optional): A string representing the message to show in the log output.
        user_subscription (str, optional): The FinancialModelingPrep plan, used when reporting
        retrieval errors. Defaults to "Free".

    Returns:
        pd.DataFrame: A pandas DataFrame object containing the statistics for the given ticker(s).
        The index of the DataFrame is the date of the data and the columns are a multi-index
        with the ticker symbol(s) as the first level and the statistics as the second level.
    """

    def worker(ticker, historical_statistics_dict):
        historical_statistics = pd.DataFrame()

        if historical_statistics.empty:
            historical_statistics = yfinance_model.get_historical_statistics(
                ticker=ticker
            )

        if api_key and historical_statistics.empty:
            historical_statistics = fmp_model.get_historical_statistics(
                ticker=ticker,
                api_key=api_key,
            )

        if historical_statistics.empty:
            no_data.append(ticker)
        if not historical_statistics.empty:
            historical_statistics_dict[ticker] = historical_statistics

    if isinstance(tickers, str):
        ticker_list = [tickers]
    elif isinstance(tickers, list):
        ticker_list = tickers
    else:
        raise ValueError(f"Type for the tickers ({type(tickers)}) variable is invalid.")

    logger.info("%s for %d ticker(s)", log_message, len(ticker_list))
    historical_statistics_dict: dict[str, pd.DataFrame] = {}
    no_data: list[str] = []
    threads = []

    for ticker in ticker_list:
        # Introduce a sleep timer to prevent rate limit errors
        time.sleep(0.1)

        thread = threading.Thread(
            target=worker,
            args=(ticker, historical_statistics_dict),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    historical_statistics_dict = (
        error_model.check_for_error_messages(
            dataset_dictionary=historical_statistics_dict,
            user_subscription=user_subscription,
        )
        if show_errors
        else historical_statistics_dict
    )

    if historical_statistics_dict:
        historical_statistics = pd.concat(historical_statistics_dict, axis=1)

        return historical_statistics, no_data

    return pd.DataFrame(), no_data
