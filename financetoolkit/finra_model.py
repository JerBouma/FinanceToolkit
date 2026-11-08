"""FINRA Short Sale Volume Model"""

__docformat__ = "google"

import io

import numpy as np
import pandas as pd
import requests

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.helpers import determine_max_workers, run_in_parallel
from financetoolkit.utilities.logger_model import get_logger
from financetoolkit.utilities.requests_model import get_request

logger = get_logger()

# FINRA publishes one file per trading day with the short sale volume of every symbol
# reported to its facilities (the Trade Reporting Facilities and the Alternative Display
# Facility). The consolidated "CNMS" file adds up all facilities and starts in August 2018.
DAILY_FILE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
FIRST_DATE = pd.Timestamp("2018-08-01")

DATASET = "short_volume"
VOLUME_COLUMNS = {
    "ShortVolume": "Short Volume",
    "ShortExemptVolume": "Short Exempt Volume",
    "TotalVolume": "Total Volume",
}

# A day without a file (a holiday, or today before FINRA publishes in the evening) is
# answered with 403 by the CDN rather than 404.
NOT_PUBLISHED = (403, 404)

# Every day is a separate file of about half a megabyte, so the downloads are spread over
# a few threads, but fewer than for an API made for many calls.
MAX_DOWNLOADS = 8


def to_finra_symbol(ticker: str) -> str | None:
    """
    Converts a ticker to the symbol FINRA reports it under: share classes are written with
    a slash (BRK/B rather than BRK-B). FINRA only covers securities listed in the United
    States, so a ticker with an exchange suffix (ASML.AS), an index (^GSPC) or a currency
    pair (EURUSD=X) has no symbol.

    Args:
        ticker (str): The ticker as the Finance Toolkit writes it.

    Returns:
        str | None: The FINRA symbol, or None when FINRA does not cover the ticker.
    """
    if any(character in ticker for character in ".^="):
        return None

    return ticker.upper().replace("-", "/")


def read_daily_file(text: str) -> pd.DataFrame:
    """
    Reads one daily short sale volume file of FINRA.

    The file is pipe delimited with the columns Date, Symbol, ShortVolume,
    ShortExemptVolume, TotalVolume and Market, and ends with a line holding the number of
    records. Volumes include fractional shares, so they are not whole numbers.

    Args:
        text (str): The content of the file.

    Returns:
        pd.DataFrame: The short volume, short exempt volume and total volume (columns) per
        symbol (index).

    Raises:
        ValueError: When a column is missing, which means FINRA changed the format.
    """
    data = pd.read_csv(
        io.StringIO(text), sep="|", dtype={"Symbol": str}, keep_default_na=False
    )

    missing = {"Symbol", *VOLUME_COLUMNS} - set(data.columns)
    if missing:
        raise ValueError(
            f"The FINRA short sale volume file has no {', '.join(sorted(missing))} column."
        )

    # The closing line only holds the record count, so it has no symbol.
    data = data[data["Symbol"] != ""]
    volumes = data[list(VOLUME_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    volumes.index = data["Symbol"]

    return volumes.rename(columns=VOLUME_COLUMNS)


def _download_day(day: pd.Timestamp, symbols: list[str]) -> pd.DataFrame | None:
    """
    Downloads the file of one trading day and keeps the rows of the requested symbols.

    Args:
        day (pd.Timestamp): The trading day.
        symbols (list[str]): The FINRA symbols to keep.

    Returns:
        pd.DataFrame | None: The volumes (columns) of the symbols found (index), empty when
        FINRA published no file for the day, or None when the file could not be retrieved.
    """
    url = DAILY_FILE_URL.format(date=day.strftime("%Y%m%d"))

    try:
        response = get_request(url, timeout=60)
    except requests.exceptions.HTTPError as error:
        if error.response is not None and error.response.status_code in NOT_PUBLISHED:
            return pd.DataFrame()

        logger.debug("Could not retrieve %s: %s", url, error)
        return None
    except requests.exceptions.RequestException as error:
        logger.debug("Could not retrieve %s: %s", url, error)
        return None

    volumes = read_daily_file(response.text)

    return volumes.loc[volumes.index.intersection(symbols)]


def _download_days(
    days: pd.DatetimeIndex, symbols: list[str]
) -> tuple[dict[str, pd.DataFrame], int]:
    """
    Downloads the files of several trading days at the same time and arranges the volumes
    per symbol.

    Args:
        days (pd.DatetimeIndex): The trading days.
        symbols (list[str]): The FINRA symbols to keep.

    Returns:
        tuple[dict[str, pd.DataFrame], int]: Per symbol, the volumes (columns) per day
        (index), and the number of days that could not be retrieved.
    """
    results = run_in_parallel(
        lambda day: (day, _download_day(day, symbols)),
        [(day,) for day in days],
        max_workers=min(MAX_DOWNLOADS, determine_max_workers()),
    )

    rows: dict[str, dict[pd.Period, pd.Series]] = {symbol: {} for symbol in symbols}
    failures = len(days) - len(results)

    for day, volumes in results:
        if volumes is None:
            failures += 1
            continue

        for symbol, row in volumes.iterrows():
            rows[str(symbol)][pd.Period(day, freq="D")] = row

    per_symbol = {}

    for symbol, symbol_rows in rows.items():
        frame = pd.DataFrame.from_dict(symbol_rows, orient="index")
        frame = frame.reindex(columns=list(VOLUME_COLUMNS.values()))
        frame.index = pd.PeriodIndex(frame.index, freq="D", name="Date")
        per_symbol[symbol] = frame.sort_index().astype(float)

    return per_symbol, failures


def _trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    The weekdays between two dates from the first FINRA file up to today; a holiday is
    requested as well and simply has no file.

    Args:
        start (pd.Timestamp): The start date.
        end (pd.Timestamp): The end date.

    Returns:
        pd.DatetimeIndex: The weekdays.
    """
    return pd.bdate_range(
        max(start, FIRST_DATE), min(end, pd.Timestamp.today().normalize())
    )


def collect_short_volume(
    symbols: list[str], start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    """
    Retrieves the daily short sale volume of symbols between two dates through the cache.

    The cache holds each symbol's days, so a rerun only downloads the days that were never
    retrieved for one of the symbols (and, once a day has passed, the last week again, as a
    file can be published late). Each daily file covers every symbol, so a day missing for
    several symbols is downloaded once.

    Args:
        symbols (list[str]): The FINRA symbols.
        start_date (str): The start date (YYYY-MM-DD).
        end_date (str): The end date (YYYY-MM-DD).

    Returns:
        dict[str, pd.DataFrame]: Per symbol, the short volume, short exempt volume and total
        volume (columns) per day (index); empty for a symbol FINRA has no data for.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    cache = get_active_cache()
    cached: dict[str, pd.DataFrame | None] = dict.fromkeys(symbols)
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = dict.fromkeys(
        symbols, (start, end)
    )

    if cache is not None:
        plan = cache.plan(
            source=policy_model.FINRA,
            dataset=DATASET,
            entities=symbols,
            start=start_date,
            end=end_date,
        )
        for symbol in symbols:
            cached[symbol] = plan.cached_frame(symbol)
            span = plan.get_fetch_span(symbol)
            spans[symbol] = (
                (pd.Timestamp(span[0]), pd.Timestamp(span[1])) if span else None
            )

    days = pd.DatetimeIndex([])
    for span in spans.values():
        if span is not None:
            days = days.union(_trading_days(*span))

    fetched, failures = _download_days(
        days, [s for s in symbols if spans[s] is not None]
    )

    if failures:
        # A day that could not be retrieved is not recorded as answered, so it is asked
        # again next time rather than remembered as a day without short sales.
        logger.warning(
            "Could not retrieve the FINRA short sale volume of %d of %d days, so those "
            "days are left out.",
            failures,
            len(days),
        )

    result = {}

    for symbol in symbols:
        span = spans[symbol]
        data = fetched.get(symbol, pd.DataFrame())

        if span is not None and cache is not None and not failures:
            cache.store(
                source=policy_model.FINRA,
                dataset=DATASET,
                entity=symbol,
                data=data,
                start=span[0].strftime("%Y-%m-%d"),
                end=span[1].strftime("%Y-%m-%d"),
            )

        frames = [
            frame
            for frame in (cached[symbol], data)
            if frame is not None and not frame.empty
        ]
        combined = pd.concat(frames) if frames else pd.DataFrame()

        if not combined.empty:
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = combined.loc[
                (combined.index >= pd.Period(start, freq="D"))
                & (combined.index <= pd.Period(end, freq="D"))
            ]

        result[symbol] = combined

    return result


def get_short_volume_ratio_statistics(
    volumes: pd.DataFrame, window: int
) -> pd.DataFrame:
    """
    Adds the short volume ratio and its z-score to the daily volumes of one symbol.

    The short volume ratio is the short volume (which includes the short exempt volume)
    divided by the total volume reported to FINRA. Its z-score compares each day with the
    `window` days before it: (ratio − mean) / standard deviation (ddof=1), so a day is
    never part of its own comparison.

    Args:
        volumes (pd.DataFrame): The Short Volume and Total Volume (columns) per day (index).
        window (int): The number of earlier days the z-score compares with.

    Returns:
        pd.DataFrame: The volumes with the Short Volume Ratio and Short Volume Ratio Z-Score
        columns added.
    """
    ratio = volumes["Short Volume"] / volumes["Total Volume"].replace(0, np.nan)
    earlier = ratio.shift(1).rolling(window, min_periods=window)

    statistics = volumes.copy()
    statistics["Short Volume Ratio"] = ratio
    statistics["Short Volume Ratio Z-Score"] = (ratio - earlier.mean()) / earlier.std(
        ddof=1
    )

    return statistics
