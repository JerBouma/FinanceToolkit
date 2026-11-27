"""
Formatting Model: converts Finance Toolkit results to JSON for LLM consumption.

JSON is preferred over Markdown tables because keys are self-labelling,
nesting is explicit, and there is no parsing ambiguity.
"""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit.economics.helpers import COUNTRY_CODES
from financetoolkit.economics.sources.oecd_model import CODE_TO_COUNTRY
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()


# Numbers in an answer are rounded to four decimals, keeping at least four significant
# digits for small values.
ANSWER_DECIMALS = 4
ANSWER_SIGNIFICANT_DIGITS = 4


def _clean(val: Any) -> int | float | str | bool | None:
    """Convert a single value to a JSON-serializable Python primitive."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, pd.Period):
        return str(val)
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, float | np.floating):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        if f == 0:
            return 0.0
        # Four decimals, but at least four significant digits, so a daily variance of
        # 0.000123 keeps its precision while larger numbers do not carry noise digits.
        decimals = max(
            ANSWER_DECIMALS,
            ANSWER_SIGNIFICANT_DIGITS - 1 - math.floor(math.log10(abs(f))),
        )
        return float(f"{round(f, decimals):.12g}")
    return val


def _clean_key(key: Any) -> str:
    """Convert a column or index label to a JSON-safe string key."""
    if isinstance(key, pd.Period):
        return str(key)
    if isinstance(key, pd.Timestamp):
        return key.strftime("%Y-%m-%d")
    return str(key)


def _idx_key(index: pd.Index) -> str:
    """Infer a sensible key name for a flat index."""
    if index.name:
        return str(index.name)
    if isinstance(index, pd.DatetimeIndex | pd.PeriodIndex):
        return "date"
    if _looks_like_tickers(index.unique()):
        return "ticker"
    if _are_countries(index.unique()):
        return "country"
    # Labels in words (Price, P-Value, Lower Tail Dependence) are what a row measures.
    if len(index) and all(isinstance(label, str) for label in index):
        return "metric"
    return "index"


# Column levels a table may have for one level to be stacked into its rows.
STACKABLE_COLUMN_LEVELS = 2


# The longest symbol counted as a ticker, with an exchange suffix (ASML.AS, BTC-USD).
MAX_SYMBOL_LENGTH = 12


@functools.cache
def country_names() -> frozenset[str]:
    """The country and region names the economic sources use."""

    return frozenset(
        {
            *COUNTRY_CODES.values(),
            *CODE_TO_COUNTRY.values(),
            "Euro Area",
            "European Union",
            "World",
        }
    )


def _are_countries(values: pd.Index) -> bool:
    """Whether every label is a country or region, as in an economic indicator."""
    return len(values) > 0 and all(str(value) in country_names() for value in values)


def _describe_level(values: pd.Index) -> str:
    """The key for an unnamed column level that becomes rows: country, ticker or series."""
    if _are_countries(values):
        return "country"

    return "ticker" if _looks_like_tickers(values) else "series"


def _looks_like_tickers(values: pd.Index) -> bool:
    """
    Whether every label is a ticker symbol (short and uppercase), 'Benchmark', 'Portfolio'
    or 'Cash' (the cash position of a fractional Kelly portfolio).
    """
    return len(values) > 0 and all(
        str(value) in ("Benchmark", "Portfolio", "Cash")
        or (
            len(str(value)) <= MAX_SYMBOL_LENGTH
            and str(value) == str(value).upper()
            and any(character.isalpha() for character in str(value))
        )
        for value in values
    )


def _df_to_records(dataframe: pd.DataFrame) -> list[dict]:
    """Convert a FinanceToolkit DataFrame to a flat list of JSON-serializable dicts.

    Three cases handled by DataFrame shape:

    1. MultiIndex *columns* ``(metric, ticker)`` — historical data.
       Stack the ticker level into the row index so each record is one
       ``(date, ticker)`` observation with all metrics as keys.

    2. MultiIndex *index* ``(ticker, metric)`` — multi-ticker financials.
       Each row is already a ``(ticker, metric)`` pair; period/date columns
       become keys inside each record.

    3. Flat index + flat columns — single-ticker financials, quotes, etc.
       The index value becomes one key (name inferred from dtype/content);
       columns become the remaining keys.
    """
    # Columns of three or more levels (such as a copula's ticker pair and ticker) would
    # leave tuple keys and mostly empty records once one level is stacked, so their
    # levels are joined into one name per column and the table is treated as flat.
    if (
        isinstance(dataframe.columns, pd.MultiIndex)
        and dataframe.columns.nlevels > STACKABLE_COLUMN_LEVELS
    ):
        dataframe = dataframe.copy()
        dataframe.columns = [
            " | ".join(str(part) for part in column if str(part))
            for column in dataframe.columns
        ]

    # Numbered columns with a name, such as the maturities of a bond calculation, carry
    # that name in their keys ("Years to Maturity 5" rather than "5.0").
    if (
        not isinstance(dataframe.columns, pd.MultiIndex)
        and dataframe.columns.name
        and len(dataframe.columns)
        and all(
            isinstance(label, int | float | np.number) and not isinstance(label, bool)
            for label in dataframe.columns
        )
    ):
        dataframe = dataframe.copy()
        dataframe.columns = [
            f"{dataframe.columns.name} {float(label):g}" for label in dataframe.columns
        ]

    # ── Case 1: MultiIndex columns (metric, ticker) ───────────────────────────
    if isinstance(dataframe.columns, pd.MultiIndex):
        orig_name = _idx_key(dataframe.index)
        # Usually (metric, ticker), but a factor model is (ticker, coefficient): the level
        # with the tickers is the one that becomes rows, so the metrics stay the keys.
        first = dataframe.columns.unique(level=0)
        last = dataframe.columns.unique(level=-1)
        if _are_countries(first) != _are_countries(last):
            ticker_level = 0 if _are_countries(first) else -1
        else:
            ticker_level = (
                0
                if _looks_like_tickers(first) and not _looks_like_tickers(last)
                else -1
            )
        ticker_name = dataframe.columns.names[ticker_level] or _describe_level(
            dataframe.columns.unique(level=ticker_level)
        )

        # Rows of two levels (a statistic per year) keep a key per level, and the stacked
        # level comes last; a name taken twice gets its position, as a record has one key
        # per name.
        row_names = (
            [
                str(name) if name else _idx_key(dataframe.index.get_level_values(i))
                for i, name in enumerate(dataframe.index.names)
            ]
            if isinstance(dataframe.index, pd.MultiIndex)
            else [orig_name]
        )
        row_names = [
            name if name not in row_names[:i] else f"{name}_{i}"
            for i, name in enumerate([*row_names, str(ticker_name)])
        ]
        stacked = dataframe.stack(level=ticker_level)
        stacked.index.names = row_names
        # When both levels hold the same names (an impulse response is shock by response,
        # both tickers), the keys say which level they are: "Shock MSFT".
        kept_name = dataframe.columns.names[-1 if ticker_level == 0 else 0]
        if kept_name and set(map(str, first)) == set(map(str, last)):
            stacked.columns = [f"{kept_name} {column}" for column in stacked.columns]

        flat = stacked.reset_index()
        return [
            {_clean_key(k): _clean(v) for k, v in rec.items()}
            for rec in flat.to_dict(orient="records")
        ]

    # ── Case 2: MultiIndex index (ticker, metric) — multi-ticker financials ───
    if isinstance(dataframe.index, pd.MultiIndex):
        level_names: list[str] = []
        for i, name in enumerate(dataframe.index.names):
            if name:
                level_names.append(str(name))
                continue
            vals = dataframe.index.get_level_values(i)
            last = i == len(dataframe.index.names) - 1
            if not last and _looks_like_tickers(vals.unique()):
                level_names.append("ticker")
            elif last:
                level_names.append("metric")
            else:
                level_names.append(f"level_{i}")

        named_last_level = bool(dataframe.index.names[-1])
        dataframe = dataframe.copy()
        dataframe.index.names = level_names
        flat = dataframe.reset_index()

        # One metric means the last column level repeats, so drop it to stay concise. A
        # level the result names itself (such as an option's expiration) is information,
        # even when it has one value, and is kept.
        last_level = level_names[-1] if level_names else None
        if last_level and not named_last_level and len(flat[last_level].unique()) == 1:
            flat = flat.drop(columns=[last_level])

        return [
            {_clean_key(k): _clean(v) for k, v in rec.items()}
            for rec in flat.to_dict(orient="records")
        ]

    # ── Case 3: Flat index, flat columns ──────────────────────────────────────
    dataframe = dataframe.copy()
    if dataframe.index.name:
        idx_name = str(dataframe.index.name)
    elif isinstance(dataframe.columns, pd.DatetimeIndex | pd.PeriodIndex):
        # Columns are time periods → rows are either tickers or metric names
        idx_name = (
            "ticker" if _looks_like_tickers(dataframe.index.unique()) else "metric"
        )
    else:
        idx_name = _idx_key(dataframe.index)
    dataframe.index.name = idx_name
    flat = dataframe.reset_index()
    return [
        {_clean_key(k): _clean(v) for k, v in rec.items()}
        for rec in flat.to_dict(orient="records")
    ]


def _no_data(
    reason: str | None,
    notes: list[str] | None = None,
    sources: list[str] | None = None,
) -> str:
    """
    Builds the response for a call without data, with the reason when one is known. The
    notes (an ignored parameter, a ticker left out) and the sources are kept, since they
    can explain the empty answer as well.

    Args:
        reason (str | None): Why there is no data, e.g. a missing API key.
        notes (list[str] | None): The notes of the answer, under "_notes".
        sources (list[str] | None): The publishers asked for the data, under "_sources".

    Returns:
        str: The JSON response.
    """
    payload: dict[str, Any] = {"error": "No data available."}

    if reason:
        payload["reason"] = reason
    if notes:
        payload["_notes"] = notes
    if sources:
        payload["_sources"] = sources

    return _dumps(payload)


# The number of rows a cut answer keeps at least, before columns are cut as well.
MINIMUM_ROWS = 5
# The fewest characters a value takes in the records ("1," or "null,"), which bounds how
# many values an answer can show.
MINIMUM_VALUE_CHARACTERS = 4

# Texts (such as an earnings call transcript) are shortened to no fewer characters than
# this before columns are cut, since a column without its text answers nothing.
MINIMUM_TEXT_CHARACTERS = 500

NARROWING_HINT = (
    "Narrow it with start_date/end_date, a longer period (e.g. period='monthly'), "
    "show_columns, fewer tickers or countries, or limit= where the indicator has it."
)


def _is_time_index(index: pd.Index) -> bool:
    """Whether the rows are dates, whose most recent ones a cut answer keeps."""
    first_level = index.levels[0] if isinstance(index, pd.MultiIndex) else index

    return isinstance(first_level, pd.DatetimeIndex | pd.PeriodIndex)


# The share of rows a trailing period needs a value for to count as the latest one.
MINIMUM_PERIOD_COVERAGE = 0.5


def select_rows(data: pd.DataFrame | pd.Series, limit: int) -> pd.DataFrame | pd.Series:
    """
    The first (positive limit) or last (negative limit) rows, like head and tail. A table
    with its periods as columns, such as a ratio per ticker and year, is a time series read
    sideways, so its first or last periods are selected instead; the last periods leave
    out trailing ones that most rows have no value for yet. Rows of a ticker and a date
    are selected per ticker.

    Args:
        data (pd.DataFrame | pd.Series): The result.
        limit (int): The number of rows or periods, negative for the last ones.

    Returns:
        pd.DataFrame | pd.Series: The selection.
    """
    if isinstance(data, pd.DataFrame) and has_period_columns(data):
        if limit > 0:
            return data.iloc[:, :limit]

        # The latest periods that most rows have (a year most companies have not reported
        # yet is not the latest one to show), unless that would leave nothing.
        reported = len(data.columns)
        coverage = data.notna().mean()
        while (
            reported > 1
            and len(data) > 1
            and coverage.iloc[reported - 1] < MINIMUM_PERIOD_COVERAGE
        ):
            reported -= 1

        return data.iloc[:, :reported].iloc[:, limit:]

    # The last (or first) rows with data: a yearly series often ends before the window
    # does, and its empty last years are not the most recent figures anyone asks for.
    if _is_time_index(data.index):
        # Rows of a date and an item (a correlation matrix per year) are selected by
        # whole periods, so a period keeps every one of its rows.
        if isinstance(data.index, pd.MultiIndex):
            periods = data.dropna(how="all").index.get_level_values(0).unique()
            chosen = periods[:limit] if limit > 0 else periods[limit:]
            return data[data.index.get_level_values(0).isin(chosen)]

        first, last = data.first_valid_index(), data.last_valid_index()
        if first is not None and last is not None:
            data = data.loc[first:last]

    # Rows of a ticker and a date (an earnings calendar of several tickers) are selected
    # per ticker: a plain tail would return the last ticker's rows only.
    elif isinstance(data.index, pd.MultiIndex) and any(
        isinstance(level, pd.DatetimeIndex | pd.PeriodIndex)
        for level in data.index.levels[1:]
    ):
        rows = data.groupby(level=0, sort=False)
        return rows.head(limit) if limit > 0 else rows.tail(-limit)

    return data.head(limit) if limit > 0 else data.tail(-limit)


def signal_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    A row per ticker and period with a signal, from a table with a column per signal and
    ticker (get_signals, get_candlestick_patterns). Those keep a period when any ticker has a
    signal, so per ticker most rows would be zeros; as rows of a ticker and a date, a limit
    selects the last signals of every ticker.

    Args:
        data (pd.DataFrame): The signals, with the signal and the ticker as column levels.

    Returns:
        pd.DataFrame: The signals of each ticker on the periods it has one, with the ticker
        and the date as row levels. A ticker without any signal is left out.
    """
    per_ticker = {}
    for ticker in dict.fromkeys(data.columns.get_level_values(1)):
        signals = data.xs(ticker, axis=1, level=1)
        signals = signals[signals.abs().gt(0).any(axis=1)]
        if not signals.empty:
            per_ticker[ticker] = signals

    if not per_ticker:
        return data.iloc[:0]

    return pd.concat(per_ticker, names=["ticker", data.index.name or "Date"])


def is_time_index_of(data: pd.DataFrame | pd.Series) -> bool:
    """Whether the rows of a result are dates."""
    return _is_time_index(data.index)


def is_yearly(index: pd.Index) -> bool:
    """Whether the rows (the first level of them) are years."""
    first_level = index.levels[0] if isinstance(index, pd.MultiIndex) else index

    return isinstance(first_level, pd.PeriodIndex) and str(
        first_level.freqstr
    ).startswith(("Y", "A"))


# The skipped periods a note names per series before it counts the rest.
# The periods an answer can be converted to, coarsest last, with their pandas frequency.
PERIOD_ORDER = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}

# The longest median spacing in days of an observed frequency, to recognise a date index.
PERIOD_SPACING_DAYS = {"daily": 4, "weekly": 10, "monthly": 45, "quarterly": 135}


def observed_period(index: pd.Index) -> str | None:
    """The frequency of a date or period index, or None when it is neither."""
    if isinstance(index, pd.PeriodIndex):
        return next(
            (
                period
                for period, code in PERIOD_ORDER.items()
                if index.freqstr[0] == code[0]
            ),
            None,
        )

    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:  # noqa: PLR2004
        return None

    spacing = pd.Series(index.sort_values()).diff().dt.days.median()

    return next(
        (period for period, days in PERIOD_SPACING_DAYS.items() if spacing <= days),
        "yearly",
    )


def convert_to_period(
    data: pd.DataFrame | pd.Series, period: str
) -> pd.DataFrame | pd.Series | None:
    """
    Converts a time series to a coarser period, taking the last value of every period (the
    end-of-period convention), for an indicator that has no period parameter of its own.

    Args:
        data (pd.DataFrame | pd.Series): The answer, indexed by date or period.
        period (str): "weekly", "monthly", "quarterly" or "yearly" ("annual" works too).

    Returns:
        pd.DataFrame | pd.Series | None: The converted series, or None when the answer is
        not a time series or is not finer than the period.
    """
    period = "yearly" if period.lower() == "annual" else period.lower()
    observed = observed_period(data.index)
    order = list(PERIOD_ORDER)

    if (
        period not in PERIOD_ORDER
        or observed is None
        or order.index(observed) >= order.index(period)
    ):
        return None

    # A period belongs to the period its last weekday falls in: a week to Friday 2 February
    # is February's first week, though it starts in January, and a week to Sunday 1 February
    # (pandas' W-SUN weeks) closes on Friday 30 January, so it is January's last week.
    index = data.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp(how="end").normalize()
        if data.index.freqstr.startswith("W"):
            index = index - pd.to_timedelta(
                np.maximum(index.dayofweek - 4, 0), unit="D"
            )
    converted = data.groupby(index.to_period(PERIOD_ORDER[period])).last()
    converted.index.name = data.index.name

    return converted


# The calendar periods an answer can be sampled to, with their pandas frequency. A week runs
# from Monday to Sunday, so a market that trades at the weekend keeps its Sunday.
SAMPLE_PERIODS = {"weekly": "W-SUN", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
SAMPLE_PERIOD_NAMES = {
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
}


def sample_rows(
    data: pd.DataFrame | pd.Series, sample: str
) -> tuple[pd.DataFrame | pd.Series, str]:
    """
    Keeps the last row with a value in each calendar week, month, quarter or year of an
    answer finer than that (daily, intraday, weekly, monthly or quarterly), including the
    latest period that has not ended yet. The rows keep
    their own dates, so a weekly sample of a 200-day moving average is that average on the
    last trading day of every week. A table with its dates as columns is sampled by column,
    and rows of a date and an item keep every row of the chosen dates.

    Args:
        data (pd.DataFrame | pd.Series): The answer.
        sample (str): "weekly", "monthly", "quarterly" or "yearly".

    Returns:
        tuple[pd.DataFrame | pd.Series, str]: The sampled answer and a note on what was done,
        or the answer unchanged and a note on why, when it has no dates or is not finer than
        the sample.
    """
    by_columns = isinstance(data, pd.DataFrame) and has_period_columns(data)
    table = data.T if by_columns else data
    axis = "columns" if by_columns else "rows"

    if not _is_time_index(table.index):
        return data, (
            f"sample={sample!r} was not applied: the answer has no dates as its rows or columns."
        )

    dates = table.index.get_level_values(0)
    observed = observed_period(dates.unique()) or (
        observed_period(dates.unique().to_timestamp())
        if isinstance(dates, pd.PeriodIndex)
        else None
    )
    order = list(PERIOD_ORDER)
    if observed is None or order.index(observed) >= order.index(sample):
        return data, (
            f"sample={sample!r} was not applied: the answer is "
            f"{observed or 'a single date'} already, as coarse as {sample} or coarser."
        )

    # A week, month or quarter belongs to the period it ends in, as in convert_to_period.
    timestamps = (
        dates.to_timestamp(how="end").normalize()
        if isinstance(dates, pd.PeriodIndex)
        else dates.tz_localize(None)
    )
    periods = timestamps.to_period(SAMPLE_PERIODS[sample])
    # The last date with a value per period; a row without any value is not a reading.
    has_value = (
        table.notna().any(axis=1) if isinstance(table, pd.DataFrame) else table.notna()
    ).to_numpy()
    if not has_value.any():
        return data, f"sample={sample!r} was not applied: the answer has no values."
    last_dates = pd.Series(timestamps[has_value]).groupby(periods[has_value]).max()
    sampled = table[timestamps.isin(last_dates)]
    before, after = len(dates.unique()), len(last_dates)

    return (sampled.T if by_columns else sampled), (
        f"Sampled to the last {axis[:-1]} of each {SAMPLE_PERIOD_NAMES[sample]} ({after} of {before} {axis})."
    )


SKIPPED_PERIODS_SHOWN = 3


def skipped_periods_note(data: pd.DataFrame | pd.Series) -> str | None:
    """
    Names the months, quarters or years a series has no value for between its first and
    last value, such as a month a statistics office did not publish. They are empty rows in
    the answer, so the rows around them are not consecutive periods with data and a change
    over them is empty too.

    Args:
        data (pd.DataFrame | pd.Series): The result, with periods as its index.

    Returns:
        str | None: The note, or None when no series skips a period.
    """
    if not isinstance(data.index, pd.PeriodIndex) or data.index.freqstr[0] not in "MQY":
        return None

    frame = data.to_frame() if isinstance(data, pd.Series) else data
    gaps = []
    for column in frame.columns:
        values = frame[column]
        first, last = values.first_valid_index(), values.last_valid_index()
        if first is None:
            continue
        skipped = values.loc[first:last]
        skipped = [str(period) for period in skipped[skipped.isna()].index]
        if skipped:
            shown = ", ".join(skipped[:SKIPPED_PERIODS_SHOWN]) + (
                f" and {len(skipped) - SKIPPED_PERIODS_SHOWN} more"
                if len(skipped) > SKIPPED_PERIODS_SHOWN
                else ""
            )
            gaps.append(f"{column} ({shown})" if frame.shape[1] > 1 else shown)

    if not gaps:
        return None

    return (
        "No value for "
        + "; ".join(gaps[:5])
        + ": the source skipped them, so they are "
        "empty rows (the rows around them are not consecutive periods) and a change over "
        "them is empty too."
    )


def has_period_columns(dataframe: pd.DataFrame) -> bool:
    """Whether the periods are the columns (and not the rows), as with ratios."""
    return not _is_time_index(dataframe.index) and _is_time_index(dataframe.columns)


# The pandas frequency of a base window date by its length: 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'.
BASE_DATE_FREQUENCIES = {4: "Y", 7: "M", 10: "D"}

# The columns a rebase note names before it counts the rest.
MAXIMUM_NAMED_COLUMNS = 10


def parse_base_window(text: str) -> tuple[pd.Period, pd.Period]:
    """
    The first and last day of a base window: 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD' for one
    year, month or day, 'START:END' for a range in those forms, or 'YYYY-YYYY' for a range
    of years. A second part of four digits is a year ('1340-1347'), one of two digits a
    month ('2020-03'), so the two never mix up.

    The days are daily periods rather than timestamps, which end in 1677 and cannot hold
    the medieval years of a long history.

    Args:
        text (str): The base window.

    Returns:
        tuple[pd.Period, pd.Period]: The first and the last day of the window.
    """
    text = str(text).strip()
    year_range = re.fullmatch(r"(\d{4})-(\d{4})", text)
    if year_range:
        parts = list(year_range.groups())
    elif ":" in text:
        parts = [part.strip() for part in text.split(":")]
    else:
        parts = [text, text]

    usage = (
        f"rebase_to={text!r} is not a date or a range: use 'YYYY', 'YYYY-MM' or "
        "'YYYY-MM-DD', a range 'START:END' in those forms (e.g. '2020-01:2020-06'), or "
        "'YYYY-YYYY' for a range of years (e.g. '1340-1347')."
    )
    if len(parts) != 2 or not all(  # noqa: PLR2004
        re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", part) for part in parts
    ):
        raise ValueError(usage)
    try:
        start, end = (
            pd.Period(part, freq=BASE_DATE_FREQUENCIES[len(part)]).asfreq("D", how=how)
            for part, how in zip(parts, ("start", "end"), strict=True)
        )
    except ValueError as exc:  # a month 13 or a day 32
        raise ValueError(usage) from exc
    if start.ordinal > end.ordinal:  # both daily periods
        raise ValueError(
            f"rebase_to={text!r} ends before it starts; give the earlier date first."
        )

    return start, end


def _row_days(index: pd.Index) -> tuple[pd.PeriodIndex, pd.PeriodIndex]:
    """The first and last day of every row of a date or period index, as daily periods."""
    if isinstance(index, pd.PeriodIndex):
        return index.asfreq("D", how="start"), index.asfreq("D", how="end")
    if index.tz is not None:
        index = index.tz_localize(None)
    days = index.to_period("D")

    return days, days


def _name_columns(columns: list) -> str:
    """The column labels for a note, a tuple label joined by spaces, the rest counted."""
    names = [
        " ".join(map(str, column)) if isinstance(column, tuple) else str(column)
        for column in columns
    ]
    if len(names) > MAXIMUM_NAMED_COLUMNS:
        return (
            ", ".join(names[:MAXIMUM_NAMED_COLUMNS])
            + f" and {len(names) - MAXIMUM_NAMED_COLUMNS:,} more"
        )

    return ", ".join(names)


def rebase_result(
    result: Any, window: tuple[pd.Period, pd.Period], text: str
) -> tuple[Any, list[str]]:
    """
    Indexes every column of a time series to 100 at its average over a base window: each
    value divided by the column's mean over the rows of the window, times 100. A row
    belongs to the window when its date, or any day of its period (a year, a quarter),
    lies in it, so a yearly series rebased to '2020-03' takes the year 2020.

    A table with its periods as columns (a ratio per ticker and year) is turned around, so
    each row is indexed to its own base. A column without a value in the window, or with
    an average of zero there, has no base and becomes empty, which a note says; text
    columns are kept as they are. Anything without dates or periods as its rows or
    columns (a list, several tables, a row per ticker and date) is returned unchanged
    with a note, rather than indexed along a guessed direction.

    Args:
        result (Any): The answer of the indicator.
        window (tuple[pd.Period, pd.Period]): The first and last day of the base window,
            from parse_base_window.
        text (str): The base window as given, for the notes.

    Returns:
        tuple[Any, list[str]]: The rebased result and the notes on it.

    Raises:
        ValueError: When no row falls in the window, saying how to widen the dates.
    """
    was_series = isinstance(result, pd.Series) and not isinstance(
        result.index, pd.MultiIndex
    )
    series_name = result.name if was_series else None
    if was_series:
        result = result.to_frame()
    if not isinstance(result, pd.DataFrame) or (
        not result.empty
        and not (
            isinstance(result.index, pd.DatetimeIndex | pd.PeriodIndex)
            or (
                has_period_columns(result)
                and isinstance(result.columns, pd.DatetimeIndex | pd.PeriodIndex)
            )
        )
    ):
        return result, [
            f"rebase_to={text!r} was ignored: the answer has no single row (or column) of "
            "dates or periods per value to index to."
        ]
    if result.empty:
        return result, []

    transposed = not isinstance(result.index, pd.DatetimeIndex | pd.PeriodIndex)
    table = result.T.copy() if transposed else result.copy()
    first_days, last_days = _row_days(table.index)
    start, end = window
    in_window = (first_days <= end) & (last_days >= start)
    if not in_window.any():
        first, last = table.index.min(), table.index.max()
        if last_days.max() < start:
            hint = f"the answer ends at {last}; set end_date on or after {start} to include it."
        elif first_days.min() > end:
            hint = (
                f"the answer starts at {first}; set start_date on or before {start} to "
                "include it (where the data goes back that far)."
            )
        else:
            hint = (
                f"the answer runs from {first} to {last} without a row in between; choose a "
                "window that holds at least one of its periods."
            )
        raise ValueError(
            f"rebase_to={text!r} ({start} to {end}) matches no row of the answer: {hint}"
        )

    numeric = [
        position
        for position in range(table.shape[1])
        if pd.api.types.is_numeric_dtype(table.iloc[:, position])
        and not pd.api.types.is_bool_dtype(table.iloc[:, position])
    ]
    values = table.iloc[:, numeric].astype(float)
    base = values[in_window].mean()
    missing = base.isna()
    # A negative base would flip the sign of every value and a zero one has no ratio, so
    # neither is indexed.
    not_positive = base <= 0
    rebased = values / base.where(~missing & ~not_positive) * 100
    for column, position in enumerate(numeric):
        table.isetitem(position, rebased.iloc[:, column])

    periods = int(in_window.sum())
    notes = [
        f"Every {'row' if transposed else 'column'} is indexed to 100 = its average over "
        f"{start} to {end} ({periods} {'period' if periods == 1 else 'periods'}), "
        "rebase_to=" + repr(text) + "."
    ]
    labels = list(values.columns)
    if missing.any():
        notes.append(
            f"{_name_columns([labels[i] for i in range(len(labels)) if missing.iloc[i]])} "
            f"{'has' if missing.sum() == 1 else 'have'} no value in the base window of "
            f"rebase_to={text!r}, so {'it is' if missing.sum() == 1 else 'they are'} empty."
        )
    if not_positive.any():
        one = not_positive.sum() == 1
        notes.append(
            f"{_name_columns([labels[i] for i in range(len(labels)) if not_positive.iloc[i]])} "
            f"{'has' if one else 'have'} an average over the base window of "
            f"rebase_to={text!r} that is not positive, so {'it' if one else 'they'} can't be "
            f"indexed to 100 and {'is' if one else 'are'} empty."
        )
    if len(numeric) < table.shape[1]:
        text_columns = [
            table.columns[position]
            for position in range(table.shape[1])
            if position not in numeric
        ]
        notes.append(
            f"{_name_columns(text_columns)} {'is' if len(text_columns) == 1 else 'are'} "
            "not numeric and kept as published."
        )

    table = table.T if transposed else table
    if was_series:
        return table.iloc[:, 0].rename(series_name), notes

    return table, notes


def _fit_to_size(
    dataframe: pd.DataFrame, max_characters: int
) -> tuple[list[dict], str | None]:
    """
    Cuts a table until its records fit the size a response may have: the most recent rows
    of a time series (the first rows otherwise), then long texts to their first characters,
    and the first columns when even a few rows are too wide.

    Args:
        dataframe (pd.DataFrame): The table.
        max_characters (int): The size the records may have as JSON.

    Returns:
        tuple[list[dict], str | None]: The records, and a note on what was cut, if anything.
    """
    rows, columns = dataframe.shape
    from_end = _is_time_index(dataframe.index)
    # Periods as columns (a ratio per ticker and year) keep their most recent ones.
    latest_columns = has_period_columns(dataframe)
    # A value takes a few characters at the least, so rows and columns beyond what could
    # fit are cut before the records are built: for a table of millions of rows the
    # records alone would take more memory than the server has.
    kept_columns = min(columns, max(1, max_characters // MINIMUM_VALUE_CHARACTERS))
    kept_rows = min(
        rows,
        max(
            MINIMUM_ROWS,
            max_characters // (MINIMUM_VALUE_CHARACTERS * (kept_columns + 1)),
        ),
    )
    text_characters: int | None = None

    def cut(part: pd.DataFrame) -> pd.DataFrame:
        part = part.iloc[-kept_rows:] if from_end else part.iloc[:kept_rows]

        return (
            part.iloc[:, -kept_columns:]
            if latest_columns
            else part.iloc[:, :kept_columns]
        )

    records = _df_to_records(cut(dataframe))
    size = len(json.dumps(records, default=str))

    if size <= max_characters and (kept_rows, kept_columns) == (rows, columns):
        return records, None

    def longest_text(part: pd.DataFrame) -> int:
        texts = part.select_dtypes(include=["object", "string"])

        return (
            int(
                texts.map(lambda value: len(value) if isinstance(value, str) else 0)
                .max()
                .max()
            )
            if not texts.empty
            else 0
        )  # noqa: E501

    while size > max_characters and (kept_rows > MINIMUM_ROWS or kept_columns > 1):
        if kept_rows > MINIMUM_ROWS:
            # Proportionally fewer rows, with a margin, and at least halving the excess.
            kept_rows = max(
                MINIMUM_ROWS,
                min(kept_rows // 2 + 1, int(kept_rows * max_characters / size * 0.9)),
            )
        elif (
            longest := text_characters or longest_text(dataframe)
        ) > MINIMUM_TEXT_CHARACTERS:
            text_characters = max(
                MINIMUM_TEXT_CHARACTERS,
                min(longest // 2 + 1, int(longest * max_characters / size * 0.9)),
            )
        else:
            kept_columns = max(1, kept_columns // 2)

        part = cut(dataframe)

        if text_characters is not None:
            part = part.map(
                lambda value: (
                    value[:text_characters] + "..."
                    if isinstance(value, str) and len(value) > text_characters
                    else value
                )
            )

        records = _df_to_records(part)
        size = len(json.dumps(records, default=str))

    shown = (
        f"the {'last' if from_end else 'first'} {kept_rows:,} of {rows:,} rows"
        if kept_rows < rows
        else "every row"
    )

    if text_characters is not None:
        shown += f", texts shortened to their first {text_characters:,} characters"

    if kept_columns < columns:
        shown += (
            f" and the {'last' if latest_columns else 'first'} {kept_columns:,} of "
            f"{columns:,} {'periods' if latest_columns else 'columns'}"
        )

    return (
        records,
        f"The answer was too large and was cut: it shows {shown}. {NARROWING_HINT}",
    )


# The most filled dates a note names per ticker.
MAXIMUM_FILLED_DATES = 5


def filled_day_notes(result: Any) -> list[str]:
    """
    Names the days a ticker did not trade, such as an exchange holiday, in daily prices that
    carry its last price over them (fill_nan). Such a day repeats the last close with a
    Volume of 0 and no Return, which reads as a flat day on a chart.

    Args:
        result (Any): The answer of a tool call, prices with (column, ticker) columns.

    Returns:
        list[str]: A note per ticker with filled days, empty for any other answer.
    """
    if not (
        isinstance(result, pd.DataFrame)
        and result.columns.nlevels == 2  # noqa: PLR2004
        and {"Close", "Volume", "Return"} <= set(result.columns.get_level_values(0))
    ):
        return []

    notes = []

    for ticker in result["Close"].columns:
        close = result["Close"][ticker]
        filled = (
            close.notna()
            & close.eq(close.shift())
            & result["Volume"][ticker].eq(0)
            & result["Return"][ticker].isna()
        )
        dates = [str(date) for date in result.index[filled.to_numpy()]]

        if dates:
            shown = ", ".join(dates[:MAXIMUM_FILLED_DATES])
            more = (
                f" and {len(dates) - MAXIMUM_FILLED_DATES} more"
                if len(dates) > MAXIMUM_FILLED_DATES
                else ""
            )
            notes.append(
                f"{ticker} did not trade on {shown}{more} (a holiday on its exchange, "
                "while another ticker or the benchmark traded): those rows repeat the last "
                "close with a Volume of 0 and no Return, so they do not count in returns or "
                "volatility. fill_nan=false leaves them empty."
            )

    return notes


def _columnar(records: list[dict]) -> dict[str, list]:
    """A table as its column names once and a list of values per row, rather than records
    that repeat every column name on every row, which about halves an answer.

    Args:
        records (list[dict]): The table as records.

    Returns:
        dict[str, list]: The ``columns`` and the ``rows``, each row's values in the order of
            the columns.
    """
    columns = list(dict.fromkeys(key for record in records for key in record))

    return {
        "columns": columns,
        "rows": [[record.get(column) for column in columns] for record in records],
    }


def _dumps(payload: Any) -> str:
    """The answer as JSON without the spaces after separators."""
    return json.dumps(payload, default=str, separators=(",", ":"))


def format_result(
    dataset: dict | pd.Series | pd.DataFrame | int | float | str | None,
    notes: list[str] | None = None,
    reason: str | None = None,
    max_characters: int | None = None,
    sources: list[str] | None = None,
    outcome: list[str] | None = None,
) -> str:
    """Format a Finance Toolkit result as a compact JSON string for LLM consumption.

    Args:
        dataset: The value returned by a controller ``get_*`` method.
        notes: Optional list of strings describing data transformations applied
            (e.g. currency conversion, fiscal-year relabelling). When provided
            they are included under a ``"_notes"`` key in the top-level object.
        reason: Optional explanation of why there is no data, such as a missing API
            key or an unreachable source, included under a ``"reason"`` key when the
            dataset turns out to be empty.
        max_characters: Optional size a response may have. A larger table is cut to the
            most recent rows (or the first columns) with a note on how to narrow it.
        sources: Optional publishers the data comes from, such as FinancialModelingPrep or
            the Bank for International Settlements, included under a ``"_sources"`` key.
        outcome: Optional list the kind of answer is appended to, for the usage statistics:
            "empty" for an answer without data, "cut" for one cut to max_characters and
            "data" for any other.

    Returns:
        A JSON string.  DataFrames become ``{"columns": [...], "rows": [[...], ...]}``
        (the ``_notes`` and ``_sources`` next to them); dicts of DataFrames become JSON
        objects keyed by the original dict keys, a table of columns and rows each;
        scalars become ``{"value": <scalar>}``; missing data becomes
        ``{"error": "No data available."}``.
    """

    def _answered(state: str, text: str) -> str:
        if outcome is not None:
            outcome.append(state)
        return text

    def _inject(payload: Any) -> Any:
        """Wrap payload in a dict with ``_notes`` and ``_sources`` when present."""
        extras = {
            key: value
            for key, value in (("_notes", notes), ("_sources", sources))
            if value
        }
        if not extras:
            return payload
        if isinstance(payload, list):
            return {"data": payload, **extras}
        # A table's columns and rows sit next to the notes and sources.
        if isinstance(payload, dict):
            payload.update(extras)
            return payload
        return payload

    if dataset is None:
        return _answered("empty", _no_data(reason, notes, sources))

    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame()

    if isinstance(dataset, pd.DataFrame):
        if dataset.empty or not int(dataset.notna().to_numpy().sum()):
            return _answered("empty", _no_data(reason, notes, sources))
        # The size a response may have is that of the table as records, so a cut keeps
        # the same rows whatever form the answer is sent in.
        if max_characters:
            records, cut_note = _fit_to_size(dataset, max_characters)
            if cut_note:
                notes = [*(notes or []), cut_note]
            return _answered(
                "cut" if cut_note else "data", _dumps(_inject(_columnar(records)))
            )
        return _answered("data", _dumps(_inject(_columnar(_df_to_records(dataset)))))

    if isinstance(dataset, dict):
        out: dict[str, Any] = {}
        tables = [
            value for value in dataset.values() if isinstance(value, pd.DataFrame)
        ]
        state = "data"
        for key, value in dataset.items():
            if isinstance(value, pd.DataFrame):
                if not value.empty and int(value.notna().to_numpy().sum()):
                    if max_characters:
                        # The tables share the size a response may have.
                        records, cut_note = _fit_to_size(
                            value, max_characters // max(1, len(tables))
                        )
                        out[str(key)] = _columnar(records)
                        if cut_note:
                            notes = [*(notes or []), f"{key}: {cut_note}"]
                            state = "cut"
                    else:
                        out[str(key)] = _columnar(_df_to_records(value))
                else:
                    out[str(key)] = []
            else:
                out[str(key)] = _clean(value)
        # Tables only, none of them with a value: a model whose every part is empty.
        if tables and len(tables) == len(dataset) and not any(out.values()):
            state = "empty"
        return _answered(state, _dumps(_inject(out)))

    if isinstance(dataset, int | float | str):
        return _answered("data", _dumps(_inject({"value": _clean(dataset)})))

    # A plain list of values, such as the expiration dates of an option chain.
    if isinstance(dataset, list | tuple):
        if not dataset:
            return _answered("empty", _no_data(reason, notes, sources))
        values = [_clean(value) for value in dataset]
        state = "data"
        # A list has no rows to cut by date, so it keeps its first values that fit.
        if max_characters and len(json.dumps(values, default=str)) > max_characters:
            kept = len(values)
            while (
                kept > 1
                and len(json.dumps(values[:kept], default=str)) > max_characters
            ):
                kept //= 2
            notes = [
                *(notes or []),
                f"The answer was cut to its first {kept} of {len(values)} values to fit the "
                "response size; use limit to choose which values are returned.",
            ]
            values = values[:kept]
            state = "cut"
        return _answered(state, _dumps(_inject(values)))

    logger.warning("Unexpected result type: %s. Returning raw string.", type(dataset))
    return _answered("data", _dumps(_inject({"value": str(dataset)})))
