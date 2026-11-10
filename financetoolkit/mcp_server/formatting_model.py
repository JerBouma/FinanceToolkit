"""
Formatting Model: converts Finance Toolkit results to JSON for LLM consumption.

JSON is preferred over Markdown tables because keys are self-labelling,
nesting is explicit, and there is no parsing ambiguity.
"""

from __future__ import annotations

import functools
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit.economics.helpers import COUNTRY_CODES
from financetoolkit.economics.oecd_model import CODE_TO_COUNTRY
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

        stacked = dataframe.stack(level=ticker_level)
        stacked.index.names = [orig_name, ticker_name]
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


def _no_data(reason: str | None) -> str:
    """
    Builds the response for a call without data, with the reason when one is known.

    Args:
        reason (str | None): Why there is no data, e.g. a missing API key.

    Returns:
        str: The JSON response.
    """
    payload = {"error": "No data available."}

    if reason:
        payload["reason"] = reason

    return json.dumps(payload)


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


def _observed_period(index: pd.Index) -> str | None:
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
    observed = _observed_period(data.index)
    order = list(PERIOD_ORDER)

    if (
        period not in PERIOD_ORDER
        or observed is None
        or order.index(observed) >= order.index(period)
    ):
        return None

    index = (
        data.index.to_timestamp()
        if isinstance(data.index, pd.PeriodIndex)
        else data.index
    )
    converted = data.groupby(index.to_period(PERIOD_ORDER[period])).last()
    converted.index.name = data.index.name

    return converted


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


def format_result(
    dataset: dict | pd.Series | pd.DataFrame | int | float | str | None,
    notes: list[str] | None = None,
    reason: str | None = None,
    max_characters: int | None = None,
    sources: list[str] | None = None,
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

    Returns:
        A JSON string.  DataFrames become lists of flat records; dicts of
        DataFrames become JSON objects keyed by the original dict keys;
        scalars become ``{"value": <scalar>}``; missing data becomes
        ``{"error": "No data available."}``.
    """

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
        if isinstance(payload, dict):
            payload.update(extras)
            return payload
        return payload

    if dataset is None:
        return _no_data(reason)

    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame()

    if isinstance(dataset, pd.DataFrame):
        if dataset.empty or not int(dataset.notna().to_numpy().sum()):
            return _no_data(reason)
        if max_characters:
            records, cut_note = _fit_to_size(dataset, max_characters)
            if cut_note:
                notes = [*(notes or []), cut_note]
            return json.dumps(_inject(records), default=str)
        return json.dumps(_inject(_df_to_records(dataset)), default=str)

    if isinstance(dataset, dict):
        out: dict[str, Any] = {}
        tables = [
            value for value in dataset.values() if isinstance(value, pd.DataFrame)
        ]
        for key, value in dataset.items():
            if isinstance(value, pd.DataFrame):
                if not value.empty and int(value.notna().to_numpy().sum()):
                    if max_characters:
                        # The tables share the size a response may have.
                        out[str(key)], cut_note = _fit_to_size(
                            value, max_characters // max(1, len(tables))
                        )
                        if cut_note:
                            notes = [*(notes or []), f"{key}: {cut_note}"]
                    else:
                        out[str(key)] = _df_to_records(value)
                else:
                    out[str(key)] = []
            else:
                out[str(key)] = _clean(value)
        return json.dumps(_inject(out), default=str)

    if isinstance(dataset, int | float | str):
        return json.dumps(_inject({"value": _clean(dataset)}))

    # A plain list of values, such as the expiration dates of an option chain.
    if isinstance(dataset, list | tuple):
        if not dataset:
            return _no_data(reason)
        values = [_clean(value) for value in dataset]
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
        return json.dumps(_inject(values), default=str)

    logger.warning("Unexpected result type: %s. Returning raw string.", type(dataset))
    return json.dumps(_inject({"value": str(dataset)}))
