"""Ticker Module"""

__docformat__ = "google"

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from financetoolkit.cache.cache_controller import Cache

# Company endpoints put the ticker on the columns or the index, so be explicit.
TICKER_ON_COLUMNS = "columns"
TICKER_ON_INDEX = "index"


def select_ticker(
    data: pd.DataFrame, ticker: str, ticker_axis: str
) -> pd.DataFrame | None:
    """
    Extract the part of a multi-ticker frame that belongs to one ticker.

    The slice keeps the ticker label in place, so that putting the pieces back
    together reproduces the original layout exactly rather than a flattened
    version of it.

    Args:
        data (pd.DataFrame): The frame covering several tickers.
        ticker (str): The ticker to extract.
        ticker_axis (str): Either TICKER_ON_COLUMNS or TICKER_ON_INDEX.

    Returns:
        pd.DataFrame | None: The ticker's slice, or None when it is not present.

    Raises:
        ValueError: If the ticker axis is not one of the two supported values.
    """
    if data is None or data.empty:
        return None

    if ticker_axis == TICKER_ON_COLUMNS:
        mask = data.columns.get_level_values(-1) == ticker
        selection = data.loc[:, mask]
    elif ticker_axis == TICKER_ON_INDEX:
        mask = data.index.get_level_values(0) == ticker
        selection = data.loc[mask]
    else:
        raise ValueError(
            f"Unsupported ticker axis ({ticker_axis}), expected "
            f"'{TICKER_ON_COLUMNS}' or '{TICKER_ON_INDEX}'."
        )

    return None if selection.empty else selection


def combine_tickers(
    frames: list[pd.DataFrame], tickers: list[str], ticker_axis: str
) -> pd.DataFrame:
    """
    Reassemble per-ticker slices into a single multi-ticker frame.

    The result is ordered by the requested ticker order rather than by whichever
    tickers happened to come from the cache, so a partially cached result is
    indistinguishable from one fetched in a single call.

    Args:
        frames (list[pd.DataFrame]): The per-ticker slices to combine.
        tickers (list[str]): The tickers in the order they were requested.
        ticker_axis (str): Either TICKER_ON_COLUMNS or TICKER_ON_INDEX.

    Returns:
        pd.DataFrame: The combined frame, empty when there is nothing to combine.

    Raises:
        ValueError: If the ticker axis is not one of the two supported values.
    """
    populated = [frame for frame in frames if frame is not None and not frame.empty]

    if not populated:
        return pd.DataFrame()

    if ticker_axis == TICKER_ON_COLUMNS:
        combined = pd.concat(populated, axis=1)
    elif ticker_axis == TICKER_ON_INDEX:
        combined = pd.concat(populated, axis=0)
    else:
        raise ValueError(
            f"Unsupported ticker axis ({ticker_axis}), expected "
            f"'{TICKER_ON_COLUMNS}' or '{TICKER_ON_INDEX}'."
        )

    return reorder_tickers(combined, tickers, ticker_axis)


def reorder_tickers(
    data: pd.DataFrame, tickers: list[str], ticker_axis: str
) -> pd.DataFrame:
    """
    Put the tickers of a combined frame back into the requested order.

    Args:
        data (pd.DataFrame): The combined frame.
        tickers (list[str]): The tickers in the order they were requested.
        ticker_axis (str): Either TICKER_ON_COLUMNS or TICKER_ON_INDEX.

    Returns:
        pd.DataFrame: The frame with its tickers ordered, or unchanged when the
            reordering does not apply cleanly to this layout.
    """
    if data.empty:
        return data

    # Reordering is presentational, so an unusual layout is left as it is.
    with contextlib.suppress(Exception):
        if ticker_axis == TICKER_ON_COLUMNS:
            present = [
                ticker
                for ticker in tickers
                if ticker in set(data.columns.get_level_values(-1))
            ]

            if data.columns.nlevels == 1:
                return data[present]

            return data.reindex(present, axis="columns", level=-1)

        present = [
            ticker
            for ticker in tickers
            if ticker in set(data.index.get_level_values(0))
        ]

        if data.index.nlevels == 1:
            return data.loc[present]

        return data.reindex(present, level=0)

    return data


def collect_per_ticker(
    cache: "Cache | None",
    source: str,
    dataset: str,
    tickers: list[str],
    ticker_axis: str,
    collector: Callable[[list[str]], Any],
    parameters: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Retrieve a per-ticker dataset, requesting only the tickers not already cached.

    The company endpoints return one frame covering every requested ticker, which
    historically meant that adding a single ticker re-requested all of them. The
    frame is therefore split per ticker on the way into the cache and reassembled
    on the way out, so a later call only pays for what it does not already have.

    Args:
        cache (Cache | None): The cache to use. None fetches everything, unchanged.
        source (str): The external data source, e.g. "fmp".
        dataset (str): The dataset name to cache under, e.g. "profile".
        tickers (list[str]): The tickers being requested.
        ticker_axis (str): Whether the ticker sits on the column or the index axis of
            the returned frame, see TICKER_ON_COLUMNS and TICKER_ON_INDEX.
        collector (Callable[[list[str]], Any]): Called with the tickers that are not
            cached. May return a frame, or a (frame, invalid_tickers) tuple.
        parameters (dict | None): Parameters that change the returned data, such as the
            period or date range the endpoint was queried for.

    Returns:
        tuple[pd.DataFrame, list[str]]: The combined frame and the tickers the source
            reported as invalid during this call.
    """
    cached_frames: list[pd.DataFrame] = []
    missing_tickers: list[str] = list(tickers)

    if cache is not None:
        missing_tickers = []

        for ticker in tickers:
            stored = cache.get(
                source=source, dataset=dataset, entity=ticker, parameters=parameters
            )

            if stored is None:
                missing_tickers.append(ticker)
            else:
                cached_frames.append(stored)

    if not missing_tickers:
        return combine_tickers(cached_frames, tickers, ticker_axis), []

    result = collector(missing_tickers)
    fetched, invalid_tickers = result if isinstance(result, tuple) else (result, [])

    if isinstance(fetched, pd.DataFrame) and not fetched.empty:
        if cache is not None:
            for ticker in missing_tickers:
                slice_for_ticker = select_ticker(fetched, ticker, ticker_axis)

                if slice_for_ticker is not None:
                    cache.set(
                        source=source,
                        dataset=dataset,
                        entity=ticker,
                        data=slice_for_ticker,
                        parameters=parameters,
                    )

        cached_frames.append(fetched)

    return combine_tickers(cached_frames, tickers, ticker_axis), invalid_tickers
