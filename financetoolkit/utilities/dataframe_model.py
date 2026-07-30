"""Dataframe Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def combine_dataframes(dataset_dictionary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine the dataframes from different companies of the same financial statement,
    e.g. the balance sheet statement, into a single dataframe.

    Args:
        dataset_dictionary (dict[str, pd.DataFrame]): A dictionary containing the
        dataframes for each company. It should have the structure key: ticker,
        value: dataframe.

    Returns:
        pd.DataFrame: A pandas DataFrame with the combined financial statements.
    """
    combined_df = pd.concat(dict(dataset_dictionary), axis=0)

    return combined_df.sort_index(level=0, sort_remaining=False)


def equal_length(dataset1: pd.Series, dataset2: pd.Series) -> pd.Series:
    """
    Equalize the length of two datasets by adding zeros to the beginning of the shorter dataset.

    Args:
        dataset1 (pd.Series): The first dataset to be equalized.
        dataset2 (pd.Series): The second dataset to be equalized.

    Returns:
        pd.Series, pd.Series: The equalized datasets.
    """
    if int(dataset1.columns[0]) > int(dataset2.columns[0]):
        for value in range(
            int(dataset1.columns[0]) - 1, int(dataset2.columns[0]) - 1, -1
        ):
            dataset1.insert(0, value, 0.0)
        dataset1 = dataset1.sort_index()
    elif int(dataset1.columns[0]) < int(dataset2.columns[0]):
        for value in range(
            int(dataset2.columns[0]) - 1, int(dataset1.columns[0]) - 1, -1
        ):
            dataset2.insert(0, value, 0.0)
        dataset2 = dataset2.sort_index()

    return dataset1, dataset2


def filter_columns(
    result: pd.DataFrame | pd.Series | dict | object,
    show_columns: list[str] | None,
) -> pd.DataFrame | pd.Series | dict | object:
    """Filter a Finance Toolkit result to only include the specified columns.

    Works on pd.DataFrame, dicts of pd.DataFrame (multi-ticker financial
    statements), and passes through pd.Series, scalars, and any other type
    unchanged.  When *show_columns* is None the result is returned unmodified.

    Args:
        result: The value returned by a controller ``get_*`` method.
        show_columns: Column names to keep.  For MultiIndex DataFrames the
            first index level is used for matching.  Invalid names are logged
            as warnings; if none of the requested columns exist the original
            result is returned unchanged.

    Returns:
        The filtered result, or *result* unchanged when filtering cannot be
        applied or *show_columns* is None.
    """
    if show_columns is None:
        return result

    if isinstance(result, pd.DataFrame):
        return _filter_dataframe_columns(result, show_columns)

    if isinstance(result, dict):
        return {
            key: (
                _filter_dataframe_columns(value, show_columns)
                if isinstance(value, pd.DataFrame)
                else value
            )
            for key, value in result.items()
        }

    return result


def _filter_dataframe_columns(
    df: pd.DataFrame,
    show_columns: list[str],
) -> pd.DataFrame:
    """Internal helper: filter a single DataFrame to *show_columns*.

    Resolution order:
        1. MultiIndex *columns* — filter by first column level (e.g. OHLCV type in
        historical data where columns are ``(metric, ticker)``).
        2. Flat *columns* — filter columns whose string representation appears in
        *show_columns*.
        3. MultiIndex *index* (fallback) — filter by the last index level (e.g.
        financial-statement line items in multi-ticker data where the row index
        is ``(ticker, line_item)``).
        4. Flat *index* (fallback) — filter by the index values whose string
        representation appears in *show_columns* (e.g. single-ticker income
        statement where rows are individual line items).

    If none of the above yield any matches the original DataFrame is returned
    unchanged and a warning is logged.
    """
    if df.empty:
        return df

    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        available = [str(c) for c in df.columns.get_level_values(0).unique()]
        valid = [c for c in show_columns if c in available]
        invalid = [c for c in show_columns if c not in available]
        for col in invalid:
            logger.warning("Column '%s' not found. Valid columns: %s", col, available)
        if valid:
            mask = df.columns.get_level_values(0).isin(valid)
            return df.loc[:, mask]
        return df

    # Flat columns
    available_cols = [str(c) for c in df.columns]
    col_map = {str(c): c for c in df.columns}
    valid_cols = [c for c in show_columns if c in available_cols]

    if valid_cols:
        return df[[col_map[c] for c in valid_cols]]

    # Row-index fallback (financial statements)
    if isinstance(df.index, pd.MultiIndex):
        level_values = df.index.get_level_values(-1)
        available_idx = [str(v) for v in level_values.unique()]
        idx_map = {str(v): v for v in level_values.unique()}
        valid_idx = [c for c in show_columns if c in available_idx]
        if valid_idx:
            mask = level_values.isin([idx_map[c] for c in valid_idx])
            filtered = df[mask]
            # When the filter reduces the last index level to one unique value
            # (e.g. show_columns=['Revenue'] on a multi-ticker statement), that
            # level repeats the same label in every row — drop it so the result
            # is indexed by ticker alone.
            if len(filtered.index.get_level_values(-1).unique()) == 1:
                filtered.index = filtered.index.droplevel(-1)
            return filtered
    else:
        available_idx = [str(v) for v in df.index.unique()]
        idx_map = {str(v): v for v in df.index.unique()}
        valid_idx = [c for c in show_columns if c in available_idx]
        if valid_idx:
            filtered = df.loc[[idx_map[c] for c in valid_idx]]
            # When only one metric row remains the index label is known from the
            # filter — squeeze to a Series so the caller gets a clean period →
            # value mapping without the redundant metric label.
            if len(filtered) == 1:
                return filtered.squeeze()
            return filtered

    all_available = available_cols + (
        available_idx
        if not isinstance(df.index, pd.MultiIndex)
        else [str(v) for v in df.index.get_level_values(-1).unique()]
    )
    logger.warning(
        "show_columns %s not matched in columns or index. Available: %s",
        show_columns,
        all_available,
    )
    return df
