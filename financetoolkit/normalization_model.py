"""Normalization Module"""

__docformat__ = "google"

import contextlib
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from financetoolkit.helpers import (
    PERIOD_END_DATE,
    convert_period_end_dates_to_calendar_periods,
)
from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


def initialize_statements_and_normalization(
    balance: pd.DataFrame,
    income: pd.DataFrame,
    cash: pd.DataFrame,
    format_location: str,
    start_date: str,
    end_date: str,
    quarterly: bool,
    tickers: list[str] | None = None,
):
    """
    Initializes financial statements by applying normalization and date conversion.
    Also loads normalization format files.

    Statements are not pre-loaded from cache here. Caching happens per ticker
    when the statements are actually requested, so that a different ticker list or
    period reuses what overlaps rather than being handed a previously assembled
    frame for a different set of companies.

    Args:
        balance (pd.DataFrame): Raw balance sheet data.
        income (pd.DataFrame): Raw income statement data.
        cash (pd.DataFrame): Raw cash flow statement data.
        format_location (str): Path to normalization file directory. A file missing there
            falls back to the packaged one.
        start_date (str): Start date for filtering.
        end_date (str): End date for filtering.
        quarterly (bool): Whether the data is quarterly.
        tickers (list[str] | None, optional): The tickers to keep from the statements.
            Defaults to None, which keeps every ticker.

    Returns:
        tuple: A tuple containing the processed statements (balance, income, cash, statistics)
               and the loaded normalization formats, for FinancialModelingPrep and for
               Yahoo Finance, including the statistics format of both sources. The statistics
               hold the Period End Date of every period when the statements are labelled with
               dates.
    """
    norm_file_names = [
        "balance",
        "balance_yf",
        "income",
        "income_yf",
        "cash",
        "cash_yf",
        "statistics",
        "statistics_yf",
    ]
    norm_formats = {
        name: read_normalization_file(name, format_location) for name in norm_file_names
    }
    fmp_balance_sheet_statement_generic = norm_formats["balance"]
    yf_balance_sheet_statement_generic = norm_formats["balance_yf"]
    fmp_income_statement_generic = norm_formats["income"]
    yf_income_statement_generic = norm_formats["income_yf"]
    fmp_cash_flow_statement_generic = norm_formats["cash"]
    yf_cash_flow_statement_generic = norm_formats["cash_yf"]
    fmp_statistics_statement_generic = norm_formats["statistics"]
    yf_statistics_statement_generic = norm_formats["statistics_yf"]

    period_end_dates: list[pd.DataFrame] = []

    def _process_statement(
        statement_df: pd.DataFrame,
        statement_format: pd.Series,
        statement_name: str,
    ) -> pd.DataFrame:
        """Normalizes a user supplied statement and relabels its dates."""
        if statement_df.empty:
            return pd.DataFrame()

        # Without the ticker level the statement cannot be told apart per company, and a
        # repeated line item or period would be counted as a separate ticker or period.
        if statement_df.index.nlevels != 2:  # noqa: PLR2004
            raise ValueError(
                f"The {statement_name} statement needs two index levels: the ticker and the line "
                "item, e.g. ('AAPL', 'Total Assets'). For one ticker, pd.concat({'AAPL': "
                "statement}) adds the ticker level."
            )

        for axis, labels in (
            ("line items", statement_df.index),
            ("periods", statement_df.columns),
        ):
            if labels.has_duplicates:
                raise ValueError(
                    f"The {statement_name} statement holds {axis} more than once: "
                    f"{', '.join(map(str, labels[labels.duplicated()].unique()[:5]))}."
                )

        if tickers is not None:
            available = statement_df.index.unique(level=0)
            missing = [ticker for ticker in tickers if ticker not in available]

            if missing:
                logger.warning(
                    "The %s statement holds no data for %s.",
                    statement_name,
                    ", ".join(missing),
                )

            statement_df = statement_df.loc[
                statement_df.index.get_level_values(0).isin(tickers)
            ]

            if statement_df.empty:
                return pd.DataFrame()

        try:
            processed_statement = convert_financial_statements(
                financial_statements=statement_df,
                statement_format=statement_format,
                adjust_financial_statements=False,
            )

            # Ensure DataFrame type
            if not isinstance(processed_statement, pd.DataFrame):
                logger.error(
                    "convert_financial_statements for %s did not return a DataFrame (got %s). Returning empty.",
                    statement_name,
                    type(processed_statement),
                )
                return pd.DataFrame()

            processed_statement, statement_end_dates = convert_date_label(
                financial_statement=processed_statement,
                start_date=start_date,
                end_date=end_date,
                quarter=quarterly,
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"The {statement_name} statement could not be read: {error}. Its columns need "
                "to be periods or dates, such as 2023 or 2023-09-30."
            ) from error

        # Supplied statements are never replaced by downloaded ones, so a window that holds
        # none of their periods says where the data does run.
        if processed_statement.empty:
            logger.warning(
                "The %s statement has no periods between %s and %s. Its columns run "
                "from %s to %s.",
                statement_name,
                start_date,
                end_date,
                statement_df.columns.min(),
                statement_df.columns.max(),
            )

        period_end_dates.append(statement_end_dates)

        return processed_statement

    balance_sheet_statement = _process_statement(
        balance,
        fmp_balance_sheet_statement_generic,
        "balance sheet",
    )
    income_statement = _process_statement(
        income, fmp_income_statement_generic, "income"
    )
    cash_flow_statement = _process_statement(
        cash, fmp_cash_flow_statement_generic, "cash flow"
    )

    # The fiscal period end dates of date labelled statements, kept as the statistic the
    # daily ratios and currency translation take each period's end from.
    statistics_statement = pd.DataFrame()
    end_dates = [dates for dates in period_end_dates if not dates.empty]

    if end_dates:
        combined = end_dates[0]
        for dates in end_dates[1:]:
            combined = combined.combine_first(dates)

        combined.index = pd.MultiIndex.from_product([combined.index, [PERIOD_END_DATE]])
        statistics_statement = combined.sort_index(axis=1)

    return (
        balance_sheet_statement,
        income_statement,
        cash_flow_statement,
        statistics_statement,
        fmp_balance_sheet_statement_generic,
        yf_balance_sheet_statement_generic,
        fmp_income_statement_generic,
        yf_income_statement_generic,
        fmp_cash_flow_statement_generic,
        yf_cash_flow_statement_generic,
        fmp_statistics_statement_generic,
        yf_statistics_statement_generic,
    )


def read_normalization_file(statement: str, format_location: str = ""):
    """
    This function copies the normalization files as found inside the environment and saves them
    to a custom location defined by the user. This is designed to make it possible to edit these files
    and supply them to the financial statement functions and within the Analyzer class.

    Args:
        statement (string): the type of statement you wish to read (balance, income, or cash).
        format_location (string): the folder to read the file from. When the folder has no file
            for the statement, the packaged file is read.
    Returns:
        A pandas Series containing the line items for the desired statement.
    """
    if statement not in [
        "balance_yf",
        "balance",
        "income_yf",
        "income",
        "cash_yf",
        "cash",
        "statistics_yf",
        "statistics",
    ]:
        raise ValueError(
            "Please provide a valid statement type (balance, income, cash or statistics)."
        )

    # A folder made with get_normalization_files holds only the FinancialModelingPrep files,
    # so a statement without its own file there uses the packaged one.
    if format_location and Path(format_location, f"{statement}.csv").is_file():
        file_location: str | Any = f"{format_location}/{statement}.csv"
    else:
        file_location = resources.files(__package__).joinpath(
            f"normalization/{statement}.csv"
        )

    try:
        # A user supplied location is a plain path while the packaged default is a
        # resource that may live inside a zip, so both are handed over as open files.
        with (
            open(file_location, encoding="utf-8")
            if isinstance(file_location, str)
            else file_location.open(encoding="utf-8")
        ) as csv_file:
            normalization = pd.read_csv(csv_file, index_col=[0]).iloc[:, 0]
    except FileNotFoundError:
        return pd.Series()

    # Stray whitespace around a normalized name makes the same concept resolve to two different labels depending on which source served the ticker, so every lookup on that name silently misses for one of them.  # noqa: E501
    normalization.index = normalization.index.astype(str).str.strip()

    return normalization.astype(str).str.strip()


def convert_financial_statements(
    financial_statements: pd.DataFrame,
    statement_format: pd.Series | pd.DataFrame = pd.DataFrame(),
    adjust_financial_statements: bool = True,
):
    """
    Converts financial statements (balance, income, or cash flow statements) based on custom input
    and returns a DataFrame containing the data with the correct naming.

    Args:
        financial_statements (pd.DataFrame): DataFrame containing the financial statement data.
        statement_format (pd.Series | pd.DataFrame): Optional mapping of the names of the financial statement line
                            items to include in the output. Rows should contain the original name of the line item,
                            and columns should contain the desired name for that line item.
        adjust_financial_statements (bool): Whether to add every line item in the format that the provider did not
                            return, filled with NaN, so that every ticker has the same rows. Defaults to True.
                            NaN keeps an item the provider never reported apart from one it reported as zero.

    Returns:
        pd.DataFrame: A DataFrame containing the financial statement data. If only one ticker is provided, the
                    returned DataFrame will have a single column containing the data for that company.
    """
    naming = {}

    if statement_format.empty:
        # If not format is provided, simply use the original financial statements
        return financial_statements

    # Compared without surrounding whitespace so that statements normalized by an older version, which had stray spaces in a handful of names, are still recognised as already normalized instead of being dropped.  # noqa: E501
    normalized_names = {str(value).strip() for value in statement_format.to_numpy()}

    unmapped_names = []

    for name in financial_statements.index.unique(level=1):
        try:
            if str(name).strip() in normalized_names:
                naming[name] = str(name).strip()
            else:
                naming[name] = statement_format.loc[name]
        except KeyError:
            # A line item the provider returned that no normalization file names. It is dropped from the output, so it is reported rather than disappearing.  # noqa: E501
            unmapped_names.append(str(name))
            continue

    if unmapped_names:
        logger.debug(
            "The following line items were returned by the data provider but are not "
            "present in the normalization format, so they are not included in the "
            "output: %s",
            ", ".join(sorted(unmapped_names)),
        )

    if adjust_financial_statements:
        # Add the line items the provider did not return, so that every ticker exposes the same rows. These are filled with NaN rather than zero: the provider not reporting an item says nothing about its value, and a fabricated zero is indistinguishable from a genuine zero while silently flowing into any ratio that uses it as a term or a denominator.  # noqa: E501
        for name in statement_format.index:
            for ticker in financial_statements.index.unique(level=0):
                if name not in financial_statements.loc[ticker].index:
                    financial_statements.loc[(ticker, name), :] = np.nan

        # All columns are present now, so the original statement format can be reused.
        naming = statement_format

    # Select only the columns it could trace back to the format
    financial_statements = financial_statements.loc[:, list(naming.keys()), :]
    financial_statements = financial_statements.rename(index=naming)

    # Oldest period first, whatever order the provider or user supplied the columns in.
    if not isinstance(financial_statements.columns, pd.MultiIndex):
        financial_statements = financial_statements.sort_index(axis=1)

    return financial_statements.sort_index(level=0, sort_remaining=False)


def convert_date_label(
    financial_statement: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    quarter: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function converts the date labels of the financial statements to a PeriodIndex. This is meant to be used
    throughout the entire Toolkit. This function is relevant for custom datasets that do not have a PeriodIndex.

    Labels that are periods (2022, "2022" or "2022Q4") keep their period, at a frequency of 'Y' or, with
    quarter = True, 'Q', and the periods that overlap the start and end date are kept. Labels that are dates
    are the last day of a fiscal period: a period ending after the end date (or before the start date) is
    left out and the others are labelled with the calendar period most of the fiscal period falls in (see
    helpers.convert_period_end_dates_to_calendar_periods), as downloaded statements are. A fiscal year ending
    30 June 2022 is labelled 2022 and one ending 31 January 2023 is labelled 2022. Columns of different
    tickers that end up in the same period are combined.

    Args:
        financial_statement (pd.DataFrame): DataFrame containing the financial statement data.
        start_date (str): The start date of the financial statement data.
        end_date (str): The end date of the financial statement data.
        quarter (bool): Whether to convert the date labels to a PeriodIndex with a frequency of 'Q' or 'Y'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The financial statement data with a PeriodIndex, and the last day of
        each ticker's fiscal period (tickers as index, periods as columns), which is empty when the labels are
        periods.
    """
    frequency = "Q" if quarter else "Y"
    columns = financial_statement.columns
    period_end_dates = pd.DataFrame()

    if isinstance(columns, pd.PeriodIndex | pd.DatetimeIndex):
        is_date_label = isinstance(columns, pd.DatetimeIndex)
    else:
        labels = pd.Index(columns.astype(str).str.strip())
        # A period label is a year ("2022") or a quarter ("2022Q4"); anything else that
        # parses is a date. Comparing either as text would truncate "2021" before
        # "2021-01-01" and "2022Q1" after "2022-12-31".
        is_date_label = not labels.str.fullmatch(r"\d{4}([Qq][1-4])?").all()
        columns = pd.DatetimeIndex(pd.to_datetime(labels)) if is_date_label else labels

    try:
        financial_statement = financial_statement.astype(np.float64)
    except ValueError as error:
        logger.error(
            "Not able to convert DataFrame to float64 due to %s. This could result in"
            "issues when values are zero and is predominantly relevant for "
            "ratio calculations.",
            error,
        )

    if is_date_label:
        dates = pd.DatetimeIndex(columns).normalize()
        within = np.ones(len(dates), dtype=bool)
        if start_date:
            within &= dates >= pd.Timestamp(start_date)
        if end_date:
            within &= dates <= pd.Timestamp(end_date)

        financial_statement = financial_statement.loc[:, within]
        dates = dates[within]
        periods = convert_period_end_dates_to_calendar_periods(dates, quarter=quarter)

        tickers = (
            financial_statement.index.get_level_values(0)
            if isinstance(financial_statement.index, pd.MultiIndex)
            else None
        )

        if tickers is not None:
            # Each ticker's fiscal end is the date of the column it has data in.
            has_data = financial_statement.notna().groupby(tickers).any()
            period_end_dates = pd.DataFrame(
                np.where(has_data.to_numpy(), np.asarray(dates)[None, :], None),
                index=has_data.index,
                columns=periods,
            )
            period_end_dates = period_end_dates.T.groupby(level=0).first().T

        financial_statement.columns = periods
    else:
        financial_statement.columns = pd.PeriodIndex(columns, freq=frequency)
        financial_statement = financial_statement.sort_index(axis=1).truncate(
            before=start_date, after=end_date, axis=1
        )

    if financial_statement.columns.duplicated().any():
        # Columns sharing a period hold different tickers (each ending its fiscal period on
        # its own date) or equal copies, so the first value of each is kept.
        financial_statement = financial_statement.T.groupby(level=0).first().T

    financial_statement = financial_statement.sort_index(axis=1)
    financial_statement.columns = pd.PeriodIndex(
        financial_statement.columns, freq=frequency
    )

    return financial_statement, period_end_dates


def copy_normalization_files(
    format_location: str = "",
    save_location: str | Path = Path(Path.home(), "Downloads"),
):
    """
    This function copies the normalization files as found inside the environment and saves them
    to a custom location defined by the user. This is designed to make it possible to edit these files
    and supply them to the financial statement functions and within the Analyzer class.

    Args:
        format_location (string): the location to read the file from.
        save_location (string): the location you wish to save the files to.
    Returns:
        Three csv files saved to the desired location.
    """
    logger.info(
        "Files are being saved to %s. Please see the following: "
        "https://www.jeroenbouma.com/projects/financetoolkit/docs/external-datasets "
        "to understand how to work with these files. In essence, all it requires is "
        "to match up the rows in your dataframe with the normalization format.",
        save_location,
    )
    save_location = Path(save_location)

    # A server often has no Downloads folder, so a missing folder is created; one that
    # cannot be falls back to the current working directory below.
    with contextlib.suppress(OSError):
        save_location.mkdir(parents=True, exist_ok=True)

    for statement in ["balance", "income", "cash", "statistics"]:
        if format_location:
            file_location = f"{format_location}/{statement}.csv"
        else:
            file_location = resources.files(__package__).joinpath(
                f"normalization/{statement}.csv"
            )

        def copy_to(target: Path) -> None:
            # A user supplied location is a plain path while the packaged default is a
            # resource that may live inside a zip, so both are copied as open files.
            with (
                (
                    open(file_location, "rb")
                    if isinstance(file_location, str)
                    else file_location.open("rb")
                ) as source,
                open(target, "wb") as sink,
            ):
                shutil.copyfileobj(source, sink)

        destination = save_location / f"{statement}.csv"
        try:
            copy_to(destination)
        except OSError:
            fallback = Path.cwd() / f"{statement}.csv"
            logger.warning(
                "Cannot write to %s. Saving to %s instead.",
                destination,
                fallback,
            )
            copy_to(fallback)


def get_normalization_files(path: str | Path = ""):
    """
    Copies the normalization files to a folder of your choice, the 'Downloads' folder by
    default. These files map the rows of your own financial statements onto the names the
    Finance Toolkit works with, so that you can supply your own datasets.

    There are four files: balance.csv, income.csv and cash.csv for the balance sheet,
    income statement and cash flow statement, and statistics.csv for the statistics such as
    the number of shares outstanding. Each has a column with the Finance Toolkit name of a
    row and a column to fill in with the matching row name of your own data. Pass the folder
    with the filled-in files to the Toolkit with `format_location`. See the external
    datasets guide for a worked example:
    https://www.jeroenbouma.com/projects/financetoolkit/docs/external-datasets

    No Toolkit instance (and no API key) is needed to get the files.

    Args:
        path (str | Path, optional): The folder to save the files to. Defaults to "", which
            saves them to the 'Downloads' folder of your home directory.

    Returns:
        None: The four csv files are saved to the chosen folder, which is created when it does
        not exist. When that folder cannot be written to, they are saved to the current
        working directory instead.

    As an example:

    ```python
    from financetoolkit import Toolkit, get_normalization_files

    get_normalization_files("normalization")

    # Fill in the files, then supply your own statements together with them.
    companies = Toolkit(
        tickers=["TSLA"],
        balance=balance_statement,
        income=income_statement,
        cash=cash_flow_statement,
        format_location="normalization",
    )
    ```
    """
    if path:
        copy_normalization_files(save_location=path)
    else:
        copy_normalization_files()
