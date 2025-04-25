"""Currencies Module"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


# pylint: disable=comparison-with-itself,too-many-locals,protected-access


def determine_currencies(
    statement_currencies: pd.DataFrame, historical_currencies: pd.DataFrame
):
    """
    Based on the statement currencies and the historical currencies, determine the
    currencies that are used in the financial statements and the historical datasets.

    This is relevant to prevent mismatches between the perceived price of the instrument
    and the numbers as found in the financial statements. If there is a mismatch, then
    the currency conversion needs to be applied.

    Args:
        statement_currencies (pd.DataFrame): A DataFrame containing the statement currencies.
        historical_currencies (pd.DataFrame): A DataFrame containing the historical currencies.

    Returns:
        pd.Series, list: a Series containing the currency symbols per ticker
        and a list containing the currencies.
    """
    currencies = []

    for period in statement_currencies.columns:
        statement_currencies.loc[:, period] = (
            statement_currencies[period] + historical_currencies + "=X"
        )

        for currency in statement_currencies[period].unique():
            # Identify the currencies that are not in the list yet
            # and that are not NaN (the currency == currency check)
            if currency not in currencies and currency == currency:  # noqa
                currencies.append(currency)

    statement_currencies = statement_currencies.bfill(axis=1).ffill(axis=1)

    statement_currencies = statement_currencies[statement_currencies.columns[-1]]

    return statement_currencies, currencies


def convert_currencies(
    financial_statement_data: pd.DataFrame,
    financial_statement_currencies: pd.Series,
    exchange_rate_data: pd.DataFrame,
    items_not_to_adjust: list[str] | None = None,
    financial_statement_name: str | None = None,
):
    """
    Based on the retrieved currency definitions (e.g. EURUSD=X) for each ticker, obtained
    through using the determine_currencies function, convert the financial statement data
    to the historical currency.

    The function reports the tickers that are converted and the currencies that they are
    converted from and to. If the currency is the same, then no conversion is applied.

    The function will also report the tickers that could not be converted. This is usually
    due to the fact that the currency is not available in the historical data.

    Args:
        financial_statement_data (pd.DataFrame): A DataFrame containing the financial statement data.
        financial_statement_currencies (pd.Series): A Series containing the currency symbols per ticker.
        exchange_rate_data (pd.DataFrame): A DataFrame containing the exchange rate data.
        items_not_to_adjust (list[str]): A list containing the items that should not be adjusted. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the converted financial statement data.
    """
    no_data = []

    periods = financial_statement_data.columns
    tickers = financial_statement_data.index.get_level_values(0).unique()
    currencies: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            currency = financial_statement_currencies.loc[ticker]

            # Only proceed if the currency is not NaN
            if currency == currency:  # noqa
                base_currency, quote_currency = currency[:3], currency[3:6]

                if base_currency != quote_currency:
                    if currency not in currencies:
                        currencies[currency] = []

                    if items_not_to_adjust is not None:
                        items_to_adjust = [
                            item
                            for item in financial_statement_data.index.get_level_values(
                                level=1
                            )
                            if item not in items_not_to_adjust
                        ]
                    else:
                        items_to_adjust = (
                            financial_statement_data.index.get_level_values(level=1)
                        )

                    financial_statement_data.loc[(ticker, items_to_adjust), :] = (
                        financial_statement_data.loc[(ticker, items_to_adjust), :].mul(
                            exchange_rate_data.loc[periods, currency], axis=1
                        )
                    ).to_numpy()

                    currencies[currency].append(ticker)
            else:
                no_data.append(ticker)
        except (KeyError, ValueError):
            no_data.append(ticker)
            continue

    if no_data:
        logger.warning(
            "The following tickers could not be verified whether the currency of the "
            "historical data matches with the financial statement data: "
            "%s",
            ", ".join(no_data),
        )

    currencies_text = []
    for currency, ticker_match in currencies.items():
        base_currency, quote_currency = currency[:3], currency[3:6]

        if base_currency != quote_currency:
            for ticker in ticker_match:
                currencies_text.append(
                    f"{ticker} ({base_currency} to {quote_currency})"
                )

    if currencies_text:
        logger.info(
            "The %s from the following tickers are converted: %s",
            (
                financial_statement_name
                if financial_statement_name
                else "financial statement"
            ),
            ", ".join(currencies_text),
        )

    return financial_statement_data
