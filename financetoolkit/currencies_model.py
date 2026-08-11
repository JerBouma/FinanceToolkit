"""Currencies Module"""

__docformat__ = "google"


import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()


# pylint: disable=comparison-with-itself,too-many-locals,protected-access

# LSE (GBp), JSE (ZAc), TASE (ILA) quote in a fractional unit; most providers only publish the major-unit rate.
MINOR_CURRENCY_UNITS: dict[str, tuple[str, int]] = {
    "GBp": ("GBP", 100),
    "ZAc": ("ZAR", 100),  # codespell:ignore zar
    "ILA": ("ILS", 100),
}

# GBp resolves via the generic "=X" construction; ZAc needs this native FMP symbol; ILA has no listing on either provider.
NATIVE_MINOR_UNIT_TICKERS: dict[tuple[str, str], str] = {
    ("USD", "ZAc"): "USDZAC",
}


def get_fx_ticker(base_currency: str, quote_currency: str) -> str:
    """
    Returns the ticker to request historical exchange rate data for from a data
    provider, for a given currency pair.

    This is usually the generic BASE+QUOTE+"=X" convention (e.g. "USDGBp=X"), but a
    handful of pairs are only listed under a provider-specific symbol instead, see
    NATIVE_MINOR_UNIT_TICKERS.

    Args:
        base_currency (str): the currency the value being converted is expressed in.
        quote_currency (str): the currency the value is being converted to.

    Returns:
        str: the ticker symbol to request historical exchange rate data for.
    """
    return NATIVE_MINOR_UNIT_TICKERS.get(
        (base_currency, quote_currency), f"{base_currency}{quote_currency}=X"
    )


def get_minor_unit_factor(base_currency: str, quote_currency: str) -> float:
    """
    Returns the factor an exchange rate has to be multiplied by to express it in the
    fractional units the quote currency is quoted in.

    An exchange rate is only ever published between major units, so converting USD
    reported statements onto a price quoted in pence needs the USD/GBP rate multiplied
    by the 100 pence in a pound. Pairs with a native minor-unit listing (see
    NATIVE_MINOR_UNIT_TICKERS) are the exception: the retrieved rate is already
    expressed in the fractional unit, so no further scaling is applied.

    Args:
        base_currency (str): the currency the value being converted is expressed in.
        quote_currency (str): the currency the value is being converted to.

    Returns:
        float: the factor to multiply the exchange rate by. 1.0 when neither side of
        the pair is quoted in a fractional unit, or when the pair is already natively
        quoted in it.
    """
    if (base_currency, quote_currency) in NATIVE_MINOR_UNIT_TICKERS:
        return 1.0

    base_factor = MINOR_CURRENCY_UNITS.get(base_currency, ("", 1))[1]
    quote_factor = MINOR_CURRENCY_UNITS.get(quote_currency, ("", 1))[1]

    return quote_factor / base_factor


def get_major_currency(currency: str) -> str:
    """
    Maps a currency code onto the major unit it belongs to, so that GBp resolves to GBP.
    Exchange rates only exist between major units, so this is the code an exchange rate
    can actually be requested for.

    Args:
        currency (str): the currency code, possibly a fractional unit such as GBp.

    Returns:
        str: the major unit currency code.
    """
    return MINOR_CURRENCY_UNITS.get(currency, (currency, 1))[0]


def is_same_currency(base_currency: str, quote_currency: str) -> bool:
    """
    Reports whether two currency codes describe the same currency in the same unit,
    so that no conversion is needed at all. GBP and GBp are the same currency but not
    the same unit, so they are not the same for this purpose.

    Args:
        base_currency (str): the first currency code.
        quote_currency (str): the second currency code.

    Returns:
        bool: whether the two codes are the same currency in the same unit.
    """
    return base_currency == quote_currency


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
            # Skip currencies already listed and NaN (the currency == currency check).
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
        financial_statement_name (str | None): The name of the statement being converted, used in the log
            message that reports which tickers were converted. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the converted financial statement data.
    """
    no_data = []

    # Cast upfront: pandas 3.0 bans silently upcasting int columns on `.loc`.
    financial_statement_data = financial_statement_data.astype("float64")

    periods = financial_statement_data.columns
    tickers = financial_statement_data.index.get_level_values(0).unique()
    currencies: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            currency = financial_statement_currencies.loc[ticker]

            # Only proceed if the currency is not NaN
            if currency == currency:  # noqa
                base_currency, quote_currency = currency[:3], currency[3:6]

                if not is_same_currency(base_currency, quote_currency):
                    if currency not in currencies:
                        currencies[currency] = []

                    # The retrieved rate is between major units even when the price is
                    # quoted in a fractional one, so it is scaled onto that unit here.
                    minor_unit_factor = get_minor_unit_factor(
                        base_currency, quote_currency
                    )

                    rates = exchange_rate_data.loc[periods, currency]

                    if rates.isna().all():
                        # Column exists (placeholder from a partly-failed batch fetch) but no provider published a rate.
                        raise ValueError(
                            f"No exchange rate data available for {currency}"
                        )

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
                            rates * minor_unit_factor,
                            axis=1,
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

        if not is_same_currency(base_currency, quote_currency):
            for ticker in ticker_match:
                currencies_text.append(
                    f"{ticker} ({base_currency} to {quote_currency})"
                )

    if currencies_text:
        logger.info(
            "Converting %s currency to match OHLC for: %s",
            financial_statement_name or "financial statement",
            ", ".join(currencies_text),
        )

    return financial_statement_data
