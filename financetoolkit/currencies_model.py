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


CURRENCY_CODE_LENGTH = 3

# The active ISO 4217 currency codes, used to recognise a forex pair such as EURUSD or
# EURUSD=X. Crypto tickers such as BTCUSD are not matched because BTC is not one of them.
CURRENCY_CODES: frozenset[str] = frozenset(
    [
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "ANG",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHF",
        "CLP",
        "CNH",
        "CNY",
        "COP",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "FKP",
        "GBP",
        "GEL",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KMF",
        "KPW",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRU",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLE",
        "SOS",
        "SRD",
        "SSP",
        "STN",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TWD",
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "UYU",
        "UZS",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XCD",
        "XOF",
        "XPF",
        "YER",
        "ZAR",
        "ZMW",
        "ZWL",
    ]
)  # codespell:ignore zar


def is_currency_pair(ticker: str) -> bool:
    """
    Returns whether a ticker is a forex pair: six letters, optionally followed by "=X" as
    Yahoo Finance writes them, of which both halves are currency codes. A stock (TSLA,
    MSFT.AS) or crypto ticker (BTCUSD) is not a currency pair.

    Args:
        ticker (str): The ticker to check, e.g. "EURUSD" or "EURUSD=X".

    Returns:
        bool: Whether the ticker is a currency pair.
    """
    pair = ticker.upper().removesuffix("=X")

    base, quote = pair[:CURRENCY_CODE_LENGTH], pair[CURRENCY_CODE_LENGTH:]

    return (
        len(pair) == 2 * CURRENCY_CODE_LENGTH
        and base in CURRENCY_CODES
        and quote in CURRENCY_CODES
    )


# The quote currencies of crypto pairs besides the fiat currencies, as in ETH-BTC.
CRYPTO_QUOTE_CURRENCIES = {"USDT", "USDC", "BTC", "ETH"}


def to_financial_modeling_prep_symbol(symbol: str) -> str:
    """
    Writes a Yahoo Finance currency or crypto symbol as FinancialModelingPrep writes it,
    so either form can be passed: EURUSD=X becomes EURUSD and BTC-USD becomes BTCUSD.
    Any other symbol, such as a share class (BRK-B), is returned unchanged.

    Args:
        symbol (str): The symbol, e.g. "EURUSD=X" or "BTC-USD".

    Returns:
        str: The symbol as FinancialModelingPrep writes it.
    """
    upper = symbol.strip().upper()

    if upper.endswith("=X") and is_currency_pair(upper):
        return upper.removesuffix("=X")

    base, separator, quote = upper.partition("-")

    if (
        separator
        and len(base) > 1
        and base.isalnum()
        and (quote in CURRENCY_CODES or quote in CRYPTO_QUOTE_CURRENCIES)
    ):
        return f"{base}{quote}"

    return symbol


def to_yahoo_finance_symbol(symbol: str) -> str:
    """
    Writes a FinancialModelingPrep currency pair as Yahoo Finance writes it: EURUSD
    becomes EURUSD=X. A crypto pair such as BTCUSD cannot be told apart from a share
    ticker reliably, so it, and any other symbol, is returned unchanged.

    Args:
        symbol (str): The symbol, e.g. "EURUSD".

    Returns:
        str: The symbol as Yahoo Finance writes it.
    """
    upper = symbol.strip().upper()

    if is_currency_pair(upper) and not upper.endswith("=X"):
        return f"{upper}=X"

    return symbol


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


def get_statement_exchange_rates(
    daily_exchange_rates: pd.DataFrame,
    period_end_dates: pd.DataFrame,
    statement_currencies: pd.Series,
    quarter: bool,
    rate: str = "closing",
) -> pd.DataFrame:
    """
    Determines the exchange rate of every statement period per ticker from the daily
    exchange rates, using the fiscal period of the ticker rather than the calendar period
    its label refers to.

    IAS 21.39 translates assets and liabilities at the closing rate, the rate on the last
    day of the fiscal period, and income, expenses and cash flows at the rates on the dates
    of the transactions, for which the average rate over the fiscal period is the usual
    approximation (IAS 21.40, IAS 7.25-27). A company with a June fiscal year end therefore
    takes the rate on 30 June and the average from July to June, not the rate on 31
    December and the calendar year average.

    The rates are:

        - closing: the rate on the period end date, or the last rate before it when no rate
          was published on that day (a weekend or holiday).
        - average: the average of the daily rates after the end of the previous period up to
          and including the period end date. The first period of a ticker has no previous
          period, so its window starts a year (or a quarter) before its end date. A window
          that starts before the daily rates do is averaged over the rates available.
        - opening: the closing rate of the previous period, i.e. the rate on the day the
          period starts from, used for balances at the beginning of the period. For the first
          period it is the rate a year (or a quarter) before its end date, which is NaN when
          the daily rates do not reach back that far.

    Args:
        daily_exchange_rates (pd.DataFrame): The daily exchange rates with a daily index and
            the currency pairs (e.g. EURUSD=X) as columns. These should not be rounded, since
            a rate such as KRWUSD needs many decimals.
        period_end_dates (pd.DataFrame): The last day of the fiscal period behind each
            statement period, with the tickers as index and the periods as columns.
        statement_currencies (pd.Series): The currency pair per ticker.
        quarter (bool): Whether the periods are quarters. Years when False.
        rate (str, optional): The rate to determine: "closing", "average" or "opening".
            Defaults to "closing".

    Returns:
        pd.DataFrame: The exchange rate per period (index) and ticker (columns). Tickers
        without a currency pair in the daily exchange rates are left out.
    """
    if rate not in ["closing", "average", "opening"]:
        raise ValueError("Please choose closing, average or opening as rate.")

    daily = daily_exchange_rates.copy()
    daily.index = (
        daily.index.to_timestamp()
        if isinstance(daily.index, pd.PeriodIndex)
        else pd.DatetimeIndex(daily.index)
    ).normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()

    period_length = pd.DateOffset(months=3) if quarter else pd.DateOffset(years=1)

    rates = {}

    for ticker in period_end_dates.index:
        currency = statement_currencies.get(ticker)

        if currency != currency or currency not in daily.columns:  # noqa
            continue

        currency_rates = daily[currency].dropna()
        end_dates = pd.Series(
            pd.DatetimeIndex(period_end_dates.loc[ticker]),
            index=period_end_dates.columns,
        ).sort_values()

        # The previous period's end (or a full period earlier for the first one) is the day
        # the period starts from.
        start_dates = end_dates.shift(1)
        start_dates = start_dates.fillna(end_dates - period_length)

        if rate == "average":
            values = [
                currency_rates.loc[
                    (currency_rates.index > start) & (currency_rates.index <= end)
                ].mean()
                for start, end in zip(start_dates, end_dates, strict=True)
            ]
        else:
            dates = end_dates if rate == "closing" else start_dates
            # The last published rate on or before each date.
            positions = currency_rates.index.searchsorted(dates, side="right") - 1
            values = [
                currency_rates.iloc[position] if position >= 0 else float("nan")
                for position in positions
            ]

        rates[ticker] = pd.Series(values, index=end_dates.index, dtype="float64")

    return pd.DataFrame(rates, index=period_end_dates.columns)


CASH_BALANCE_ITEMS = {"Cash Beginning of Period", "Cash End of Period"}


FOREX_CHANGES_ON_CASH = "Forex Changes on Cash"


def get_cash_flows_excluding_forex(
    cash_flow_statement: pd.DataFrame, ticker: str
) -> pd.Series | None:
    """
    Determines the operating, investing and financing cash flows of each period together,
    without the effect of exchange rate changes on the cash held.

    The sources report Net Change in Cash in two ways. FinancialModelingPrep includes the
    effect of exchange rate changes (Cash End of Period - Cash Beginning of Period = Net Change
    in Cash), Yahoo Finance excludes it (Cash End of Period - Cash Beginning of Period = Net
    Change in Cash + Forex Changes on Cash). Which of the two applies is determined per period
    from the cash balances: the convention under which the balances reconcile best is used.
    Without a reported Forex Changes on Cash, Net Change in Cash is taken as the cash flows.

    Args:
        cash_flow_statement (pd.DataFrame): The cash flow statement in its reporting currency
            with a (ticker, item) index and the periods as columns.
        ticker (str): The ticker to determine the cash flows for.

    Returns:
        pd.Series | None: The cash flows per period, or None when Net Change in Cash, Cash
        Beginning of Period or Cash End of Period is not in the statement.
    """
    required = ["Cash Beginning of Period", "Cash End of Period", "Net Change in Cash"]

    if not all((ticker, item) in cash_flow_statement.index for item in required):
        return None

    beginning, end, net_change = (
        cash_flow_statement.loc[(ticker, item), :].astype(float) for item in required
    )
    forex_changes = (
        cash_flow_statement.loc[(ticker, FOREX_CHANGES_ON_CASH), :].astype(float)
        if (ticker, FOREX_CHANGES_ON_CASH) in cash_flow_statement.index
        else pd.Series(float("nan"), index=cash_flow_statement.columns)
    )

    # The part of the change in cash that Net Change in Cash does not explain: zero when it
    # includes the exchange rate effect, the exchange rate effect itself when it does not.
    residual = end - beginning - net_change
    excludes_forex = forex_changes.notna() & (
        (residual - forex_changes).abs() < residual.abs()
    )

    return (net_change - forex_changes.fillna(0)).where(~excludes_forex, net_change)


def reconcile_cash_translation(
    cash_flow_statement: pd.DataFrame,
    ticker: str,
    cash_flows: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Restores the reconciliation of the cash at the beginning and the end of a period after
    a translation in which the cash flows are converted at the average rate of the period
    and the cash balances at closing rates.

    Translated at different rates, the cash flows no longer add up to the change in the
    translated cash balances. IAS 7.28 reports that difference, the effect of exchange rate
    changes on the cash held, as a separate item. Net Change in Cash therefore becomes the
    change in the translated cash balances (so it includes the exchange rate effect, whatever
    the convention of the source) and Forex Changes on Cash takes up the difference with the
    translated operating, investing and financing cash flows. The Forex Changes on Cash row
    is added when the statement does not have it. A period for which one of the items is
    missing is left as it is, which includes the first period when the opening rate of its
    Cash Beginning of Period is not available.

    Args:
        cash_flow_statement (pd.DataFrame): The translated cash flow statement with a
            (ticker, item) index and the periods as columns.
        ticker (str): The ticker to reconcile.
        cash_flows (pd.Series | None, optional): The translated operating, investing and
            financing cash flows per period, as determined with get_cash_flows_excluding_forex
            before the translation and converted at the average rate. Defaults to None, which
            takes them as Net Change in Cash minus Forex Changes on Cash of the translated
            statement.

    Returns:
        pd.DataFrame: The cash flow statement with Net Change in Cash and Forex Changes on
        Cash of the ticker reconciled to the translated cash balances.
    """
    items = ["Cash Beginning of Period", "Cash End of Period", "Net Change in Cash"]

    if not all((ticker, item) in cash_flow_statement.index for item in items):
        return cash_flow_statement

    if (ticker, FOREX_CHANGES_ON_CASH) not in cash_flow_statement.index:
        cash_flow_statement.loc[(ticker, FOREX_CHANGES_ON_CASH), :] = float("nan")

    beginning, end, net_change, forex_changes = (
        cash_flow_statement.loc[(ticker, item), :]
        for item in [*items, FOREX_CHANGES_ON_CASH]
    )

    if cash_flows is None:
        # The operating, investing and financing cash flows, all translated at the average rate.
        cash_flows = net_change - forex_changes.fillna(0)

    reconciled_net_change = end - beginning
    reconciled_forex_changes = reconciled_net_change - cash_flows

    cash_flow_statement.loc[(ticker, "Net Change in Cash"), :] = (
        reconciled_net_change.where(reconciled_forex_changes.notna(), net_change)
    ).to_numpy()
    cash_flow_statement.loc[(ticker, FOREX_CHANGES_ON_CASH), :] = (
        reconciled_forex_changes.where(reconciled_forex_changes.notna(), forex_changes)
    ).to_numpy()

    return cash_flow_statement


def convert_currencies(
    financial_statement_data: pd.DataFrame,
    financial_statement_currencies: pd.Series,
    exchange_rate_data: pd.DataFrame,
    items_not_to_adjust: list[str] | None = None,
    financial_statement_name: str | None = None,
    item_exchange_rate_data: dict[str, pd.DataFrame] | None = None,
):
    """
    Based on the retrieved currency definitions (e.g. EURUSD=X) for each ticker, obtained
    through using the determine_currencies function, convert the financial statement data
    to the historical currency.

    Each period is converted at the rate of exchange_rate_data for that period. Following
    IAS 21.39-40 and IAS 7.25-28, the caller passes the closing rate (the rate at the end of
    the period) for the balance sheet and the average rate of the period for the income and
    cash flow statements. Items that describe a balance at a moment in time within a flow
    statement, such as the cash at the beginning and the end of the period, take their own
    rate through item_exchange_rate_data.

    The function reports the tickers that are converted and the currencies that they are
    converted from and to. If the currency is the same, then no conversion is applied.

    The function will also report the tickers that could not be converted. This is usually
    due to the fact that the currency is not available in the historical data.

    Args:
        financial_statement_data (pd.DataFrame): A DataFrame containing the financial statement data.
        financial_statement_currencies (pd.Series): A Series containing the currency symbols per ticker.
        exchange_rate_data (pd.DataFrame): The exchange rate per period (index), with as columns either
            the currency pairs (e.g. EURUSD=X) or the tickers, the latter for rates that depend on each
            ticker's fiscal periods (see get_statement_exchange_rates). A ticker column takes precedence
            over the column of its currency pair.
        items_not_to_adjust (list[str]): A list containing the items that should not be adjusted. Defaults to None.
        financial_statement_name (str | None): The name of the statement being converted, used in the log
            message that reports which tickers were converted. Defaults to None.
        item_exchange_rate_data (dict[str, pd.DataFrame] | None): The exchange rate data for specific items,
            as {item: exchange rate data}, used instead of exchange_rate_data for those items, e.g. the closing
            rate of the previous period for Cash Beginning of Period. Defaults to None.

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

                    # The retrieved rate is between major units even when the price is quoted in a fractional one, so it is scaled onto that unit here.  # noqa: E501
                    minor_unit_factor = get_minor_unit_factor(
                        base_currency, quote_currency
                    )

                    # reindex (not .loc) since older periods can predate the FX pair's own history, e.g. a statement stretching back to 1999 when TWDUSD=X only starts in 2004 — those periods get NaN rather than raising and dropping the whole ticker from conversion.  # noqa: E501
                    rates = exchange_rate_data[
                        ticker if ticker in exchange_rate_data.columns else currency
                    ].reindex(periods)

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

                    # Items with a rate of their own are converted separately below.
                    item_rates = {
                        item: rate_data[
                            ticker if ticker in rate_data.columns else currency
                        ].reindex(periods)
                        for item, rate_data in (item_exchange_rate_data or {}).items()
                        if item in items_to_adjust
                        and (ticker, item) in financial_statement_data.index
                        and (ticker in rate_data.columns or currency in rate_data)
                    }
                    items_to_adjust = [
                        item for item in items_to_adjust if item not in item_rates
                    ]

                    # The cash flows without the exchange rate effect, determined before the
                    # translation since the convention of the source shows in the local balances.
                    local_cash_flows = (
                        get_cash_flows_excluding_forex(financial_statement_data, ticker)
                        if set(item_rates) >= CASH_BALANCE_ITEMS
                        else None
                    )

                    financial_statement_data.loc[(ticker, items_to_adjust), :] = (
                        financial_statement_data.loc[(ticker, items_to_adjust), :].mul(
                            rates * minor_unit_factor,
                            axis=1,
                        )
                    ).to_numpy()

                    for item, item_rate in item_rates.items():
                        financial_statement_data.loc[(ticker, item), :] = (
                            financial_statement_data.loc[(ticker, item), :]
                            * item_rate
                            * minor_unit_factor
                        ).to_numpy()

                    if set(item_rates) >= CASH_BALANCE_ITEMS:
                        financial_statement_data = reconcile_cash_translation(
                            financial_statement_data,
                            ticker,
                            cash_flows=(
                                local_cash_flows * rates * minor_unit_factor
                                if local_cash_flows is not None
                                else None
                            ),
                        )

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
