"""Options Base Module"""

__docformat__ = "google"

from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from financetoolkit.options import options_model
from financetoolkit.ratios import valuation_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
)

# ruff: noqa: E501

logger = logger_model.get_logger()

# The volatility is measured over the trading year before the pricing date, once at least a
# month of returns is available.
TRAILING_DAYS = 252

MINIMUM_VOLATILITY_DAYS = 21


def _days_to_expiration(expiration_date: str) -> int:
    """
    Counts the calendar days from today to an option's expiration date. Counting whole
    days, rather than from the current time of day, keeps an option that expires tomorrow
    at one day instead of rounding it down to none.

    Args:
        expiration_date (str): The expiration date (YYYY-MM-DD).

    Returns:
        int: The number of days, zero for an option that expires today.
    """
    return (
        pd.Timestamp(expiration_date).normalize() - pd.Timestamp.today().normalize()
    ).days


@filter_shown_columns
@track_sources
class OptionsBase:
    """
    The data every options topic shares: prices, volatility, the risk-free rate, dividend yields and the expiries.
    """

    def __init__(
        self,
        tickers: list[str],
        daily_historical: pd.DataFrame | Callable[[], pd.DataFrame] = pd.DataFrame(),
        annual_historical: pd.DataFrame | Callable[[], pd.DataFrame] = pd.DataFrame(),
        risk_free_rate: pd.DataFrame | Callable[[], pd.DataFrame] = pd.DataFrame(),
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
        earnings_function: Callable[[], dict[str, str]] | None = None,
    ):
        """
        Initializes the Options Controller Class. The Options module is meant to calculate important options
        metrics such as the First, Second and Third Order Greeks, the Black Scholes Model and the Option
        Chains as well as Implied Volatilities, Breeden—Litzenberger and more.

        Each dataset is either the data itself or a function without arguments that retrieves
        it, which is called when a calculation first needs that dataset. The Toolkit passes
        functions, so no prices or Treasury yields are retrieved before a calculation asks for
        them.

        Args:
            tickers (str | list[str]): The tickers to use.
            daily_historical (pd.DataFrame | Callable[[], pd.DataFrame], optional): The daily
                historical data. Defaults to pd.DataFrame().
            annual_historical (pd.DataFrame | Callable[[], pd.DataFrame], optional): The annual
                historical data. Defaults to pd.DataFrame().
            risk_free_rate (pd.DataFrame | Callable[[], pd.DataFrame], optional): The risk free rate, a Treasury yield quoted on a
                semi-annual bond-equivalent basis ("Adj Close" column) that is converted to continuous
                compounding. Every maturity goes through this conversion, so the 13-week bill rate,
                quoted as a discount rate, is approximated as a bond-equivalent yield.
                Defaults to pd.DataFrame().
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            earnings_function (Callable | None, optional): Returns the next earnings date per
                ticker, used by get_implied_move. Defaults to None.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "MU"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        all_greeks = toolkit.options.collect_all_greeks(start_date='2024-01-03')

        all_greeks.loc['TSLA', '2024-01-04']
        ```

        Which returns:

        |   Strike Price |   Delta |   Dual Delta |   Vega |   Theta |    Rho |   Epsilon |   Lambda |   Gamma |   Dual Gamma |   Vanna |    Charm |   Vomma |    Vera |    Veta |     PD |   Speed |   Zomma |   Color |   Ultima |
        |---------------:|--------:|-------------:|-------:|--------:|-------:|----------:|---------:|--------:|-------------:|--------:|---------:|--------:|--------:|--------:|-------:|--------:|--------:|--------:|---------:|
        |            180 |  1      |      -0.9999 | 0      | -0.0193 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0198 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0204 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0209 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0.0002 |
        |            200 |  1      |      -0.9999 | 0      | -0.0214 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      | -0      |   0.0003 |  0.0002 | -0      | -0      | 0      | -0      |  0      | -0.0002 |   0.0065 |
        |            205 |  1      |      -0.9999 | 0      | -0.022  | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      | -0.0001 |   0.0097 |  0.0048 | -0.0001 | -0      | 0      | -0      |  0      | -0.0053 |   0.1335 |
        |            210 |  0.9999 |      -0.9998 | 0      | -0.0235 | 0.5752 |   -0.6532 |   8.3742 |  0      |       0      | -0.0015 |   0.1722 |  0.0714 | -0.001  | -0.0002 | 0      | -0      |  0.0007 | -0.0778 |   1.3082 |
        |            215 |  0.9991 |      -0.9989 | 0.0004 | -0.0346 | 0.5884 |   -0.6527 |  10.1489 |  0.0004 |       0.0005 | -0.0143 |   1.6563 |  0.5604 | -0.0095 | -0.002  | 0.0005 | -0.0001 |  0.0051 | -0.5876 |   5.9321 |
        |            220 |  0.9927 |      -0.9919 | 0.0025 | -0.1034 | 0.5979 |   -0.6485 |  12.8004 |  0.0025 |       0.003  | -0.0766 |   8.8524 |  2.3355 | -0.0507 | -0.0087 | 0.003  | -0.0008 |  0.0196 | -2.264  |  10.617  |
        |            225 |  0.9614 |      -0.9584 | 0.0105 | -0.355  | 0.5908 |   -0.628  |  16.8575 |  0.0106 |       0.0119 | -0.2287 |  26.4045 |  5.0433 | -0.1523 | -0.0212 | 0.0119 | -0.0024 |  0.0343 | -3.9573 |   0.4951 |
        |            230 |  0.8655 |      -0.8581 | 0.027  | -0.8792 | 0.5407 |   -0.5654 |  22.8791 |  0.0273 |       0.0294 | -0.3657 |  42.119  |  5.0451 | -0.2463 | -0.0294 | 0.0294 | -0.0039 |  0.008  | -0.8883 | -14.4266 |
        |            235 |  0.6767 |      -0.6646 | 0.0448 | -1.4399 | 0.4279 |   -0.442  |  31.1644 |  0.0453 |       0.0467 | -0.2405 |  27.4427 |  1.3756 | -0.1694 | -0.0267 | 0.0467 | -0.0028 | -0.0575 |  6.6838 |  -6.0896 |
        |            240 |  0.4305 |      -0.4174 | 0.049  | -1.5675 | 0.2745 |   -0.2812 |  41.601  |  0.0496 |       0.0489 |  0.1289 | -15.4004 |  0.2817 |  0.0708 | -0.0254 | 0.0489 |  0.0009 | -0.0752 |  8.707  |  -1.3284 |
        |            245 |  0.2132 |      -0.2036 | 0.0363 | -1.1574 | 0.1367 |   -0.1393 |  53.7837 |  0.0367 |       0.0348 |  0.3795 | -44.314  |  3.7676 |  0.238  | -0.0302 | 0.0348 |  0.0035 | -0.0197 |  2.2468 | -13.8988 |
        |            250 |  0.0803 |      -0.0754 | 0.0186 | -0.5925 | 0.0516 |   -0.0524 |  67.2285 |  0.0188 |       0.0171 |  0.3372 | -39.2457 |  5.9056 |  0.2152 | -0.0281 | 0.0171 |  0.0033 |  0.0301 | -3.518  |  -9.1559 |
        |            255 |  0.0228 |      -0.0211 | 0.0067 | -0.2149 | 0.0147 |   -0.0149 |  81.5108 |  0.0068 |       0.006  |  0.1731 | -20.1217 |  4.3191 |  0.1112 | -0.0171 | 0.006  |  0.0017 |  0.0329 | -3.8307 |   7.2307 |
        |            260 |  0.0049 |      -0.0044 | 0.0018 | -0.0563 | 0.0032 |   -0.0032 |  96.308  |  0.0018 |       0.0015 |  0.0584 |  -6.7872 |  1.8839 |  0.0377 | -0.0069 | 0.0015 |  0.0006 |  0.0162 | -1.8861 |  11.1559 |
        |            265 |  0.0008 |      -0.0007 | 0.0003 | -0.0109 | 0.0005 |   -0.0005 | 111.391  |  0.0003 |       0.0003 |  0.0137 |  -1.5964 |  0.5417 |  0.0089 | -0.0019 | 0.0003 |  0.0001 |  0.0049 | -0.5728 |   6.0301 |
        |            270 |  0.0001 |      -0.0001 | 0      | -0.0016 | 0.0001 |   -0.0001 | 126.604  |  0      |       0      |  0.0023 |  -0.2715 |  0.1086 |  0.0015 | -0.0004 | 0      |  0      |  0.001  | -0.1183 |   1.8733 |
        |            275 |  0      |      -0      | 0      | -0.0002 | 0      |   -0      | 141.839  |  0      |       0      |  0.0003 |  -0.0343 |  0.0158 |  0.0002 | -0.0001 | 0      |  0      |  0.0002 | -0.0175 |   0.3819 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      | 157.025  |  0      |       0      |  0      |  -0.0033 |  0.0017 |  0      | -0      | 0      |  0      |  0      | -0.0019 |   0.0546 |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      | 172.114  |  0      |       0      |  0      |  -0.0002 |  0.0001 |  0      | -0      | 0      |  0      |  0      | -0.0002 |   0.0057 |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      | 187.072  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0004 |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      | 201.877  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        """
        self._tickers = tickers
        self._daily_historical_source = daily_historical
        self._annual_historical_source = annual_historical
        self._risk_free_rate_source = risk_free_rate
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date
        self._earnings_function = earnings_function

    @cached_property
    def _daily_historical(self) -> pd.DataFrame:
        """The daily historical data, retrieved when a calculation first needs it."""
        if callable(self._daily_historical_source):
            return self._daily_historical_source()
        return self._daily_historical_source

    @cached_property
    def _annual_historical(self) -> pd.DataFrame:
        """The annual historical data, retrieved when a calculation first needs it."""
        if callable(self._annual_historical_source):
            return self._annual_historical_source()
        return self._annual_historical_source

    @cached_property
    def _prices(self) -> pd.DataFrame:
        """The spot prices."""
        # The spot price is the traded close. On a past pricing date the Adj Close is lowered by
        # every dividend paid after that date, which was not known yet and which the dividend
        # yield q already accounts for. Returns (and so the volatility) do use the Adj Close.
        return self._daily_historical[
            "Close" if "Close" in self._daily_historical else "Adj Close"
        ]

    @cached_property
    def _volatility(self) -> pd.DataFrame:
        """The annualized volatility of the year of daily returns up to each date."""
        # The volatility on a date is that of the year of daily returns up to it, so a past
        # pricing date uses only what was known on that date.
        return self._daily_historical["Return"].rolling(
            TRAILING_DAYS, min_periods=MINIMUM_VOLATILITY_DAYS
        ).std() * np.sqrt(TRAILING_DAYS)

    @cached_property
    def _risk_free_rate(self) -> pd.Series:
        """The continuously compounded risk free rate, retrieved when first needed."""
        risk_free_rate = self._risk_free_rate_source
        if not isinstance(risk_free_rate, pd.DataFrame):
            risk_free_rate = risk_free_rate()
        # The Toolkit's risk-free rate is a US Treasury yield (the 10-year by default, set with the
        # Toolkit's risk_free_rate parameter: '13w', '5y', '10y' or '30y'), quoted on a semi-annual
        # bond-equivalent basis. The option formulas discount continuously through e^(-r * t), so
        # the yield is converted to its continuously compounded equivalent once, here. The '13w'
        # rate (^IRX) is a bill discount rate rather than a bond-equivalent yield, and the rate
        # arrives here without its maturity, so it goes through the same semi-annual conversion;
        # a 13-week discount rate of 5% sits 13 basis points below its bond-equivalent yield,
        # which makes the '13w' rate an approximation.  # noqa: E501
        return options_model.get_continuously_compounded_rate(
            risk_free_rate["Adj Close"]
        )

    @cached_property
    def _dividend_yield(self) -> dict[str, pd.Series]:
        """The dividend yield of every year per ticker, as the Black Scholes formula uses it."""
        # Calculate Dividend Yield, relevant for Black Scholes formula, relative to the traded
        # close like the spot price.
        dividend_yield = valuation_model.get_dividend_yield(
            dividends=self._annual_historical.loc[:, "Dividends"].T,
            stock_price=self._annual_historical.loc[
                :, "Close" if "Close" in self._annual_historical else "Adj Close"
            ].T,
        )

        dividend_yields = {ticker: pd.Series() for ticker in self._tickers}

        for ticker in self._tickers:
            dividend_yield_cleaned = dividend_yield.loc[
                ticker, dividend_yield.loc[ticker] != 0
            ]

            if dividend_yield_cleaned.empty:
                # An empty value means the company pays no dividends, so 0 is correct.
                dividend_yield_cleaned = dividend_yield.loc[ticker]

            dividend_yields[ticker] = dividend_yield_cleaned

        return dividend_yields

    @cached_property
    def _daily_dividend_yield(self) -> pd.DataFrame:
        """The trailing dividend yield on every date."""
        # The dividend yield on a date: the dividends of the 365 days up to it over the close of
        # that date, see options_model.get_trailing_dividend_yield.
        if "Close" not in self._daily_historical:
            return pd.DataFrame()

        return options_model.get_trailing_dividend_yield(
            close=self._daily_historical["Close"],
            adjusted_close=self._daily_historical["Adj Close"],
            dividends=self._daily_historical.get("Dividends"),
        )

    def _pricing_date(self, start_date: str | None) -> pd.Period:
        """
        The date the theoretical option values start from: the last date of the historical
        data, or the last trading day on or before start_date, so that a weekend or holiday
        uses the close before it rather than being rejected.

        Args:
            start_date (str | None): The requested date, None for the last date.

        Returns:
            pd.Period: The date of the prices to use.

        Raises:
            ValueError: If start_date is not a date or lies before the historical data.
        """
        dates = self._prices.index

        if start_date is None:
            return dates[-1]

        try:
            requested = pd.Period(start_date, freq="D")
        except ValueError as error:
            raise ValueError(
                f"The start date {start_date} is not a valid date."
            ) from error

        position = dates.searchsorted(requested, side="right") - 1

        if position < 0:
            raise ValueError(
                f"The start date {start_date} lies before the historical data, which starts on "
                f"{dates[0]}."
            )

        if dates[position] != requested:
            logger.info(
                "There are no prices on %s, so the option values start from %s.",
                start_date,
                dates[position],
            )

        return dates[position]

    def _dividend_yield_on(self, ticker: str, date) -> float:
        """
        The dividend yield of a ticker on a pricing date: the dividends of the 365 days up to
        that date over its close, or the latest annual yield when there are no daily prices,
        converted to the continuously compounded ln(1 + q) the option formulas discount with.

        Args:
            ticker (str): The ticker.
            date: The pricing date.

        Returns:
            float: The dividend yield as a decimal.
        """
        if (
            not self._daily_dividend_yield.empty
            and ticker in self._daily_dividend_yield
            and date in self._daily_dividend_yield.index
        ):
            value = self._daily_dividend_yield.loc[date, ticker]
            if pd.notna(value):
                return float(np.log1p(value))

        # The trailing yield is a simple annual rate (a year of dividends over the price) while
        # the option formulas carry it continuously through e^(-q * t); ln(1 + q) grows the
        # stock by the same amount over a year, as the risk-free rate conversion does.  # noqa: E501
        return float(np.log1p(self._dividend_yield[ticker].iloc[-1]))

    @handle_errors
    def get_option_chains(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        show_expiration_dates: bool = False,
        rounding: int | None = None,
        strike_price_range: float | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Get the Option Chains which gives information about the currently available
        options as reported by Yahoo Finance. This returns the Contract Symbol, Last
        Trade Date, Last Price, Bid, Ask, Change, Percent Change, Volume, Open Interest,
        Implied Volatility, whether the option is In The Money and the Currency.

        The data comes from Yahoo Finance. When Yahoo Finance has no options for a ticker,
        Cboe's delayed quotes (15 minutes) are used, which cover every option listed on a US
        equity or index, such as "^SPX" with expiries several years out. If neither has
        data, it is advised to use the theoretical calculations as provided by the
        Black Scholes Model as well as the Greeks to get a better understanding of the
        option prices over time.

        Also known as: calls, puts, strike prices, expiry dates, option data.

        Args:
            expiration_date (str | None, optional): The expiration date to use. Defaults to None which means it will
                use the first available expiration date.
            put_option (bool, optional): Whether to show the put options instead of the call options.
                Defaults to False.
            show_expiration_dates (bool, optional): Whether to return the available expiration dates instead of
                the option chains. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            strike_price_range (float | None, optional): Only the strikes within this fraction of the
                current stock price, e.g. 0.05 for strikes within 5% of it. Defaults to None, which
                keeps every strike.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: the option chains with the tickers and strike prices as the index
            and the contract details (Contract Symbol, Last Trade Date, Last Price, Bid, Ask,
            Change, Percent Change, Volume, Open Interest, Implied Volatility, In The Money
            and Currency) as the columns. The expiration date is stored as the DataFrame's
            name. If show_expiration_dates is True, it returns the list of expiration dates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        option_chains = toolkit.options.get_option_chains(strike_price_range=0.05)

        option_chains.loc["MSFT", ["Bid", "Ask"]]
        ```

        Which returns:

        |   Strike Price |   Bid |   Ask |
        |---------------:|------:|------:|
        |          500   | 23.45 | 25.15 |
        |          502.5 | 21    | 23.35 |
        |          505   | 18.5  | 21    |
        |          507.5 | 16.15 | 18.5  |
        |          510   | 13.7  | 16.1  |
        |          512.5 | 11.85 | 13.65 |
        |          515   |  9.7  | 11.6  |
        |          517.5 |  7.65 |  8.9  |
        |          520   |  5.8  |  6.75 |
        |          522.5 |  4.2  |  4.9  |
        |          525   |  2.87 |  3.15 |
        |          527.5 |  1.83 |  2.18 |
        |          530   |  1.07 |  1.19 |
        |          532.5 |  0.69 |  0.7  |
        |          535   |  0.32 |  0.37 |
        |          537.5 |  0.16 |  0.24 |
        |          540   |  0.09 |  0.12 |
        |          542.5 |  0.05 |  0.08 |
        |          545   |  0.03 |  0.05 |
        |          547.5 |  0.02 |  0.03 |
        |          550   |  0.01 |  0.03 |
        """
        expiry_dates = options_model.get_option_expiry_dates(ticker=self._tickers[0])

        if show_expiration_dates:
            return expiry_dates

        if expiration_date is None:
            expiration_date = expiry_dates[0]
        elif expiration_date not in expiry_dates:
            raise ValueError(
                f"The expiration date {expiration_date} is not a valid date. Choose from {', '.join(expiry_dates)}"
            )

        option_chains = options_model.get_option_chains(
            tickers=self._tickers,
            expiration_date=expiration_date,
            put_option=put_option,
        )

        option_chains["Change"] = option_chains["Change"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )
        option_chains["Percent Change"] = option_chains["Percent Change"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )
        option_chains["Implied Volatility"] = option_chains["Implied Volatility"].pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if strike_price_range is not None:
            # The strikes around the current price, where the trading (and the question) is.
            current_prices = pd.Series(
                {
                    ticker: self._spot_price(ticker, option_chains.loc[ticker])
                    for ticker in option_chains.index.get_level_values(0).unique()
                }
            )
            spot = option_chains.index.get_level_values(0).map(current_prices)
            strikes = option_chains.index.get_level_values(1)
            option_chains = option_chains[
                (strikes >= spot * (1 - strike_price_range))
                & (strikes <= spot * (1 + strike_price_range))
            ]

        option_chains = option_chains.drop(columns="Underlying Price", errors="ignore")

        option_chains.name = expiration_date

        return option_chains

    def _spot_price(self, ticker: str, *chains: pd.DataFrame) -> float:
        """
        The price of the underlying to combine with live option chains: the price quoted
        with the chain (Yahoo Finance's regular market price or Cboe's current price) when
        the source provides one, otherwise the traded Close of the last historical date.
        The Adjusted Close is not used, since it is scaled down for later dividends and so
        is not a price the strikes can be compared with.
        """
        for chain in chains:
            if "Underlying Price" in chain:
                quoted = pd.to_numeric(chain["Underlying Price"], errors="coerce")
                quoted = quoted[quoted > 0]
                if not quoted.empty:
                    return float(quoted.iloc[-1])
        prices = (
            self._daily_historical["Close"]
            if "Close" in self._daily_historical.columns.get_level_values(0)
            else self._prices
        )
        return float(prices[ticker].dropna().iloc[-1])
