"""Options Module"""

__docformat__ = "google"


from collections.abc import Callable
from functools import cached_property

import numpy as np
import pandas as pd

from financetoolkit.options import (
    binomial_trees_model,
    black_scholes_model,
    exotics_model,
    greeks_model,
    helpers,
    market_metrics_model,
    options_model,
    risk_neutral_density_model,
    svi_model,
)
from financetoolkit.ratios import valuation_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_standardization,
    to_period_index,
)

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals,cell-var-from-loop
# pylint: disable=line-too-long,too-many-public-methods
# ruff: noqa: E501

logger = logger_model.get_logger()

# The volatility is measured over the trading year before the pricing date, once at least a
# month of returns is available.
TRAILING_DAYS = 252
MINIMUM_VOLATILITY_DAYS = 21

MINIMUM_OBSERVATIONS_FOR_SVI_FIT = 5

# The number of calendar days the last historical price may lag today before the spot price
# is treated as stale next to the live option chains (a weekend plus a holiday is four).
MAXIMUM_SPOT_AGE_DAYS = 5
MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK = 2

# What the market metrics take from the historical data while the option chains are live.
_CHAIN_INPUTS_FROM_HISTORY = (
    "the risk free rate, the dividend yield and, when a chain quotes no price of "
    "the underlying, the stock price come from that date"
)


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


class Options:
    """
    The Options module is meant to calculate important options metrics such as the
    First, Second and Third Order Greeks, the Black Scholes Model and the Option Chains as well as
    Implied Volatilities, Breeden—Litzenberger and more.
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

    def _warn_if_spot_is_stale(self) -> None:
        """
        Logs a warning when the last historical date lies more than MAXIMUM_SPOT_AGE_DAYS
        calendar days before today. The methods that work with the live option chains
        (implied volatility, the volatility surface, the risk-neutral density and the option
        chains filtered on strike) take the quotes and the days to expiration from today but
        the spot price from the last historical date, so an end_date in the past or an
        outdated price history pairs today's option prices with an old stock price. The
        history of an instance does not change, so the warning is shown once per instance
        rather than once for every expiration date a surface walks through.
        """
        if getattr(self, "_stale_spot_warning_shown", False):
            return

        last_date = self._daily_historical.index[-1]
        last_timestamp = (
            last_date.to_timestamp()
            if isinstance(last_date, pd.Period)
            else pd.Timestamp(last_date)
        )
        spot_age = (pd.Timestamp.today().normalize() - last_timestamp.normalize()).days

        if spot_age > MAXIMUM_SPOT_AGE_DAYS:
            self._stale_spot_warning_shown = True
            logger.warning(
                "The stock price is the close of %s, %d days ago, while the option chains and "
                "the days to expiration are live. Implied volatilities and densities from these "
                "chains are therefore based on an outdated stock price; set end_date to today "
                "or leave it empty for current results.",
                last_timestamp.date(),
                spot_age,
            )

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

    def get_option_chains(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        show_expiration_dates: bool = False,
        rounding: int | None = None,
        strike_price_range: float | None = None,
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            strike_price_range (float | None, optional): Only the strikes within this fraction of the
                current stock price, e.g. 0.05 for strikes within 5% of it. Defaults to None, which
                keeps every strike.

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

    def get_black_scholes_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the Black Scholes Model, a mathematical model used to estimate the price of European—style options.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options.
        It is widely used by traders and investors to determine the theoretical value of an option, and to
        assess the potential risks and rewards of a position.

        Within Risk Management, defining the theoretical value of an option is important to assess the potential
        risk and rewards of an option position. A position that could be used to hedge a portfolio, for example,
        is a long put option. The theoretical value of this option can be used to determine the potential risk
        and rewards of this position.

        The Black Scholes Model is based on several assumptions, including the following:

        - The option is European and can only be exercised at expiration.
        - The underlying stock follows a lognormal distribution.
        - The risk—free rate and volatility of the underlying stock are known and constant.
        - The returns on the underlying stock are normally distributed.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        The formulas are as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Option Price = S * e^(—q * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The model price is only as good as its volatility input, and here that input is realised (historical)
        volatility, while traded options are priced off implied volatility, which usually differs from it and varies
        by strike and expiry. Practitioners therefore use the Black Scholes value as a benchmark and compare it with
        market prices from `get_option_chains`, or back the market's volatility out with `get_implied_volatility`.
        Deep out-of-the-money options are most sensitive to that choice, since the model's flat volatility ignores
        the skew.

        Also known as: BSM, Black-Scholes-Merton, option pricing model.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Black Scholes values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        black_scholes = toolkit.options.get_black_scholes_model(
            strike_price_range=0.15, strike_step_size=5, expiration_time_range=91
        )

        black_scholes.loc["NVDA", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            160 |      28.7159 |      31.761  |      34.5215 |
        |            165 |      24.6249 |      28.1288 |      31.1403 |
        |            170 |      20.8373 |      24.7538 |      27.9815 |
        |            175 |      17.3911 |      21.6463 |      25.0482 |
        |            180 |      14.3123 |      18.8111 |      22.34   |
        |            185 |      11.6125 |      16.2474 |      19.8537 |
        |            190 |       9.2892 |      13.9492 |      17.5835 |
        |            195 |       7.3269 |      11.9066 |      15.5214 |
        |            200 |       5.6997 |      10.106  |      13.6576 |
        |            205 |       4.3742 |       8.5311 |      11.9811 |
        |            210 |       3.313  |       7.1639 |      10.4798 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        black_scholes: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            black_scholes[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            black_scholes[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: black_scholes_model.get_black_scholes(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    time_to_expiration=time_to_expiration,
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        black_scholes_df = helpers.create_greek_dataframe(
            greek_dictionary=black_scholes,
            start_date=start_date,
        )

        black_scholes_df = black_scholes_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            black_scholes_df = calculate_standardization(
                dataset=black_scholes_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return black_scholes_df

    def get_implied_volatility(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_expiration_dates: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the Implied Volatility (IV) based on the Black Scholes Model and the actual option prices for
        any of the available expiration dates.

        Implied Volatility (IV) is a measure of how much the market expects the price of the underlying asset to
        fluctuate in the future. It is a key component of options pricing and can also be used to calculate the
        theoretical value of an option.

        The most recent stock price, risk free rate and dividend yield are used, and the implied volatility is the
        volatility at which the Black Scholes price equals the option's last traded price.

        The formulas are as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Option Price = S * e^(—q * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        In which the Implied Volatility is then calculated as follows:

        - Implied Volatility = the σ for which Black Scholes Theoretical Price(σ) — Actual Option Price = 0

        To determine the Implied Volatility, the Black Scholes Model is used to calculate the theoretical option price in
        which sigma (σ) is the only unknown variable. Since that price rises strictly with σ, the implied volatility is
        the root of the difference between the theoretical and actual option price, found with Brent's method on the
        interval [1e-6, 5]. Prices outside the no-arbitrage bounds have no root and return NaN.

        The implied volatility is solved from each option's last traded price, which can be hours or days old for
        illiquid strikes, while the stock price is the latest close. Deep in-the-money and far out-of-the-money
        strikes therefore often show extreme values (above 100%) that say more about stale quotes than about the
        market's view. Practitioners read the smile around the money, where trading is liquid, and use
        `get_volatility_surface`, which filters outliers and fits a smooth curve, when they need values across all
        strikes.

        The quotes and the days to expiration are live, while the stock price is the close of the last historical
        date. When that date lies more than five days before today (for example with an end_date in the past), a
        warning is logged, since the implied volatilities then pair today's option prices with an old stock price.

        Also known as: IV, option-implied volatility.

        Args:
            expiration_date (str | None, optional): The expiration date to use for the calculation. Defaults to None
            which means it will use the most recent expiration date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_expiration_dates (bool, optional): Whether to show the expiration dates. Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.Series | list[str]: Implied Volatility values indexed by ticker and strike price, with the expiration
            date as the Series name. If show_expiration_dates is True, it returns the list of expiration dates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        implied_volatility = toolkit.options.get_implied_volatility()

        implied_volatility.loc["AAPL"].tail(20)
        ```

        Which returns:

        |   Strike Price |   2026-10-26 |
        |---------------:|-------------:|
        |          337.5 |       0.1868 |
        |          340   |       0.1847 |
        |          342.5 |       0.1965 |
        |          345   |       0.1994 |
        |          347.5 |       0.1877 |
        |          350   |       0.1839 |
        |          352.5 |       0.179  |
        |          355   |       0.1778 |
        |          357.5 |       0.1914 |
        |          360   |       0.1865 |
        |          362.5 |       0.2381 |
        |          365   |       0.2723 |
        |          367.5 |       0.2829 |
        |          370   |       0.3138 |
        |          372.5 |       0.3441 |
        |          375   |       0.4376 |
        |          377.5 |       0.4035 |
        |          380   |       0.4326 |
        |          385   |       0.4896 |
        |          410   |       0.7561 |
        """
        if expiration_date is None and not show_expiration_dates:
            # An expiry of today or tomorrow has no time left to price, so the first
            # expiry at least a day away is used.
            expiration_date = next(
                (
                    date
                    for date in self.get_option_chains(show_expiration_dates=True)
                    if _days_to_expiration(date) >= 1
                ),
                None,
            )

        option_chains = self.get_option_chains(
            expiration_date=expiration_date,
            show_expiration_dates=show_expiration_dates,
            put_option=put_option,
        )

        if show_expiration_dates:
            # While the name implies option chains, it actually returns the expiration dates
            return option_chains

        self._warn_if_spot_is_stale()

        current_period = self._daily_historical.index[-1]
        stock_price = self._prices.loc[current_period]
        volatility = self._volatility.loc[current_period]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[current_period]
        )

        tickers = option_chains.index.get_level_values(0).unique()
        dividend_yield_value: dict[str, float] = {}
        implied_volatility: dict[str, dict[float, float]] = {}

        for ticker in tickers:
            implied_volatility[ticker] = {}
            option_chain = option_chains.loc[ticker]
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, current_period)
            )

            for strike_price, row in option_chain.iterrows():
                # Days to expiration feed the time to expiration in the Black-Scholes model.
                days_to_expiration = _days_to_expiration(option_chains.name)

                # Numerically finds the volatility matching the market option price.
                implied_volatility_value = black_scholes_model.get_implied_volatility(
                    market_price=row["Last Price"],
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    time_to_expiration=days_to_expiration / 365,
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                )

                # The solver returns NaN when the observed price admits no solution, for instance when it sits outside the no-arbitrage bounds or never traded.
                if not pd.isna(implied_volatility_value):
                    implied_volatility[ticker][strike_price] = implied_volatility_value

        # Ordered by ticker and strike price, as the option chain itself is.
        implied_volatility_df = (
            pd.DataFrame(implied_volatility).unstack().dropna().sort_index()
        )
        # Named like the option chains, so the strike price is not mistaken for a metric.
        implied_volatility_df.index.names = ["Ticker", "Strike Price"]

        implied_volatility_df = apply_rounding(
            implied_volatility_df, rounding if rounding is not None else self._rounding
        )

        if standardize:
            implied_volatility_df = calculate_standardization(
                dataset=implied_volatility_df,
                rounding=rounding if rounding is not None else self._rounding,
            )

        # The Expiration date is used as the name of the DataFrame
        implied_volatility_df.name = option_chains.name

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return implied_volatility_df

    def get_volatility_surface(
        self,
        expiration_dates: list[str] | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        number_of_expirations: int = 6,
        outlier_threshold: float = 5.0,
        rounding: int | None = None,
    ):
        """
        Calibrate an arbitrage-checked implied volatility surface across multiple
        expiries, by fitting a raw SVI (Stochastic Volatility Inspired, Gatheral
        2004) curve to the market-implied smile (see `get_implied_volatility`) at
        each expiry, and checking the fitted surface for calendar-spread arbitrage.

        A single per-expiry smile only tells you the shape of the market's
        volatility skew at that one maturity. Stitching several calibrated SVI
        slices together instead gives a full surface, which is what is needed to
        price/interpolate options at maturities or strikes that don't trade
        directly, and to check for term-structure inconsistencies (see Notes).

        Before fitting, strikes whose market-implied volatility is a statistical
        outlier relative to its neighbors (a common symptom of a stale or
        wide-bid/ask illiquid quote) are dropped via a median-absolute-deviation
        filter, since a single bad quote can otherwise dominate the least-squares
        SVI fit for that whole expiry. Each fit keeps the total variance non-negative
        through Gatheral's condition a + b * sigma * sqrt(1 - rho^2) >= 0.

        The option chains and the days to expiration are live, while the stock price
        is the close of the last historical date; a warning is logged when that date
        lies more than five days before today.

        See: Gatheral, J. (2004), "A parsimonious arbitrage-free implied volatility
        parameterization with application to the valuation of volatility
        derivatives", and Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI
        volatility surfaces", Quantitative Finance, 14(1), 59-71.

        Also known as: SVI surface, implied volatility surface.

        Notes:
            A warning is logged (not raised) if the fitted surface has any
            calendar-spread arbitrage violations, i.e. total implied variance
            decreasing with time to expiration at some log-moneyness -- this
            reflects genuine inconsistency in the underlying market quotes across
            expiries, not a fitting error, and is only checked, not corrected.

        Args:
            expiration_dates (list[str] | None, optional): The expiration dates to
                fit the surface over. Defaults to None, meaning the first
                `number_of_expirations` available expiration dates.
            put_option (bool, optional): Whether to use put options instead of call
                options. Defaults to False.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
                uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
                risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
                quote to continuous compounding. A rate passed here is used as given, as a continuously compounded
                rate.
            dividend_yield (float, optional): The dividend yield to use for the
                calculation. Defaults to None, which uses the dividend yield on the
                pricing date (the dividends of the 365 days up to it relative to its close,
                as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            number_of_expirations (int, optional): The number of near-term
                expiration dates to fit when `expiration_dates` is not given.
                Defaults to 6.
            outlier_threshold (float, optional): The number of median absolute
                deviations from the median implied volatility beyond which a quote
                is treated as an outlier and dropped before fitting. Defaults to
                5.0.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.

        Returns:
            pd.DataFrame: The SVI-fitted implied volatility, indexed by (ticker,
            strike price), with one column per expiration date. NaN where a given
            strike wasn't part of that expiry's calibration.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["JPM"], api_key="FINANCIAL_MODELING_PREP_KEY")

        volatility_surface = toolkit.options.get_volatility_surface(number_of_expirations=4)

        volatility_surface.loc["JPM"]
        ```

        Which returns:

        |   Strike Price |   2026-10-30 |   2026-11-06 |   2026-11-13 |   2026-11-20 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|
        |          155   |     nan      |     nan      |     nan      |       1.3993 |
        |          160   |     nan      |     nan      |     nan      |       1.3572 |
        |          165   |     nan      |     nan      |     nan      |       1.3159 |
        |          170   |     nan      |     nan      |     nan      |       1.2756 |
        |          180   |     nan      |     nan      |     nan      |       1.1971 |
        |          185   |     nan      |     nan      |     nan      |       1.159  |
        |          190   |     nan      |     nan      |     nan      |       1.1216 |
        |          195   |     nan      |     nan      |     nan      |       1.0848 |
        |          200   |     nan      |     nan      |     nan      |       1.0487 |
        |          205   |     nan      |     nan      |     nan      |       1.0132 |
        |          210   |     nan      |     nan      |     nan      |       0.9782 |
        |          215   |     nan      |     nan      |     nan      |       0.9438 |
        |          220   |     nan      |     nan      |     nan      |       0.91   |
        |          230   |     nan      |     nan      |     nan      |       0.8439 |
        |          235   |     nan      |     nan      |     nan      |       0.8117 |
        |          240   |     nan      |     nan      |     nan      |       0.7799 |
        |          245   |     nan      |     nan      |     nan      |       0.7487 |
        |          250   |     nan      |     nan      |     nan      |       0.7179 |
        |          255   |     nan      |     nan      |     nan      |       0.6876 |
        |          260   |     nan      |     nan      |     nan      |       0.6578 |
        |          265   |     nan      |     nan      |     nan      |       0.6286 |
        |          270   |     nan      |     nan      |     nan      |       0.5998 |
        |          280   |     nan      |       0.7398 |     nan      |       0.5438 |
        |          285   |     nan      |     nan      |     nan      |       0.5166 |
        |          290   |     nan      |     nan      |     nan      |       0.49   |
        |          295   |     nan      |     nan      |     nan      |       0.464  |
        |          300   |       0.4147 |     nan      |       0.33   |       0.4387 |
        |          305   |       0.3937 |       0.5749 |     nan      |       0.4142 |
        |          310   |       0.3722 |       0.5386 |     nan      |       0.3904 |
        |          312.5 |       0.3613 |     nan      |     nan      |     nan      |
        |          315   |       0.3502 |       0.5006 |     nan      |       0.3675 |
        |          320   |       0.3276 |       0.4609 |       0.2749 |       0.3456 |
        |          322.5 |       0.3161 |     nan      |     nan      |     nan      |
        |          325   |       0.3045 |       0.4193 |       0.262  |       0.3249 |
        |          327.5 |       0.2929 |     nan      |     nan      |     nan      |
        |          330   |       0.2812 |       0.3755 |       0.25   |       0.3054 |
        |          332.5 |       0.2698 |     nan      |     nan      |     nan      |
        |          335   |       0.2586 |       0.3298 |       0.2392 |       0.2875 |
        |          337.5 |       0.2479 |     nan      |     nan      |     nan      |
        |          340   |       0.2382 |       0.2836 |       0.2303 |       0.2712 |
        |          342.5 |       0.2299 |     nan      |     nan      |     nan      |
        |          345   |       0.224  |       0.2408 |       0.2239 |       0.257  |
        |          347.5 |       0.2215 |     nan      |     nan      |     nan      |
        |          350   |       0.2238 |       0.2104 |       0.2211 |       0.245  |
        |          352.5 |       0.2321 |     nan      |     nan      |     nan      |
        |          355   |       0.2469 |       0.2032 |       0.2229 |       0.2355 |
        |          357.5 |       0.2679 |     nan      |     nan      |     nan      |
        |          360   |       0.2936 |       0.22   |       0.2303 |       0.2288 |
        |          365   |       0.3515 |       0.2504 |       0.2439 |       0.2251 |
        |          370   |       0.4099 |       0.2853 |       0.264  |       0.2244 |
        |          375   |       0.4647 |       0.3202 |       0.2899 |       0.2267 |
        |          380   |       0.5153 |       0.3537 |       0.3206 |       0.2318 |
        |          385   |       0.5619 |       0.3852 |       0.3546 |       0.2396 |
        |          390   |       0.6051 |       0.4149 |     nan      |       0.2497 |
        |          395   |       0.6452 |     nan      |     nan      |       0.2618 |
        |          400   |       0.6829 |     nan      |     nan      |       0.2756 |
        |          405   |       0.7183 |     nan      |     nan      |       0.2908 |
        |          410   |       0.7517 |       0.5176 |     nan      |       0.3072 |
        |          415   |       0.7835 |     nan      |     nan      |       0.3246 |
        |          420   |     nan      |     nan      |     nan      |       0.3428 |
        |          425   |       0.8425 |     nan      |     nan      |       0.3616 |
        |          430   |     nan      |     nan      |     nan      |       0.3809 |
        |          435   |     nan      |     nan      |     nan      |       0.4007 |
        |          440   |     nan      |     nan      |     nan      |       0.4208 |
        |          445   |     nan      |     nan      |     nan      |       0.4412 |
        |          450   |     nan      |     nan      |     nan      |       0.4618 |
        |          455   |     nan      |     nan      |     nan      |       0.4826 |
        |          460   |     nan      |     nan      |     nan      |       0.5035 |
        |          465   |     nan      |     nan      |     nan      |       0.5246 |
        |          470   |     nan      |     nan      |     nan      |       0.5457 |
        """
        all_expiration_dates = self.get_option_chains(show_expiration_dates=True)

        if expiration_dates is None:
            expiration_dates = list(all_expiration_dates[:number_of_expirations])
        else:
            invalid_dates = [
                expiration_date
                for expiration_date in expiration_dates
                if expiration_date not in all_expiration_dates
            ]
            if invalid_dates:
                raise ValueError(
                    f"The expiration date(s) {', '.join(invalid_dates)} are not valid. "
                    f"Choose from {', '.join(all_expiration_dates)}"
                )

        self._warn_if_spot_is_stale()

        current_period = self._daily_historical.index[-1]
        stock_price = self._prices.loc[current_period]
        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[current_period]
        )
        surface: dict[str, dict[float, dict[str, float]]] = {}
        svi_parameters_per_ticker: dict[str, dict[float, dict[str, float]]] = {
            ticker: {} for ticker in self._tickers
        }

        for expiration_date in expiration_dates:
            time_to_expiration = _days_to_expiration(expiration_date) / 365

            if time_to_expiration <= 0:
                continue

            implied_volatility = self.get_implied_volatility(
                expiration_date=expiration_date,
                put_option=put_option,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )

            if implied_volatility.empty:
                continue

            for ticker in implied_volatility.index.get_level_values(0).unique():
                ticker_iv = implied_volatility.loc[ticker]

                median_iv = ticker_iv.median()
                deviation = (ticker_iv - median_iv).abs()
                mad = deviation.median()
                if mad > 0:
                    ticker_iv = ticker_iv[deviation <= outlier_threshold * mad]

                if len(ticker_iv) < MINIMUM_OBSERVATIONS_FOR_SVI_FIT:
                    continue

                dividend_yield_value = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, current_period)
                )

                forward_price = stock_price.loc[ticker] * np.exp(
                    (risk_free_rate - dividend_yield_value) * time_to_expiration
                )

                log_moneyness = np.log(
                    ticker_iv.index.to_numpy(dtype=float) / forward_price
                )
                total_variance = (ticker_iv.to_numpy() ** 2) * time_to_expiration

                try:
                    fitted_parameters = svi_model.get_svi_parameters(
                        log_moneyness, total_variance
                    )
                except ValueError:
                    continue

                svi_parameters_per_ticker[ticker][
                    time_to_expiration
                ] = fitted_parameters

                fitted_volatility = svi_model.get_svi_implied_volatility(
                    log_moneyness, time_to_expiration, **fitted_parameters
                )

                surface.setdefault(ticker, {})
                for strike_price, volatility in zip(ticker_iv.index, fitted_volatility):
                    surface[ticker].setdefault(strike_price, {})[
                        expiration_date
                    ] = volatility

        for ticker, expirations in svi_parameters_per_ticker.items():
            if len(expirations) < MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK:
                continue

            violations = svi_model.check_calendar_arbitrage(expirations)

            if not violations.empty:
                logger.warning(
                    "The calibrated volatility surface for %s has %d calendar-spread "
                    "arbitrage violation(s) across the checked log-moneyness grid -- "
                    "this reflects genuine inconsistency in the underlying market "
                    "quotes across expiries, not a fitting error.",
                    ticker,
                    len(violations),
                )

        if not surface:
            return pd.DataFrame()

        volatility_surface = pd.concat(
            {
                ticker: pd.DataFrame(strikes).T.sort_index()
                for ticker, strikes in surface.items()
            }
        )
        volatility_surface.index.names = ["Ticker", "Strike Price"]
        # In expiration order, whatever order the tickers' expiries were collected in.
        volatility_surface = volatility_surface[sorted(volatility_surface.columns)]

        return apply_rounding(
            volatility_surface, rounding if rounding is not None else self._rounding
        )

    def get_risk_neutral_density(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        strike_price_range: float = 0.5,
        number_of_strikes: int = 200,
        outlier_threshold: float = 5.0,
        rounding: int | None = None,
    ):
        """
        Extract the market-implied risk-neutral probability density of the
        underlying's price at expiration, via the Breeden-Litzenberger (1978)
        theorem, applied to a volatility smile calibrated to real market option
        prices (see `get_implied_volatility`) rather than a single flat assumed
        volatility.

        `get_partial_derivative` computes the same second-derivative relationship
        but with one flat volatility value applied at every strike -- with a flat
        volatility input, the second derivative can only ever recover a lognormal
        density regardless of what the real market smile looks like, which defeats
        the entire purpose of the theorem. This method instead first calibrates a
        raw SVI (Gatheral 2004) curve to the actual market smile (see
        `get_volatility_surface`) and evaluates the density from that.

        The formula is as follows:

        - f(K) = e^(r * t) * d^2 C(K) / dK^2

        Where C(K) is the Black-Scholes call price at strike K, using the
        SVI-smoothed implied volatility at that strike, r is the risk-free rate and
        t is the time to expiration. The second derivative is approximated
        numerically via the three-point central difference
        (C(K - h) - 2 * C(K) + C(K + h)) / h^2 on a fine, evenly-spaced strike grid
        with spacing h, since the smile only gives implied volatility at a sparse set
        of traded strikes. The first and last grid points have no neighbor on one side
        and are left out.

        The option chains and the days to expiration are live, while the stock price is
        the close of the last historical date; a warning is logged when that date lies
        more than five days before today.

        See the paper: Breeden, D.T., & Litzenberger, R.H. (1978), "Prices of
        State-Contingent Claims Implicit in Option Prices", Journal of Business,
        51(4), 621-651. https://www.jstor.org/stable/2352653

        Also known as: Breeden-Litzenberger, implied risk-neutral distribution.

        Notes:
            A warning is logged (not raised) for any ticker whose density goes
            negative at some strike -- this indicates a butterfly-arbitrage
            violation (the call price is not convex in the strike) in the
            underlying market quotes or the SVI fit, which a well-calibrated,
            liquid smile should not produce.

        Args:
            expiration_date (str | None, optional): The expiration date to use.
                Defaults to None which means it will use the first available
                expiration date.
            put_option (bool, optional): Whether to use put options instead of call
                options. Defaults to False.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
                uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
                risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
                quote to continuous compounding. A rate passed here is used as given, as a continuously compounded
                rate.
            dividend_yield (float, optional): The dividend yield to use for the
                calculation. Defaults to None, which uses the dividend yield on the
                pricing date (the dividends of the 365 days up to it relative to its close,
                as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            strike_price_range (float, optional): The range of strikes to evaluate
                the density over, as a fraction of the forward price in each
                direction. Defaults to 0.5, i.e. from 50% to 150% of the forward
                price.
            number_of_strikes (int, optional): The number of strikes in the
                evaluation grid. Defaults to 200.
            outlier_threshold (float, optional): The number of median absolute
                deviations from the median implied volatility beyond which a quote
                is treated as an outlier and dropped before fitting. Defaults to
                5.0.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.

        Raises:
            ValueError: If no implied volatility could be determined for the given
                expiration date.

        Returns:
            pd.DataFrame: The risk-neutral probability density, indexed by strike
            price, with one column per ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["SPY"], api_key="FINANCIAL_MODELING_PREP_KEY", benchmark_ticker=None)

        toolkit.options.get_risk_neutral_density(strike_price_range=0.1, number_of_strikes=81)
        ```

        Which returns:

        |   Strike Price |    SPY |
        |---------------:|-------:|
        |        686.021 | 0      |
        |        687.921 | 0      |
        |        689.821 | 0      |
        |        691.722 | 0      |
        |        693.622 | 0.0001 |
        |        695.522 | 0.0001 |
        |        697.423 | 0.0001 |
        |        699.323 | 0.0001 |
        |        701.223 | 0.0001 |
        |        703.124 | 0.0001 |
        |        705.024 | 0.0001 |
        |        706.924 | 0.0001 |
        |        708.825 | 0.0001 |
        |        710.725 | 0.0001 |
        |        712.625 | 0.0002 |
        |        714.526 | 0.0002 |
        |        716.426 | 0.0002 |
        |        718.326 | 0.0002 |
        |        720.227 | 0.0003 |
        |        722.127 | 0.0003 |
        |        724.028 | 0.0004 |
        |        725.928 | 0.0005 |
        |        727.828 | 0.0006 |
        |        729.729 | 0.0007 |
        |        731.629 | 0.0009 |
        |        733.529 | 0.0011 |
        |        735.43  | 0.0015 |
        |        737.33  | 0.0019 |
        |        739.23  | 0.0024 |
        |        741.131 | 0.0032 |
        |        743.031 | 0.0043 |
        |        744.931 | 0.0058 |
        |        746.832 | 0.008  |
        |        748.732 | 0.011  |
        |        750.632 | 0.0151 |
        |        752.533 | 0.0209 |
        |        754.433 | 0.0284 |
        |        756.333 | 0.0377 |
        |        758.234 | 0.048  |
        |        760.134 | 0.0573 |
        |        762.034 | 0.0626 |
        |        763.935 | 0.0612 |
        |        765.835 | 0.0525 |
        |        767.735 | 0.0392 |
        |        769.636 | 0.0256 |
        |        771.536 | 0.015  |
        |        773.436 | 0.0081 |
        |        775.337 | 0.0042 |
        |        777.237 | 0.0022 |
        |        779.137 | 0.0011 |
        |        781.038 | 0.0006 |
        |        782.938 | 0.0003 |
        |        784.838 | 0.0002 |
        |        786.739 | 0.0001 |
        |        788.639 | 0.0001 |
        |        790.539 | 0.0001 |
        |        792.44  | 0      |
        |        794.34  | 0      |
        |        796.24  | 0      |
        |        798.141 | 0      |
        |        800.041 | 0      |
        |        801.941 | 0      |
        |        803.842 | 0      |
        |        805.742 | 0      |
        |        807.642 | 0      |
        |        809.543 | 0      |
        |        811.443 | 0      |
        |        813.343 | 0      |
        |        815.244 | 0      |
        |        817.144 | 0      |
        |        819.044 | 0      |
        |        820.945 | 0      |
        |        822.845 | 0      |
        |        824.745 | 0      |
        |        826.646 | 0      |
        |        828.546 | 0      |
        |        830.446 | 0      |
        |        832.347 | 0      |
        |        834.247 | 0      |
        """
        if expiration_date is not None:
            candidate_dates = [expiration_date]
        else:
            # Illiquid near-term expiries resolve nothing, so pick the first usable date.
            candidate_dates = self.get_option_chains(show_expiration_dates=True)

        for candidate_date in candidate_dates:
            implied_volatility = self.get_implied_volatility(
                expiration_date=candidate_date,
                put_option=put_option,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )

            if not implied_volatility.empty:
                expiration_date = candidate_date
                break
        else:
            raise ValueError(
                f"No implied volatility could be determined for expiration date(s) "
                f"{', '.join(candidate_dates)}."
            )

        self._warn_if_spot_is_stale()

        current_period = self._daily_historical.index[-1]
        stock_price = self._prices.loc[current_period]
        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[current_period]
        )
        time_to_expiration = _days_to_expiration(expiration_date) / 365

        density: dict[str, pd.Series] = {}

        for ticker in implied_volatility.index.get_level_values(0).unique():
            ticker_iv = implied_volatility.loc[ticker]

            median_iv = ticker_iv.median()
            deviation = (ticker_iv - median_iv).abs()
            mad = deviation.median()
            if mad > 0:
                ticker_iv = ticker_iv[deviation <= outlier_threshold * mad]

            if len(ticker_iv) < MINIMUM_OBSERVATIONS_FOR_SVI_FIT:
                logger.warning(
                    "Not enough option quotes remain for %s after outlier "
                    "filtering to calibrate a smile, skipping.",
                    ticker,
                )
                continue

            dividend_yield_value = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, current_period)
            )

            forward_price = stock_price.loc[ticker] * np.exp(
                (risk_free_rate - dividend_yield_value) * time_to_expiration
            )

            log_moneyness = np.log(
                ticker_iv.index.to_numpy(dtype=float) / forward_price
            )
            total_variance = (ticker_iv.to_numpy() ** 2) * time_to_expiration

            svi_parameters = svi_model.get_svi_parameters(log_moneyness, total_variance)

            ticker_density = risk_neutral_density_model.get_risk_neutral_density(
                stock_price=stock_price.loc[ticker],
                forward_price=forward_price,
                time_to_expiration=time_to_expiration,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
                svi_parameters=svi_parameters,
                strike_price_range=strike_price_range,
                number_of_strikes=number_of_strikes,
            )

            if (ticker_density < 0).any():
                logger.warning(
                    "The risk-neutral density for %s has negative values at some "
                    "strikes, indicating a butterfly-arbitrage violation in the "
                    "underlying market quotes or the SVI fit.",
                    ticker,
                )

            density[ticker] = ticker_density

        if not density:
            return pd.DataFrame()

        density_df = pd.concat(density, axis=1)
        density_df.index.name = "Strike Price"

        return apply_rounding(
            density_df, rounding if rounding is not None else self._rounding
        )

    def get_binomial_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        american_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        time_to_expiration: float = 1,
        timesteps: int = 10,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the Binomial Option Pricing Model, a mathematical model used to estimate the price of European and
        American style options. It does so by creating a binomial tree of price paths for the underlying asset, and
        then working backwards through the tree to determine the price of the option at each node.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        The formulas are as follows:

        - time step (Δt) = T / n
        - up movement (u) = e^(σ * sqrt(Δt))
        - down movement (d) = 1 / u
        - risk neutral probability (p) = (e^((r — q) * Δt) — d) / (u — d)
        - stock price at each node = S * u^j * d^(n — j)
        - call option price at expiration date = max(S — K, 0)
        - put option price at expiration date = max(K — S, 0)

        For European Style options:

        - call option price at each node = (p * C_u + (1 — p) * C_d) * e^(—r * Δt)
        - put option price at each node = (p * P_u + (1 — p) * P_d) * e^(—r * Δt)

        For American Style options:

        - call option price at each node = max(S — K, (p * C_u + (1 — p) * C_d) * e^(—r * Δt))
        - put option price at each node = max(K — S, (p * P_u + (1 — p) * P_d) * e^(—r * Δt))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, T is the time to expiration, Δt is the length of one time step, j is the number of up movements,
        n is the number of time steps, C_u is the call option price at the up
        movement, C_d is the call option price at the down movement, P_u is the put option price at the up movement and
        P_d is the put option price at the down movement.

        The resulting output is a DataFrame containing the tickers, strike prices and movements as the index and the
        time to expiration as the columns. The movements index contains the number of up movements and the number of
        down movements. The output is the binomial tree displayed in a table. E.g. when using 10 time steps, the
        table for each strike price from each company will contain the actual binomial tree as also depicted
        in the image found here: https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method

        Also known as: binomial tree, lattice model, option pricing.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            american_option (bool, optional): Whether to value an American option, which can be exercised at every
                node, instead of a European option. Defaults to False.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            time_to_expiration (float): The number of years to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Binomial Trees values containing the tickers, strike prices and movements as the index and the
            timesteps as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        binomial_model = toolkit.options.get_binomial_model(
            strike_price_range=0, strike_step_size=1, timesteps=5
        )

        binomial_model.loc["KO"]
        ```

        Which returns:

        |               |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:--------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | (69, 'UUUUU') |        5.821 |       9.067  |      13.6268 |      19.594  |      26.7232 |      34.524  |
        | (69, 'DUUUU') |      nan     |       2.7196 |       4.7251 |       7.9735 |      12.8942 |      19.479  |
        | (69, 'DDUUU') |      nan     |     nan      |       0.7891 |       1.6034 |       3.2581 |       6.6205 |
        | (69, 'DDDUU') |      nan     |     nan      |     nan      |       0      |       0      |       0      |
        | (69, 'DDDDU') |      nan     |     nan      |     nan      |     nan      |       0      |       0      |
        | (69, 'DDDDD') |      nan     |     nan      |     nan      |     nan      |     nan      |       0      |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        binomial_trees: dict[str, dict[float, pd.DataFrame]] = {}
        binomial_trees_statistics: dict[str, dict[float, dict[str, float]]] = {
            "Up Movement": {},
            "Down Movement": {},
            "Risk Neutral Probability": {},
        }

        logger.info("Calculating Binomial Trees")
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            binomial_trees[ticker] = {}

            for strike_price in strike_prices:
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )

                if show_input_info:
                    (
                        binomial_trees[ticker][strike_price],
                        binomial_trees_statistics["Up Movement"][ticker],
                        binomial_trees_statistics["Down Movement"][ticker],
                        binomial_trees_statistics["Risk Neutral Probability"][ticker],
                    ) = binomial_trees_model.get_option_payoffs(
                        stock_price=stock_price.loc[ticker],
                        strike_price=strike_price,
                        years=time_to_expiration,
                        timesteps=timesteps,
                        risk_free_rate=risk_free_rate,
                        volatility=volatility.loc[ticker],
                        dividend_yield=dividend_yield_value[ticker],
                        put_option=put_option,
                        american_option=american_option,
                        show_input_info=show_input_info,
                    )

                binomial_trees[ticker][strike_price] = (
                    binomial_trees_model.get_option_payoffs(
                        stock_price=stock_price.loc[ticker],
                        strike_price=strike_price,
                        years=time_to_expiration,
                        timesteps=timesteps,
                        risk_free_rate=risk_free_rate,
                        volatility=volatility.loc[ticker],
                        dividend_yield=dividend_yield_value[ticker],
                        put_option=put_option,
                        american_option=american_option,
                    )
                )

        binomial_trees_df = helpers.create_binomial_tree_dataframe(
            binomial_tree_dictionary=binomial_trees,
            start_date=start_date,
            time_to_expiration=time_to_expiration,
        )

        binomial_trees_df = binomial_trees_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            binomial_trees_df = calculate_standardization(
                dataset=binomial_trees_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
                up_movement_dict=binomial_trees_statistics["Up Movement"],
                down_movement_dict=binomial_trees_statistics["Down Movement"],
                risk_neutral_probability_dict=binomial_trees_statistics[
                    "Risk Neutral Probability"
                ],
            )

        return binomial_trees_df

    def get_stock_price_simulation(
        self,
        start_date: str | None = None,
        time_to_expiration: float = 1,
        timesteps: int = 10,
        risk_free_rate: float | None = None,
        show_unique_combinations: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Simulate the Stock Price based on the Binomial Model, a mathematical model used to estimate the price of European
        and American style options. It does so by creating a binomial tree of price paths for the underlying asset based
        on the stock price, its volatility and the time to expiration: each step multiplies the price by
        u = e^(σ * sqrt(Δt)) or by d = 1 / u. The stock price is then simulated based on these up and down movements.

        By default the most recent stock price is used, you can alter this by changing the start date. The volatility
        is the annualized standard deviation of the daily returns over the year before that date, so an earlier start
        date only uses what was known on that date.

        The formulas are as follows:

        - up movement (u) = e^(σ * sqrt(Δt))
        - down movement (d) = 1 / u
        - stock price at each node = S * u^j * d^(n — j)

        Where S is the stock price, σ is the volatility, Δt is the length of a time step (the time to
        expiration divided by the number of time steps), j is the number of up movements and n is the
        number of time steps. The risk free rate and the dividend yield do not affect the tree.

        The resulting output is a DataFrame containing the tickers and movements as the index and the
        time to expiration as the columns. The movements index contains the number of up movements and the number of
        down movements. The output is the binomial tree displayed in a table. E.g. when using 10 time steps, the
        table from each company will contain the actual binomial tree's stock prices as also depicted
        in the image found here: https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method

        **Hint:** consider plotting the resulting DataFrame for each company to visualize the binomial tree.
        For example for below's example use `stock_price_simulation.loc['AMD'].T.plot(legend=False)`

        Also known as: binomial tree, stock price tree, stock price path, lattice.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            time_to_expiration (float): The number of years to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate shown with show_input_info; it does not affect the
            simulated prices. Defaults to None, which uses the Toolkit's risk-free rate on the pricing date: the
            10-year Treasury yield unless the Toolkit's risk_free_rate parameter selects '13w', '5y' or '30y',
            converted from its semi-annual bond-equivalent quote to continuous compounding. A rate passed here is used
            as given, as a continuously compounded rate.
            show_unique_combinations (bool, optional): Whether to show the unique combinations of the stock prices.
            Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Simulated Stock Price values containing the tickers and movements as the index and the
            timesteps as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        stock_price_simulation = toolkit.options.get_stock_price_simulation(
            timesteps=5, show_unique_combinations=True
        )

        stock_price_simulation.loc["AMD"]
        ```

        Which returns:

        | Movement   |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUUU      |       214.16 |      280.988 |      368.67  |     483.712  |      634.654 |     832.696  |
        | DUUUU      |       214.16 |      163.226 |      214.16  |     280.988  |      368.67  |     483.712  |
        | DDUUU      |       214.16 |      163.226 |      124.405 |     163.226  |      214.16  |     280.988  |
        | DDDUU      |       214.16 |      163.226 |      124.405 |      94.8177 |      124.405 |     163.226  |
        | DDDDU      |       214.16 |      163.226 |      124.405 |      94.8177 |       72.267 |      94.8177 |
        | DDDDD      |       214.16 |      163.226 |      124.405 |      94.8177 |       72.267 |      55.0796 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        stock_price_simulation: dict[str, pd.DataFrame] = {}
        stock_price_statistics: dict[str, dict[str, float]] = {
            "Up Movement": {},
            "Down Movement": {},
        }

        logger.info("Simulating Stock Prices")
        for ticker in self._tickers:
            (
                up_movement,
                down_movement,
            ) = binomial_trees_model.calculate_up_and_down_movements(
                volatility=volatility.loc[ticker],
                time_delta=time_to_expiration / timesteps,
            )

            stock_price_simulation[ticker] = (
                binomial_trees_model.calculate_stock_prices(
                    stock_price=stock_price.loc[ticker],
                    up_movement=up_movement,
                    down_movement=down_movement,
                    period_length=timesteps,
                    show_unique_combinations=show_unique_combinations,
                )
            )

            stock_price_statistics["Up Movement"][ticker] = up_movement
            stock_price_statistics["Down Movement"][ticker] = down_movement

        stock_price_simulation_df = helpers.create_stock_simulation_dataframe(
            stock_simulation_dictonary=stock_price_simulation,
            start_date=start_date,
            time_to_expiration=time_to_expiration,
        )

        stock_price_simulation_df = stock_price_simulation_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            stock_price_simulation_df = calculate_standardization(
                dataset=stock_price_simulation_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=None,
                up_movement_dict=stock_price_statistics["Up Movement"],
                down_movement_dict=stock_price_statistics["Down Movement"],
                risk_neutral_probability_dict=None,
            )

        return stock_price_simulation_df

    def get_put_call_parity(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the Put-Call Parity gap, the amount by which Black-Scholes call and put
        prices deviate from the no-arbitrage relationship between them.

        Put-Call Parity states that, for European options sharing the same strike price
        and time to expiration, the following relationship must hold in order to prevent
        arbitrage: C - P = S * e^(-q * t) - K * e^(-r * t).

        The formula is as follows:

        - Parity Gap = (C - P) - (S * e^(-q * t) - K * e^(-r * t))

        Where C is the Black-Scholes call option price, P is the Black-Scholes put option
        price, S is the stock price, K is the strike price, r is the continuously compounded
        risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)) and t is
        the time to expiration in years (days / 365).

        This method computes the Black-Scholes call and put price for each ticker,
        strike price and time to expiration and then calculates the parity gap, i.e. the
        amount by which (C - P) deviates from S * e^(-qt) - K * e^(-rt). Because both
        prices come from the same Black-Scholes model and inputs, the gap is (up to
        floating point precision) always zero — this is a useful diagnostic to confirm
        that a set of option prices is internally consistent, or, when plugging in
        externally observed call and put prices, to detect potential arbitrage.

        Also known as: Put-Call Parity, PCP, the no-arbitrage relationship between calls
        and puts.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: The Put-Call Parity gap containing the tickers and strike prices as the index and the
            time to expiration as the columns. Values should be (approximately) zero.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JNJ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        parity_gap = toolkit.options.get_put_call_parity(
            strike_price_range=0.1, strike_step_size=5
        )

        parity_gap.loc["JNJ", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |            185 |            0 |            0 |            0 |
        |            190 |            0 |            0 |            0 |
        |            195 |            0 |            0 |            0 |
        |            200 |            0 |            0 |            0 |
        |            205 |            0 |            0 |            0 |
        |            210 |            0 |            0 |            0 |
        |            215 |            0 |            0 |            0 |
        |            220 |            0 |            0 |            0 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        parity_gap: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            parity_gap[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            for strike_price in strike_prices:
                parity_gap[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    call_price = black_scholes_model.get_black_scholes(
                        stock_price=stock_price.loc[ticker],
                        strike_price=strike_price,
                        risk_free_rate=risk_free_rate,
                        volatility=volatility.loc[ticker],
                        time_to_expiration=time_to_expiration,
                        dividend_yield=dividend_yield_value[ticker],
                        put_option=False,
                    )
                    put_price = black_scholes_model.get_black_scholes(
                        stock_price=stock_price.loc[ticker],
                        strike_price=strike_price,
                        risk_free_rate=risk_free_rate,
                        volatility=volatility.loc[ticker],
                        time_to_expiration=time_to_expiration,
                        dividend_yield=dividend_yield_value[ticker],
                        put_option=True,
                    )

                    parity_gap[ticker][strike_price][time_to_expiration] = (
                        black_scholes_model.get_put_call_parity(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            call_price=call_price,
                            put_price=put_price,
                        )
                    )

        parity_gap_df = helpers.create_greek_dataframe(
            greek_dictionary=parity_gap,
            start_date=start_date,
        )

        parity_gap_df = parity_gap_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            parity_gap_df = calculate_standardization(
                dataset=parity_gap_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return parity_gap_df

    def get_garman_kohlhagen(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        foreign_risk_free_rate: float = 0.0,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the Garman-Kohlhagen Model, a variant of the Black-Scholes model used
        to price European-style options on foreign exchange (FX) rates.

        Because holding foreign currency earns the foreign risk-free rate (analogous to a
        continuous dividend yield on a stock), the Garman-Kohlhagen model uses the
        foreign risk-free rate in place of the dividend yield used in the standard
        Black-Scholes model.

        The formulas are as follows:

        - d1 = (ln(S / K) + (r — r_f + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Option Price = S * e^(—r_f * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—r_f * t) * N(—d1)

        Where S is the spot exchange rate, K is the strike price, r is the domestic
        risk-free rate, r_f is the foreign risk-free rate, σ is the volatility, t is the
        time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2)
        is the cumulative normal distribution of d2.

        Also known as: the Black-Scholes model for currency options, FX option pricing
        model.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The domestic risk free rate to use for the calculation. Defaults to
            None, which uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the
            Toolkit's risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual
            bond-equivalent quote to continuous compounding. A rate passed here is used as given, as a continuously
            compounded rate.
            foreign_risk_free_rate (float, optional): The foreign risk free rate to use for the calculation, which
            plays the role of the dividend yield in the standard Black-Scholes model. Defaults to 0.0.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Garman-Kohlhagen values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["USDJPY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        garman_kohlhagen = toolkit.options.get_garman_kohlhagen(
            strike_price_range=0.05,
            strike_step_size=2,
            expiration_time_range=91,
            risk_free_rate=0.0075,
            foreign_risk_free_rate=0.0375,
        )

        garman_kohlhagen.loc["USDJPY", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            148 |       8.4856 |       8.266  |       8.1462 |
        |            150 |       6.5551 |       6.4903 |       6.498  |
        |            152 |       4.7422 |       4.8793 |       5.0189 |
        |            154 |       3.1505 |       3.4884 |       3.741  |
        |            156 |       1.8845 |       2.3571 |       2.683  |
        |            158 |       0.9971 |       1.4973 |       1.8467 |
        |            160 |       0.4601 |       0.8903 |       1.2176 |
        |            162 |       0.1832 |       0.494  |       0.7678 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        garman_kohlhagen: dict[str, dict[float, dict[float, float]]] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            garman_kohlhagen[ticker] = {}

            for strike_price in strike_prices:
                garman_kohlhagen[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    garman_kohlhagen[ticker][strike_price][time_to_expiration] = (
                        black_scholes_model.get_garman_kohlhagen(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            foreign_risk_free_rate=foreign_risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            put_option=put_option,
                        )
                    )

        garman_kohlhagen_df = helpers.create_greek_dataframe(
            greek_dictionary=garman_kohlhagen,
            start_date=start_date,
        )

        garman_kohlhagen_df = garman_kohlhagen_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            garman_kohlhagen_df = calculate_standardization(
                dataset=garman_kohlhagen_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
            )

        return garman_kohlhagen_df

    def get_binary_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        option_type: str = "cash-or-nothing",
        cash_payout: float = 1.0,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the price of a Binary (Digital) Option using the Black-Scholes
        framework.

        A binary option pays out a fixed amount if the option expires in-the-money and
        nothing otherwise. Two variants are supported through the ``option_type``
        parameter: "cash-or-nothing" (the default) pays a fixed cash amount and
        "asset-or-nothing" pays the value of the underlying asset if the option expires
        in-the-money.

        The formula is as follows:

        - d1 = (ln(S / K) + (r - q + σ² / 2) * t) / (σ * √t)
        - d2 = d1 - σ * √t
        - Cash-or-Nothing Call = Cash Payout * e^(-r * t) * N(d2), with Cash Payout defaulting to 1
        - Cash-or-Nothing Put = Cash Payout * e^(-r * t) * N(-d2)
        - Asset-or-Nothing Call = S * e^(-q * t) * N(d1)
        - Asset-or-Nothing Put = S * e^(-q * t) * N(-d1)

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, r is the continuously compounded risk-free rate, q is the continuous dividend
        yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252), t is the time to
        expiration in years (days / 365) and N is the cumulative standard normal distribution.

        Also known as: digital option, all-or-nothing option, cash-or-nothing option,
        asset-or-nothing option.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            option_type (str, optional): Either "cash-or-nothing" or "asset-or-nothing". Defaults to
            "cash-or-nothing".
            cash_payout (float, optional): The fixed cash amount paid out by a cash-or-nothing option when it
            expires in-the-money. Ignored for asset-or-nothing options. Defaults to 1.0.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Binary Option values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GOOGL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        binary_option = toolkit.options.get_binary_option(
            strike_price_range=0.1, strike_step_size=5, expiration_time_range=91
        )

        binary_option.loc["GOOGL", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            280 |       0.8793 |       0.7914 |       0.7413 |
        |            285 |       0.8376 |       0.7516 |       0.7053 |
        |            290 |       0.7881 |       0.7086 |       0.6678 |
        |            295 |       0.7315 |       0.6631 |       0.6291 |
        |            300 |       0.6693 |       0.6157 |       0.5896 |
        |            305 |       0.6029 |       0.5674 |       0.5498 |
        |            310 |       0.5345 |       0.5187 |       0.5101 |
        |            315 |       0.4662 |       0.4705 |       0.4709 |
        |            320 |       0.3998 |       0.4234 |       0.4325 |
        |            325 |       0.3371 |       0.3781 |       0.3952 |
        |            330 |       0.2795 |       0.335  |       0.3595 |
        |            335 |       0.2278 |       0.2945 |       0.3254 |
        |            340 |       0.1827 |       0.257  |       0.2931 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        binary_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            binary_option[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            for strike_price in strike_prices:
                binary_option[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    binary_option[ticker][strike_price][time_to_expiration] = (
                        black_scholes_model.get_binary_option(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                            option_type=option_type,
                            cash_payout=cash_payout,
                        )
                    )

        binary_option_df = helpers.create_greek_dataframe(
            greek_dictionary=binary_option,
            start_date=start_date,
        )

        binary_option_df = binary_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            binary_option_df = calculate_standardization(
                dataset=binary_option_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return binary_option_df

    def get_bjerksund_stensland(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate American option prices using the Bjerksund-Stensland (2002)
        closed-form analytical approximation.

        Unlike European options, American options can be exercised at any time up to
        and including expiration, which normally requires a numerical approach such as
        the Binomial Tree model (see ``get_binomial_model``). The Bjerksund-Stensland
        model instead derives a closed-form approximation by assuming the early-exercise
        boundary is flat in steps: once the stock price crosses the trigger price,
        immediate exercise is assumed optimal. The option's life is split at
        t1 = ½(√5 - 1) * T into two periods, each with its own flat trigger price, which
        brings the price closer to the exact American value than a single flat trigger.

        The formula is as follows (for a call, with cost of carry b = r - q smaller than r):

        - β = (0.5 - b / σ²) + √((b / σ² - 0.5)² + 2r / σ²)
        - B∞ = β / (β - 1) * K
        - B0 = MAX(K, r / (r - b) * K)
        - h(t) = -(b * t + 2σ√t) * K² / ((B∞ - B0) * B0)
        - I(t) = B0 + (B∞ - B0) * (1 - e^h(t)), with I1 = I(t1) and I2 = I(T)
        - Call = MAX(Approximation, European Black-Scholes Call, MAX(S - K, 0)), where the Approximation is
          S - K when S ≥ I2 and otherwise combines univariate (first period) and bivariate (both periods)
          normal terms, see ``black_scholes_model.get_bjerksund_stensland``
        - Put (S, K, T, r, b, σ) = Call (K, S, T, r - b, -b, σ), the put-call transformation

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, r is the continuously compounded risk-free rate, q is the continuous dividend
        yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252) and T is the time
        to expiration in years (days / 365).

        When b ≥ r early exercise of a call is not optimal and the value is the Black-Scholes
        price (floored at the exercise value). Where the boundary I2 falls to or below the
        strike, a 1,000-step binomial tree prices the option instead. At zero days to
        expiration the value is the exercise value.

        See: Bjerksund, P., & Stensland, G. (2002), "Closed Form Valuation of American
        Options", Working paper, NHH, as presented in Haug, E. G. (2007), "The Complete
        Guide to Option Pricing Formulas" (2nd ed.), section 3.1.4.

        Also known as: BS2002, Bjerksund-Stensland approximation, American option
        approximation.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Bjerksund-Stensland American option values containing the tickers and strike prices as
            the index and the time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        bjerksund_stensland = toolkit.options.get_bjerksund_stensland(
            strike_price_range=0.15, strike_step_size=1, expiration_time_range=91
        )

        bjerksund_stensland.loc["VZ", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             34 |       6.73   |       6.73   |       6.73   |
        |             35 |       5.73   |       5.73   |       5.7417 |
        |             36 |       4.73   |       4.753  |       4.8234 |
        |             37 |       3.7409 |       3.8477 |       3.9719 |
        |             38 |       2.8269 |       3.0217 |       3.202  |
        |             39 |       2.0118 |       2.2958 |       2.5242 |
        |             40 |       1.3361 |       1.6834 |       1.9439 |
        |             41 |       0.8214 |       1.189  |       1.4615 |
        |             42 |       0.4646 |       0.8079 |       1.0723 |
        |             43 |       0.241  |       0.5278 |       0.7678 |
        |             44 |       0.1144 |       0.3314 |       0.5365 |
        |             45 |       0.0497 |       0.2001 |       0.366  |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        bjerksund_stensland: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            bjerksund_stensland[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            for strike_price in strike_prices:
                bjerksund_stensland[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    bjerksund_stensland[ticker][strike_price][time_to_expiration] = (
                        black_scholes_model.get_bjerksund_stensland(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        bjerksund_stensland_df = helpers.create_greek_dataframe(
            greek_dictionary=bjerksund_stensland,
            start_date=start_date,
        )

        bjerksund_stensland_df = bjerksund_stensland_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            bjerksund_stensland_df = calculate_standardization(
                dataset=bjerksund_stensland_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return bjerksund_stensland_df

    def get_monte_carlo_option_price(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        simulations: int = 10_000,
        time_steps: int = 100,
        seed: int | None = None,
        show_standard_error: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate European option prices through Monte Carlo simulation of Geometric
        Brownian Motion (GBM) stock price paths.

        The Monte Carlo method prices an option by simulating a large number of possible
        future paths for the underlying stock price under the risk-neutral measure,
        computing the option's payoff at expiration for each simulated path, and then
        discounting the average payoff back to the present.

        The formula is as follows:

        - Δt = T / Time Steps, with Time Steps defaulting to 100
        - S(t + Δt) = S(t) * e^((r - q - σ² / 2) * Δt + σ * √Δt * Z)
        - Call Price = e^(-r * T) * MEAN(MAX(S(T) - K, 0)), averaged over the simulated paths (10,000 by default)
        - Put Price = e^(-r * T) * MEAN(MAX(K - S(T), 0))
        - Standard Error = Sample Standard Deviation (ddof=1) of the Discounted Payoffs / √Simulations

        Where S(t) is the stock price at time t (starting from the traded close on the pricing
        date), K is the strike price, r is the continuously compounded risk-free rate, q is the
        continuous dividend yield (ln(1 + dividend yield)), σ is the annualized volatility (the
        standard deviation of the daily returns over the trailing 252 trading days times √252),
        T is the time to expiration in years (days / 365) and Z is a standard normal random
        variable.

        All strike prices of an expiry are priced on the same simulated paths (common
        random numbers), which keeps the prices consistent across strikes and means the
        paths are simulated once per expiry rather than once per strike.

        As it is a simulation, the result comes with sampling error. Set
        ``show_standard_error=True`` to additionally return the standard error of each
        estimate — as a rule of thumb, the true price lies within plus or minus 2 times
        the standard error roughly 95% of the time. Pass a ``seed`` (as the example does)
        to make the results reproducible; without one every call draws new paths.

        Also known as: Monte Carlo option pricing, simulation-based option pricing.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            simulations (int, optional): The number of simulated stock price paths. Defaults to 10,000.
            time_steps (int, optional): The number of time steps used to build each simulated path. Defaults to 100.
            seed (int | None, optional): The seed used to initialize the random number generator, ensuring
            reproducible results. Defaults to None, which means the results will not be reproducible.
            show_standard_error (bool, optional): Whether to also return the standard error of each Monte Carlo
            estimate as a second DataFrame. Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Monte Carlo option values containing the tickers and strike prices as the index and the
            time to expiration as the columns. If show_standard_error is True, a tuple of (prices, standard_errors)
            is returned instead.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["COST"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        monte_carlo = toolkit.options.get_monte_carlo_option_price(
            strike_price_range=0.1, strike_step_size=10, expiration_time_range=31, seed=42
        )

        monte_carlo.loc["COST", ["2026-01-07", "2026-01-14", "2026-01-30"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-30 |
        |---------------:|-------------:|-------------:|-------------:|
        |            780 |      82.8804 |      83.4929 |      85.5951 |
        |            790 |      72.8944 |      73.6474 |      76.2377 |
        |            800 |      62.9375 |      63.9365 |      67.1784 |
        |            810 |      53.0608 |      54.449  |      58.4909 |
        |            820 |      43.3668 |      45.3486 |      50.2675 |
        |            830 |      34.0544 |      36.7989 |      42.5958 |
        |            840 |      25.4476 |      28.9941 |      35.5519 |
        |            850 |      17.9002 |      22.0923 |      29.1778 |
        |            860 |      11.7131 |      16.2131 |      23.5449 |
        |            870 |       7.0931 |      11.4623 |      18.6844 |
        |            880 |       3.9774 |       7.8047 |      14.5854 |
        |            890 |       2.0399 |       5.1208 |      11.2062 |
        |            900 |       0.9613 |       3.2137 |       8.477  |
        |            910 |       0.4226 |       1.935  |       6.285  |
        |            920 |       0.1699 |       1.1307 |       4.5788 |
        |            930 |       0.0583 |       0.6377 |       3.2783 |
        |            940 |       0.018  |       0.3433 |       2.3104 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        monte_carlo_price: dict[str, dict[float, dict[float, float]]] = {}
        monte_carlo_error: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        logger.info("Running Monte Carlo Simulations")
        for ticker, strike_prices in strike_prices_per_ticker.items():
            monte_carlo_price[ticker] = {}
            monte_carlo_error[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            for strike_price in strike_prices:
                monte_carlo_price[ticker][strike_price] = {}
                monte_carlo_error[ticker][strike_price] = {}

            # Every strike is priced on the same simulated paths of an expiry, which with
            # a seed gives the same prices as simulating per strike, once per expiry.
            for time_to_expiration in time_to_expiration_list:
                prices, standard_errors = options_model.get_monte_carlo_option_prices(
                    stock_price=float(stock_price.loc[ticker]),
                    strike_prices=np.asarray(strike_prices, dtype=float),
                    risk_free_rate=risk_free_rate,
                    volatility=float(volatility.loc[ticker]),
                    time_to_expiration=time_to_expiration,
                    dividend_yield=float(dividend_yield_value[ticker]),
                    put_option=put_option,
                    simulations=simulations,
                    time_steps=time_steps,
                    seed=seed,
                )
                for strike_price, price, standard_error in zip(
                    strike_prices, prices, standard_errors, strict=True
                ):
                    monte_carlo_price[ticker][strike_price][time_to_expiration] = float(
                        price
                    )
                    monte_carlo_error[ticker][strike_price][time_to_expiration] = float(
                        standard_error
                    )

        monte_carlo_price_df = helpers.create_greek_dataframe(
            greek_dictionary=monte_carlo_price,
            start_date=start_date,
        )
        monte_carlo_price_df = monte_carlo_price_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            monte_carlo_price_df = calculate_standardization(
                dataset=monte_carlo_price_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        if show_standard_error:
            monte_carlo_error_df = helpers.create_greek_dataframe(
                greek_dictionary=monte_carlo_error,
                start_date=start_date,
            )
            monte_carlo_error_df = monte_carlo_error_df.pipe(
                apply_rounding, rounding if rounding is not None else self._rounding
            )

            return monte_carlo_price_df, monte_carlo_error_df

        return monte_carlo_price_df

    def get_barrier_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        barrier_percentage: float = 0.9,
        barrier_direction: str = "down",
        knock_type: str = "out",
        rebate: float = 0.0,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the closed-form price of a single-barrier (knock-in or knock-out, up
        or down) European option using the Reiner & Rubinstein (1991) formulas.

        A barrier option is a path-dependent option whose payoff (and existence)
        depends on whether the underlying stock price touches a pre-specified barrier
        level at any point before expiration:

        - Knock-out: the option becomes worthless if the barrier is touched.
        - Knock-in: the option only comes into existence if the barrier is touched.
        - Down barrier: the barrier is below the current stock price.
        - Up barrier: the barrier is above the current stock price.

        The barrier level is defined relative to the current stock price through
        ``barrier_percentage``, e.g. a value of 0.9 sets the barrier at 90% of the
        current stock price (a sensible default for a down barrier).

        The formula is as follows (Reiner & Rubinstein, 1991, in the notation of Haug, 2007):

        - H = Barrier Percentage * S, with Barrier Percentage defaulting to 0.9
        - b = r - q, μ = (b - σ² / 2) / σ², λ = √(μ² + 2r / σ²), φ = 1 for a call and -1 for a put, η = 1 for a
          down barrier and -1 for an up barrier
        - x1 = ln(S / K) / (σ√T) + (1 + μ)σ√T, x2 = ln(S / H) / (σ√T) + (1 + μ)σ√T
        - y1 = ln(H² / (S * K)) / (σ√T) + (1 + μ)σ√T, y2 = ln(H / S) / (σ√T) + (1 + μ)σ√T
        - z = ln(H / S) / (σ√T) + λσ√T
        - A = φS * e^((b - r)T) * N(φx1) - φK * e^(-rT) * N(φx1 - φσ√T), and B the same with x2 in place of x1
        - C = φS * e^((b - r)T) * (H / S)^(2(μ + 1)) * N(ηy1) - φK * e^(-rT) * (H / S)^(2μ) * N(ηy1 - ησ√T),
          and D the same with y2 in place of y1
        - E = Rebate * e^(-rT) * (N(ηx2 - ησ√T) - (H / S)^(2μ) * N(ηy2 - ησ√T))
        - F = Rebate * ((H / S)^(μ + λ) * N(ηz) + (H / S)^(μ - λ) * N(ηz - 2ηλσ√T))
        - Down-and-Out Call (the default) = A - C + F if K > H, else B - D + F
        - Down-and-In Call = C + E if K > H, else A - B + D + E
        - Up-and-Out Call = F if K > H, else A - B + C - D + F
        - Up-and-In Call = A + E if K > H, else B - C + D + E
        - Down-and-Out Put = A - B + C - D + F if K > H, else F
        - Down-and-In Put = B - C + D + E if K > H, else A + E
        - Up-and-Out Put = B - D + F if K > H, else A - C + F
        - Up-and-In Put = A - B + D + E if K > H, else C + E

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, H is the barrier, r is the continuously compounded risk-free rate, q is the
        continuous dividend yield (ln(1 + dividend yield)), σ is the annualized volatility (the
        standard deviation of the daily returns over the trailing 252 trading days times √252),
        T is the time to expiration in years (days / 365), N is the cumulative standard normal
        distribution and the Rebate defaults to 0.

        A useful identity is in-out parity: without a rebate, a knock-in option plus its
        corresponding knock-out option (same direction and barrier) equals the price of
        the equivalent vanilla Black-Scholes option, since the underlying either does or
        does not touch the barrier. With a rebate, both legs pay it in the scenario where
        they do not exist, so the sum exceeds the vanilla price by the value of those
        rebates.

        Also known as: knock-in option, knock-out option, down-and-out, down-and-in,
        up-and-out, up-and-in option.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            barrier_percentage (float, optional): The barrier level as a percentage of the current stock price.
            Defaults to 0.9 which equals 90% of the current stock price. A down barrier needs a value below 1
            and an up barrier a value above 1; otherwise the barrier is already breached.
            barrier_direction (str, optional): Either "down" or "up". Defaults to "down".
            knock_type (str, optional): Either "in" or "out". Defaults to "out".
            rebate (float, optional): The fixed cash amount paid out if the option knocks out (or fails to knock
            in). Defaults to 0.0.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Barrier option values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        Raises:
            ValueError: If barrier_percentage puts the barrier at or beyond the current stock price for the
                given barrier_direction (at or below 1 for "up", at or above 1 for "down"), since such a
                barrier is breached at the start and every price would be the rebate (knock-out) or the
                vanilla price (knock-in).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["META"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        barrier_option = toolkit.options.get_barrier_option(
            strike_price_range=0.1, strike_step_size=10, expiration_time_range=91
        )

        barrier_option.loc["META", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            590 |      70.211  |      70.5594 |      71.0161 |
        |            600 |      63.6599 |      65.6623 |      66.9713 |
        |            610 |      57.1775 |      60.7959 |      62.9446 |
        |            620 |      50.8693 |      56.0077 |      58.9639 |
        |            630 |      44.8284 |      51.341  |      55.0551 |
        |            640 |      39.1312 |      46.8335 |      51.2414 |
        |            650 |      33.8365 |      42.5169 |      47.543  |
        |            660 |      28.9845 |      38.4166 |      43.9768 |
        |            670 |      24.5979 |      34.5518 |      40.557  |
        |            680 |      20.6836 |      30.9358 |      37.2945 |
        |            690 |      17.2345 |      27.5763 |      34.1974 |
        |            700 |      14.2322 |      24.4761 |      31.2713 |
        |            710 |      11.6493 |      21.6334 |      28.519  |
        |            720 |       9.4527 |      19.043  |      25.9413 |
        """
        # The barrier is set relative to the spot, so a barrier on the wrong side of it is breached
        # before the option starts. The prices would then be a flat rebate or the vanilla price,
        # which hides a mistaken barrier_percentage or barrier_direction instead of reporting it.
        if barrier_direction == "up" and barrier_percentage <= 1:
            raise ValueError(
                f"An up barrier must lie above the current stock price, so barrier_percentage must be above 1 "
                f"(received {barrier_percentage}). Use barrier_direction='down' for a barrier below the stock price."
            )
        if barrier_direction == "down" and barrier_percentage >= 1:
            raise ValueError(
                f"A down barrier must lie below the current stock price, so barrier_percentage must be below 1 "
                f"(received {barrier_percentage}). Use barrier_direction='up' for a barrier above the stock price."
            )

        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        barrier_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            barrier_option[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )
            barrier = stock_price.loc[ticker] * barrier_percentage

            for strike_price in strike_prices:
                barrier_option[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    barrier_option[ticker][strike_price][time_to_expiration] = (
                        exotics_model.get_barrier_option(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            barrier=barrier,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                            barrier_direction=barrier_direction,
                            knock_type=knock_type,
                            rebate=rebate,
                        )
                    )

        barrier_option_df = helpers.create_greek_dataframe(
            greek_dictionary=barrier_option,
            start_date=start_date,
        )

        barrier_option_df = barrier_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            barrier_option_df = calculate_standardization(
                dataset=barrier_option_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return barrier_option_df

    def get_asian_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the closed-form price of a geometric-average Asian option using the
        Kemna & Vorst (1990) formula.

        An Asian option's payoff depends on the average price of the underlying stock
        over the option's life, rather than the price at a single point in time, which
        typically makes it cheaper than the equivalent vanilla option (the averaging
        reduces variance). This prices the continuously sampled geometric-average Asian
        option only; the arithmetic-average version has no closed form and is not
        offered. The geometric-average version has a closed-form solution based on an
        adjusted volatility and cost of carry.

        The formula is as follows:

        - σ_A = σ / √3
        - b_A = 0.5 * (b - σ² / 6), where b = r - q is the cost of carry
        - d1 = (ln(S / K) + (b_A + σ_A² / 2) * t) / (σ_A * √t)
        - d2 = d1 - σ_A * √t
        - Call Price = S * e^((b_A - r) * t) * N(d1) - K * e^(-r * t) * N(d2)
        - Put Price = K * e^(-r * t) * N(-d2) - S * e^((b_A - r) * t) * N(-d1)

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, r is the continuously compounded risk-free rate, q is the continuous dividend
        yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252), t is the time to
        expiration in years (days / 365) and N is the cumulative standard normal distribution.

        Also known as: geometric Asian option, average rate option, average price
        option.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Geometric-average Asian option values containing the tickers and strike prices as the
            index and the time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["XOM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        asian_option = toolkit.options.get_asian_option(
            strike_price_range=0.1, strike_step_size=2, expiration_time_range=91
        )

        asian_option.loc["XOM", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            108 |      12.2984 |      12.3072 |      12.3784 |
        |            110 |      10.3173 |      10.3976 |      10.5491 |
        |            112 |       8.365  |       8.5609 |       8.8116 |
        |            114 |       6.4839 |       6.8383 |       7.197  |
        |            116 |       4.7442 |       5.2756 |       5.7352 |
        |            118 |       3.2336 |       3.9144 |       4.4502 |
        |            120 |       2.0276 |       2.7832 |       3.3568 |
        |            122 |       1.1573 |       1.8907 |       2.4581 |
        |            124 |       0.5964 |       1.2243 |       1.7458 |
        |            126 |       0.276  |       0.7545 |       1.2016 |
        |            128 |       0.1143 |       0.4421 |       0.8013 |
        |            130 |       0.0423 |       0.2462 |       0.5176 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        asian_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            asian_option[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            for strike_price in strike_prices:
                asian_option[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    asian_option[ticker][strike_price][time_to_expiration] = (
                        exotics_model.get_asian_option(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        asian_option_df = helpers.create_greek_dataframe(
            greek_dictionary=asian_option,
            start_date=start_date,
        )

        asian_option_df = asian_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            asian_option_df = calculate_standardization(
                dataset=asian_option_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return asian_option_df

    def get_strategy_payoff(
        self,
        legs: list[dict[str, float | bool | str]],
        start_date: str | None = None,
        stock_price_range: float = 0.5,
        stock_price_step_size: int = 1,
        rounding: int | None = None,
    ):
        """
        Calculate the net expiration profit and loss (P&L) profile of a multi-leg
        option (and, optionally, stock) strategy across a range of stock prices.

        A strategy is expressed as a list of "legs". Each leg is a dictionary
        describing either an option position or a stock position:

        - For an option leg: "instrument": "option" (default), "strike_price" (float,
          required), "put_option" (bool, defaults to False), "position" ("long" or
          "short", defaults to "long"), "premium" (float, defaults to 0).
        - For a stock leg: "instrument": "stock", "position" ("long" or "short",
          defaults to "long"), "premium" (float, the entry price, defaults to 0).

        This single, generic building block can express many common strategies by
        combining legs, for example:

        - Straddle: long call + long put, same strike.
        - Strangle: long call + long put, different (OTM) strikes.
        - Bull call spread: long call (lower strike) + short call (higher strike).
        - Bear put spread: long put (higher strike) + short put (lower strike).
        - Covered call: long stock + short call.
        - Protective put: long stock + long put.
        - Iron condor: short put + long put (lower strikes) + short call + long call
          (higher strikes).

        The formula is as follows, for each stock price at expiration S(T) and one unit per leg:

        - Call Leg Value = MAX(S(T) - K, 0)
        - Put Leg Value = MAX(K - S(T), 0)
        - Stock Leg Value = S(T)
        - Leg P&L = Leg Value - Premium for a long leg, Premium - Leg Value for a short leg
        - Strategy P&L = SUM(Leg P&L) over all legs

        Where K is the leg's strike price and the premium is the price paid (long) or received
        (short) for the leg, or the entry price of a stock leg. The P&L is not discounted.

        Also known as: option strategy payoff diagram, P&L profile.

        Args:
            legs (list[dict]): A list of leg dictionaries as described above. Must
            contain at least one leg. The same legs are applied to every ticker, so
            strike prices should be chosen with the relevant tickers' price levels in
            mind.
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            stock_price_range (float): The percentage range to use for the stock prices at expiration. Defaults
            to 0.5 which equals 50% and thus results in stock prices from 50 to 150 if the current stock price is
            100.
            stock_price_step_size (int): The step size to use for the stock prices at expiration. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: The strategy's net P&L with the range of stock prices at expiration as the index and the
            tickers as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        straddle_legs = [
            {"strike_price": 450, "put_option": False, "position": "long", "premium": 30},
            {"strike_price": 450, "put_option": True, "position": "long", "premium": 28},
        ]

        toolkit.options.get_strategy_payoff(
            legs=straddle_legs, stock_price_range=0.25, stock_price_step_size=5
        )
        ```

        Which returns:

        |   Stock Price |   TSLA |
        |--------------:|-------:|
        |           335 |     57 |
        |           340 |     52 |
        |           345 |     47 |
        |           350 |     42 |
        |           355 |     37 |
        |           360 |     32 |
        |           365 |     27 |
        |           370 |     22 |
        |           375 |     17 |
        |           380 |     12 |
        |           385 |      7 |
        |           390 |      2 |
        |           395 |     -3 |
        |           400 |     -8 |
        |           405 |    -13 |
        |           410 |    -18 |
        |           415 |    -23 |
        |           420 |    -28 |
        |           425 |    -33 |
        |           430 |    -38 |
        |           435 |    -43 |
        |           440 |    -48 |
        |           445 |    -53 |
        |           450 |    -58 |
        |           455 |    -53 |
        |           460 |    -48 |
        |           465 |    -43 |
        |           470 |    -38 |
        |           475 |    -33 |
        |           480 |    -28 |
        |           485 |    -23 |
        |           490 |    -18 |
        |           495 |    -13 |
        |           500 |     -8 |
        |           505 |     -3 |
        |           510 |      2 |
        |           515 |      7 |
        |           520 |     12 |
        |           525 |     17 |
        |           530 |     22 |
        |           535 |     27 |
        |           540 |     32 |
        |           545 |     37 |
        |           550 |     42 |
        |           555 |     47 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]

        stock_price_range_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=stock_price_step_size,
            strike_price_range=stock_price_range,
        )

        strategy_payoff: dict[str, pd.Series] = {}

        for ticker, price_range in stock_price_range_per_ticker.items():
            price_range_series = pd.Series(price_range, index=price_range, name=ticker)

            strategy_payoff[ticker] = binomial_trees_model.get_strategy_payoff(
                stock_price=price_range_series,
                legs=legs,
            )

        strategy_payoff_df = pd.DataFrame(strategy_payoff)
        strategy_payoff_df.index.name = "Stock Price"

        strategy_payoff_df = apply_rounding(
            strategy_payoff_df, rounding if rounding is not None else self._rounding
        )

        return strategy_payoff_df

    def collect_all_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate all Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        **First Order Greeks:**

        - Delta: measures the rate of change of the theoretical option value with respect to changes in the underlying
        asset's price.
        - Dual Delta: the first derivative of the option price with respect to the strike price. Up to the discount
        factor and a sign it is the risk-neutral probability of the option finishing in the money, negative for a
        call and positive for a put.
        - Vega: measures sensitivity to volatility. Vega is the derivative of the option value with respect to the volatility
        of the underlying asset.
        - Theta: measures the sensitivity of the value of the derivative to the passage of time, the "time decay."
        - Rho: measures sensitivity to the interest rate: it is the derivative of the option value with respect to
        the risk-free interest rate (for the relevant outstanding term).
        - Epsilon: measures the change in option value per unit (1.00) change in the dividend yield, the derivative of
        the value with respect to the dividend yield, a measure of the dividend risk.
        - Lambda: measures the percentage change in option value per percentage change in the underlying price, a measure of
        leverage, sometimes called gearing. This greek is also sometimes called Omega or Elasticity.

        **Second Order Greeks:**

        - Gamma: measures the rate of change in the delta with respect to changes in the underlying price. Gamma is
        the second derivative of the value function with respect to the underlying price.
        - Dual Gamma: the second derivative of the option value with respect to the strike price rather than the
        underlying price. It is the discounted risk-neutral probability density of the underlying at expiration.
        - Vanna: also referred to as DvegaDspot and DdeltaDvol, is a second—order derivative of the option value,
        once to the underlying spot price and once to volatility.
        - Charm: Charm  or delta decay measures the instantaneous rate of change of delta over the passage of time.
        - Vomma: also referred to as volga, vega convexity, or DvegaDvol measures second—order sensitivity to
        volatility. Vomma is the second derivative of the option value with respect to the volatility, or,
        stated another way, vomma measures the rate of change to vega as volatility changes.
        - Veta: also referred to as DvegaDtime, measures the rate of change in the vega with respect to
        the passage of time. Veta is the second derivative of the value function; once to volatility and once to time.
        - Vera: also referred to as rhova, measures the rate of change in rho with respect to volatility. Vera is the
        second derivative of the value function; once to volatility and once to interest rate.
        - Partial Derivative: the second derivative of the option price with respect to the strike price, the rate
        at which the dual delta changes with the strike.

        **Third Order Greeks:**

        - Speed: measures the rate of change in Gamma with respect to changes in the underlying price.
        - Zomma: measures the rate of change of Gamma with respect to changes in volatility.
        - Color: also referred to as gamma decay or DgammaDtime measures the rate of change of gamma over
        the passage of time.
        - Ultima: measures the sensitivity of the option vomma with respect to change in volatility.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option delta. Defaults to False which means
            it will calculate the call option delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

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
        |            180 |  1      |      -0.9999 | 0      | -0.0191 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0196 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0202 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0207 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            200 |  1      |      -0.9999 | 0      | -0.0212 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0.0001 |
        |            205 |  1      |      -0.9999 | 0      | -0.0217 | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      | -0      |   0.0002 |  0.0001 | -0      | -0      | 0      | -0      |  0      | -0.0002 |   0.0071 |
        |            210 |  1      |      -0.9999 | 0      | -0.0223 | 0.5753 |   -0.6533 |   8.3748 |  0      |       0      | -0.0001 |   0.0124 |  0.0071 | -0.0001 | -0      | 0      | -0      |  0.0001 | -0.0079 |   0.2261 |
        |            215 |  0.9999 |      -0.9998 | 0.0001 | -0.0243 | 0.5889 |   -0.6532 |  10.1573 |  0.0001 |       0.0001 | -0.0031 |   0.3048 |  0.1423 | -0.002  | -0.0004 | 0.0001 | -0      |  0.0016 | -0.1539 |   2.7483 |
        |            220 |  0.9979 |      -0.9976 | 0.0008 | -0.0454 | 0.6013 |   -0.6519 |  12.8781 |  0.001  |       0.0011 | -0.0345 |   3.3891 |  1.2332 | -0.0227 | -0.0037 | 0.0011 | -0.0004 |  0.0129 | -1.2633 |  11.7212 |
        |            225 |  0.9808 |      -0.9794 | 0.0058 | -0.1808 | 0.6037 |   -0.6408 |  17.297  |  0.0069 |       0.0078 | -0.1766 |  17.3444 |  4.5655 | -0.117  | -0.0152 | 0.0078 | -0.0022 |  0.0414 | -4.0652 |  10.4124 |
        |            230 |  0.9022 |      -0.8971 | 0.0216 | -0.604  | 0.5653 |   -0.5894 |  24.4765 |  0.0256 |       0.0275 | -0.4047 |  39.6576 |  6.5361 | -0.2703 | -0.0283 | 0.0275 | -0.005  |  0.0303 | -2.9369 | -16.4877 |
        |            235 |  0.7031 |      -0.6932 | 0.0432 | -1.1838 | 0.4463 |   -0.4593 |  35.2268 |  0.0513 |       0.0528 | -0.3236 |  31.4245 |  2.1541 | -0.2232 | -0.0273 | 0.0528 | -0.0043 | -0.0694 |  6.8815 | -10.901  |
        |            240 |  0.4163 |      -0.4053 | 0.0487 | -1.3254 | 0.2665 |   -0.272  |  49.5371 |  0.0579 |       0.0571 |  0.173  | -17.5904 |  0.4562 |  0.0997 | -0.0256 | 0.0571 |  0.0016 | -0.1017 | 10.0098 |  -2.5045 |
        |            245 |  0.1735 |      -0.1663 | 0.032  | -0.8684 | 0.1116 |   -0.1133 |  66.7011 |  0.038  |       0.036  |  0.4598 | -45.6744 |  5.3973 |  0.2916 | -0.0307 | 0.036  |  0.0051 | -0.0062 |  0.5668 | -20.8797 |
        |            250 |  0.049  |      -0.0461 | 0.0127 | -0.343  | 0.0316 |   -0.032  |  85.8277 |  0.015  |       0.0137 |  0.316  | -31.292  |  6.5283 |  0.203  | -0.024  | 0.0137 |  0.0036 |  0.0497 | -4.9373 |  -2.5872 |
        |            255 |  0.0092 |      -0.0086 | 0.0031 | -0.0841 | 0.006  |   -0.006  | 106.169  |  0.0037 |       0.0032 |  0.1099 | -10.8635 |  3.2297 |  0.0709 | -0.0103 | 0.0032 |  0.0013 |  0.0315 | -3.1218 |  15.635  |
        |            260 |  0.0012 |      -0.0011 | 0.0005 | -0.0132 | 0.0008 |   -0.0008 | 127.197  |  0.0006 |       0.0005 |  0.0222 |  -2.1908 |  0.8418 |  0.0143 | -0.0025 | 0.0005 |  0.0003 |  0.0089 | -0.8831 |   9.8831 |
        |            265 |  0.0001 |      -0.0001 | 0      | -0.0014 | 0.0001 |   -0.0001 | 148.561  |  0.0001 |       0      |  0.0028 |  -0.2741 |  0.1287 |  0.0018 | -0.0004 | 0      |  0      |  0.0014 | -0.1402 |   2.6    |
        |            270 |  0      |      -0      | 0      | -0.0001 | 0      |   -0      | 170.037  |  0      |       0      |  0.0002 |  -0.0222 |  0.0123 |  0.0001 | -0      | 0      |  0      |  0.0001 | -0.0137 |   0.3703 |
        |            275 |  0      |      -0      | 0      | -0      | 0      |   -0      | 191.483  |  0      |       0      |  0      |  -0.0012 |  0.0008 |  0      | -0      | 0      |  0      |  0      | -0.0009 |   0.0317 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      | 212.807  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0017 |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      | 233.949  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0001 |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      | 254.872  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      | 275.552  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        """
        first_order_greeks = self.collect_first_order_greeks(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        second_order_greeks = self.collect_second_order_greeks(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        third_order_greeks = self.collect_third_order_greeks(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        all_greeks = pd.concat(
            [first_order_greeks, second_order_greeks, third_order_greeks], axis=1
        )

        return all_greeks

    def collect_first_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the first order Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        - Delta: measures the rate of change of the theoretical option value with respect to changes in the underlying
        asset's price.
        - Dual Delta: the first derivative of the option price with respect to the strike price. Up to the discount
        factor and a sign it is the risk-neutral probability of the option finishing in the money, negative for a
        call and positive for a put.
        - Vega: measures sensitivity to volatility. Vega is the derivative of the option value with respect to the volatility
        of the underlying asset.
        - Theta: measures the sensitivity of the value of the derivative to the passage of time, the "time decay."
        - Rho: measures sensitivity to the interest rate: it is the derivative of the option value with respect to
        the risk—free interest rate (for the relevant outstanding term).
        - Epsilon: measures the change in option value per unit (1.00) change in the dividend yield, the derivative of
        the value with respect to the dividend yield, a measure of the dividend risk.
        - Lambda: measures the percentage change in option value per percentage change in the underlying price, a measure of
        leverage, sometimes called gearing. This greek is also sometimes called Omega or Elasticity.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option delta. Defaults to False which means
            it will calculate the call option delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the first order greek values containing the tickers and strike prices as the index
            and the time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.collect_first_order_greeks().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   (Period('2026-01-01', 'D'), 'Delta') |   (Period('2026-01-01', 'D'), 'Dual Delta') |   (Period('2026-01-01', 'D'), 'Vega') |   (Period('2026-01-01', 'D'), 'Theta') |   (Period('2026-01-01', 'D'), 'Rho') |   (Period('2026-01-01', 'D'), 'Epsilon') |   (Period('2026-01-01', 'D'), 'Lambda') |   (Period('2026-01-02', 'D'), 'Delta') |   (Period('2026-01-02', 'D'), 'Dual Delta') |   (Period('2026-01-02', 'D'), 'Vega') |   (Period('2026-01-02', 'D'), 'Theta') |   (Period('2026-01-02', 'D'), 'Rho') |   (Period('2026-01-02', 'D'), 'Epsilon') |   (Period('2026-01-02', 'D'), 'Lambda') |   (Period('2026-01-03', 'D'), 'Delta') |   (Period('2026-01-03', 'D'), 'Dual Delta') |   (Period('2026-01-03', 'D'), 'Vega') |   (Period('2026-01-03', 'D'), 'Theta') |   (Period('2026-01-03', 'D'), 'Rho') |   (Period('2026-01-03', 'D'), 'Epsilon') |   (Period('2026-01-03', 'D'), 'Lambda') |   (Period('2026-01-04', 'D'), 'Delta') |   (Period('2026-01-04', 'D'), 'Dual Delta') |   (Period('2026-01-04', 'D'), 'Vega') |   (Period('2026-01-04', 'D'), 'Theta') |   (Period('2026-01-04', 'D'), 'Rho') |   (Period('2026-01-04', 'D'), 'Epsilon') |   (Period('2026-01-04', 'D'), 'Lambda') |   (Period('2026-01-05', 'D'), 'Delta') |   (Period('2026-01-05', 'D'), 'Dual Delta') |   (Period('2026-01-05', 'D'), 'Vega') |   (Period('2026-01-05', 'D'), 'Theta') |   (Period('2026-01-05', 'D'), 'Rho') |   (Period('2026-01-05', 'D'), 'Epsilon') |   (Period('2026-01-05', 'D'), 'Lambda') |   (Period('2026-01-06', 'D'), 'Delta') |   (Period('2026-01-06', 'D'), 'Dual Delta') |   (Period('2026-01-06', 'D'), 'Vega') |   (Period('2026-01-06', 'D'), 'Theta') |   (Period('2026-01-06', 'D'), 'Rho') |   (Period('2026-01-06', 'D'), 'Epsilon') |   (Period('2026-01-06', 'D'), 'Lambda') |   (Period('2026-01-07', 'D'), 'Delta') |   (Period('2026-01-07', 'D'), 'Dual Delta') |   (Period('2026-01-07', 'D'), 'Vega') |   (Period('2026-01-07', 'D'), 'Theta') |   (Period('2026-01-07', 'D'), 'Rho') |   (Period('2026-01-07', 'D'), 'Epsilon') |   (Period('2026-01-07', 'D'), 'Lambda') |   (Period('2026-01-08', 'D'), 'Delta') |   (Period('2026-01-08', 'D'), 'Dual Delta') |   (Period('2026-01-08', 'D'), 'Vega') |   (Period('2026-01-08', 'D'), 'Theta') |   (Period('2026-01-08', 'D'), 'Rho') |   (Period('2026-01-08', 'D'), 'Epsilon') |   (Period('2026-01-08', 'D'), 'Lambda') |   (Period('2026-01-09', 'D'), 'Delta') |   (Period('2026-01-09', 'D'), 'Dual Delta') |   (Period('2026-01-09', 'D'), 'Vega') |   (Period('2026-01-09', 'D'), 'Theta') |   (Period('2026-01-09', 'D'), 'Rho') |   (Period('2026-01-09', 'D'), 'Epsilon') |   (Period('2026-01-09', 'D'), 'Lambda') |   (Period('2026-01-10', 'D'), 'Delta') |   (Period('2026-01-10', 'D'), 'Dual Delta') |   (Period('2026-01-10', 'D'), 'Vega') |   (Period('2026-01-10', 'D'), 'Theta') |   (Period('2026-01-10', 'D'), 'Rho') |   (Period('2026-01-10', 'D'), 'Epsilon') |   (Period('2026-01-10', 'D'), 'Lambda') |   (Period('2026-01-11', 'D'), 'Delta') |   (Period('2026-01-11', 'D'), 'Dual Delta') |   (Period('2026-01-11', 'D'), 'Vega') |   (Period('2026-01-11', 'D'), 'Theta') |   (Period('2026-01-11', 'D'), 'Rho') |   (Period('2026-01-11', 'D'), 'Epsilon') |   (Period('2026-01-11', 'D'), 'Lambda') |   (Period('2026-01-12', 'D'), 'Delta') |   (Period('2026-01-12', 'D'), 'Dual Delta') |   (Period('2026-01-12', 'D'), 'Vega') |   (Period('2026-01-12', 'D'), 'Theta') |   (Period('2026-01-12', 'D'), 'Rho') |   (Period('2026-01-12', 'D'), 'Epsilon') |   (Period('2026-01-12', 'D'), 'Lambda') |   (Period('2026-01-13', 'D'), 'Delta') |   (Period('2026-01-13', 'D'), 'Dual Delta') |   (Period('2026-01-13', 'D'), 'Vega') |   (Period('2026-01-13', 'D'), 'Theta') |   (Period('2026-01-13', 'D'), 'Rho') |   (Period('2026-01-13', 'D'), 'Epsilon') |   (Period('2026-01-13', 'D'), 'Lambda') |   (Period('2026-01-14', 'D'), 'Delta') |   (Period('2026-01-14', 'D'), 'Dual Delta') |   (Period('2026-01-14', 'D'), 'Vega') |   (Period('2026-01-14', 'D'), 'Theta') |   (Period('2026-01-14', 'D'), 'Rho') |   (Period('2026-01-14', 'D'), 'Epsilon') |   (Period('2026-01-14', 'D'), 'Lambda') |   (Period('2026-01-15', 'D'), 'Delta') |   (Period('2026-01-15', 'D'), 'Dual Delta') |   (Period('2026-01-15', 'D'), 'Vega') |   (Period('2026-01-15', 'D'), 'Theta') |   (Period('2026-01-15', 'D'), 'Rho') |   (Period('2026-01-15', 'D'), 'Epsilon') |   (Period('2026-01-15', 'D'), 'Lambda') |   (Period('2026-01-16', 'D'), 'Delta') |   (Period('2026-01-16', 'D'), 'Dual Delta') |   (Period('2026-01-16', 'D'), 'Vega') |   (Period('2026-01-16', 'D'), 'Theta') |   (Period('2026-01-16', 'D'), 'Rho') |   (Period('2026-01-16', 'D'), 'Epsilon') |   (Period('2026-01-16', 'D'), 'Lambda') |   (Period('2026-01-17', 'D'), 'Delta') |   (Period('2026-01-17', 'D'), 'Dual Delta') |   (Period('2026-01-17', 'D'), 'Vega') |   (Period('2026-01-17', 'D'), 'Theta') |   (Period('2026-01-17', 'D'), 'Rho') |   (Period('2026-01-17', 'D'), 'Epsilon') |   (Period('2026-01-17', 'D'), 'Lambda') |   (Period('2026-01-18', 'D'), 'Delta') |   (Period('2026-01-18', 'D'), 'Dual Delta') |   (Period('2026-01-18', 'D'), 'Vega') |   (Period('2026-01-18', 'D'), 'Theta') |   (Period('2026-01-18', 'D'), 'Rho') |   (Period('2026-01-18', 'D'), 'Epsilon') |   (Period('2026-01-18', 'D'), 'Lambda') |   (Period('2026-01-19', 'D'), 'Delta') |   (Period('2026-01-19', 'D'), 'Dual Delta') |   (Period('2026-01-19', 'D'), 'Vega') |   (Period('2026-01-19', 'D'), 'Theta') |   (Period('2026-01-19', 'D'), 'Rho') |   (Period('2026-01-19', 'D'), 'Epsilon') |   (Period('2026-01-19', 'D'), 'Lambda') |   (Period('2026-01-20', 'D'), 'Delta') |   (Period('2026-01-20', 'D'), 'Dual Delta') |   (Period('2026-01-20', 'D'), 'Vega') |   (Period('2026-01-20', 'D'), 'Theta') |   (Period('2026-01-20', 'D'), 'Rho') |   (Period('2026-01-20', 'D'), 'Epsilon') |   (Period('2026-01-20', 'D'), 'Lambda') |   (Period('2026-01-21', 'D'), 'Delta') |   (Period('2026-01-21', 'D'), 'Dual Delta') |   (Period('2026-01-21', 'D'), 'Vega') |   (Period('2026-01-21', 'D'), 'Theta') |   (Period('2026-01-21', 'D'), 'Rho') |   (Period('2026-01-21', 'D'), 'Epsilon') |   (Period('2026-01-21', 'D'), 'Lambda') |   (Period('2026-01-22', 'D'), 'Delta') |   (Period('2026-01-22', 'D'), 'Dual Delta') |   (Period('2026-01-22', 'D'), 'Vega') |   (Period('2026-01-22', 'D'), 'Theta') |   (Period('2026-01-22', 'D'), 'Rho') |   (Period('2026-01-22', 'D'), 'Epsilon') |   (Period('2026-01-22', 'D'), 'Lambda') |   (Period('2026-01-23', 'D'), 'Delta') |   (Period('2026-01-23', 'D'), 'Dual Delta') |   (Period('2026-01-23', 'D'), 'Vega') |   (Period('2026-01-23', 'D'), 'Theta') |   (Period('2026-01-23', 'D'), 'Rho') |   (Period('2026-01-23', 'D'), 'Epsilon') |   (Period('2026-01-23', 'D'), 'Lambda') |   (Period('2026-01-24', 'D'), 'Delta') |   (Period('2026-01-24', 'D'), 'Dual Delta') |   (Period('2026-01-24', 'D'), 'Vega') |   (Period('2026-01-24', 'D'), 'Theta') |   (Period('2026-01-24', 'D'), 'Rho') |   (Period('2026-01-24', 'D'), 'Epsilon') |   (Period('2026-01-24', 'D'), 'Lambda') |   (Period('2026-01-25', 'D'), 'Delta') |   (Period('2026-01-25', 'D'), 'Dual Delta') |   (Period('2026-01-25', 'D'), 'Vega') |   (Period('2026-01-25', 'D'), 'Theta') |   (Period('2026-01-25', 'D'), 'Rho') |   (Period('2026-01-25', 'D'), 'Epsilon') |   (Period('2026-01-25', 'D'), 'Lambda') |   (Period('2026-01-26', 'D'), 'Delta') |   (Period('2026-01-26', 'D'), 'Dual Delta') |   (Period('2026-01-26', 'D'), 'Vega') |   (Period('2026-01-26', 'D'), 'Theta') |   (Period('2026-01-26', 'D'), 'Rho') |   (Period('2026-01-26', 'D'), 'Epsilon') |   (Period('2026-01-26', 'D'), 'Lambda') |   (Period('2026-01-27', 'D'), 'Delta') |   (Period('2026-01-27', 'D'), 'Dual Delta') |   (Period('2026-01-27', 'D'), 'Vega') |   (Period('2026-01-27', 'D'), 'Theta') |   (Period('2026-01-27', 'D'), 'Rho') |   (Period('2026-01-27', 'D'), 'Epsilon') |   (Period('2026-01-27', 'D'), 'Lambda') |   (Period('2026-01-28', 'D'), 'Delta') |   (Period('2026-01-28', 'D'), 'Dual Delta') |   (Period('2026-01-28', 'D'), 'Vega') |   (Period('2026-01-28', 'D'), 'Theta') |   (Period('2026-01-28', 'D'), 'Rho') |   (Period('2026-01-28', 'D'), 'Epsilon') |   (Period('2026-01-28', 'D'), 'Lambda') |   (Period('2026-01-29', 'D'), 'Delta') |   (Period('2026-01-29', 'D'), 'Dual Delta') |   (Period('2026-01-29', 'D'), 'Vega') |   (Period('2026-01-29', 'D'), 'Theta') |   (Period('2026-01-29', 'D'), 'Rho') |   (Period('2026-01-29', 'D'), 'Epsilon') |   (Period('2026-01-29', 'D'), 'Lambda') |
        |---------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|---------------------------------------:|-------------------------------------:|-----------------------------------------:|----------------------------------------:|
        |            290 |                                 0.0001 |                                     -0.0001 |                                     0 |                                -0.0007 |                               0.0001 |                                  -0.0001 |                                 250.951 |                                 0.0038 |                                     -0.0035 |                                0.0023 |                                -0.0185 |                               0.0056 |                                  -0.0056 |                                 136.163 |                                 0.0149 |                                     -0.0138 |                                0.0093 |                                -0.0506 |                               0.033  |                                  -0.0333 |                                 96.9305 |                                 0.0305 |                                     -0.0282 |                                0.0196 |                                -0.0803 |                               0.0895 |                                  -0.0907 |                                 76.8302 |                                 0.0475 |                                     -0.0438 |                                0.0315 |                                -0.1034 |                               0.1741 |                                  -0.1768 |                                 64.4873 |                                 0.0645 |                                     -0.0593 |                                0.0439 |                                -0.1204 |                               0.2829 |                                  -0.288  |                                 56.0765 |                                 0.0807 |                                     -0.0741 |                                0.0563 |                                -0.1327 |                               0.4124 |                                  -0.4208 |                                 49.9432 |                                 0.096  |                                     -0.088  |                                0.0685 |                                -0.1415 |                               0.5593 |                                  -0.572  |                                 45.2523 |                                 0.1102 |                                     -0.1008 |                                0.0804 |                                -0.1478 |                               0.7211 |                                  -0.7389 |                                 41.5355 |                                 0.1235 |                                     -0.1127 |                                0.0918 |                                -0.1522 |                               0.8957 |                                  -0.9196 |                                 38.5092 |                                 0.1358 |                                     -0.1237 |                                0.1029 |                                -0.1552 |                               1.0813 |                                  -1.1122 |                                 35.9912 |                                 0.1472 |                                     -0.1339 |                                0.1135 |                                -0.1573 |                               1.2766 |                                  -1.3155 |                                 33.8589 |                                 0.1578 |                                     -0.1433 |                                0.1237 |                                -0.1585 |                               1.4804 |                                  -1.5281 |                                 32.0267 |                                 0.1678 |                                     -0.1521 |                                0.1336 |                                -0.1592 |                               1.6918 |                                  -1.7493 |                                 30.4329 |                                 0.177  |                                     -0.1603 |                                0.1431 |                                -0.1594 |                               1.9099 |                                  -1.978  |                                 29.032  |                                 0.1858 |                                     -0.1679 |                                0.1523 |                                -0.1593 |                               2.134  |                                  -2.2137 |                                 27.7894 |                                 0.1939 |                                     -0.175  |                                0.1612 |                                -0.1589 |                               2.3637 |                                  -2.4557 |                                 26.6785 |                                 0.2017 |                                     -0.1817 |                                0.1698 |                                -0.1583 |                               2.5983 |                                  -2.7036 |                                 25.6784 |                                 0.2089 |                                     -0.188  |                                0.1782 |                                -0.1576 |                               2.8375 |                                  -2.9568 |                                 24.7725 |                                 0.2158 |                                     -0.1939 |                                0.1863 |                                -0.1568 |                               3.0808 |                                  -3.2151 |                                 23.9475 |                                 0.2224 |                                     -0.1995 |                                0.1942 |                                -0.1558 |                               3.328  |                                  -3.478  |                                 23.1925 |                                 0.2286 |                                     -0.2047 |                                0.2019 |                                -0.1548 |                               3.5787 |                                  -3.7452 |                                 22.4984 |                                 0.2345 |                                     -0.2097 |                                0.2094 |                                -0.1538 |                               3.8327 |                                  -4.0165 |                                 21.8578 |                                 0.2401 |                                     -0.2145 |                                0.2167 |                                -0.1527 |                               4.0898 |                                  -4.2916 |                                 21.2643 |                                 0.2454 |                                     -0.219  |                                0.2239 |                                -0.1516 |                               4.3497 |                                  -4.5704 |                                 20.7127 |                                 0.2506 |                                     -0.2233 |                                0.2308 |                                -0.1505 |                               4.6123 |                                  -4.8525 |                                 20.1984 |                                 0.2555 |                                     -0.2274 |                                0.2376 |                                -0.1494 |                               4.8773 |                                  -5.1379 |                                 19.7176 |                                 0.2602 |                                     -0.2313 |                                0.2443 |                                -0.1483 |                               5.1447 |                                  -5.4264 |                                 19.2669 |                                 0.2647 |                                     -0.235  |                                0.2508 |                                -0.1471 |                               5.4143 |                                  -5.7177 |                                 18.8433 |
        |            295 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 305.838 |                                 0.0004 |                                     -0.0003 |                                0.0003 |                                -0.0021 |                               0.0005 |                                  -0.0005 |                                 162.333 |                                 0.0029 |                                     -0.0027 |                                0.0022 |                                -0.0121 |                               0.0065 |                                  -0.0066 |                                113.798  |                                 0.0087 |                                     -0.0079 |                                0.0067 |                                -0.0275 |                               0.0256 |                                  -0.0259 |                                 89.1566 |                                 0.017  |                                     -0.0154 |                                0.0134 |                                -0.0439 |                               0.0624 |                                  -0.0633 |                                 74.1443 |                                 0.0268 |                                     -0.0243 |                                0.0216 |                                -0.0591 |                               0.118  |                                  -0.1199 |                                 63.985  |                                 0.0375 |                                     -0.0339 |                                0.0307 |                                -0.0722 |                               0.1918 |                                  -0.1953 |                                 56.6221 |                                 0.0484 |                                     -0.0437 |                                0.0404 |                                -0.0832 |                               0.2825 |                                  -0.2882 |                                 51.0216 |                                 0.0593 |                                     -0.0534 |                                0.0504 |                                -0.0923 |                               0.3887 |                                  -0.3972 |                                 46.606  |                                 0.0699 |                                     -0.0629 |                                0.0604 |                                -0.0997 |                               0.5087 |                                  -0.5208 |                                 43.0268 |                                 0.0803 |                                     -0.0721 |                                0.0703 |                                -0.1058 |                               0.6412 |                                  -0.6576 |                                 40.061  |                                 0.0902 |                                     -0.0809 |                                0.0802 |                                -0.1108 |                               0.7849 |                                  -0.8064 |                                 37.5589 |                                 0.0998 |                                     -0.0894 |                                0.0899 |                                -0.1148 |                               0.9388 |                                  -0.9661 |                                 35.4166 |                                 0.1089 |                                     -0.0974 |                                0.0994 |                                -0.118  |                               1.1019 |                                  -1.1358 |                                 33.559  |                                 0.1177 |                                     -0.105  |                                0.1087 |                                -0.1207 |                               1.2734 |                                  -1.3145 |                                 31.9312 |                                 0.126  |                                     -0.1123 |                                0.1178 |                                -0.1228 |                               1.4524 |                                  -1.5017 |                                 30.4914 |                                 0.134  |                                     -0.1192 |                                0.1267 |                                -0.1244 |                               1.6384 |                                  -1.6965 |                                 29.2075 |                                 0.1416 |                                     -0.1258 |                                0.1354 |                                -0.1257 |                               1.8308 |                                  -1.8985 |                                 28.0547 |                                 0.1489 |                                     -0.1321 |                                0.1439 |                                -0.1267 |                               2.0291 |                                  -2.1071 |                                 27.0129 |                                 0.1559 |                                     -0.1381 |                                0.1522 |                                -0.1275 |                               2.2328 |                                  -2.3218 |                                 26.0663 |                                 0.1625 |                                     -0.1438 |                                0.1603 |                                -0.1281 |                               2.4415 |                                  -2.5424 |                                 25.2017 |                                 0.1689 |                                     -0.1493 |                                0.1682 |                                -0.1284 |                               2.6549 |                                  -2.7683 |                                 24.4086 |                                 0.1751 |                                     -0.1545 |                                0.176  |                                -0.1287 |                               2.8726 |                                  -2.9993 |                                 23.6779 |                                 0.181  |                                     -0.1595 |                                0.1835 |                                -0.1288 |                               3.0945 |                                  -3.2351 |                                 23.0022 |                                 0.1866 |                                     -0.1643 |                                0.1909 |                                -0.1288 |                               3.3201 |                                  -3.4754 |                                 22.3753 |                                 0.1921 |                                     -0.1689 |                                0.1982 |                                -0.1287 |                               3.5492 |                                  -3.7199 |                                 21.7918 |                                 0.1973 |                                     -0.1733 |                                0.2053 |                                -0.1285 |                               3.7818 |                                  -3.9685 |                                 21.2471 |                                 0.2024 |                                     -0.1775 |                                0.2123 |                                -0.1282 |                               4.0174 |                                  -4.221  |                                 20.7373 |                                 0.2073 |                                     -0.1816 |                                0.2191 |                                -0.1279 |                               4.2561 |                                  -4.477  |                                 20.2589 |
        |            300 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 360.989 |                                 0      |                                     -0      |                                0      |                                -0.0002 |                               0      |                                  -0      |                                 188.858 |                                 0.0004 |                                     -0.0004 |                                0.0004 |                                -0.0021 |                               0.001  |                                  -0.001  |                                130.968  |                                 0.002  |                                     -0.0018 |                                0.0018 |                                -0.0075 |                               0.006  |                                  -0.0061 |                                101.735  |                                 0.0052 |                                     -0.0046 |                                0.0048 |                                -0.0155 |                               0.0191 |                                  -0.0193 |                                 84.0119 |                                 0.0098 |                                     -0.0088 |                                0.0091 |                                -0.0249 |                               0.0432 |                                  -0.0438 |                                 72.0724 |                                 0.0156 |                                     -0.0139 |                                0.0147 |                                -0.0345 |                               0.0799 |                                  -0.0812 |                                 63.4548 |                                 0.0222 |                                     -0.0197 |                                0.0213 |                                -0.0437 |                               0.1297 |                                  -0.132  |                                 56.9248 |                                 0.0293 |                                     -0.026  |                                0.0285 |                                -0.0521 |                               0.1926 |                                  -0.1964 |                                 51.7942 |                                 0.0368 |                                     -0.0326 |                                0.0362 |                                -0.0597 |                               0.2681 |                                  -0.2738 |                                 47.6489 |                                 0.0444 |                                     -0.0393 |                                0.0442 |                                -0.0664 |                               0.3554 |                                  -0.3636 |                                 44.2242 |                                 0.052  |                                     -0.046  |                                0.0525 |                                -0.0723 |                               0.454  |                                  -0.4652 |                                 41.3432 |                                 0.0597 |                                     -0.0527 |                                0.0609 |                                -0.0775 |                               0.5629 |                                  -0.5778 |                                 38.8827 |                                 0.0672 |                                     -0.0592 |                                0.0693 |                                -0.082  |                               0.6816 |                                  -0.7007 |                                 36.7546 |                                 0.0746 |                                     -0.0656 |                                0.0777 |                                -0.086  |                               0.8094 |                                  -0.8333 |                                 34.8939 |                                 0.0818 |                                     -0.0719 |                                0.086  |                                -0.0894 |                               0.9455 |                                  -0.9748 |                                 33.2518 |                                 0.0888 |                                     -0.078  |                                0.0944 |                                -0.0924 |                               1.0895 |                                  -1.1248 |                                 31.7905 |                                 0.0957 |                                     -0.0839 |                                0.1026 |                                -0.0949 |                               1.2407 |                                  -1.2828 |                                 30.481  |                                 0.1023 |                                     -0.0896 |                                0.1107 |                                -0.0972 |                               1.3987 |                                  -1.4481 |                                 29.2997 |                                 0.1088 |                                     -0.0951 |                                0.1187 |                                -0.0991 |                               1.563  |                                  -1.6204 |                                 28.2283 |                                 0.115  |                                     -0.1004 |                                0.1266 |                                -0.1008 |                               1.7333 |                                  -1.7993 |                                 27.2514 |                                 0.1211 |                                     -0.1056 |                                0.1344 |                                -0.1022 |                               1.9091 |                                  -1.9844 |                                 26.3565 |                                 0.127  |                                     -0.1106 |                                0.142  |                                -0.1035 |                               2.0901 |                                  -2.1753 |                                 25.5335 |                                 0.1327 |                                     -0.1154 |                                0.1496 |                                -0.1045 |                               2.2761 |                                  -2.3718 |                                 24.7735 |                                 0.1382 |                                     -0.12   |                                0.157  |                                -0.1055 |                               2.4666 |                                  -2.5735 |                                 24.0694 |                                 0.1436 |                                     -0.1245 |                                0.1643 |                                -0.1062 |                               2.6615 |                                  -2.7802 |                                 23.4149 |                                 0.1488 |                                     -0.1289 |                                0.1714 |                                -0.1069 |                               2.8605 |                                  -2.9917 |                                 22.8047 |                                 0.1538 |                                     -0.1331 |                                0.1785 |                                -0.1074 |                               3.0634 |                                  -3.2076 |                                 22.2343 |                                 0.1587 |                                     -0.1372 |                                0.1854 |                                -0.1078 |                               3.2699 |                                  -3.4279 |                                 21.6998 |
        |            305 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 415.99  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 215.496 |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0003 |                               0.0001 |                                  -0.0001 |                                148.283  |                                 0.0004 |                                     -0.0003 |                                0.0004 |                                -0.0016 |                               0.0012 |                                  -0.0012 |                                114.451  |                                 0.0014 |                                     -0.0012 |                                0.0014 |                                -0.0046 |                               0.005  |                                  -0.005  |                                 94.005  |                                 0.0032 |                                     -0.0028 |                                0.0033 |                                -0.0091 |                               0.0139 |                                  -0.0141 |                                 80.272  |                                 0.0058 |                                     -0.0051 |                                0.0062 |                                -0.0146 |                               0.0299 |                                  -0.0303 |                                 70.3878 |                                 0.0093 |                                     -0.0081 |                                0.01   |                                -0.0206 |                               0.0542 |                                  -0.0551 |                                 62.9178 |                                 0.0133 |                                     -0.0117 |                                0.0146 |                                -0.0267 |                               0.0879 |                                  -0.0895 |                                 57.0633 |                                 0.018  |                                     -0.0157 |                                0.0199 |                                -0.0327 |                               0.1313 |                                  -0.1338 |                                 52.3441 |                                 0.023  |                                     -0.0201 |                                0.0257 |                                -0.0385 |                               0.1845 |                                  -0.1883 |                                 48.4539 |                                 0.0283 |                                     -0.0247 |                                0.0319 |                                -0.0439 |                               0.2473 |                                  -0.2529 |                                 45.1881 |                                 0.0338 |                                     -0.0294 |                                0.0385 |                                -0.049  |                               0.3196 |                                  -0.3273 |                                 42.4045 |                                 0.0394 |                                     -0.0343 |                                0.0454 |                                -0.0536 |                               0.4009 |                                  -0.4112 |                                 40.0013 |                                 0.0451 |                                     -0.0392 |                                0.0524 |                                -0.0578 |                               0.491  |                                  -0.5043 |                                 37.9039 |                                 0.0509 |                                     -0.0441 |                                0.0595 |                                -0.0617 |                               0.5893 |                                  -0.6061 |                                 36.0559 |                                 0.0566 |                                     -0.049  |                                0.0667 |                                -0.0652 |                               0.6955 |                                  -0.7164 |                                 34.4142 |                                 0.0623 |                                     -0.0538 |                                0.074  |                                -0.0683 |                               0.8093 |                                  -0.8346 |                                 32.9451 |                                 0.0679 |                                     -0.0586 |                                0.0813 |                                -0.0712 |                               0.9301 |                                  -0.9605 |                                 31.6219 |                                 0.0734 |                                     -0.0633 |                                0.0886 |                                -0.0738 |                               1.0576 |                                  -1.0936 |                                 30.4234 |                                 0.0789 |                                     -0.0679 |                                0.0959 |                                -0.0761 |                               1.1916 |                                  -1.2336 |                                 29.332  |                                 0.0842 |                                     -0.0724 |                                0.1031 |                                -0.0783 |                               1.3315 |                                  -1.3802 |                                 28.3337 |                                 0.0895 |                                     -0.0769 |                                0.1104 |                                -0.0802 |                               1.4772 |                                  -1.5332 |                                 27.4166 |                                 0.0947 |                                     -0.0812 |                                0.1175 |                                -0.0819 |                               1.6284 |                                  -1.6921 |                                 26.5708 |                                 0.0997 |                                     -0.0854 |                                0.1246 |                                -0.0834 |                               1.7847 |                                  -1.8567 |                                 25.788  |                                 0.1047 |                                     -0.0896 |                                0.1316 |                                -0.0848 |                               1.9459 |                                  -2.0268 |                                 25.0612 |                                 0.1095 |                                     -0.0936 |                                0.1386 |                                -0.0861 |                               2.1118 |                                  -2.2021 |                                 24.3844 |                                 0.1142 |                                     -0.0975 |                                0.1454 |                                -0.0872 |                               2.2821 |                                  -2.3824 |                                 23.7523 |                                 0.1189 |                                     -0.1014 |                                0.1523 |                                -0.0883 |                               2.4566 |                                  -2.5675 |                                 23.1606 |
        |            310 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 470.605 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 242.094 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                165.632  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                127.224  |                                 0.0003 |                                     -0.0003 |                                0.0004 |                                -0.0012 |                               0.0011 |                                  -0.0011 |                                104.06   |                                 0.0009 |                                     -0.0008 |                                0.0011 |                                -0.0029 |                               0.004  |                                  -0.004  |                                 88.5331 |                                 0.002  |                                     -0.0017 |                                0.0023 |                                -0.0055 |                               0.01   |                                  -0.0102 |                                 77.3795 |                                 0.0035 |                                     -0.003  |                                0.0043 |                                -0.0087 |                               0.0207 |                                  -0.021  |                                 68.9659 |                                 0.0056 |                                     -0.0048 |                                0.0068 |                                -0.0125 |                               0.0371 |                                  -0.0377 |                                 62.3837 |                                 0.0082 |                                     -0.0071 |                                0.0101 |                                -0.0165 |                               0.0599 |                                  -0.061  |                                 57.087  |                                 0.0112 |                                     -0.0096 |                                0.0139 |                                -0.0207 |                               0.0898 |                                  -0.0915 |                                 52.7279 |                                 0.0145 |                                     -0.0125 |                                0.0181 |                                -0.0249 |                               0.1271 |                                  -0.1297 |                                 49.0741 |                                 0.0182 |                                     -0.0156 |                                0.0229 |                                -0.029  |                               0.172  |                                  -0.1758 |                                 45.9645 |                                 0.022  |                                     -0.0189 |                                0.028  |                                -0.033  |                               0.2245 |                                  -0.2298 |                                 43.2838 |                                 0.0261 |                                     -0.0223 |                                0.0334 |                                -0.0368 |                               0.2845 |                                  -0.2917 |                                 40.9473 |                                 0.0303 |                                     -0.0259 |                                0.0391 |                                -0.0404 |                               0.352  |                                  -0.3613 |                                 38.8913 |                                 0.0346 |                                     -0.0296 |                                0.045  |                                -0.0438 |                               0.4268 |                                  -0.4386 |                                 37.0671 |                                 0.039  |                                     -0.0333 |                                0.051  |                                -0.047  |                               0.5086 |                                  -0.5233 |                                 35.4367 |                                 0.0435 |                                     -0.037  |                                0.0572 |                                -0.05   |                               0.5972 |                                  -0.6153 |                                 33.97   |                                 0.0479 |                                     -0.0408 |                                0.0635 |                                -0.0527 |                               0.6924 |                                  -0.7142 |                                 32.6428 |                                 0.0524 |                                     -0.0445 |                                0.0698 |                                -0.0553 |                               0.7939 |                                  -0.82   |                                 31.4357 |                                 0.0569 |                                     -0.0482 |                                0.0762 |                                -0.0577 |                               0.9015 |                                  -0.9322 |                                 30.3326 |                                 0.0613 |                                     -0.052  |                                0.0827 |                                -0.0599 |                               1.0149 |                                  -1.0507 |                                 29.3203 |                                 0.0658 |                                     -0.0556 |                                0.0892 |                                -0.062  |                               1.134  |                                  -1.1754 |                                 28.3876 |                                 0.0701 |                                     -0.0593 |                                0.0957 |                                -0.0639 |                               1.2584 |                                  -1.3058 |                                 27.5252 |                                 0.0745 |                                     -0.0629 |                                0.1021 |                                -0.0657 |                               1.388  |                                  -1.442  |                                 26.7252 |                                 0.0787 |                                     -0.0664 |                                0.1086 |                                -0.0673 |                               1.5226 |                                  -1.5835 |                                 25.9808 |                                 0.083  |                                     -0.0699 |                                0.1151 |                                -0.0688 |                               1.6619 |                                  -1.7303 |                                 25.2863 |                                 0.0871 |                                     -0.0733 |                                0.1215 |                                -0.0702 |                               1.8057 |                                  -1.8821 |                                 24.6366 |
        |            315 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 524.702 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 268.551 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                182.942  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                139.995  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                114.131  |                                 0.0002 |                                     -0.0002 |                                0.0003 |                                -0.0008 |                               0.001  |                                  -0.001  |                                 96.8174 |                                 0.0006 |                                     -0.0005 |                                0.0008 |                                -0.0018 |                               0.0031 |                                  -0.0031 |                                 84.3978 |                                 0.0012 |                                     -0.001  |                                0.0016 |                                -0.0034 |                               0.0072 |                                  -0.0073 |                                 75.0417 |                                 0.0022 |                                     -0.0019 |                                0.0029 |                                -0.0054 |                               0.0145 |                                  -0.0147 |                                 67.7318 |                                 0.0035 |                                     -0.003  |                                0.0047 |                                -0.0077 |                               0.0255 |                                  -0.0259 |                                 61.8571 |                                 0.0051 |                                     -0.0043 |                                0.0069 |                                -0.0104 |                               0.0411 |                                  -0.0418 |                                 57.0282 |                                 0.007  |                                     -0.006  |                                0.0097 |                                -0.0132 |                               0.0617 |                                  -0.0629 |                                 52.9853 |                                 0.0093 |                                     -0.0078 |                                0.0128 |                                -0.0162 |                               0.0879 |                                  -0.0897 |                                 49.5486 |                                 0.0117 |                                     -0.0099 |                                0.0163 |                                -0.0192 |                               0.1198 |                                  -0.1225 |                                 46.5891 |                                 0.0145 |                                     -0.0122 |                                0.0202 |                                -0.0222 |                               0.1578 |                                  -0.1614 |                                 44.0124 |                                 0.0174 |                                     -0.0146 |                                0.0244 |                                -0.0252 |                               0.2018 |                                  -0.2068 |                                 41.7473 |                                 0.0204 |                                     -0.0172 |                                0.0289 |                                -0.0281 |                               0.252  |                                  -0.2585 |                                 39.7396 |                                 0.0236 |                                     -0.0198 |                                0.0336 |                                -0.0309 |                               0.3083 |                                  -0.3166 |                                 37.9469 |                                 0.0269 |                                     -0.0226 |                                0.0386 |                                -0.0336 |                               0.3706 |                                  -0.3811 |                                 36.3357 |                                 0.0303 |                                     -0.0254 |                                0.0437 |                                -0.0362 |                               0.4389 |                                  -0.4519 |                                 34.8791 |                                 0.0338 |                                     -0.0283 |                                0.049  |                                -0.0387 |                               0.513  |                                  -0.5288 |                                 33.5554 |                                 0.0373 |                                     -0.0312 |                                0.0544 |                                -0.0411 |                               0.5929 |                                  -0.6118 |                                 32.3468 |                                 0.0409 |                                     -0.0342 |                                0.0599 |                                -0.0433 |                               0.6784 |                                  -0.7008 |                                 31.2385 |                                 0.0445 |                                     -0.0371 |                                0.0655 |                                -0.0454 |                               0.7692 |                                  -0.7955 |                                 30.2182 |                                 0.0481 |                                     -0.0401 |                                0.0712 |                                -0.0474 |                               0.8654 |                                  -0.896  |                                 29.2756 |                                 0.0517 |                                     -0.0431 |                                0.0769 |                                -0.0493 |                               0.9666 |                                  -1.0019 |                                 28.4017 |                                 0.0554 |                                     -0.046  |                                0.0827 |                                -0.0511 |                               1.0728 |                                  -1.1131 |                                 27.5893 |                                 0.059  |                                     -0.049  |                                0.0885 |                                -0.0528 |                               1.1838 |                                  -1.2296 |                                 26.8318 |                                 0.0626 |                                     -0.0519 |                                0.0943 |                                -0.0544 |                               1.2994 |                                  -1.3512 |                                 26.1236 |
        |            320 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 578.202 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 294.805 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                200.161  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                152.724  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                124.183  |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                105.096  |                                 0.0002 |                                     -0.0001 |                                0.0002 |                                -0.0006 |                               0.0008 |                                  -0.0009 |                                 91.4179 |                                 0.0004 |                                     -0.0003 |                                0.0006 |                                -0.0012 |                               0.0023 |                                  -0.0024 |                                 81.124  |                                 0.0008 |                                     -0.0007 |                                0.0012 |                                -0.0021 |                               0.0052 |                                  -0.0053 |                                 73.0892 |                                 0.0014 |                                     -0.0012 |                                0.002  |                                -0.0033 |                               0.0102 |                                  -0.0103 |                                 66.6379 |                                 0.0022 |                                     -0.0018 |                                0.0033 |                                -0.0049 |                               0.0177 |                                  -0.018  |                                 61.3401 |                                 0.0032 |                                     -0.0027 |                                0.0048 |                                -0.0066 |                               0.0284 |                                  -0.0289 |                                 56.9087 |                                 0.0045 |                                     -0.0037 |                                0.0067 |                                -0.0085 |                               0.0427 |                                  -0.0435 |                                 53.145  |                                 0.006  |                                     -0.005  |                                0.009  |                                -0.0106 |                               0.0611 |                                  -0.0623 |                                 49.9067 |                                 0.0077 |                                     -0.0064 |                                0.0116 |                                -0.0128 |                               0.0838 |                                  -0.0856 |                                 47.0895 |                                 0.0095 |                                     -0.0079 |                                0.0146 |                                -0.015  |                               0.1112 |                                  -0.1137 |                                 44.6152 |                                 0.0116 |                                     -0.0096 |                                0.0178 |                                -0.0173 |                               0.1433 |                                  -0.1468 |                                 42.4237 |                                 0.0138 |                                     -0.0114 |                                0.0213 |                                -0.0195 |                               0.1804 |                                  -0.185  |                                 40.4683 |                                 0.0161 |                                     -0.0134 |                                0.025  |                                -0.0218 |                               0.2225 |                                  -0.2284 |                                 38.7122 |                                 0.0186 |                                     -0.0154 |                                0.029  |                                -0.024  |                               0.2697 |                                  -0.2772 |                                 37.1259 |                                 0.0212 |                                     -0.0175 |                                0.0331 |                                -0.0262 |                               0.3219 |                                  -0.3312 |                                 35.6852 |                                 0.0238 |                                     -0.0197 |                                0.0375 |                                -0.0283 |                               0.3792 |                                  -0.3906 |                                 34.3707 |                                 0.0266 |                                     -0.0219 |                                0.042  |                                -0.0303 |                               0.4415 |                                  -0.4552 |                                 33.1661 |                                 0.0294 |                                     -0.0242 |                                0.0466 |                                -0.0323 |                               0.5087 |                                  -0.5251 |                                 32.0579 |                                 0.0322 |                                     -0.0265 |                                0.0514 |                                -0.0342 |                               0.5808 |                                  -0.6001 |                                 31.0346 |                                 0.0351 |                                     -0.0289 |                                0.0562 |                                -0.036  |                               0.6577 |                                  -0.6803 |                                 30.0867 |                                 0.0381 |                                     -0.0312 |                                0.0612 |                                -0.0378 |                               0.7392 |                                  -0.7654 |                                 29.2058 |                                 0.041  |                                     -0.0336 |                                0.0662 |                                -0.0394 |                               0.8254 |                                  -0.8555 |                                 28.385  |                                 0.044  |                                     -0.036  |                                0.0713 |                                -0.041  |                               0.9161 |                                  -0.9505 |                                 27.6181 |
        |            325 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 631.061 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 320.813 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                217.254  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                165.381  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                134.19   |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                113.347  |                                 0      |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                 98.4212 |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0004 |                               0.0007 |                                  -0.0007 |                                 87.1963 |                                 0.0003 |                                     -0.0002 |                                0.0004 |                                -0.0008 |                               0.0018 |                                  -0.0018 |                                 78.4411 |                                 0.0005 |                                     -0.0004 |                                0.0008 |                                -0.0014 |                               0.0038 |                                  -0.0039 |                                 71.4166 |                                 0.0009 |                                     -0.0007 |                                0.0014 |                                -0.0021 |                               0.0072 |                                  -0.0073 |                                 65.6521 |                                 0.0014 |                                     -0.0012 |                                0.0023 |                                -0.0031 |                               0.0124 |                                  -0.0126 |                                 60.8337 |                                 0.0021 |                                     -0.0017 |                                0.0034 |                                -0.0043 |                               0.0198 |                                  -0.0201 |                                 56.7441 |                                 0.0029 |                                     -0.0024 |                                0.0047 |                                -0.0056 |                               0.0298 |                                  -0.0304 |                                 53.2278 |                                 0.0039 |                                     -0.0032 |                                0.0064 |                                -0.007  |                               0.0427 |                                  -0.0436 |                                 50.1708 |                                 0.0051 |                                     -0.0041 |                                0.0083 |                                -0.0086 |                               0.0589 |                                  -0.0602 |                                 47.4875 |                                 0.0064 |                                     -0.0052 |                                0.0105 |                                -0.0102 |                               0.0787 |                                  -0.0804 |                                 45.1125 |                                 0.0078 |                                     -0.0064 |                                0.0129 |                                -0.0119 |                               0.1021 |                                  -0.1045 |                                 42.9947 |                                 0.0094 |                                     -0.0077 |                                0.0156 |                                -0.0136 |                               0.1294 |                                  -0.1327 |                                 41.0938 |                                 0.0111 |                                     -0.009  |                                0.0185 |                                -0.0153 |                               0.1608 |                                  -0.165  |                                 39.3776 |                                 0.0129 |                                     -0.0105 |                                0.0217 |                                -0.0171 |                               0.1963 |                                  -0.2017 |                                 37.82   |                                 0.0148 |                                     -0.012  |                                0.025  |                                -0.0188 |                               0.236  |                                  -0.2427 |                                 36.3996 |                                 0.0168 |                                     -0.0137 |                                0.0285 |                                -0.0206 |                               0.28   |                                  -0.2882 |                                 35.0986 |                                 0.0189 |                                     -0.0154 |                                0.0322 |                                -0.0223 |                               0.3282 |                                  -0.3382 |                                 33.9024 |                                 0.0211 |                                     -0.0171 |                                0.036  |                                -0.0239 |                               0.3808 |                                  -0.3927 |                                 32.7984 |                                 0.0233 |                                     -0.0189 |                                0.04   |                                -0.0256 |                               0.4375 |                                  -0.4517 |                                 31.7762 |                                 0.0256 |                                     -0.0207 |                                0.0441 |                                -0.0272 |                               0.4985 |                                  -0.5152 |                                 30.8269 |                                 0.028  |                                     -0.0226 |                                0.0483 |                                -0.0287 |                               0.5637 |                                  -0.5832 |                                 29.9426 |                                 0.0304 |                                     -0.0245 |                                0.0526 |                                -0.0302 |                               0.6331 |                                  -0.6556 |                                 29.1169 |
        |            330 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 683.256 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 346.55  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                234.197  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                177.944  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                144.136  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                121.555  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                105.393  |                                 0      |                                     -0      |                                0.0001 |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 93.2459 |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0006 |                                  -0.0006 |                                 83.7763 |                                 0.0002 |                                     -0.0001 |                                0.0003 |                                -0.0005 |                               0.0013 |                                  -0.0014 |                                 76.1828 |                                 0.0003 |                                     -0.0003 |                                0.0006 |                                -0.0009 |                               0.0028 |                                  -0.0028 |                                 69.9548 |                                 0.0006 |                                     -0.0005 |                                0.001  |                                -0.0014 |                               0.0051 |                                  -0.0052 |                                 64.7519 |                                 0.0009 |                                     -0.0007 |                                0.0016 |                                -0.002  |                               0.0088 |                                  -0.0089 |                                 60.3383 |                                 0.0014 |                                     -0.0011 |                                0.0024 |                                -0.0028 |                               0.0139 |                                  -0.0142 |                                 56.5454 |                                 0.0019 |                                     -0.0015 |                                0.0034 |                                -0.0037 |                               0.0209 |                                  -0.0213 |                                 53.2497 |                                 0.0026 |                                     -0.0021 |                                0.0045 |                                -0.0047 |                               0.0301 |                                  -0.0307 |                                 50.3584 |                                 0.0034 |                                     -0.0027 |                                0.0059 |                                -0.0058 |                               0.0417 |                                  -0.0426 |                                 47.8004 |                                 0.0043 |                                     -0.0034 |                                0.0076 |                                -0.0069 |                               0.0559 |                                  -0.0572 |                                 45.5207 |                                 0.0053 |                                     -0.0043 |                                0.0094 |                                -0.0082 |                               0.073  |                                  -0.0747 |                                 43.4755 |                                 0.0064 |                                     -0.0052 |                                0.0115 |                                -0.0095 |                               0.0932 |                                  -0.0954 |                                 41.6299 |                                 0.0076 |                                     -0.0061 |                                0.0137 |                                -0.0108 |                               0.1165 |                                  -0.1195 |                                 39.9556 |                                 0.009  |                                     -0.0072 |                                0.0162 |                                -0.0122 |                               0.1431 |                                  -0.147  |                                 38.4294 |                                 0.0104 |                                     -0.0083 |                                0.0188 |                                -0.0135 |                               0.1732 |                                  -0.178  |                                 37.0323 |                                 0.0119 |                                     -0.0095 |                                0.0216 |                                -0.0149 |                               0.2068 |                                  -0.2128 |                                 35.7481 |                                 0.0135 |                                     -0.0108 |                                0.0246 |                                -0.0163 |                               0.244  |                                  -0.2513 |                                 34.5636 |                                 0.0152 |                                     -0.0121 |                                0.0277 |                                -0.0177 |                               0.2848 |                                  -0.2936 |                                 33.4673 |                                 0.0169 |                                     -0.0135 |                                0.031  |                                -0.0191 |                               0.3293 |                                  -0.3397 |                                 32.4494 |                                 0.0187 |                                     -0.0149 |                                0.0344 |                                -0.0204 |                               0.3774 |                                  -0.3898 |                                 31.5019 |                                 0.0205 |                                     -0.0164 |                                0.0379 |                                -0.0218 |                               0.4292 |                                  -0.4437 |                                 30.6173 |
        |            335 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 734.777 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 371.998 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                250.975  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                190.398  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                154.005  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                129.707  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                112.324  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 99.263  |                                 0      |                                     -0      |                                0      |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 89.0859 |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0004 |                                  -0.0004 |                                 80.9286 |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0003 |                               0.001  |                                  -0.001  |                                 74.241  |                                 0.0002 |                                     -0.0002 |                                0.0004 |                                -0.0006 |                               0.002  |                                  -0.0021 |                                 68.6565 |                                 0.0004 |                                     -0.0003 |                                0.0007 |                                -0.0009 |                               0.0037 |                                  -0.0038 |                                 63.9212 |                                 0.0006 |                                     -0.0005 |                                0.0011 |                                -0.0013 |                               0.0062 |                                  -0.0063 |                                 59.8537 |                                 0.0009 |                                     -0.0007 |                                0.0017 |                                -0.0018 |                               0.0099 |                                  -0.0101 |                                 56.3208 |                                 0.0013 |                                     -0.001  |                                0.0024 |                                -0.0025 |                               0.0148 |                                  -0.0151 |                                 53.2227 |                                 0.0017 |                                     -0.0014 |                                0.0032 |                                -0.0031 |                               0.0214 |                                  -0.0218 |                                 50.4829 |                                 0.0023 |                                     -0.0018 |                                0.0043 |                                -0.0039 |                               0.0297 |                                  -0.0303 |                                 48.0421 |                                 0.0029 |                                     -0.0023 |                                0.0055 |                                -0.0048 |                               0.04   |                                  -0.0409 |                                 45.8533 |                                 0.0036 |                                     -0.0029 |                                0.0069 |                                -0.0057 |                               0.0525 |                                  -0.0537 |                                 43.8788 |                                 0.0044 |                                     -0.0035 |                                0.0084 |                                -0.0066 |                               0.0673 |                                  -0.0689 |                                 42.0884 |                                 0.0053 |                                     -0.0042 |                                0.0102 |                                -0.0076 |                               0.0847 |                                  -0.0868 |                                 40.4569 |                                 0.0063 |                                     -0.005  |                                0.0121 |                                -0.0087 |                               0.1046 |                                  -0.1074 |                                 38.9639 |                                 0.0073 |                                     -0.0058 |                                0.0141 |                                -0.0097 |                               0.1274 |                                  -0.1309 |                                 37.5922 |                                 0.0085 |                                     -0.0067 |                                0.0164 |                                -0.0108 |                               0.153  |                                  -0.1573 |                                 36.3273 |                                 0.0097 |                                     -0.0076 |                                0.0187 |                                -0.0119 |                               0.1816 |                                  -0.1869 |                                 35.1571 |                                 0.0109 |                                     -0.0086 |                                0.0213 |                                -0.0131 |                               0.2132 |                                  -0.2196 |                                 34.071  |                                 0.0123 |                                     -0.0096 |                                0.0239 |                                -0.0142 |                               0.2478 |                                  -0.2556 |                                 33.0602 |                                 0.0136 |                                     -0.0107 |                                0.0268 |                                -0.0153 |                               0.2856 |                                  -0.2948 |                                 32.1169 |
        """
        greeks = {}

        greeks["Delta"] = self.get_delta(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Dual Delta"] = self.get_dual_delta(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Vega"] = self.get_vega(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Theta"] = self.get_theta(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Rho"] = self.get_rho(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Epsilon"] = self.get_epsilon(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Lambda"] = self.get_lambda(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    def get_delta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the delta of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The delta is
        the rate of change of the option price with respect to the price of the underlying asset.

        The delta calculation is the theoretical value of the delta. The actual delta can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - Call Option Delta = e^(—q * t) * N(d1)
        - Put Option Delta = —e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Delta can be interpreted as follows:

        - For call options, Delta is positive, indicating that the option price tends to move in the same direction as the
        underlying asset's price.
        - For put options, Delta is negative, indicating that the option price tends to move in the opposite direction to the
        underlying asset's price.

        Note that the delta of a call option is always between 0 and e^(—q * t), while the delta of a put option
        is always between —e^(—q * t) and 0. Without a dividend yield those bounds collapse to the familiar
        0 to 1 and —1 to 0.

        Delta is read as the hedge ratio: a call with a delta of 0.6 moves about $0.60 for a $1 move in the stock,
        so a market maker that is short calls on 100 shares buys 60 shares to stay neutral. It is often used as a
        quick gauge of how far in or out of the money an option is (traders talk of "25-delta" puts), and loosely
        as the chance of finishing in the money, although that (risk-neutral) probability is N(d2), which
        `get_dual_delta` gives up to discounting and sign. Delta changes with the stock price and with time (see
        gamma and charm), so a delta hedge has to be rebalanced.

        Also known as: option price sensitivity to underlying, hedge ratio.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option delta. Defaults to False which means
            it will calculate the call option delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the delta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_delta(
            strike_price_range=0.15, strike_step_size=5, expiration_time_range=91
        ).loc["AMZN", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            195 |       0.9632 |       0.9073 |       0.8707 |
        |            200 |       0.9374 |       0.8736 |       0.8369 |
        |            205 |       0.9004 |       0.8332 |       0.7989 |
        |            210 |       0.8508 |       0.7865 |       0.7571 |
        |            215 |       0.7887 |       0.7344 |       0.7121 |
        |            220 |       0.7154 |       0.6779 |       0.6647 |
        |            225 |       0.6338 |       0.6183 |       0.6156 |
        |            230 |       0.5476 |       0.5572 |       0.5658 |
        |            235 |       0.461  |       0.496  |       0.516  |
        |            240 |       0.378  |       0.4362 |       0.467  |
        |            245 |       0.3017 |       0.379  |       0.4195 |
        |            250 |       0.2346 |       0.3254 |       0.374  |
        |            255 |       0.1777 |       0.2762 |       0.3311 |
        |            260 |       0.1313 |       0.2317 |       0.2911 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        delta: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            delta[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            delta[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_delta(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        delta_df = helpers.create_greek_dataframe(
            greek_dictionary=delta,
            start_date=start_date,
        )

        delta_df = delta_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            delta_df = calculate_standardization(
                dataset=delta_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return delta_df

    def get_dual_delta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the dual delta of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The dual delta is
        the first derivative of the option price with respect to the strike price; up to the discount
        factor and its sign it is the risk-neutral probability of the option finishing in the money.

        The dual delta calculation is the theoretical value of the dual delta. The actual dual delta can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Dual Delta = —e^(—r * t) * N(d2)
        - Put Dual Delta = e^(—r * t) * N(—d2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Dual Delta is the sensitivity of the option value to the strike price rather than to the underlying price.
        Up to the discount factor and a sign it is the risk-neutral probability that the option finishes in the money:
        a call Dual Delta of —0.5 corresponds to a roughly 50% chance of finishing in the money. It is negative for a
        call, since raising the strike lowers the call's value, and positive for a put.

        Also known as: strike delta, binary option delta.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option dual delta. Defaults to False which means
            it will calculate the call option dual delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the dual delta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["JPM"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_dual_delta(
            strike_price_range=0.15, strike_step_size=5, expiration_time_range=91
        ).loc["JPM", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            275 |      -0.9832 |      -0.9341 |      -0.8885 |
        |            280 |      -0.9717 |      -0.91   |      -0.8604 |
        |            285 |      -0.9533 |      -0.8799 |      -0.8278 |
        |            290 |      -0.9258 |      -0.8434 |      -0.791  |
        |            295 |      -0.8874 |      -0.8005 |      -0.7504 |
        |            300 |      -0.8368 |      -0.7519 |      -0.7065 |
        |            305 |      -0.7742 |      -0.6982 |      -0.6599 |
        |            310 |      -0.7009 |      -0.6406 |      -0.6113 |
        |            315 |      -0.6195 |      -0.5805 |      -0.5616 |
        |            320 |      -0.5336 |      -0.5193 |      -0.5116 |
        |            325 |      -0.4474 |      -0.4585 |      -0.4621 |
        |            330 |      -0.3649 |      -0.3995 |      -0.4139 |
        |            335 |      -0.2892 |      -0.3435 |      -0.3676 |
        |            340 |      -0.2227 |      -0.2915 |      -0.3237 |
        |            345 |      -0.1667 |      -0.2441 |      -0.2827 |
        |            350 |      -0.1213 |      -0.2018 |      -0.2449 |
        |            355 |      -0.0858 |      -0.1646 |      -0.2105 |
        |            360 |      -0.059  |      -0.1327 |      -0.1794 |
        |            365 |      -0.0395 |      -0.1056 |      -0.1518 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        dual_delta: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            dual_delta[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            dual_delta[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_dual_delta(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        dual_delta_df = helpers.create_greek_dataframe(
            greek_dictionary=dual_delta,
            start_date=start_date,
        )

        dual_delta_df = dual_delta_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            dual_delta_df = calculate_standardization(
                dataset=dual_delta_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return dual_delta_df

    def get_vega(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the vega of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The vega is
        the rate of change of the option price with respect to the volatility of the underlying asset.

        The vega calculation is the theoretical value of the vega. The actual vega can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - Vega = S * e^(—q * t) * N'(d1) * sqrt(t) / 100

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d1) is the standard normal probability density at d1 and N(d2) is
        the cumulative normal distribution of d2.

        The division by 100 expresses Vega per 1 percentage point change in volatility, the usual market quote. The
        higher order volatility Greeks (Vanna, Vomma, Zomma, Vera, Ultima) are reported unscaled, per 1.00 of
        volatility; only Vega and Veta carry this factor.

        The Vega can be interpreted as follows:

        - If Vega is positive, it indicates that the option value will increase as the volatility increases,
        and vice versa.
        - If Vega is negative, it implies that the option value will decrease as the volatility increases,
        and vice versa.

        Note that the vega of a call option and put option are equal to each other.

        Vega is quoted per volatility point: a vega of 0.25 means the option gains about $0.25 when volatility rises
        from 20% to 21%. It is largest for at-the-money options and grows with the square root of the time to
        expiry, so long-dated options are mostly a bet on volatility while short-dated ones are mostly a bet on
        gamma. Desks sum vega across positions per expiry, since short- and long-dated implied volatilities rarely
        move by the same amount.

        Also known as: option sensitivity to volatility changes.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the vega values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NFLX"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vega(
            strike_price_range=0.15, strike_step_size=2, expiration_time_range=91
        ).loc["NFLX", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             80 |       0.0248 |       0.0678 |       0.1032 |
        |             82 |       0.037  |       0.0837 |       0.1196 |
        |             84 |       0.0515 |       0.0998 |       0.1354 |
        |             86 |       0.067  |       0.1149 |       0.1498 |
        |             88 |       0.0819 |       0.1283 |       0.1623 |
        |             90 |       0.0945 |       0.1392 |       0.1723 |
        |             92 |       0.1032 |       0.1468 |       0.1797 |
        |             94 |       0.1071 |       0.1509 |       0.1842 |
        |             96 |       0.1059 |       0.1514 |       0.1857 |
        |             98 |       0.1001 |       0.1486 |       0.1845 |
        |            100 |       0.0908 |       0.1427 |       0.1806 |
        |            102 |       0.0791 |       0.1343 |       0.1744 |
        |            104 |       0.0664 |       0.1241 |       0.1664 |
        |            106 |       0.0538 |       0.1126 |       0.1568 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        vega: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            vega[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            vega[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_vega(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        vega_df = helpers.create_greek_dataframe(
            greek_dictionary=vega,
            start_date=start_date,
        )

        vega_df = vega_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            vega_df = calculate_standardization(
                dataset=vega_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return vega_df

    def get_theta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the theta of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The theta is
        the rate of change of the option price with respect to the passage of time.

        The theta calculation is the theoretical value of the theta. The actual theta can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Theta = [—e^(—q * t) * (S * N'(d1) * σ) / (2 * sqrt(t)) — r * K * e^(—r * t) * N(d2)
        + q * S * e^(—q * t) * N(d1)] / 365
        - Put Theta = [—e^(—q * t) * (S * N'(d1) * σ) / (2 * sqrt(t)) + r * K * e^(—r * t) * N(—d2)
        — q * S * e^(—q * t) * N(—d1)] / 365

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d1) is the standard normal probability density at d1 and N(d2) is
        the cumulative normal distribution of d2.

        Theta is the derivative with respect to calendar time elapsed, not with respect to the remaining time to
        maturity, and the division by 365 expresses it per calendar day rather than per year. Charm, Veta and Color
        measure the same passage of time in the same direction.

        The Theta can be interpreted as follows:

        - If Theta is negative, the option loses value with each day that passes, all else equal. This is the normal
        case for a long option, whose time value erodes towards expiration.
        - If Theta is positive, the option gains value with each day that passes. This happens for instance on a deep
        in-the-money European put, where the discounting of the strike dominates.

        Theta is the cost of holding an option for one more calendar day, all else equal, and is the other side of
        gamma: a long at-the-money option close to expiry has the most gamma and pays for it with the most negative
        theta. Because it is per calendar day, a weekend counts as two days of decay; desks that use a trading-day
        convention divide the yearly figure by 252 instead, which gives a larger daily number.

        Also known as: time decay, option time value erosion.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option theta. Defaults to False which means
            it will calculate the call option theta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the theta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AVGO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_theta(
            strike_price_range=0.15, strike_step_size=10
        ).loc["AVGO", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0622 |      -0.1332 |      -0.18   |
        |            300 |      -0.1272 |      -0.2073 |      -0.2284 |
        |            310 |      -0.2474 |      -0.2972 |      -0.2762 |
        |            320 |      -0.4168 |      -0.3897 |      -0.3184 |
        |            330 |      -0.5934 |      -0.4679 |      -0.3511 |
        |            340 |      -0.7151 |      -0.5165 |      -0.371  |
        |            350 |      -0.7358 |      -0.5271 |      -0.3771 |
        |            360 |      -0.6538 |      -0.4999 |      -0.3696 |
        |            370 |      -0.5071 |      -0.4432 |      -0.3502 |
        |            380 |      -0.347  |      -0.369  |      -0.3216 |
        |            390 |      -0.2114 |      -0.2901 |      -0.287  |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        theta: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            theta[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            theta[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_theta(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        theta_df = helpers.create_greek_dataframe(
            greek_dictionary=theta,
            start_date=start_date,
        )

        theta_df = theta_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            theta_df = calculate_standardization(
                dataset=theta_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return theta_df

    def get_rho(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the rho of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The rho is
        the rate of change of the option price with respect to the risk free interest rate.

        The rho calculation is the theoretical value of the rho. The actual rho can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Rho = K * t * e^(—r * t) * N(d2)
        - Put Rho = —K * t * e^(—r * t) * N(—d2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Rho can be interpreted as follows:

        - If Rho is positive, it indicates that the option value will increase as the risk free rate increases,
        and vice versa.
        - If Rho is negative, it implies that the option value will decrease as the risk free rate increases,
        and vice versa.

        Rho is reported unscaled, as the amount of money per share of the underlying that the value of the option
        gains or loses per 1.00 change in the risk—free rate. Divide by 100 for the more commonly quoted move per
        1.0% per annum (100 basis points). Epsilon and Vera follow the same unscaled convention, while Vega and Veta
        are already divided by 100.

        Also known as: option sensitivity to interest rate.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option rho. Defaults to False which means
            it will calculate the call option rho.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the rho values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PG"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_rho(
            strike_price_range=0.15, strike_step_size=5, expiration_time_range=181
        ).loc["PG", ["2026-01-30", "2026-03-31", "2026-06-29"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-31 |   2026-06-29 |
        |---------------:|-------------:|-------------:|-------------:|
        |            120 |       9.8242 |      28.3961 |      52.5163 |
        |            125 |      10.1773 |      28.2204 |      50.9291 |
        |            130 |      10.255  |      26.8642 |      47.8651 |
        |            135 |       9.5384 |      24.1409 |      43.4449 |
        |            140 |       7.609  |      20.248  |      38.0022 |
        |            145 |       4.8851 |      15.7391 |      32.0076 |
        |            150 |       2.4336 |      11.3025 |      25.961  |
        |            155 |       0.9292 |       7.4974 |      20.2949 |
        |            160 |       0.2728 |       4.6028 |      15.3122 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        rho: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            rho[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            rho[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_rho(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        rho_df = helpers.create_greek_dataframe(
            greek_dictionary=rho,
            start_date=start_date,
        )

        rho_df = rho_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            rho_df = calculate_standardization(
                dataset=rho_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return rho_df

    def get_epsilon(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the epsilon of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The epsilon is
        the rate of change of the option price with respect to the dividend yield.

        The epsilon calculation is the theoretical value of the epsilon. The actual epsilon can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - Call Epsilon = —S * t * e^(—q * t) * N(d1)
        - Put Epsilon = S * t * e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        Epsilon is reported unscaled, per 1.00 change in the dividend yield, matching Rho and Vera.

        The Epsilon can be interpreted as follows:

        - If Epsilon is positive, it indicates that the option value will increase as the dividend yield increases,
        and vice versa.
        - If Epsilon is negative, it implies that the option value will decrease as the dividend yield increases,
        and vice versa.

        Also known as: option sensitivity to dividend yield.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option epsilon. Defaults to False which means
            it will calculate the call option epsilon.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the epsilon values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_epsilon(
            strike_price_range=0.15, strike_step_size=2, expiration_time_range=181
        ).loc["KO", ["2026-01-30", "2026-03-31", "2026-06-29"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-31 |   2026-06-29 |
        |---------------:|-------------:|-------------:|-------------:|
        |             60 |      -5.7266 |     -16.5433 |     -30.9881 |
        |             62 |      -5.6893 |     -15.8719 |     -29.2684 |
        |             64 |      -5.5266 |     -14.7634 |     -27.0588 |
        |             66 |      -5.0603 |     -13.1795 |     -24.4187 |
        |             68 |      -4.1522 |     -11.1958 |     -21.4667 |
        |             70 |      -2.9117 |      -8.9944 |     -18.3606 |
        |             72 |      -1.6892 |      -6.8079 |     -15.2693 |
        |             74 |      -0.7977 |      -4.8468 |     -12.3463 |
        |             76 |      -0.3054 |      -3.2454 |      -9.7093 |
        |             78 |      -0.0951 |      -2.0461 |      -7.4315 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        epsilon: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            epsilon[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            epsilon[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_epsilon(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        epsilon_df = helpers.create_greek_dataframe(
            greek_dictionary=epsilon,
            start_date=start_date,
        )

        epsilon_df = epsilon_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            epsilon_df = calculate_standardization(
                dataset=epsilon_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return epsilon_df

    def get_lambda(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the lambda of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The lambda, or
        elasticity, is the percentage change in the option price for a 1% change in the price of the
        underlying asset, i.e. the leverage the option gives.

        The lambda calculation is the theoretical value of the lambda. The actual lambda can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Delta = e^(—q * t) * N(d1), Put Delta = —e^(—q * t) * N(—d1)
        - Call Option Price = S * e^(—q * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—q * t) * N(—d1)
        - Lambda = Delta * (Stock Price / Call Option Price or Put Option Price)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Lambda can be interpreted as follows:

        - If Lambda is positive, it indicates that the option value will increase as the underlying price increases,
        and vice versa.
        - If Lambda is negative, it implies that the option value will decrease as the underlying price increases,
        and vice versa.

        Also known as: option elasticity, leverage factor.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            put_option (bool, optional): Whether to calculate the put option lambda. Defaults to False which means
            it will calculate the call option lambda.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the lambda values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["V"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_lambda(
            strike_price_range=0.1, strike_step_size=5, expiration_time_range=91
        ).loc["V", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            315 |       9.0898 |       8.0785 |       7.3271 |
        |            320 |      10.1189 |       8.7379 |       7.8111 |
        |            325 |      11.2877 |       9.4477 |       8.321  |
        |            330 |      12.5998 |      10.2062 |       8.8559 |
        |            335 |      14.0544 |      11.0116 |       9.4143 |
        |            340 |      15.647  |      11.8613 |       9.9949 |
        |            345 |      17.37   |      12.7522 |      10.5961 |
        |            350 |      19.2134 |      13.6812 |      11.2162 |
        |            355 |      21.1656 |      14.6449 |      11.8537 |
        |            360 |      23.2145 |      15.6398 |      12.5069 |
        |            365 |      25.3476 |      16.6627 |      13.1742 |
        |            370 |      27.553  |      17.7102 |      13.8541 |
        |            375 |      29.8195 |      18.7791 |      14.5451 |
        |            380 |      32.1368 |      19.8667 |      15.2458 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        lambda_greek: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            lambda_greek[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            lambda_greek[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_lambda(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        lambda_df = helpers.create_greek_dataframe(
            greek_dictionary=lambda_greek,
            start_date=start_date,
        )

        lambda_df = lambda_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            lambda_df = calculate_standardization(
                dataset=lambda_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return lambda_df

    def collect_second_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the second order Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        - Gamma: measures the rate of change in the delta with respect to changes in the underlying price. Gamma is
        the second derivative of the value function with respect to the underlying price.
        - Dual Gamma: the second derivative of the option value with respect to the strike price rather than the
        underlying price. It is the discounted risk-neutral probability density of the underlying at expiration.
        - Vanna: also referred to as DvegaDspot and DdeltaDvol, is a second—order derivative of the option value,
        once to the underlying spot price and once to volatility.
        - Charm: Charm  or delta decay measures the instantaneous rate of change of delta over the passage of time.
        - Vomma: also referred to as volga, vega convexity, or DvegaDvol measures second—order sensitivity to
        volatility. Vomma is the second derivative of the option value with respect to the volatility, or,
        stated another way, vomma measures the rate of change to vega as volatility changes.
        - Veta: also referred to as DvegaDtime, measures the rate of change in the vega with respect to
        the passage of time. Veta is the second derivative of the value function; once to volatility and once to time.
        - Vera: also referred to as rhova, measures the rate of change in rho with respect to volatility. Vera is the
        second derivative of the value function; once to volatility and once to interest rate.
        - Partial Derivative: the second derivative of the option price with respect to the strike price, the rate
        at which the dual delta changes with the strike.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            put_option (bool, optional): Whether to calculate the put option delta. Defaults to False which means
            it will calculate the call option delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the second order greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.collect_second_order_greeks().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   (Period('2026-01-01', 'D'), 'Gamma') |   (Period('2026-01-01', 'D'), 'Dual Gamma') |   (Period('2026-01-01', 'D'), 'Vanna') |   (Period('2026-01-01', 'D'), 'Charm') |   (Period('2026-01-01', 'D'), 'Vomma') |   (Period('2026-01-01', 'D'), 'Vera') |   (Period('2026-01-01', 'D'), 'Veta') |   (Period('2026-01-01', 'D'), 'PD') |   (Period('2026-01-02', 'D'), 'Gamma') |   (Period('2026-01-02', 'D'), 'Dual Gamma') |   (Period('2026-01-02', 'D'), 'Vanna') |   (Period('2026-01-02', 'D'), 'Charm') |   (Period('2026-01-02', 'D'), 'Vomma') |   (Period('2026-01-02', 'D'), 'Vera') |   (Period('2026-01-02', 'D'), 'Veta') |   (Period('2026-01-02', 'D'), 'PD') |   (Period('2026-01-03', 'D'), 'Gamma') |   (Period('2026-01-03', 'D'), 'Dual Gamma') |   (Period('2026-01-03', 'D'), 'Vanna') |   (Period('2026-01-03', 'D'), 'Charm') |   (Period('2026-01-03', 'D'), 'Vomma') |   (Period('2026-01-03', 'D'), 'Vera') |   (Period('2026-01-03', 'D'), 'Veta') |   (Period('2026-01-03', 'D'), 'PD') |   (Period('2026-01-04', 'D'), 'Gamma') |   (Period('2026-01-04', 'D'), 'Dual Gamma') |   (Period('2026-01-04', 'D'), 'Vanna') |   (Period('2026-01-04', 'D'), 'Charm') |   (Period('2026-01-04', 'D'), 'Vomma') |   (Period('2026-01-04', 'D'), 'Vera') |   (Period('2026-01-04', 'D'), 'Veta') |   (Period('2026-01-04', 'D'), 'PD') |   (Period('2026-01-05', 'D'), 'Gamma') |   (Period('2026-01-05', 'D'), 'Dual Gamma') |   (Period('2026-01-05', 'D'), 'Vanna') |   (Period('2026-01-05', 'D'), 'Charm') |   (Period('2026-01-05', 'D'), 'Vomma') |   (Period('2026-01-05', 'D'), 'Vera') |   (Period('2026-01-05', 'D'), 'Veta') |   (Period('2026-01-05', 'D'), 'PD') |   (Period('2026-01-06', 'D'), 'Gamma') |   (Period('2026-01-06', 'D'), 'Dual Gamma') |   (Period('2026-01-06', 'D'), 'Vanna') |   (Period('2026-01-06', 'D'), 'Charm') |   (Period('2026-01-06', 'D'), 'Vomma') |   (Period('2026-01-06', 'D'), 'Vera') |   (Period('2026-01-06', 'D'), 'Veta') |   (Period('2026-01-06', 'D'), 'PD') |   (Period('2026-01-07', 'D'), 'Gamma') |   (Period('2026-01-07', 'D'), 'Dual Gamma') |   (Period('2026-01-07', 'D'), 'Vanna') |   (Period('2026-01-07', 'D'), 'Charm') |   (Period('2026-01-07', 'D'), 'Vomma') |   (Period('2026-01-07', 'D'), 'Vera') |   (Period('2026-01-07', 'D'), 'Veta') |   (Period('2026-01-07', 'D'), 'PD') |   (Period('2026-01-08', 'D'), 'Gamma') |   (Period('2026-01-08', 'D'), 'Dual Gamma') |   (Period('2026-01-08', 'D'), 'Vanna') |   (Period('2026-01-08', 'D'), 'Charm') |   (Period('2026-01-08', 'D'), 'Vomma') |   (Period('2026-01-08', 'D'), 'Vera') |   (Period('2026-01-08', 'D'), 'Veta') |   (Period('2026-01-08', 'D'), 'PD') |   (Period('2026-01-09', 'D'), 'Gamma') |   (Period('2026-01-09', 'D'), 'Dual Gamma') |   (Period('2026-01-09', 'D'), 'Vanna') |   (Period('2026-01-09', 'D'), 'Charm') |   (Period('2026-01-09', 'D'), 'Vomma') |   (Period('2026-01-09', 'D'), 'Vera') |   (Period('2026-01-09', 'D'), 'Veta') |   (Period('2026-01-09', 'D'), 'PD') |   (Period('2026-01-10', 'D'), 'Gamma') |   (Period('2026-01-10', 'D'), 'Dual Gamma') |   (Period('2026-01-10', 'D'), 'Vanna') |   (Period('2026-01-10', 'D'), 'Charm') |   (Period('2026-01-10', 'D'), 'Vomma') |   (Period('2026-01-10', 'D'), 'Vera') |   (Period('2026-01-10', 'D'), 'Veta') |   (Period('2026-01-10', 'D'), 'PD') |   (Period('2026-01-11', 'D'), 'Gamma') |   (Period('2026-01-11', 'D'), 'Dual Gamma') |   (Period('2026-01-11', 'D'), 'Vanna') |   (Period('2026-01-11', 'D'), 'Charm') |   (Period('2026-01-11', 'D'), 'Vomma') |   (Period('2026-01-11', 'D'), 'Vera') |   (Period('2026-01-11', 'D'), 'Veta') |   (Period('2026-01-11', 'D'), 'PD') |   (Period('2026-01-12', 'D'), 'Gamma') |   (Period('2026-01-12', 'D'), 'Dual Gamma') |   (Period('2026-01-12', 'D'), 'Vanna') |   (Period('2026-01-12', 'D'), 'Charm') |   (Period('2026-01-12', 'D'), 'Vomma') |   (Period('2026-01-12', 'D'), 'Vera') |   (Period('2026-01-12', 'D'), 'Veta') |   (Period('2026-01-12', 'D'), 'PD') |   (Period('2026-01-13', 'D'), 'Gamma') |   (Period('2026-01-13', 'D'), 'Dual Gamma') |   (Period('2026-01-13', 'D'), 'Vanna') |   (Period('2026-01-13', 'D'), 'Charm') |   (Period('2026-01-13', 'D'), 'Vomma') |   (Period('2026-01-13', 'D'), 'Vera') |   (Period('2026-01-13', 'D'), 'Veta') |   (Period('2026-01-13', 'D'), 'PD') |   (Period('2026-01-14', 'D'), 'Gamma') |   (Period('2026-01-14', 'D'), 'Dual Gamma') |   (Period('2026-01-14', 'D'), 'Vanna') |   (Period('2026-01-14', 'D'), 'Charm') |   (Period('2026-01-14', 'D'), 'Vomma') |   (Period('2026-01-14', 'D'), 'Vera') |   (Period('2026-01-14', 'D'), 'Veta') |   (Period('2026-01-14', 'D'), 'PD') |   (Period('2026-01-15', 'D'), 'Gamma') |   (Period('2026-01-15', 'D'), 'Dual Gamma') |   (Period('2026-01-15', 'D'), 'Vanna') |   (Period('2026-01-15', 'D'), 'Charm') |   (Period('2026-01-15', 'D'), 'Vomma') |   (Period('2026-01-15', 'D'), 'Vera') |   (Period('2026-01-15', 'D'), 'Veta') |   (Period('2026-01-15', 'D'), 'PD') |   (Period('2026-01-16', 'D'), 'Gamma') |   (Period('2026-01-16', 'D'), 'Dual Gamma') |   (Period('2026-01-16', 'D'), 'Vanna') |   (Period('2026-01-16', 'D'), 'Charm') |   (Period('2026-01-16', 'D'), 'Vomma') |   (Period('2026-01-16', 'D'), 'Vera') |   (Period('2026-01-16', 'D'), 'Veta') |   (Period('2026-01-16', 'D'), 'PD') |   (Period('2026-01-17', 'D'), 'Gamma') |   (Period('2026-01-17', 'D'), 'Dual Gamma') |   (Period('2026-01-17', 'D'), 'Vanna') |   (Period('2026-01-17', 'D'), 'Charm') |   (Period('2026-01-17', 'D'), 'Vomma') |   (Period('2026-01-17', 'D'), 'Vera') |   (Period('2026-01-17', 'D'), 'Veta') |   (Period('2026-01-17', 'D'), 'PD') |   (Period('2026-01-18', 'D'), 'Gamma') |   (Period('2026-01-18', 'D'), 'Dual Gamma') |   (Period('2026-01-18', 'D'), 'Vanna') |   (Period('2026-01-18', 'D'), 'Charm') |   (Period('2026-01-18', 'D'), 'Vomma') |   (Period('2026-01-18', 'D'), 'Vera') |   (Period('2026-01-18', 'D'), 'Veta') |   (Period('2026-01-18', 'D'), 'PD') |   (Period('2026-01-19', 'D'), 'Gamma') |   (Period('2026-01-19', 'D'), 'Dual Gamma') |   (Period('2026-01-19', 'D'), 'Vanna') |   (Period('2026-01-19', 'D'), 'Charm') |   (Period('2026-01-19', 'D'), 'Vomma') |   (Period('2026-01-19', 'D'), 'Vera') |   (Period('2026-01-19', 'D'), 'Veta') |   (Period('2026-01-19', 'D'), 'PD') |   (Period('2026-01-20', 'D'), 'Gamma') |   (Period('2026-01-20', 'D'), 'Dual Gamma') |   (Period('2026-01-20', 'D'), 'Vanna') |   (Period('2026-01-20', 'D'), 'Charm') |   (Period('2026-01-20', 'D'), 'Vomma') |   (Period('2026-01-20', 'D'), 'Vera') |   (Period('2026-01-20', 'D'), 'Veta') |   (Period('2026-01-20', 'D'), 'PD') |   (Period('2026-01-21', 'D'), 'Gamma') |   (Period('2026-01-21', 'D'), 'Dual Gamma') |   (Period('2026-01-21', 'D'), 'Vanna') |   (Period('2026-01-21', 'D'), 'Charm') |   (Period('2026-01-21', 'D'), 'Vomma') |   (Period('2026-01-21', 'D'), 'Vera') |   (Period('2026-01-21', 'D'), 'Veta') |   (Period('2026-01-21', 'D'), 'PD') |   (Period('2026-01-22', 'D'), 'Gamma') |   (Period('2026-01-22', 'D'), 'Dual Gamma') |   (Period('2026-01-22', 'D'), 'Vanna') |   (Period('2026-01-22', 'D'), 'Charm') |   (Period('2026-01-22', 'D'), 'Vomma') |   (Period('2026-01-22', 'D'), 'Vera') |   (Period('2026-01-22', 'D'), 'Veta') |   (Period('2026-01-22', 'D'), 'PD') |   (Period('2026-01-23', 'D'), 'Gamma') |   (Period('2026-01-23', 'D'), 'Dual Gamma') |   (Period('2026-01-23', 'D'), 'Vanna') |   (Period('2026-01-23', 'D'), 'Charm') |   (Period('2026-01-23', 'D'), 'Vomma') |   (Period('2026-01-23', 'D'), 'Vera') |   (Period('2026-01-23', 'D'), 'Veta') |   (Period('2026-01-23', 'D'), 'PD') |   (Period('2026-01-24', 'D'), 'Gamma') |   (Period('2026-01-24', 'D'), 'Dual Gamma') |   (Period('2026-01-24', 'D'), 'Vanna') |   (Period('2026-01-24', 'D'), 'Charm') |   (Period('2026-01-24', 'D'), 'Vomma') |   (Period('2026-01-24', 'D'), 'Vera') |   (Period('2026-01-24', 'D'), 'Veta') |   (Period('2026-01-24', 'D'), 'PD') |   (Period('2026-01-25', 'D'), 'Gamma') |   (Period('2026-01-25', 'D'), 'Dual Gamma') |   (Period('2026-01-25', 'D'), 'Vanna') |   (Period('2026-01-25', 'D'), 'Charm') |   (Period('2026-01-25', 'D'), 'Vomma') |   (Period('2026-01-25', 'D'), 'Vera') |   (Period('2026-01-25', 'D'), 'Veta') |   (Period('2026-01-25', 'D'), 'PD') |   (Period('2026-01-26', 'D'), 'Gamma') |   (Period('2026-01-26', 'D'), 'Dual Gamma') |   (Period('2026-01-26', 'D'), 'Vanna') |   (Period('2026-01-26', 'D'), 'Charm') |   (Period('2026-01-26', 'D'), 'Vomma') |   (Period('2026-01-26', 'D'), 'Vera') |   (Period('2026-01-26', 'D'), 'Veta') |   (Period('2026-01-26', 'D'), 'PD') |   (Period('2026-01-27', 'D'), 'Gamma') |   (Period('2026-01-27', 'D'), 'Dual Gamma') |   (Period('2026-01-27', 'D'), 'Vanna') |   (Period('2026-01-27', 'D'), 'Charm') |   (Period('2026-01-27', 'D'), 'Vomma') |   (Period('2026-01-27', 'D'), 'Vera') |   (Period('2026-01-27', 'D'), 'Veta') |   (Period('2026-01-27', 'D'), 'PD') |   (Period('2026-01-28', 'D'), 'Gamma') |   (Period('2026-01-28', 'D'), 'Dual Gamma') |   (Period('2026-01-28', 'D'), 'Vanna') |   (Period('2026-01-28', 'D'), 'Charm') |   (Period('2026-01-28', 'D'), 'Vomma') |   (Period('2026-01-28', 'D'), 'Vera') |   (Period('2026-01-28', 'D'), 'Veta') |   (Period('2026-01-28', 'D'), 'PD') |   (Period('2026-01-29', 'D'), 'Gamma') |   (Period('2026-01-29', 'D'), 'Dual Gamma') |   (Period('2026-01-29', 'D'), 'Vanna') |   (Period('2026-01-29', 'D'), 'Charm') |   (Period('2026-01-29', 'D'), 'Vomma') |   (Period('2026-01-29', 'D'), 'Vera') |   (Period('2026-01-29', 'D'), 'Veta') |   (Period('2026-01-29', 'D'), 'PD') |
        |---------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------:|------------------------------------:|
        |            290 |                                 0.0001 |                                      0.0001 |                                 0.0035 |                                -0.2099 |                                 0.1908 |                                0.0026 |                               -0.0003 |                              0.0001 |                                 0.0017 |                                      0.0015 |                                 0.0936 |                                -2.7876 |                                 5.033  |                                0.1382 |                               -0.0047 |                              0.0015 |                                 0.0047 |                                      0.0041 |                                 0.2559 |                                -5.0939 |                                13.6998 |                                0.5641 |                               -0.009  |                              0.0041 |                                 0.0075 |                                      0.0066 |                                 0.4055 |                                -6.0731 |                                21.6262 |                                1.1865 |                               -0.0113 |                              0.0066 |                                 0.0096 |                                      0.0084 |                                 0.5212 |                                -6.2652 |                                27.6936 |                                1.898  |                               -0.0123 |                              0.0084 |                                 0.0111 |                                      0.0098 |                                 0.6061 |                                -6.0898 |                                32.0774 |                                2.6364 |                               -0.0125 |                              0.0098 |                                 0.0123 |                                      0.0108 |                                 0.6671 |                                -5.7631 |                                35.1695 |                                3.3701 |                               -0.0123 |                              0.0108 |                                 0.0131 |                                      0.0115 |                                 0.7106 |                                -5.3877 |                                37.3143 |                                4.0839 |                               -0.012  |                              0.0115 |                                 0.0136 |                                      0.012  |                                 0.7413 |                                -5.011  |                                38.7714 |                                4.7707 |                               -0.0116 |                              0.012  |                                 0.014  |                                      0.0123 |                                 0.7626 |                                -4.6535 |                                39.7283 |                                5.4281 |                               -0.0112 |                              0.0123 |                                 0.0142 |                                      0.0125 |                                 0.777  |                                -4.3235 |                                40.3187 |                                6.0558 |                               -0.0108 |                              0.0125 |                                 0.0144 |                                      0.0127 |                                 0.7863 |                                -4.0227 |                                40.6389 |                                6.6545 |                               -0.0104 |                              0.0127 |                                 0.0145 |                                      0.0127 |                                 0.7918 |                                -3.7502 |                                40.7583 |                                7.2257 |                               -0.01   |                              0.0127 |                                 0.0145 |                                      0.0128 |                                 0.7943 |                                -3.5042 |                                40.7278 |                                7.7707 |                               -0.0097 |                              0.0128 |                                 0.0145 |                                      0.0128 |                                 0.7948 |                                -3.282  |                                40.5849 |                                8.2912 |                               -0.0094 |                              0.0128 |                                 0.0145 |                                      0.0127 |                                 0.7935 |                                -3.0811 |                                40.3576 |                                8.7888 |                               -0.0091 |                              0.0127 |                                 0.0144 |                                      0.0127 |                                 0.791  |                                -2.8992 |                                40.0669 |                                9.265  |                               -0.0088 |                              0.0127 |                                 0.0144 |                                      0.0126 |                                 0.7876 |                                -2.7341 |                                39.7289 |                                9.7211 |                               -0.0085 |                              0.0126 |                                 0.0143 |                                      0.0126 |                                 0.7834 |                                -2.5839 |                                39.3558 |                               10.1583 |                               -0.0082 |                              0.0126 |                                 0.0142 |                                      0.0125 |                                 0.7786 |                                -2.4469 |                                38.9571 |                               10.5779 |                               -0.008  |                              0.0125 |                                 0.0141 |                                      0.0124 |                                 0.7735 |                                -2.3217 |                                38.5399 |                               10.9809 |                               -0.0078 |                              0.0124 |                                 0.014  |                                      0.0123 |                                 0.7681 |                                -2.2069 |                                38.1101 |                               11.3682 |                               -0.0076 |                              0.0123 |                                 0.0139 |                                      0.0122 |                                 0.7624 |                                -2.1013 |                                37.6719 |                               11.7409 |                               -0.0074 |                              0.0122 |                                 0.0138 |                                      0.0121 |                                 0.7566 |                                -2.0041 |                                37.229  |                               12.0996 |                               -0.0072 |                              0.0121 |                                 0.0136 |                                      0.012  |                                 0.7507 |                                -1.9144 |                                36.7839 |                               12.4452 |                               -0.007  |                              0.012  |                                 0.0135 |                                      0.0119 |                                 0.7448 |                                -1.8313 |                                36.3388 |                               12.7783 |                               -0.0069 |                              0.0119 |                                 0.0134 |                                      0.0118 |                                 0.7388 |                                -1.7543 |                                35.8954 |                               13.0996 |                               -0.0067 |                              0.0118 |                                 0.0133 |                                      0.0117 |                                 0.7329 |                                -1.6827 |                                35.4549 |                               13.4096 |                               -0.0066 |                              0.0117 |                                 0.0132 |                                      0.0116 |                                 0.7269 |                                -1.616  |                                35.0184 |                               13.7088 |                               -0.0065 |                              0.0116 |
        |            295 |                                 0      |                                      0      |                                 0.0001 |                                -0.0035 |                                 0.004  |                                0      |                               -0      |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0137 |                                -0.4074 |                                 0.933  |                                0.0203 |                               -0.0008 |                              0.0002 |                                 0.0011 |                                      0.001  |                                 0.0772 |                                -1.5339 |                                 5.2401 |                                0.1707 |                               -0.0032 |                              0.001  |                                 0.0026 |                                      0.0022 |                                 0.1756 |                                -2.624  |                                11.886  |                                0.5159 |                               -0.0057 |                              0.0022 |                                 0.0041 |                                      0.0035 |                                 0.2805 |                                -3.3607 |                                18.9241 |                                1.0262 |                               -0.0076 |                              0.0035 |                                 0.0055 |                                      0.0047 |                                 0.3769 |                                -3.7724 |                                25.3508 |                                1.6489 |                               -0.0088 |                              0.0047 |                                 0.0067 |                                      0.0057 |                                 0.46   |                                -3.9558 |                                30.8432 |                                2.3393 |                               -0.0095 |                              0.0057 |                                 0.0077 |                                      0.0065 |                                 0.5294 |                                -3.9933 |                                35.3873 |                                3.0658 |                               -0.0098 |                              0.0065 |                                 0.0085 |                                      0.0072 |                                 0.5865 |                                -3.942  |                                39.0826 |                                3.8072 |                               -0.01   |                              0.0072 |                                 0.0092 |                                      0.0078 |                                 0.6331 |                                -3.8389 |                                42.0567 |                                4.5499 |                               -0.01   |                              0.0078 |                                 0.0097 |                                      0.0083 |                                 0.6709 |                                -3.7074 |                                44.4324 |                                5.2849 |                               -0.0099 |                              0.0083 |                                 0.0102 |                                      0.0086 |                                 0.7016 |                                -3.5622 |                                46.3167 |                                6.0068 |                               -0.0098 |                              0.0086 |                                 0.0105 |                                      0.0089 |                                 0.7263 |                                -3.4122 |                                47.7991 |                                6.7122 |                               -0.0096 |                              0.0089 |                                 0.0108 |                                      0.0092 |                                 0.7462 |                                -3.263  |                                48.953  |                                7.3993 |                               -0.0094 |                              0.0092 |                                 0.011  |                                      0.0094 |                                 0.762  |                                -3.1176 |                                49.8377 |                                8.067  |                               -0.0092 |                              0.0094 |                                 0.0112 |                                      0.0095 |                                 0.7746 |                                -2.9781 |                                50.5015 |                                8.715  |                               -0.009  |                              0.0095 |                                 0.0114 |                                      0.0096 |                                 0.7845 |                                -2.8454 |                                50.9835 |                                9.3434 |                               -0.0088 |                              0.0096 |                                 0.0115 |                                      0.0097 |                                 0.7921 |                                -2.7198 |                                51.3153 |                                9.9524 |                               -0.0086 |                              0.0097 |                                 0.0115 |                                      0.0098 |                                 0.7979 |                                -2.6014 |                                51.523  |                               10.5425 |                               -0.0084 |                              0.0098 |                                 0.0116 |                                      0.0098 |                                 0.8021 |                                -2.4901 |                                51.6275 |                               11.1143 |                               -0.0082 |                              0.0098 |                                 0.0116 |                                      0.0099 |                                 0.805  |                                -2.3855 |                                51.6463 |                               11.6683 |                               -0.008  |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8067 |                                -2.2874 |                                51.5937 |                               12.2054 |                               -0.0078 |                              0.0099 |                                 0.0117 |                                      0.0099 |                                 0.8076 |                                -2.1953 |                                51.4815 |                               12.726  |                               -0.0077 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8077 |                                -2.1088 |                                51.3196 |                               13.2309 |                               -0.0075 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8071 |                                -2.0277 |                                51.1162 |                               13.7207 |                               -0.0073 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.806  |                                -1.9514 |                                50.8782 |                               14.1959 |                               -0.0072 |                              0.0099 |                                 0.0116 |                                      0.0098 |                                 0.8044 |                                -1.8797 |                                50.6114 |                               14.6573 |                               -0.007  |                              0.0098 |                                 0.0115 |                                      0.0098 |                                 0.8024 |                                -1.8121 |                                50.3206 |                               15.1052 |                               -0.0069 |                              0.0098 |                                 0.0115 |                                      0.0098 |                                 0.8001 |                                -1.7485 |                                50.0101 |                               15.5403 |                               -0.0068 |                              0.0098 |
        |            300 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0012 |                                -0.0359 |                                 0.0994 |                                0.0018 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0164 |                                -0.3247 |                                 1.3411 |                                0.0362 |                               -0.0008 |                              0.0002 |                                 0.0007 |                                      0.0006 |                                 0.0577 |                                -0.8604 |                                 4.7168 |                                0.1699 |                               -0.0022 |                              0.0006 |                                 0.0014 |                                      0.0012 |                                 0.1198 |                                -1.4326 |                                 9.7724 |                                0.4397 |                               -0.0037 |                              0.0012 |                                 0.0023 |                                      0.0019 |                                 0.1918 |                                -1.9152 |                                15.6059 |                                0.8423 |                               -0.005  |                              0.0019 |                                 0.0032 |                                      0.0026 |                                 0.2653 |                                -2.275  |                                21.5289 |                                1.355  |                               -0.0061 |                              0.0026 |                                 0.004  |                                      0.0033 |                                 0.3354 |                                -2.5214 |                                27.1446 |                                1.9517 |                               -0.0069 |                              0.0033 |                                 0.0048 |                                      0.004  |                                 0.3997 |                                -2.6762 |                                32.2652 |                                2.6088 |                               -0.0075 |                              0.004  |                                 0.0055 |                                      0.0045 |                                 0.4573 |                                -2.7617 |                                36.8262 |                                3.307  |                               -0.0079 |                              0.0045 |                                 0.0061 |                                      0.005  |                                 0.5083 |                                -2.7964 |                                40.8302 |                                4.0316 |                               -0.0082 |                              0.005  |                                 0.0067 |                                      0.0055 |                                 0.5531 |                                -2.7947 |                                44.312  |                                4.7711 |                               -0.0083 |                              0.0055 |                                 0.0071 |                                      0.0059 |                                 0.5922 |                                -2.7675 |                                47.3199 |                                5.5172 |                               -0.0084 |                              0.0059 |                                 0.0075 |                                      0.0062 |                                 0.6262 |                                -2.7227 |                                49.9062 |                                6.2637 |                               -0.0084 |                              0.0062 |                                 0.0079 |                                      0.0065 |                                 0.6557 |                                -2.6661 |                                52.121  |                                7.0061 |                               -0.0084 |                              0.0065 |                                 0.0082 |                                      0.0067 |                                 0.6812 |                                -2.602  |                                54.0108 |                                7.7408 |                               -0.0083 |                              0.0067 |                                 0.0085 |                                      0.0069 |                                 0.7033 |                                -2.5334 |                                55.6173 |                                8.4657 |                               -0.0083 |                              0.0069 |                                 0.0087 |                                      0.0071 |                                 0.7224 |                                -2.4624 |                                56.9771 |                                9.179  |                               -0.0082 |                              0.0071 |                                 0.0089 |                                      0.0073 |                                 0.7388 |                                -2.3906 |                                58.1222 |                                9.8796 |                               -0.0081 |                              0.0073 |                                 0.009  |                                      0.0074 |                                 0.753  |                                -2.3191 |                                59.0806 |                               10.5666 |                               -0.008  |                              0.0074 |                                 0.0092 |                                      0.0075 |                                 0.7652 |                                -2.2487 |                                59.8762 |                               11.2396 |                               -0.0078 |                              0.0075 |                                 0.0093 |                                      0.0076 |                                 0.7756 |                                -2.1799 |                                60.5298 |                               11.8984 |                               -0.0077 |                              0.0076 |                                 0.0094 |                                      0.0077 |                                 0.7844 |                                -2.113  |                                61.0597 |                               12.5429 |                               -0.0076 |                              0.0077 |                                 0.0095 |                                      0.0078 |                                 0.7919 |                                -2.0483 |                                61.4812 |                               13.1731 |                               -0.0075 |                              0.0078 |                                 0.0096 |                                      0.0079 |                                 0.7983 |                                -1.9859 |                                61.8079 |                               13.7891 |                               -0.0074 |                              0.0079 |                                 0.0096 |                                      0.0079 |                                 0.8036 |                                -1.9259 |                                62.0515 |                               14.3912 |                               -0.0072 |                              0.0079 |                                 0.0097 |                                      0.0079 |                                 0.8079 |                                -1.8682 |                                62.2222 |                               14.9796 |                               -0.0071 |                              0.0079 |                                 0.0097 |                                      0.008  |                                 0.8115 |                                -1.8129 |                                62.3288 |                               15.5545 |                               -0.007  |                              0.008  |                                 0.0097 |                                      0.008  |                                 0.8143 |                                -1.7598 |                                62.3791 |                               16.1163 |                               -0.0069 |                              0.008  |
        |            305 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.002  |                                 0.0064 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0498 |                                 0.2408 |                                0.0056 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0001 |                                 0.0148 |                                -0.2199 |                                 1.4116 |                                0.0435 |                               -0.0006 |                              0.0001 |                                 0.0004 |                                      0.0003 |                                 0.0417 |                                -0.4973 |                                 3.9745 |                                0.1532 |                               -0.0014 |                              0.0003 |                                 0.0008 |                                      0.0007 |                                 0.0818 |                                -0.8152 |                                 7.7878 |                                0.36   |                               -0.0024 |                              0.0007 |                                 0.0014 |                                      0.0011 |                                 0.1309 |                                -1.12   |                                12.435  |                                0.6705 |                               -0.0034 |                              0.0011 |                                 0.0019 |                                      0.0015 |                                 0.1846 |                                -1.3844 |                                17.4972 |                                1.0778 |                               -0.0042 |                              0.0015 |                                 0.0025 |                                      0.002  |                                 0.2395 |                                -1.5994 |                                22.6526 |                                1.5692 |                               -0.005  |                              0.002  |                                 0.003  |                                      0.0024 |                                 0.2933 |                                -1.7662 |                                27.6854 |                                2.1302 |                               -0.0056 |                              0.0024 |                                 0.0036 |                                      0.0028 |                                 0.3447 |                                -1.8902 |                                32.4643 |                                2.7467 |                               -0.006  |                              0.0028 |                                 0.0041 |                                      0.0032 |                                 0.3929 |                                -1.9781 |                                36.9187 |                                3.4063 |                               -0.0064 |                              0.0032 |                                 0.0045 |                                      0.0036 |                                 0.4374 |                                -2.0367 |                                41.0175 |                                4.0984 |                               -0.0067 |                              0.0036 |                                 0.0049 |                                      0.0039 |                                 0.4784 |                                -2.0716 |                                44.7551 |                                4.8141 |                               -0.0069 |                              0.0039 |                                 0.0053 |                                      0.0042 |                                 0.5157 |                                -2.088  |                                48.141  |                                5.5462 |                               -0.0071 |                              0.0042 |                                 0.0057 |                                      0.0045 |                                 0.5496 |                                -2.0898 |                                51.1935 |                                6.2888 |                               -0.0072 |                              0.0045 |                                 0.006  |                                      0.0048 |                                 0.5803 |                                -2.0803 |                                53.935  |                                7.0371 |                               -0.0073 |                              0.0048 |                                 0.0063 |                                      0.005  |                                 0.6081 |                                -2.0622 |                                56.39   |                                7.7874 |                               -0.0073 |                              0.005  |                                 0.0065 |                                      0.0052 |                                 0.6331 |                                -2.0376 |                                58.5828 |                                8.5366 |                               -0.0073 |                              0.0052 |                                 0.0067 |                                      0.0054 |                                 0.6557 |                                -2.0082 |                                60.5371 |                                9.2824 |                               -0.0073 |                              0.0054 |                                 0.007  |                                      0.0055 |                                 0.6761 |                                -1.9752 |                                62.275  |                               10.0227 |                               -0.0073 |                              0.0055 |                                 0.0071 |                                      0.0057 |                                 0.6944 |                                -1.9397 |                                63.8171 |                               10.7561 |                               -0.0072 |                              0.0057 |                                 0.0073 |                                      0.0058 |                                 0.7108 |                                -1.9025 |                                65.1823 |                               11.4815 |                               -0.0072 |                              0.0058 |                                 0.0075 |                                      0.0059 |                                 0.7256 |                                -1.8642 |                                66.3877 |                               12.1978 |                               -0.0071 |                              0.0059 |                                 0.0076 |                                      0.006  |                                 0.7389 |                                -1.8254 |                                67.4488 |                               12.9046 |                               -0.0071 |                              0.006  |                                 0.0077 |                                      0.0061 |                                 0.7508 |                                -1.7864 |                                68.3798 |                               13.6011 |                               -0.007  |                              0.0061 |                                 0.0078 |                                      0.0062 |                                 0.7614 |                                -1.7476 |                                69.1932 |                               14.2871 |                               -0.0069 |                              0.0062 |                                 0.0079 |                                      0.0063 |                                 0.7709 |                                -1.7091 |                                69.9004 |                               14.9624 |                               -0.0068 |                              0.0063 |                                 0.008  |                                      0.0064 |                                 0.7795 |                                -1.6711 |                                70.5116 |                               15.6267 |                               -0.0068 |                              0.0064 |
        |            310 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0003 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0003 |                                -0.0057 |                                 0.0313 |                                0.0006 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.003  |                                -0.0447 |                                 0.3281 |                                0.0089 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.012  |                                -0.143  |                                 1.3081 |                                0.0442 |                               -0.0005 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0297 |                                -0.2956 |                                 3.2325 |                                0.131  |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0561 |                                -0.4791 |                                 6.0922 |                                0.2879 |                               -0.0016 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.0895 |                                -0.6704 |                                 9.7079 |                                0.5241 |                               -0.0023 |                              0.0006 |                                 0.0012 |                                      0.0009 |                                 0.1279 |                                -0.8528 |                                13.8459 |                                0.8407 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0012 |                                 0.1693 |                                -1.0171 |                                18.2857 |                                1.2332 |                               -0.0035 |                              0.0012 |                                 0.0019 |                                      0.0015 |                                 0.2119 |                                -1.1593 |                                22.8467 |                                1.6944 |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0018 |                                 0.2546 |                                -1.2785 |                                27.3933 |                                2.2156 |                               -0.0045 |                              0.0018 |                                 0.0027 |                                      0.0021 |                                 0.2964 |                                -1.376  |                                31.8295 |                                2.788  |                               -0.0049 |                              0.0021 |                                 0.003  |                                      0.0023 |                                 0.3367 |                                -1.4538 |                                36.0911 |                                3.4034 |                               -0.0053 |                              0.0023 |                                 0.0034 |                                      0.0026 |                                 0.3752 |                                -1.5142 |                                40.1381 |                                4.0542 |                               -0.0056 |                              0.0026 |                                 0.0037 |                                      0.0029 |                                 0.4116 |                                -1.5597 |                                43.9485 |                                4.7335 |                               -0.0058 |                              0.0029 |                                 0.004  |                                      0.0031 |                                 0.4458 |                                -1.5924 |                                47.5125 |                                5.4354 |                               -0.006  |                              0.0031 |                                 0.0043 |                                      0.0033 |                                 0.4779 |                                -1.6145 |                                50.8292 |                                6.155  |                               -0.0061 |                              0.0033 |                                 0.0046 |                                      0.0035 |                                 0.5077 |                                -1.6276 |                                53.9035 |                                6.8877 |                               -0.0062 |                              0.0035 |                                 0.0048 |                                      0.0037 |                                 0.5355 |                                -1.6333 |                                56.7443 |                                7.6299 |                               -0.0063 |                              0.0037 |                                 0.0051 |                                      0.0039 |                                 0.5613 |                                -1.6329 |                                59.3623 |                                8.3784 |                               -0.0064 |                              0.0039 |                                 0.0053 |                                      0.0041 |                                 0.5853 |                                -1.6274 |                                61.7701 |                                9.1305 |                               -0.0064 |                              0.0041 |                                 0.0055 |                                      0.0042 |                                 0.6074 |                                -1.6179 |                                63.9803 |                                9.884  |                               -0.0065 |                              0.0042 |                                 0.0057 |                                      0.0044 |                                 0.6279 |                                -1.6051 |                                66.006  |                               10.637  |                               -0.0065 |                              0.0044 |                                 0.0058 |                                      0.0045 |                                 0.6468 |                                -1.5896 |                                67.8598 |                               11.3878 |                               -0.0065 |                              0.0045 |                                 0.006  |                                      0.0046 |                                 0.6642 |                                -1.5721 |                                69.5538 |                               12.1351 |                               -0.0065 |                              0.0046 |                                 0.0061 |                                      0.0047 |                                 0.6803 |                                -1.5528 |                                71.0998 |                               12.8779 |                               -0.0065 |                              0.0047 |                                 0.0063 |                                      0.0048 |                                 0.6952 |                                -1.5323 |                                72.5086 |                               13.6152 |                               -0.0064 |                              0.0048 |                                 0.0064 |                                      0.0049 |                                 0.7089 |                                -1.5108 |                                73.7905 |                               14.3463 |                               -0.0064 |                              0.0049 |
        |            315 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0005 |                                 0.003  |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0073 |                                 0.0605 |                                0.0015 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0346 |                                 0.3553 |                                0.0107 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0093 |                                -0.0924 |                                 1.1367 |                                0.0411 |                               -0.0003 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0211 |                                -0.1801 |                                 2.5762 |                                0.1085 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0387 |                                -0.2893 |                                 4.7144 |                                0.2269 |                               -0.0011 |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0616 |                                -0.4097 |                                 7.4883 |                                0.4054 |                               -0.0015 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.0888 |                                -0.5325 |                                10.7793 |                                0.6482 |                               -0.002  |                              0.0005 |                                 0.001  |                                      0.0007 |                                 0.1192 |                                -0.651  |                                14.4523 |                                0.9558 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1519 |                                -0.7611 |                                18.378  |                                1.3255 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0011 |                                 0.1858 |                                -0.8607 |                                22.4443 |                                1.7532 |                               -0.0033 |                              0.0011 |                                 0.0018 |                                      0.0013 |                                 0.2202 |                                -0.9486 |                                26.5596 |                                2.2336 |                               -0.0037 |                              0.0013 |                                 0.0021 |                                      0.0015 |                                 0.2546 |                                -1.025  |                                30.6523 |                                2.7611 |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0017 |                                 0.2884 |                                -1.0901 |                                34.6683 |                                3.3301 |                               -0.0044 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3214 |                                -1.1449 |                                38.5685 |                                3.9352 |                               -0.0046 |                              0.0019 |                                 0.0028 |                                      0.0021 |                                 0.3533 |                                -1.1903 |                                42.3255 |                                4.5713 |                               -0.0048 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.384  |                                -1.2272 |                                45.9212 |                                5.2337 |                               -0.005  |                              0.0023 |                                 0.0033 |                                      0.0025 |                                 0.4134 |                                -1.2566 |                                49.345  |                                5.9183 |                               -0.0052 |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4413 |                                -1.2794 |                                52.5915 |                                6.6212 |                               -0.0053 |                              0.0026 |                                 0.0038 |                                      0.0028 |                                 0.4679 |                                -1.2965 |                                55.6594 |                                7.339  |                               -0.0055 |                              0.0028 |                                 0.004  |                                      0.003  |                                 0.493  |                                -1.3085 |                                58.5505 |                                8.0689 |                               -0.0056 |                              0.003  |                                 0.0042 |                                      0.0031 |                                 0.5168 |                                -1.3162 |                                61.2686 |                                8.8081 |                               -0.0056 |                              0.0031 |                                 0.0043 |                                      0.0032 |                                 0.5393 |                                -1.3202 |                                63.8191 |                                9.5544 |                               -0.0057 |                              0.0032 |                                 0.0045 |                                      0.0034 |                                 0.5605 |                                -1.321  |                                66.2083 |                               10.3057 |                               -0.0058 |                              0.0034 |                                 0.0047 |                                      0.0035 |                                 0.5804 |                                -1.319  |                                68.4433 |                               11.0602 |                               -0.0058 |                              0.0035 |                                 0.0048 |                                      0.0036 |                                 0.5991 |                                -1.3147 |                                70.5311 |                               11.8164 |                               -0.0058 |                              0.0036 |                                 0.005  |                                      0.0037 |                                 0.6168 |                                -1.3085 |                                72.4794 |                               12.573  |                               -0.0058 |                              0.0037 |
        |            320 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0002 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.001  |                                 0.009  |                                0.0002 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0071 |                                 0.0809 |                                0.0022 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0252 |                                 0.3435 |                                0.0112 |                               -0.0001 |                              0      |                                 0.0001 |                                      0      |                                 0.007  |                                -0.0601 |                                 0.9529 |                                0.0363 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.015  |                                -0.1122 |                                 2.0293 |                                0.0883 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0269 |                                -0.1788 |                                 3.627  |                                0.1774 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0426 |                                -0.2553 |                                 5.738  |                                0.3118 |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0003 |                                 0.0618 |                                -0.3371 |                                 8.3113 |                                0.4967 |                               -0.0014 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.084  |                                -0.4203 |                                11.2722 |                                0.7347 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.1085 |                                -0.5017 |                                14.5378 |                                1.0263 |                               -0.0021 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1347 |                                -0.5793 |                                18.0271 |                                1.3701 |                               -0.0024 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1622 |                                -0.6516 |                                21.6659 |                                1.7638 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.1903 |                                -0.7179 |                                25.39   |                                2.2043 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0012 |                                 0.2188 |                                -0.7777 |                                29.1456 |                                2.6878 |                               -0.0034 |                              0.0012 |                                 0.0018 |                                      0.0013 |                                 0.2473 |                                -0.8311 |                                32.8888 |                                3.2105 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.2755 |                                -0.8783 |                                36.5845 |                                3.7688 |                               -0.0039 |                              0.0014 |                                 0.0022 |                                      0.0016 |                                 0.3033 |                                -0.9195 |                                40.2057 |                                4.3587 |                               -0.0041 |                              0.0016 |                                 0.0024 |                                      0.0017 |                                 0.3304 |                                -0.9552 |                                43.7318 |                                4.9767 |                               -0.0043 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3567 |                                -0.9857 |                                47.1477 |                                5.6196 |                               -0.0044 |                              0.0019 |                                 0.0028 |                                      0.002  |                                 0.3823 |                                -1.0115 |                                50.4428 |                                6.284  |                               -0.0046 |                              0.002  |                                 0.003  |                                      0.0021 |                                 0.4069 |                                -1.0331 |                                53.6102 |                                6.9672 |                               -0.0047 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.4306 |                                -1.0508 |                                56.6456 |                                7.6665 |                               -0.0048 |                              0.0023 |                                 0.0033 |                                      0.0024 |                                 0.4534 |                                -1.0651 |                                59.5473 |                                8.3794 |                               -0.0049 |                              0.0024 |                                 0.0035 |                                      0.0025 |                                 0.4752 |                                -1.0763 |                                62.3153 |                                9.1039 |                               -0.005  |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4961 |                                -1.0848 |                                64.9507 |                                9.8379 |                               -0.0051 |                              0.0026 |                                 0.0037 |                                      0.0027 |                                 0.516  |                                -1.0908 |                                67.4562 |                               10.5796 |                               -0.0051 |                              0.0027 |
        |            325 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0011 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0156 |                                0.0004 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.006  |                                 0.0902 |                                0.0027 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0021 |                                -0.0179 |                                 0.3116 |                                0.0108 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0053 |                                -0.0394 |                                 0.782  |                                0.0311 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0107 |                                -0.0713 |                                 1.5883 |                                0.071  |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0189 |                                -0.1128 |                                 2.7834 |                                0.1382 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0297 |                                -0.1619 |                                 4.3836 |                                0.2393 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0433 |                                -0.2164 |                                 6.3749 |                                0.3795 |                               -0.001  |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0593 |                                -0.274  |                                 8.722  |                                0.5624 |                               -0.0012 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.0775 |                                -0.3327 |                                11.3771 |                                0.7898 |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0005 |                                 0.0974 |                                -0.391  |                                14.2872 |                                1.0625 |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0006 |                                 0.1188 |                                -0.4475 |                                17.399  |                                1.3798 |                               -0.0021 |                              0.0006 |                                 0.0009 |                                      0.0007 |                                 0.1413 |                                -0.5014 |                                20.6617 |                                1.7406 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.1646 |                                -0.5521 |                                24.0293 |                                2.1429 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.1884 |                                -0.5993 |                                27.4611 |                                2.5843 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.2124 |                                -0.6427 |                                30.9221 |                                3.0625 |                               -0.003  |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2365 |                                -0.6823 |                                34.3825 |                                3.5746 |                               -0.0032 |                              0.0011 |                                 0.0017 |                                      0.0012 |                                 0.2605 |                                -0.7182 |                                37.8177 |                                4.1181 |                               -0.0034 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.2843 |                                -0.7504 |                                41.2076 |                                4.6901 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.3077 |                                -0.7792 |                                44.536  |                                5.2881 |                               -0.0038 |                              0.0014 |                                 0.0022 |                                      0.0015 |                                 0.3306 |                                -0.8047 |                                47.79   |                                5.9095 |                               -0.0039 |                              0.0015 |                                 0.0023 |                                      0.0016 |                                 0.3531 |                                -0.8272 |                                50.96   |                                6.552  |                               -0.004  |                              0.0016 |                                 0.0025 |                                      0.0017 |                                 0.3749 |                                -0.8468 |                                54.0383 |                                7.2134 |                               -0.0042 |                              0.0017 |                                 0.0026 |                                      0.0018 |                                 0.3962 |                                -0.8638 |                                57.0199 |                                7.8915 |                               -0.0043 |                              0.0018 |                                 0.0028 |                                      0.0019 |                                 0.4168 |                                -0.8783 |                                59.901  |                                8.5843 |                               -0.0044 |                              0.0019 |
        |            330 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0001 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0026 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0208 |                                0.0006 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0048 |                                 0.0909 |                                0.0029 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0017 |                                -0.0126 |                                 0.2721 |                                0.01   |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0039 |                                -0.0262 |                                 0.6338 |                                0.0261 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0077 |                                -0.0462 |                                 1.2395 |                                0.0567 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0133 |                                -0.0725 |                                 2.1356 |                                0.1074 |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0209 |                                -0.1044 |                                 3.3471 |                                0.1836 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0305 |                                -0.1408 |                                 4.879  |                                0.2899 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0421 |                                -0.1805 |                                 6.7196 |                                0.4299 |                               -0.0009 |                              0.0002 |                                 0.0003 |                                      0.0002 |                                 0.0555 |                                -0.2223 |                                 8.8454 |                                0.6061 |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0705 |                                -0.2651 |                                11.2252 |                                0.8203 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.0869 |                                -0.308  |                                13.8241 |                                1.0732 |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0004 |                                 0.1046 |                                -0.3502 |                                16.606  |                                1.3647 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0005 |                                 0.1232 |                                -0.3912 |                                19.5351 |                                1.6942 |                               -0.0019 |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.1426 |                                -0.4306 |                                22.5782 |                                2.0607 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1625 |                                -0.4679 |                                25.7044 |                                2.4628 |                               -0.0024 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.1829 |                                -0.5031 |                                28.8859 |                                2.8988 |                               -0.0025 |                              0.0008 |                                 0.0012 |                                      0.0008 |                                 0.2035 |                                -0.536  |                                32.0983 |                                3.3669 |                               -0.0027 |                              0.0008 |                                 0.0014 |                                      0.0009 |                                 0.2242 |                                -0.5666 |                                35.3202 |                                3.8651 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.001  |                                 0.2449 |                                -0.5948 |                                38.5334 |                                4.3915 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2655 |                                -0.6207 |                                41.7221 |                                4.9441 |                               -0.0032 |                              0.0011 |                                 0.0017 |                                      0.0012 |                                 0.2859 |                                -0.6443 |                                44.8734 |                                5.5208 |                               -0.0033 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.3061 |                                -0.6658 |                                47.9762 |                                6.1199 |                               -0.0035 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.326  |                                -0.6852 |                                51.0219 |                                6.7394 |                               -0.0036 |                              0.0014 |
        |            335 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0004 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0042 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0012 |                                 0.0238 |                                0.0007 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0037 |                                 0.086  |                                0.0029 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0013 |                                -0.0089 |                                 0.232  |                                0.0089 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0176 |                                 0.5099 |                                0.0216 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0056 |                                -0.0304 |                                 0.9667 |                                0.0451 |                               -0.0002 |                              0      |                                 0.0001 |                                      0      |                                 0.0095 |                                -0.0474 |                                 1.6409 |                                0.0835 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0148 |                                -0.0684 |                                 2.5589 |                                0.1411 |                               -0.0004 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0217 |                                -0.0929 |                                 3.7341 |                                0.2217 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.03   |                                -0.1202 |                                 5.1681 |                                0.3287 |                               -0.0006 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0399 |                                -0.1498 |                                 6.8528 |                                0.4648 |                               -0.0008 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0511 |                                -0.1809 |                                 8.7727 |                                0.6321 |                               -0.0009 |                              0.0002 |                                 0.0004 |                                      0.0002 |                                 0.0636 |                                -0.2128 |                                10.9072 |                                0.832  |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0773 |                                -0.2452 |                                13.2328 |                                1.0653 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0003 |                                 0.092  |                                -0.2774 |                                15.7244 |                                1.3322 |                               -0.0015 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.1075 |                                -0.309  |                                18.3569 |                                1.6327 |                               -0.0016 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.1237 |                                -0.3399 |                                21.1055 |                                1.9662 |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0005 |                                 0.1406 |                                -0.3697 |                                23.947  |                                2.3318 |                               -0.002  |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.1578 |                                -0.3982 |                                26.8596 |                                2.7286 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1755 |                                -0.4254 |                                29.8235 |                                3.1553 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0007 |                                 0.1933 |                                -0.4511 |                                32.8207 |                                3.6106 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0008 |                                 0.2114 |                                -0.4753 |                                35.8352 |                                4.0931 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.2294 |                                -0.498  |                                38.8528 |                                4.6012 |                               -0.0027 |                              0.0009 |                                 0.0014 |                                      0.0009 |                                 0.2475 |                                -0.5192 |                                41.8612 |                                5.1335 |                               -0.0029 |                              0.0009 |
        """
        greeks = {}

        greeks["Gamma"] = self.get_gamma(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Dual Gamma"] = self.get_dual_gamma(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Vanna"] = self.get_vanna(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Charm"] = self.get_charm(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            put_option=put_option,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Vomma"] = self.get_vomma(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Vera"] = self.get_vera(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Veta"] = self.get_veta(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["PD"] = self.get_partial_derivative(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    def get_gamma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the gamma of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The gamma is
        the rate of change of the delta with respect to the price of the underlying asset.

        The gamma calculation is the theoretical value of the gamma. The actual gamma can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - Gamma = e^(—q * t) * N'(d1) / (S * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d1) is the standard normal probability density at d1 and N(d2) is
        the cumulative normal distribution of d2.

        The Gamma can be interpreted as follows:

        - If Gamma is high, it indicates that the option's Delta is highly sensitive to changes in the underlying
        asset's price. The option's Delta will change more significantly with small movements in the stock price.
        - If Gamma is low, it suggests that the option's Delta is relatively insensitive to changes in the
        underlying asset's price. The option's Delta changes more gradually with movements in the stock price.

        Note that the gamma of a call option and put option are equal to each other.

        Gamma is largest for at-the-money options close to expiry, which is where a delta hedge needs the most
        frequent rebalancing and where a short option position can lose money fastest on a sharp move. Because it
        is per $1 move in the underlying, it is not comparable across stocks with different prices; risk reports
        therefore use dollar gamma (Gamma x S^2 / 100, the change in the dollar delta for a 1% move) when they add
        up gamma across underlyings.

        Also known as: rate of change of delta, option convexity.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
                means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the gamma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PFE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_gamma(
            strike_price_range=0.2, strike_step_size=1
        ).loc["PFE", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |             19 |       0      |       0      |       0.0002 |
        |             20 |       0      |       0      |       0.0022 |
        |             21 |       0      |       0.0012 |       0.0135 |
        |             22 |       0.0013 |       0.0165 |       0.0502 |
        |             23 |       0.0395 |       0.0929 |       0.1195 |
        |             24 |       0.2623 |       0.2399 |       0.1926 |
        |             25 |       0.4395 |       0.3114 |       0.2199 |
        |             26 |       0.2178 |       0.2197 |       0.1852 |
        |             27 |       0.0366 |       0.0903 |       0.119  |
        |             28 |       0.0023 |       0.0229 |       0.0601 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        gamma: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            gamma[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            gamma[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_gamma(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        gamma_df = helpers.create_greek_dataframe(
            greek_dictionary=gamma,
            start_date=start_date,
        )

        gamma_df = gamma_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            gamma_df = calculate_standardization(
                dataset=gamma_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return gamma_df

    def get_dual_gamma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the dual gamma of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The dual gamma is
        the second derivative of the option price with respect to the strike price, i.e. the rate of change
        of the dual delta as the strike price changes. It is the strike-space counterpart of gamma and
        describes the (discounted) risk-neutral probability density of the underlying finishing at the strike.

        The dual gamma calculation is the theoretical value of the dual gamma. The actual dual gamma can differ
        from this value due to several factors such as the volatility of the underlying asset, the time to
        expiration, the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Dual Gamma = e^(—r * t) * N'(d2) / (K * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d2) is the standard normal probability density at d2 and N(d1) is
        the cumulative normal distribution of d1. Note that Dual Gamma is a second derivative with respect to the
        strike price, so it is the strike and not the stock price that appears in the denominator.

        Note that the dual gamma of a call option and put option are equal to each other.

        Also known as: strike gamma, binary option gamma.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the dual gamma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["T"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_dual_gamma(
            strike_price_range=0.2, strike_step_size=1
        ).loc["T", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |             19 |       0      |       0      |       0      |
        |             20 |       0      |       0      |       0.0007 |
        |             21 |       0      |       0.0003 |       0.008  |
        |             22 |       0.0003 |       0.0089 |       0.0445 |
        |             23 |       0.0254 |       0.0864 |       0.1339 |
        |             24 |       0.2943 |       0.2846 |       0.235  |
        |             25 |       0.5063 |       0.3616 |       0.2566 |
        |             26 |       0.1612 |       0.198  |       0.1842 |
        |             27 |       0.0115 |       0.0513 |       0.0911 |
        |             28 |       0.0002 |       0.0068 |       0.0323 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        dual_gamma: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            dual_gamma[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            dual_gamma[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_dual_gamma(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        dual_gamma_df = helpers.create_greek_dataframe(
            greek_dictionary=dual_gamma,
            start_date=start_date,
        )

        dual_gamma_df = dual_gamma_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            dual_gamma_df = calculate_standardization(
                dataset=dual_gamma_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return dual_gamma_df

    def get_vanna(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the vanna of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The vanna is
        the rate of change of the vega with respect to the price of the underlying asset.

        The vanna calculation is the theoretical value of the vanna. The actual vanna can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Vanna = —e^(—q * t) * N'(d1) * (d2 / σ)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Vanna can be interpreted as follows:

        - If Vanna is positive, it indicates that the Delta of the option becomes more positive as both the underlying
        asset's price and implied volatility increase, and more negative as they both decrease.
        - If Vanna is negative, it suggests that the Delta of the option becomes more negative as both the underlying
        asset's price and implied volatility increase, and more positive as they both decrease.

        Note that the vanna of a call option and put option are equal to each other.

        Also known as: delta-vega cross-derivative.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the vanna values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["DIS"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vanna(
            strike_price_range=0.15, strike_step_size=2, expiration_time_range=91
        ).loc["DIS", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             96 |      -0.3179 |      -0.595  |      -0.6673 |
        |             98 |      -0.4447 |      -0.6644 |      -0.6908 |
        |            100 |      -0.5731 |      -0.706  |      -0.6877 |
        |            102 |      -0.6795 |      -0.7113 |      -0.6555 |
        |            104 |      -0.7374 |      -0.675  |      -0.5938 |
        |            106 |      -0.7246 |      -0.5954 |      -0.504  |
        |            108 |      -0.6296 |      -0.4753 |      -0.3894 |
        |            110 |      -0.455  |      -0.3212 |      -0.2548 |
        |            112 |      -0.2189 |      -0.1429 |      -0.1063 |
        |            114 |       0.0498 |       0.048  |       0.0496 |
        |            116 |       0.3172 |       0.2394 |       0.2062 |
        |            118 |       0.5518 |       0.4196 |       0.3572 |
        |            120 |       0.73   |       0.5791 |       0.4969 |
        |            122 |       0.8397 |       0.7104 |       0.621  |
        |            124 |       0.8799 |       0.8092 |       0.7259 |
        |            126 |       0.8593 |       0.874  |       0.8096 |
        |            128 |       0.7923 |       0.9055 |       0.8713 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        vanna: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            vanna[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            vanna[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_vanna(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        vanna_df = helpers.create_greek_dataframe(
            greek_dictionary=vanna,
            start_date=start_date,
        )

        vanna_df = vanna_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            vanna_df = calculate_standardization(
                dataset=vanna_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return vanna_df

    def get_charm(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the charm of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The charm is
        the rate of change of the delta with respect to the time to expiration.

        The charm calculation is the theoretical value of the charm. The actual charm can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Charm = q * e^(—q * t) * N(d1) — e^(—q * t) * N'(d1) * (2 * (r — q) * t — d2 * σ * sqrt(t)) / (2 * t * σ * sqrt(t))
        - Put Charm = —q * e^(—q * t) * N(—d1) — e^(—q * t) * N'(d1) * (2 * (r — q) * t — d2 * σ * sqrt(t)) / (2 * t * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d1) is the standard normal probability density at d1 and N(d1) is
        the cumulative normal distribution of d1.

        Charm is the derivative with respect to calendar time elapsed, in the same direction as Theta, but it is
        reported per year rather than per day. Divide by 365 for delta decay per calendar day. Color follows the same
        per-year convention.

        The Charm can be interpreted as follows:

        - If Charm is positive, it suggests that the option's Delta is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in the underlying asset's price as time passes.
        - If Charm is negative, it indicates that the option's Delta is becoming more negative over time. The
        option is losing sensitivity to changes in the underlying asset's price as time passes.

        Also known as: delta time decay, delta bleed.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            put_option (bool, optional): Whether to calculate the put option charm. Defaults to False which means
            it will calculate the call option charm.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the charm values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["HD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_charm(
            strike_price_range=0.15, strike_step_size=5
        ).loc["HD", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0264 |       0.0405 |       0.1976 |
        |            295 |       0.0268 |       0.0759 |       0.3318 |
        |            300 |       0.0307 |       0.1725 |       0.5237 |
        |            305 |       0.057  |       0.3912 |       0.7661 |
        |            310 |       0.1842 |       0.8002 |       1.0313 |
        |            315 |       0.6282 |       1.4225 |       1.2703 |
        |            320 |       1.7329 |       2.1637 |       1.4208 |
        |            325 |       3.6145 |       2.778  |       1.4233 |
        |            330 |       5.5276 |       2.9353 |       1.2407 |
        |            335 |       5.8386 |       2.3899 |       0.8728 |
        |            340 |       3.3619 |       1.1522 |       0.3608 |
        |            345 |      -1.0897 |      -0.4685 |      -0.2214 |
        |            350 |      -5.0645 |      -1.9986 |      -0.7863 |
        |            355 |      -6.6129 |      -3.026  |      -1.2543 |
        |            360 |      -5.7284 |      -3.3722 |      -1.5715 |
        |            365 |      -3.7535 |      -3.1179 |      -1.7176 |
        |            370 |      -1.9567 |      -2.5018 |      -1.7047 |
        |            375 |      -0.8339 |      -1.7829 |      -1.5677 |
        |            380 |      -0.2958 |      -1.1446 |      -1.3518 |
        |            385 |      -0.0885 |      -0.6684 |      -1.1018 |
        |            390 |      -0.0225 |      -0.3577 |      -0.8537 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        charm: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            charm[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            charm[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_charm(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )

        charm_df = helpers.create_greek_dataframe(
            greek_dictionary=charm,
            start_date=start_date,
        )

        charm_df = charm_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            charm_df = calculate_standardization(
                dataset=charm_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return charm_df

    def get_vomma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the vomma of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The vomma is
        the rate of change of the vega with respect to the volatility of the underlying asset.

        The vomma calculation is the theoretical value of the vomma. The actual vomma can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Vomma = S * e^(—q * t) * N'(d1) * sqrt(t) * (d1 * d2) / σ

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The vomma can be interpreted as follows:

        - If Vomma is high, it indicates that the option's Vega is highly sensitive to changes in implied
        volatility. The option's value will experience more significant fluctuations with variations in
        implied volatility.
        - If Vomma is low, it suggests that the option's Vega is relatively less sensitive to changes in
        implied volatility.

        Also known as: volga, vega convexity.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the vomma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["UNH"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vomma(
            strike_price_range=0.15, strike_step_size=5, expiration_time_range=91
        ).loc["UNH", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            280 |      51.2037 |      52.1525 |      47.8087 |
        |            285 |      47.8315 |      45.2094 |      40.3451 |
        |            290 |      42.7785 |      37.8599 |      32.9563 |
        |            295 |      36.4224 |      30.4288 |      25.8629 |
        |            300 |      29.2618 |      23.2357 |      19.2678 |
        |            305 |      21.867  |      16.58   |      13.3503 |
        |            310 |      14.8246 |      10.7274 |       8.2623 |
        |            315 |       8.6826 |       5.8997 |       4.1252 |
        |            320 |       3.903  |       2.267  |       1.0283 |
        |            325 |       0.8248 |      -0.0555 |      -0.9713 |
        |            330 |      -0.3565 |      -1.0091 |      -1.8478 |
        |            335 |       0.4043 |      -0.5893 |      -1.605  |
        |            340 |       3.0105 |       1.1584 |      -0.274  |
        |            345 |       7.2433 |       4.1444 |       2.0906 |
        |            350 |      12.7898 |       8.2425 |       5.4135 |
        |            355 |      19.2757 |      13.2979 |       9.6027 |
        |            360 |      26.2986 |      19.1359 |      14.5533 |
        |            365 |      33.459  |      25.5694 |      20.1513 |
        |            370 |      40.3869 |      32.4074 |      26.2771 |
        |            375 |      46.762  |      39.4615 |      32.8086 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        vomma: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            vomma[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            vomma[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_vomma(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        vomma_df = helpers.create_greek_dataframe(
            greek_dictionary=vomma,
            start_date=start_date,
        )

        vomma_df = vomma_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            vomma_df = calculate_standardization(
                dataset=vomma_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return vomma_df

    def get_vera(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the vera of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The vera is
        the rate of change of the rho with respect to volatility.

        The vera calculation is the theoretical value of the vera. The actual vera can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Vera = —K * t * e^(—r * t) * N'(d2) * (d1 / σ)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d2) is the standard normal probability density at d2 and N(d1) is
        the cumulative normal distribution of d1.

        Vera is reported unscaled, per 1.00 of volatility and per 1.00 of the risk free rate, matching Rho.

        The Vera can be interpreted as follows:

        - If Vera is positive, it indicates that the option's Rho becomes more positive as implied volatility rises.
        In other words, the option gains sensitivity to the risk free rate when volatility increases.
        - If Vera is negative, it suggests that the option's Rho becomes more negative as implied volatility rises.
        The option loses sensitivity to the risk free rate when volatility increases.

        Note that the vera of a call option and put option are equal to each other.

        Also known as: rho-vega cross-derivative.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the vera values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["LLY"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vera(
            strike_price_range=0.15, strike_step_size=20, expiration_time_range=181
        ).loc["LLY", ["2026-01-30", "2026-03-31", "2026-06-29"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-31 |   2026-06-29 |
        |---------------:|-------------:|-------------:|-------------:|
        |            920 |     -45.0097 |    -157.22   |    -295.3    |
        |            940 |     -50.1028 |    -152.347  |    -280.257  |
        |            960 |     -52.655  |    -143.627  |    -261.579  |
        |            980 |     -52.0124 |    -131.274  |    -239.616  |
        |           1000 |     -47.8631 |    -115.645  |    -214.77   |
        |           1020 |     -40.3005 |     -97.2153 |    -187.477  |
        |           1040 |     -29.8155 |     -76.5467 |    -158.194  |
        |           1060 |     -17.222  |     -54.2559 |    -127.389  |
        |           1080 |      -3.5353 |     -30.9814 |     -95.5245 |
        |           1100 |      10.1718 |      -7.3538 |     -63.0514 |
        |           1120 |      22.905  |      16.0301 |     -30.3978 |
        |           1140 |      33.8572 |      38.6281 |       2.0364 |
        |           1160 |      42.4753 |      59.9686 |      33.8855 |
        |           1180 |      48.4829 |      79.6604 |      64.8218 |
        |           1200 |      51.8631 |      97.3981 |      94.5578 |
        |           1220 |      52.8138 |     112.963  |     122.847  |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        vera: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            vera[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            vera[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_vera(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        vera_df = helpers.create_greek_dataframe(
            greek_dictionary=vera,
            start_date=start_date,
        )

        vera_df = vera_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            vera_df = calculate_standardization(
                dataset=vera_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return vera_df

    def get_veta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the veta of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The veta is
        the rate of change of the vega with respect to the time to expiration.

        The veta calculation is the theoretical value of the veta. The actual veta can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Veta = S * e^(—q * t) * N'(d1) * sqrt(t) * (q + ((r — q) * d1) / (σ * sqrt(t)) — (1 + d1 * d2) / (2 * t)) / (100 * 365)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N'(d1) is the standard normal probability density at d1 and N(d2) is
        the cumulative normal distribution of d2.

        This expression equals dVega/dt, the change in Vega as calendar time passes (the negative of the derivative
        with respect to the remaining time to maturity t), and is used as is, without a change of sign. Veta thus,
        like Theta, Charm and Color, measures the change per unit of time that passes: an option close to the money
        loses Vega as expiry approaches, so its Veta is typically negative.

        It is common practice to divide the mathematical result of veta by 100 times the number of days per year to
        express it as the change in Vega (per percentage point of volatility) per calendar day. This is also done here.

        The Veta can be interpreted as follows:

        - If Veta is positive, it indicates that the option's Vega is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in implied volatility as time passes.
        - If Veta is negative, it suggests that the option's Vega is becoming more negative over time. The
        option is losing sensitivity to changes in implied volatility as time passes.

        Also known as: vega time decay.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the veta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AZO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_veta(
            strike_price_range=0.1, strike_step_size=50
        ).loc["AZO", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |           3050 |      -0.0051 |      -0.0325 |      -0.0541 |
        |           3100 |      -0.0189 |      -0.0561 |      -0.0651 |
        |           3150 |      -0.051  |      -0.0816 |      -0.0722 |
        |           3200 |      -0.1    |      -0.1011 |      -0.0746 |
        |           3250 |      -0.1436 |      -0.1082 |      -0.0728 |
        |           3300 |      -0.1551 |      -0.1037 |      -0.0692 |
        |           3350 |      -0.1403 |      -0.096  |      -0.0664 |
        |           3400 |      -0.1344 |      -0.0946 |      -0.0665 |
        |           3450 |      -0.1525 |      -0.1026 |      -0.0701 |
        |           3500 |      -0.1688 |      -0.1146 |      -0.0762 |
        |           3550 |      -0.154  |      -0.1213 |      -0.0826 |
        |           3600 |      -0.1108 |      -0.1163 |      -0.0869 |
        |           3650 |      -0.063  |      -0.0994 |      -0.0875 |
        |           3700 |      -0.0288 |      -0.0759 |      -0.0835 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        veta: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            veta[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            veta[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_veta(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        veta_df = helpers.create_greek_dataframe(
            greek_dictionary=veta,
            start_date=start_date,
        )

        veta_df = veta_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            veta_df = calculate_standardization(
                dataset=veta_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return veta_df

    def get_partial_derivative(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the partial derivative of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The partial derivative is
        the second derivative of the option price with respect to the strike price, the rate at which the dual
        delta changes with the strike.

        Note that this uses a single, flat assumed volatility (the same value at every strike price) rather than
        the market's actual implied volatility smile. This means it is NOT the Breeden-Litzenberger risk-neutral
        density -- with a flat volatility input the second derivative can only ever recover a lognormal density,
        regardless of what the real market smile looks like, which defeats the entire purpose of that theorem. For
        the actual market-implied (smile-consistent) risk-neutral density, see `get_risk_neutral_density`, which
        uses this same second-derivative relationship but applied to a volatility surface calibrated to real
        market option prices instead of a flat assumption.

        The formula is as follows:

        - Partial Derivative (PD) = e^(—r * t) * (1 / K) * (1 / sqrt(2 * pi * σ ** 2 * t)) *
        e^(—(1 / (2 * σ ** 2 * t)) * (ln(K / S) — ((r — q) — (0.5 * σ ** 2)) * t) ** 2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility and t is the time to expiration. This expression is algebraically identical to
        e^(—r * t) * N'(d2) / (K * σ * sqrt(t)), i.e. to the Dual Gamma, since both are the second derivative of the
        option price with respect to the strike price.

        Also known as: second strike derivative, lognormal risk-neutral density.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the partial derivative values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["BAC"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_partial_derivative(
            strike_price_range=0.2, strike_step_size=1
        ).loc["BAC", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |             44 |       0      |       0      |       0.0014 |
        |             45 |       0      |       0.0001 |       0.0032 |
        |             46 |       0      |       0.0005 |       0.0066 |
        |             47 |       0      |       0.0019 |       0.0124 |
        |             48 |       0.0003 |       0.0056 |       0.0213 |
        |             49 |       0.0017 |       0.014  |       0.0332 |
        |             50 |       0.008  |       0.0296 |       0.0478 |
        |             51 |       0.0269 |       0.0536 |       0.0635 |
        |             52 |       0.0665 |       0.0833 |       0.0783 |
        |             53 |       0.1241 |       0.1125 |       0.09   |
        |             54 |       0.1769 |       0.1328 |       0.0967 |
        |             55 |       0.1956 |       0.1382 |       0.0975 |
        |             56 |       0.1701 |       0.1275 |       0.0927 |
        |             57 |       0.1178 |       0.105  |       0.0832 |
        |             58 |       0.0657 |       0.0776 |       0.0708 |
        |             59 |       0.0299 |       0.0518 |       0.0573 |
        |             60 |       0.0112 |       0.0314 |       0.0441 |
        |             61 |       0.0035 |       0.0173 |       0.0325 |
        |             62 |       0.0009 |       0.0088 |       0.0229 |
        |             63 |       0.0002 |       0.0041 |       0.0155 |
        |             64 |       0      |       0.0018 |       0.0101 |
        |             65 |       0      |       0.0007 |       0.0063 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        partial_derivative: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            partial_derivative[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            partial_derivative[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_second_order_partial_derivative(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        partial_derivative_df = helpers.create_greek_dataframe(
            greek_dictionary=partial_derivative,
            start_date=start_date,
        )

        partial_derivative_df = partial_derivative_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            partial_derivative_df = calculate_standardization(
                dataset=partial_derivative_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return partial_derivative_df

    def collect_third_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the third order Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        - Speed: measures the rate of change in Gamma with respect to changes in the underlying price.
        - Zomma: measures the rate of change of gamma with respect to changes in volatility.
        - Color: also referred to as gamma decay or DgammaDtime measures the rate of change of gamma over
        the passage of time.
        - Ultima: measures the sensitivity of the option vomma with respect to change in volatility.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the stock price (the traded close), risk free rate, dividend yield and volatility of the most recent
        date are used; changing the start date prices the option as of that date instead. The volatility is the
        annualized standard deviation of the daily returns over the year before that date and the dividend yield the
        dividends of the 365 days up to that date relative to its close, so an earlier start date only uses what was
        known on that date. The exception is a date in the first year of the price history: without a full year
        before it, it takes the dividend yield of the first date that has one.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the third order greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["MU", "AMZN"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.collect_third_order_greeks().loc["MU"]
        ```

        Which returns:

        |   Strike Price |   (Period('2026-01-01', 'D'), 'Speed') |   (Period('2026-01-01', 'D'), 'Zomma') |   (Period('2026-01-01', 'D'), 'Color') |   (Period('2026-01-01', 'D'), 'Ultima') |   (Period('2026-01-02', 'D'), 'Speed') |   (Period('2026-01-02', 'D'), 'Zomma') |   (Period('2026-01-02', 'D'), 'Color') |   (Period('2026-01-02', 'D'), 'Ultima') |   (Period('2026-01-03', 'D'), 'Speed') |   (Period('2026-01-03', 'D'), 'Zomma') |   (Period('2026-01-03', 'D'), 'Color') |   (Period('2026-01-03', 'D'), 'Ultima') |   (Period('2026-01-04', 'D'), 'Speed') |   (Period('2026-01-04', 'D'), 'Zomma') |   (Period('2026-01-04', 'D'), 'Color') |   (Period('2026-01-04', 'D'), 'Ultima') |   (Period('2026-01-05', 'D'), 'Speed') |   (Period('2026-01-05', 'D'), 'Zomma') |   (Period('2026-01-05', 'D'), 'Color') |   (Period('2026-01-05', 'D'), 'Ultima') |   (Period('2026-01-06', 'D'), 'Speed') |   (Period('2026-01-06', 'D'), 'Zomma') |   (Period('2026-01-06', 'D'), 'Color') |   (Period('2026-01-06', 'D'), 'Ultima') |   (Period('2026-01-07', 'D'), 'Speed') |   (Period('2026-01-07', 'D'), 'Zomma') |   (Period('2026-01-07', 'D'), 'Color') |   (Period('2026-01-07', 'D'), 'Ultima') |   (Period('2026-01-08', 'D'), 'Speed') |   (Period('2026-01-08', 'D'), 'Zomma') |   (Period('2026-01-08', 'D'), 'Color') |   (Period('2026-01-08', 'D'), 'Ultima') |   (Period('2026-01-09', 'D'), 'Speed') |   (Period('2026-01-09', 'D'), 'Zomma') |   (Period('2026-01-09', 'D'), 'Color') |   (Period('2026-01-09', 'D'), 'Ultima') |   (Period('2026-01-10', 'D'), 'Speed') |   (Period('2026-01-10', 'D'), 'Zomma') |   (Period('2026-01-10', 'D'), 'Color') |   (Period('2026-01-10', 'D'), 'Ultima') |   (Period('2026-01-11', 'D'), 'Speed') |   (Period('2026-01-11', 'D'), 'Zomma') |   (Period('2026-01-11', 'D'), 'Color') |   (Period('2026-01-11', 'D'), 'Ultima') |   (Period('2026-01-12', 'D'), 'Speed') |   (Period('2026-01-12', 'D'), 'Zomma') |   (Period('2026-01-12', 'D'), 'Color') |   (Period('2026-01-12', 'D'), 'Ultima') |   (Period('2026-01-13', 'D'), 'Speed') |   (Period('2026-01-13', 'D'), 'Zomma') |   (Period('2026-01-13', 'D'), 'Color') |   (Period('2026-01-13', 'D'), 'Ultima') |   (Period('2026-01-14', 'D'), 'Speed') |   (Period('2026-01-14', 'D'), 'Zomma') |   (Period('2026-01-14', 'D'), 'Color') |   (Period('2026-01-14', 'D'), 'Ultima') |   (Period('2026-01-15', 'D'), 'Speed') |   (Period('2026-01-15', 'D'), 'Zomma') |   (Period('2026-01-15', 'D'), 'Color') |   (Period('2026-01-15', 'D'), 'Ultima') |   (Period('2026-01-16', 'D'), 'Speed') |   (Period('2026-01-16', 'D'), 'Zomma') |   (Period('2026-01-16', 'D'), 'Color') |   (Period('2026-01-16', 'D'), 'Ultima') |   (Period('2026-01-17', 'D'), 'Speed') |   (Period('2026-01-17', 'D'), 'Zomma') |   (Period('2026-01-17', 'D'), 'Color') |   (Period('2026-01-17', 'D'), 'Ultima') |   (Period('2026-01-18', 'D'), 'Speed') |   (Period('2026-01-18', 'D'), 'Zomma') |   (Period('2026-01-18', 'D'), 'Color') |   (Period('2026-01-18', 'D'), 'Ultima') |   (Period('2026-01-19', 'D'), 'Speed') |   (Period('2026-01-19', 'D'), 'Zomma') |   (Period('2026-01-19', 'D'), 'Color') |   (Period('2026-01-19', 'D'), 'Ultima') |   (Period('2026-01-20', 'D'), 'Speed') |   (Period('2026-01-20', 'D'), 'Zomma') |   (Period('2026-01-20', 'D'), 'Color') |   (Period('2026-01-20', 'D'), 'Ultima') |   (Period('2026-01-21', 'D'), 'Speed') |   (Period('2026-01-21', 'D'), 'Zomma') |   (Period('2026-01-21', 'D'), 'Color') |   (Period('2026-01-21', 'D'), 'Ultima') |   (Period('2026-01-22', 'D'), 'Speed') |   (Period('2026-01-22', 'D'), 'Zomma') |   (Period('2026-01-22', 'D'), 'Color') |   (Period('2026-01-22', 'D'), 'Ultima') |   (Period('2026-01-23', 'D'), 'Speed') |   (Period('2026-01-23', 'D'), 'Zomma') |   (Period('2026-01-23', 'D'), 'Color') |   (Period('2026-01-23', 'D'), 'Ultima') |   (Period('2026-01-24', 'D'), 'Speed') |   (Period('2026-01-24', 'D'), 'Zomma') |   (Period('2026-01-24', 'D'), 'Color') |   (Period('2026-01-24', 'D'), 'Ultima') |   (Period('2026-01-25', 'D'), 'Speed') |   (Period('2026-01-25', 'D'), 'Zomma') |   (Period('2026-01-25', 'D'), 'Color') |   (Period('2026-01-25', 'D'), 'Ultima') |   (Period('2026-01-26', 'D'), 'Speed') |   (Period('2026-01-26', 'D'), 'Zomma') |   (Period('2026-01-26', 'D'), 'Color') |   (Period('2026-01-26', 'D'), 'Ultima') |   (Period('2026-01-27', 'D'), 'Speed') |   (Period('2026-01-27', 'D'), 'Zomma') |   (Period('2026-01-27', 'D'), 'Color') |   (Period('2026-01-27', 'D'), 'Ultima') |   (Period('2026-01-28', 'D'), 'Speed') |   (Period('2026-01-28', 'D'), 'Zomma') |   (Period('2026-01-28', 'D'), 'Color') |   (Period('2026-01-28', 'D'), 'Ultima') |   (Period('2026-01-29', 'D'), 'Speed') |   (Period('2026-01-29', 'D'), 'Zomma') |   (Period('2026-01-29', 'D'), 'Color') |   (Period('2026-01-29', 'D'), 'Ultima') |
        |---------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|----------------------------------------:|
        |            215 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0      |                                  0.0002 |                                -0      |                                 0      |                                -0.0001 |                                  0.0462 |                                -0      |                                 0      |                                -0.0012 |                                  0.6517 |                                -0      |                                 0.0002 |                                -0.0046 |                                  2.8961 |                                -0      |                                 0.0005 |                                -0.01   |                                  7.3099 |                                -0      |                                 0.001  |                                -0.0163 |                                 13.4344 |                                -0      |                                 0.0016 |                                -0.0224 |                                 20.3143 |                                -0      |                                 0.0022 |                                -0.0277 |                                 27.0161 |                                -0      |                                 0.0028 |                                -0.0318 |                                 32.8627 |                                -0      |                                 0.0034 |                                -0.0346 |                                 37.4633 |                                -0      |                                 0.0039 |                                -0.0364 |                                 40.6574 |                                -0      |                                 0.0043 |                                -0.0373 |                                 42.4423 |                                -0      |                                 0.0047 |                                -0.0374 |                                 42.9099 |                                -0.0001 |                                 0.0049 |                                -0.037  |                                 42.2021 |                                -0.0001 |                                 0.0052 |                                -0.0362 |                                 40.4799 |                                -0.0001 |                                 0.0053 |                                -0.0351 |                                 37.9056 |                                -0.0001 |                                 0.0054 |                                -0.0338 |                                 34.6322 |                                -0.0001 |                                 0.0055 |                                -0.0323 |                                 30.7978 |                                -0.0001 |                                 0.0055 |                                -0.0308 |                                 26.5243 |                                -0.0001 |                                 0.0055 |                                -0.0292 |                                 21.9168 |                                -0.0001 |                                 0.0055 |                                -0.0277 |                                 17.0646 |                                -0.0001 |                                 0.0054 |                                -0.0262 |                                 12.043  |                                -0.0001 |                                 0.0053 |                                -0.0247 |                                  6.9146 |                                -0.0001 |                                 0.0052 |                                -0.0232 |                                  1.731  |                                -0.0001 |                                 0.0051 |                                -0.0219 |                                 -3.4654 |                                -0.0001 |                                 0.005  |                                -0.0205 |                                 -8.6404 |                                -0.0001 |                                 0.0049 |                                -0.0193 |                                -13.7662 |                                -0.0001 |                                 0.0048 |                                -0.0181 |                                -18.8211 |
        |            220 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0      |                                  0.0023 |                                -0      |                                 0      |                                -0.0007 |                                  0.2279 |                                -0      |                                 0.0002 |                                -0.0045 |                                  1.9556 |                                -0      |                                 0.0005 |                                -0.0124 |                                  6.4192 |                                -0      |                                 0.0012 |                                -0.0222 |                                 13.1726 |                                -0      |                                 0.0019 |                                -0.0315 |                                 20.7745 |                                -0      |                                 0.0028 |                                -0.0389 |                                 27.857  |                                -0      |                                 0.0035 |                                -0.0442 |                                 33.5462 |                                -0      |                                 0.0042 |                                -0.0473 |                                 37.4485 |                                -0.0001 |                                 0.0048 |                                -0.0488 |                                 39.5046 |                                -0.0001 |                                 0.0052 |                                -0.0489 |                                 39.847  |                                -0.0001 |                                 0.0056 |                                -0.0481 |                                 38.6989 |                                -0.0001 |                                 0.0058 |                                -0.0465 |                                 36.3117 |                                -0.0001 |                                 0.006  |                                -0.0446 |                                 32.9304 |                                -0.0001 |                                 0.0061 |                                -0.0424 |                                 28.7768 |                                -0.0001 |                                 0.0061 |                                -0.04   |                                 24.043  |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                 18.8906 |                                -0.0001 |                                 0.006  |                                -0.0352 |                                 13.4528 |                                -0.0001 |                                 0.0059 |                                -0.0329 |                                  7.8377 |                                -0.0001 |                                 0.0058 |                                -0.0307 |                                  2.132  |                                -0.0001 |                                 0.0057 |                                -0.0285 |                                 -3.5961 |                                -0.0001 |                                 0.0055 |                                -0.0265 |                                 -9.2928 |                                -0.0001 |                                 0.0053 |                                -0.0246 |                                -14.917  |                                -0.0001 |                                 0.0052 |                                -0.0228 |                                -20.4373 |                                -0.0001 |                                 0.005  |                                -0.0211 |                                -25.8304 |                                -0.0001 |                                 0.0048 |                                -0.0195 |                                -31.0792 |                                -0.0001 |                                 0.0046 |                                -0.018  |                                -36.1721 |                                -0.0001 |                                 0.0044 |                                -0.0167 |                                -41.1014 |
        |            225 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0.0001 |                                  0.0214 |                                -0      |                                 0.0001 |                                -0.0032 |                                  0.8894 |                                -0      |                                 0.0005 |                                -0.014  |                                  4.8702 |                                -0      |                                 0.0013 |                                -0.0293 |                                 12.1181 |                                -0      |                                 0.0023 |                                -0.0439 |                                 20.5173 |                                -0      |                                 0.0034 |                                -0.0548 |                                 27.9779 |                                -0      |                                 0.0044 |                                -0.0615 |                                 33.339  |                                -0.0001 |                                 0.0051 |                                -0.0646 |                                 36.2509 |                                -0.0001 |                                 0.0058 |                                -0.065  |                                 36.8391 |                                -0.0001 |                                 0.0062 |                                -0.0636 |                                 35.4429 |                                -0.0001 |                                 0.0065 |                                -0.061  |                                 32.4609 |                                -0.0001 |                                 0.0067 |                                -0.0577 |                                 28.2735 |                                -0.0001 |                                 0.0067 |                                -0.054  |                                 23.212  |                                -0.0001 |                                 0.0067 |                                -0.0502 |                                 17.5492 |                                -0.0001 |                                 0.0066 |                                -0.0464 |                                 11.5031 |                                -0.0001 |                                 0.0065 |                                -0.0427 |                                  5.2433 |                                -0.0001 |                                 0.0063 |                                -0.0392 |                                 -1.1003 |                                -0.0001 |                                 0.0061 |                                -0.0359 |                                 -7.4303 |                                -0.0001 |                                 0.0059 |                                -0.0328 |                                -13.6747 |                                -0.0001 |                                 0.0057 |                                -0.03   |                                -19.7816 |                                -0.0001 |                                 0.0055 |                                -0.0273 |                                -25.7145 |                                -0.0001 |                                 0.0052 |                                -0.0249 |                                -31.4487 |                                -0.0001 |                                 0.005  |                                -0.0227 |                                -36.9686 |                                -0.0001 |                                 0.0047 |                                -0.0206 |                                -42.2655 |                                -0.0001 |                                 0.0045 |                                -0.0188 |                                -47.3358 |                                -0.0001 |                                 0.0043 |                                -0.017  |                                -52.1796 |                                -0.0001 |                                 0.004  |                                -0.0155 |                                -56.8001 |                                -0.0001 |                                 0.0038 |                                -0.014  |                                -61.2021 |
        |            230 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0.0009 |                                  0.1466 |                                -0      |                                 0.0003 |                                -0.0126 |                                  2.7616 |                                -0      |                                 0.0013 |                                -0.0367 |                                 10.0594 |                                -0      |                                 0.0027 |                                -0.0608 |                                 19.3768 |                                -0.0001 |                                 0.0041 |                                -0.0774 |                                 27.319  |                                -0.0001 |                                 0.0053 |                                -0.086  |                                 32.2266 |                                -0.0001 |                                 0.0063 |                                -0.0883 |                                 33.8627 |                                -0.0001 |                                 0.0069 |                                -0.0865 |                                 32.6688 |                                -0.0001 |                                 0.0073 |                                -0.0821 |                                 29.2873 |                                -0.0001 |                                 0.0075 |                                -0.0765 |                                 24.34   |                                -0.0001 |                                 0.0075 |                                -0.0703 |                                 18.3477 |                                -0.0001 |                                 0.0074 |                                -0.064  |                                 11.717  |                                -0.0001 |                                 0.0073 |                                -0.0579 |                                  4.7522 |                                -0.0001 |                                 0.007  |                                -0.0522 |                                 -2.3262 |                                -0.0001 |                                 0.0068 |                                -0.0469 |                                 -9.3622 |                                -0.0001 |                                 0.0064 |                                -0.042  |                                -16.2485 |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                -22.9138 |                                -0.0001 |                                 0.0058 |                                -0.0336 |                                -29.3128 |                                -0.0001 |                                 0.0055 |                                -0.03   |                                -35.4195 |                                -0.0001 |                                 0.0051 |                                -0.0267 |                                -41.221  |                                -0.0001 |                                 0.0048 |                                -0.0238 |                                -46.7141 |                                -0.0001 |                                 0.0045 |                                -0.0212 |                                -51.9022 |                                -0.0001 |                                 0.0042 |                                -0.0188 |                                -56.793  |                                -0.0001 |                                 0.0039 |                                -0.0167 |                                -61.3971 |                                -0.0001 |                                 0.0036 |                                -0.0148 |                                -65.7271 |                                -0.0001 |                                 0.0033 |                                -0.013  |                                -69.7963 |                                -0.0001 |                                 0.0031 |                                -0.0115 |                                -73.6186 |                                -0.0001 |                                 0.0028 |                                -0.0101 |                                -77.208  |
        |            235 |                                -0      |                                 0      |                                -0      |                                  0.0003 |                                -0      |                                 0.0001 |                                -0.006  |                                  0.7369 |                                -0      |                                 0.0011 |                                -0.0402 |                                  6.8209 |                                -0      |                                 0.0029 |                                -0.0824 |                                 17.1105 |                                -0.0001 |                                 0.0049 |                                -0.1103 |                                 25.8465 |                                -0.0001 |                                 0.0064 |                                -0.1216 |                                 30.2851 |                                -0.0001 |                                 0.0075 |                                -0.1215 |                                 30.3907 |                                -0.0001 |                                 0.0082 |                                -0.115  |                                 27.1328 |                                -0.0001 |                                 0.0084 |                                -0.1054 |                                 21.5931 |                                -0.0001 |                                 0.0084 |                                -0.0947 |                                 14.6651 |                                -0.0001 |                                 0.0083 |                                -0.0841 |                                  7.0056 |                                -0.0001 |                                 0.008  |                                -0.0741 |                                 -0.932  |                                -0.0001 |                                 0.0076 |                                -0.0649 |                                 -8.8474 |                                -0.0001 |                                 0.0071 |                                -0.0567 |                                -16.5502 |                                -0.0001 |                                 0.0067 |                                -0.0494 |                                -23.9254 |                                -0.0001 |                                 0.0062 |                                -0.043  |                                -30.9092 |                                -0.0001 |                                 0.0058 |                                -0.0373 |                                -37.4717 |                                -0.0001 |                                 0.0053 |                                -0.0324 |                                -43.6053 |                                -0.0001 |                                 0.0049 |                                -0.028  |                                -49.3163 |                                -0.0001 |                                 0.0045 |                                -0.0242 |                                -54.62   |                                -0.0001 |                                 0.0041 |                                -0.0208 |                                -59.5365 |                                -0.0001 |                                 0.0037 |                                -0.0178 |                                -64.0887 |                                -0.0001 |                                 0.0033 |                                -0.0152 |                                -68.3004 |                                -0.0001 |                                 0.003  |                                -0.013  |                                -72.1954 |                                -0.0001 |                                 0.0027 |                                -0.0109 |                                -75.7969 |                                -0.0001 |                                 0.0023 |                                -0.0091 |                                -79.127  |                                -0.0001 |                                 0.0021 |                                -0.0076 |                                -82.2065 |                                -0.0001 |                                 0.0018 |                                -0.0062 |                                -85.0551 |                                -0.0001 |                                 0.0015 |                                -0.0049 |                                -87.6906 |
        |            240 |                                -0      |                                 0      |                                -0.0002 |                                  0.0079 |                                -0      |                                 0.0005 |                                -0.0292 |                                  2.7238 |                                -0      |                                 0.0028 |                                -0.1053 |                                 13.2964 |                                -0.0001 |                                 0.0056 |                                -0.1584 |                                 23.4962 |                                -0.0001 |                                 0.0077 |                                -0.1751 |                                 27.6953 |                                -0.0002 |                                 0.009  |                                -0.1694 |                                 26.0801 |                                -0.0002 |                                 0.0095 |                                -0.1535 |                                 20.5216 |                                -0.0002 |                                 0.0096 |                                -0.1344 |                                 12.7461 |                                -0.0002 |                                 0.0093 |                                -0.1155 |                                  3.9649 |                                -0.0002 |                                 0.0088 |                                -0.0982 |                                 -5.0606 |                                -0.0002 |                                 0.0082 |                                -0.083  |                                -13.8839 |                                -0.0002 |                                 0.0075 |                                -0.0698 |                                -22.261  |                                -0.0002 |                                 0.0069 |                                -0.0586 |                                -30.0728 |                                -0.0002 |                                 0.0062 |                                -0.049  |                                -37.2753 |                                -0.0002 |                                 0.0056 |                                -0.0409 |                                -43.8678 |                                -0.0002 |                                 0.005  |                                -0.034  |                                -49.8739 |                                -0.0002 |                                 0.0044 |                                -0.0281 |                                -55.33   |                                -0.0002 |                                 0.0039 |                                -0.0232 |                                -60.2776 |                                -0.0001 |                                 0.0034 |                                -0.0189 |                                -64.7599 |                                -0.0001 |                                 0.0029 |                                -0.0153 |                                -68.819  |                                -0.0001 |                                 0.0025 |                                -0.0122 |                                -72.4946 |                                -0.0001 |                                 0.0021 |                                -0.0095 |                                -75.8236 |                                -0.0001 |                                 0.0017 |                                -0.0072 |                                -78.8397 |                                -0.0001 |                                 0.0014 |                                -0.0053 |                                -81.5736 |                                -0.0001 |                                 0.0011 |                                -0.0036 |                                -84.0528 |                                -0.0001 |                                 0.0007 |                                -0.0021 |                                -86.3023 |                                -0.0001 |                                 0.0005 |                                -0.0008 |                                -88.3444 |                                -0.0001 |                                 0.0002 |                                 0.0003 |                                -90.1989 |                                -0.0001 |                                -0      |                                 0.0012 |                                -91.8839 |
        |            245 |                                -0      |                                 0      |                                -0.0028 |                                  0.107  |                                -0      |                                 0.0019 |                                -0.1076 |                                  7.3595 |                                -0.0001 |                                 0.006  |                                -0.2274 |                                 19.9799 |                                -0.0002 |                                 0.0092 |                                -0.2596 |                                 24.7342 |                                -0.0002 |                                 0.0107 |                                -0.2416 |                                 21.3397 |                                -0.0002 |                                 0.011  |                                -0.2069 |                                 13.2918 |                                -0.0002 |                                 0.0106 |                                -0.1707 |                                  3.3131 |                                -0.0002 |                                 0.0099 |                                -0.1383 |                                 -7.0299 |                                -0.0002 |                                 0.009  |                                -0.111  |                                -16.9457 |                                -0.0002 |                                 0.008  |                                -0.0886 |                                -26.0824 |                                -0.0002 |                                 0.007  |                                -0.0705 |                                -34.3209 |                                -0.0002 |                                 0.0061 |                                -0.0558 |                                -41.6597 |                                -0.0002 |                                 0.0052 |                                -0.0438 |                                -48.1524 |                                -0.0002 |                                 0.0044 |                                -0.0342 |                                -53.8753 |                                -0.0002 |                                 0.0037 |                                -0.0263 |                                -58.9105 |                                -0.0002 |                                 0.003  |                                -0.0199 |                                -63.3376 |                                -0.0002 |                                 0.0024 |                                -0.0146 |                                -67.2303 |                                -0.0002 |                                 0.0019 |                                -0.0102 |                                -70.6543 |                                -0.0002 |                                 0.0014 |                                -0.0067 |                                -73.6681 |                                -0.0002 |                                 0.0009 |                                -0.0037 |                                -76.3227 |                                -0.0001 |                                 0.0005 |                                -0.0012 |                                -78.6623 |                                -0.0001 |                                 0.0001 |                                 0.0008 |                                -80.7258 |                                -0.0001 |                                -0.0002 |                                 0.0025 |                                -82.5464 |                                -0.0001 |                                -0.0006 |                                 0.0039 |                                -84.1533 |                                -0.0001 |                                -0.0008 |                                 0.0051 |                                -85.5717 |                                -0.0001 |                                -0.0011 |                                 0.0061 |                                -86.8236 |                                -0.0001 |                                -0.0014 |                                 0.0069 |                                -87.928  |                                -0.0001 |                                -0.0016 |                                 0.0076 |                                -88.9017 |                                -0.0001 |                                -0.0018 |                                 0.0081 |                                -89.7592 |
        |            250 |                                -0      |                                 0.0003 |                                -0.0306 |                                  0.8572 |                                -0.0001 |                                 0.0054 |                                -0.3043 |                                 14.2006 |                                -0.0002 |                                 0.0106 |                                -0.4024 |                                 21.6781 |                                -0.0003 |                                 0.0127 |                                -0.3589 |                                 16.7711 |                                -0.0003 |                                 0.0126 |                                -0.2843 |                                  6.0745 |                                -0.0003 |                                 0.0115 |                                -0.2159 |                                 -5.9653 |                                -0.0003 |                                 0.0101 |                                -0.1611 |                                -17.4029 |                                -0.0003 |                                 0.0086 |                                -0.1192 |                                -27.5631 |                                -0.0003 |                                 0.0071 |                                -0.0875 |                                -36.3191 |                                -0.0003 |                                 0.0058 |                                -0.0635 |                                -43.7586 |                                -0.0003 |                                 0.0046 |                                -0.0454 |                                -50.0394 |                                -0.0002 |                                 0.0036 |                                -0.0315 |                                -55.3291 |                                -0.0002 |                                 0.0026 |                                -0.0208 |                                -59.7824 |                                -0.0002 |                                 0.0018 |                                -0.0126 |                                -63.5337 |                                -0.0002 |                                 0.0011 |                                -0.0062 |                                -66.6966 |                                -0.0002 |                                 0.0004 |                                -0.0012 |                                -69.3662 |                                -0.0002 |                                -0.0001 |                                 0.0027 |                                -71.6209 |                                -0.0002 |                                -0.0006 |                                 0.0058 |                                -73.5259 |                                -0.0002 |                                -0.0011 |                                 0.0082 |                                -75.1353 |                                -0.0002 |                                -0.0015 |                                 0.01   |                                -76.4939 |                                -0.0002 |                                -0.0018 |                                 0.0115 |                                -77.639  |                                -0.0001 |                                -0.0022 |                                 0.0126 |                                -78.6019 |                                -0.0001 |                                -0.0024 |                                 0.0135 |                                -79.4086 |                                -0.0001 |                                -0.0027 |                                 0.0142 |                                -80.0814 |                                -0.0001 |                                -0.0029 |                                 0.0146 |                                -80.6387 |                                -0.0001 |                                -0.0031 |                                 0.015  |                                -81.0964 |                                -0.0001 |                                -0.0033 |                                 0.0152 |                                -81.468  |                                -0.0001 |                                -0.0035 |                                 0.0154 |                                -81.7649 |                                -0.0001 |                                -0.0036 |                                 0.0154 |                                -81.997  |
        |            255 |                                -0      |                                 0.0019 |                                -0.2109 |                                  4.0061 |                                -0.0003 |                                 0.0115 |                                -0.655  |                                 18.2481 |                                -0.0004 |                                 0.0152 |                                -0.5732 |                                 13.2127 |                                -0.0004 |                                 0.0144 |                                -0.4056 |                                 -0.3183 |                                -0.0004 |                                 0.0121 |                                -0.2723 |                                -14.2005 |                                -0.0004 |                                 0.0097 |                                -0.1794 |                                -26.0918 |                                -0.0004 |                                 0.0074 |                                -0.1163 |                                -35.7065 |                                -0.0003 |                                 0.0054 |                                -0.0733 |                                -43.3256 |                                -0.0003 |                                 0.0037 |                                -0.0437 |                                -49.3266 |                                -0.0003 |                                 0.0023 |                                -0.023  |                                -54.0501 |                                -0.0003 |                                 0.0011 |                                -0.0085 |                                -57.7725 |                                -0.0003 |                                 0.0001 |                                 0.0018 |                                -60.7099 |                                -0.0002 |                                -0.0008 |                                 0.0092 |                                -63.0291 |                                -0.0002 |                                -0.0015 |                                 0.0144 |                                -64.8588 |                                -0.0002 |                                -0.0021 |                                 0.0182 |                                -66.2983 |                                -0.0002 |                                -0.0026 |                                 0.0208 |                                -67.4251 |                                -0.0002 |                                -0.0031 |                                 0.0226 |                                -68.2996 |                                -0.0002 |                                -0.0035 |                                 0.0238 |                                -68.9696 |                                -0.0002 |                                -0.0038 |                                 0.0246 |                                -69.473  |                                -0.0002 |                                -0.0041 |                                 0.025  |                                -69.8401 |                                -0.0002 |                                -0.0044 |                                 0.0252 |                                -70.0953 |                                -0.0001 |                                -0.0046 |                                 0.0252 |                                -70.2583 |                                -0.0001 |                                -0.0048 |                                 0.0251 |                                -70.3454 |                                -0.0001 |                                -0.005  |                                 0.0248 |                                -70.3695 |                                -0.0001 |                                -0.0051 |                                 0.0245 |                                -70.3417 |                                -0.0001 |                                -0.0052 |                                 0.0241 |                                -70.2708 |                                -0.0001 |                                -0.0053 |                                 0.0237 |                                -70.1642 |                                -0.0001 |                                -0.0054 |                                 0.0233 |                                -70.0283 |                                -0.0001 |                                -0.0055 |                                 0.0228 |                                -69.8682 |
        |            260 |                                -0.0002 |                                 0.008  |                                -0.9052 |                                 10.3989 |                                -0.0006 |                                 0.0185 |                                -1.0466 |                                 11.7619 |                                -0.0006 |                                 0.0166 |                                -0.6246 |                                 -4.8617 |                                -0.0006 |                                 0.0123 |                                -0.3431 |                                -20.4075 |                                -0.0005 |                                 0.0082 |                                -0.1818 |                                -31.9628 |                                -0.0005 |                                 0.005  |                                -0.0893 |                                -40.1561 |                                -0.0004 |                                 0.0024 |                                -0.035  |                                -45.9203 |                                -0.0004 |                                 0.0004 |                                -0.0023 |                                -49.9805 |                                -0.0003 |                                -0.0011 |                                 0.0176 |                                -52.8434 |                                -0.0003 |                                -0.0023 |                                 0.0298 |                                -54.8558 |                                -0.0003 |                                -0.0033 |                                 0.0371 |                                -56.2562 |                                -0.0003 |                                -0.0041 |                                 0.0414 |                                -57.2105 |                                -0.0002 |                                -0.0047 |                                 0.0437 |                                -57.8362 |                                -0.0002 |                                -0.0052 |                                 0.0447 |                                -58.2172 |                                -0.0002 |                                -0.0056 |                                 0.0448 |                                -58.4148 |                                -0.0002 |                                -0.006  |                                 0.0444 |                                -58.4741 |                                -0.0002 |                                -0.0062 |                                 0.0437 |                                -58.4287 |                                -0.0002 |                                -0.0065 |                                 0.0427 |                                -58.304  |                                -0.0002 |                                -0.0067 |                                 0.0415 |                                -58.1193 |                                -0.0002 |                                -0.0068 |                                 0.0403 |                                -57.8893 |                                -0.0001 |                                -0.0069 |                                 0.0391 |                                -57.6255 |                                -0.0001 |                                -0.007  |                                 0.0378 |                                -57.3368 |                                -0.0001 |                                -0.0071 |                                 0.0366 |                                -57.0301 |                                -0.0001 |                                -0.0072 |                                 0.0354 |                                -56.7108 |                                -0.0001 |                                -0.0072 |                                 0.0342 |                                -56.3833 |                                -0.0001 |                                -0.0073 |                                 0.033  |                                -56.0509 |                                -0.0001 |                                -0.0073 |                                 0.0319 |                                -55.7163 |                                -0.0001 |                                -0.0073 |                                 0.0309 |                                -55.3816 |                                -0.0001 |                                -0.0073 |                                 0.0298 |                                -55.0486 |
        |            265 |                                -0.0008 |                                 0.0208 |                                -2.3611 |                                 12.3941 |                                -0.001  |                                 0.0202 |                                -1.1439 |                                 -6.1305 |                                -0.0009 |                                 0.0116 |                                -0.4337 |                                -23.5888 |                                -0.0007 |                                 0.0051 |                                -0.1389 |                                -33.9314 |                                -0.0006 |                                 0.0007 |                                -0.0104 |                                -39.8486 |                                -0.0005 |                                -0.0023 |                                 0.0481 |                                -43.2328 |                                -0.0004 |                                -0.0043 |                                 0.0747 |                                -45.1384 |                                -0.0004 |                                -0.0058 |                                 0.086  |                                -46.1536 |                                -0.0003 |                                -0.0068 |                                 0.0896 |                                -46.6172 |                                -0.0003 |                                -0.0076 |                                 0.0891 |                                -46.732  |                                -0.0003 |                                -0.0081 |                                 0.0867 |                                -46.6232 |                                -0.0002 |                                -0.0085 |                                 0.0832 |                                -46.3706 |                                -0.0002 |                                -0.0088 |                                 0.0794 |                                -46.0259 |                                -0.0002 |                                -0.009  |                                 0.0754 |                                -45.6236 |                                -0.0002 |                                -0.0092 |                                 0.0715 |                                -45.1868 |                                -0.0002 |                                -0.0093 |                                 0.0677 |                                -44.7314 |                                -0.0002 |                                -0.0093 |                                 0.0642 |                                -44.2681 |                                -0.0002 |                                -0.0094 |                                 0.0608 |                                -43.8043 |                                -0.0001 |                                -0.0094 |                                 0.0577 |                                -43.3453 |                                -0.0001 |                                -0.0094 |                                 0.0548 |                                -42.8944 |                                -0.0001 |                                -0.0094 |                                 0.0521 |                                -42.4539 |                                -0.0001 |                                -0.0093 |                                 0.0496 |                                -42.0255 |                                -0.0001 |                                -0.0093 |                                 0.0473 |                                -41.61   |                                -0.0001 |                                -0.0093 |                                 0.0451 |                                -41.208  |                                -0.0001 |                                -0.0092 |                                 0.0431 |                                -40.8196 |                                -0.0001 |                                -0.0092 |                                 0.0412 |                                -40.4448 |                                -0.0001 |                                -0.0091 |                                 0.0394 |                                -40.0836 |                                -0.0001 |                                -0.0091 |                                 0.0378 |                                -39.7356 |                                -0.0001 |                                -0.009  |                                 0.0362 |                                -39.4006 |
        |            270 |                                -0.0018 |                                 0.0296 |                                -3.3565 |                                 -1.1372 |                                -0.0014 |                                 0.0102 |                                -0.568  |                                -22.8451 |                                -0.001  |                                -0.0007 |                                 0.0379 |                                -31.161  |                                -0.0007 |                                -0.0062 |                                 0.1856 |                                -34.1546 |                                -0.0006 |                                -0.0092 |                                 0.2155 |                                -35.0519 |                                -0.0005 |                                -0.0108 |                                 0.2108 |                                -35.0499 |                                -0.0004 |                                -0.0118 |                                 0.1959 |                                -34.6438 |                                -0.0003 |                                -0.0123 |                                 0.1789 |                                -34.058  |                                -0.0003 |                                -0.0126 |                                 0.1625 |                                -33.4006 |                                -0.0003 |                                -0.0127 |                                 0.1476 |                                -32.7261 |                                -0.0002 |                                -0.0127 |                                 0.1344 |                                -32.0618 |                                -0.0002 |                                -0.0127 |                                 0.1228 |                                -31.4217 |                                -0.0002 |                                -0.0126 |                                 0.1126 |                                -30.8123 |                                -0.0002 |                                -0.0125 |                                 0.1036 |                                -30.2359 |                                -0.0002 |                                -0.0124 |                                 0.0957 |                                -29.6929 |                                -0.0002 |                                -0.0122 |                                 0.0887 |                                -29.1825 |                                -0.0001 |                                -0.0121 |                                 0.0825 |                                -28.7031 |                                -0.0001 |                                -0.0119 |                                 0.077  |                                -28.2529 |                                -0.0001 |                                -0.0118 |                                 0.072  |                                -27.8301 |                                -0.0001 |                                -0.0116 |                                 0.0675 |                                -27.4329 |                                -0.0001 |                                -0.0115 |                                 0.0635 |                                -27.0595 |                                -0.0001 |                                -0.0113 |                                 0.0598 |                                -26.7082 |                                -0.0001 |                                -0.0112 |                                 0.0565 |                                -26.3775 |                                -0.0001 |                                -0.0111 |                                 0.0535 |                                -26.0659 |                                -0.0001 |                                -0.0109 |                                 0.0507 |                                -25.7721 |                                -0.0001 |                                -0.0108 |                                 0.0482 |                                -25.4949 |                                -0.0001 |                                -0.0107 |                                 0.0458 |                                -25.2331 |                                -0.0001 |                                -0.0105 |                                 0.0437 |                                -24.9857 |                                -0.0001 |                                -0.0104 |                                 0.0417 |                                -24.7519 |
        |            275 |                                -0.0028 |                                 0.0104 |                                -1.1552 |                                -17.2898 |                                -0.0014 |                                -0.0119 |                                 0.6953 |                                -23.4003 |                                -0.0009 |                                -0.0175 |                                 0.6759 |                                -23.3625 |                                -0.0006 |                                -0.0191 |                                 0.5505 |                                -22.3833 |                                -0.0005 |                                -0.0193 |                                 0.4456 |                                -21.3102 |                                -0.0004 |                                -0.0191 |                                 0.3664 |                                -20.3192 |                                -0.0003 |                                -0.0186 |                                 0.3069 |                                -19.4414 |                                -0.0003 |                                -0.0181 |                                 0.2613 |                                -18.6713 |                                -0.0002 |                                -0.0176 |                                 0.2257 |                                -17.9957 |                                -0.0002 |                                -0.0171 |                                 0.1974 |                                -17.4007 |                                -0.0002 |                                -0.0166 |                                 0.1744 |                                -16.8739 |                                -0.0002 |                                -0.0162 |                                 0.1556 |                                -16.4053 |                                -0.0002 |                                -0.0158 |                                 0.1399 |                                -15.9864 |                                -0.0002 |                                -0.0154 |                                 0.1266 |                                -15.6101 |                                -0.0001 |                                -0.015  |                                 0.1153 |                                -15.2707 |                                -0.0001 |                                -0.0146 |                                 0.1056 |                                -14.9635 |                                -0.0001 |                                -0.0143 |                                 0.0971 |                                -14.6846 |                                -0.0001 |                                -0.014  |                                 0.0898 |                                -14.4305 |                                -0.0001 |                                -0.0137 |                                 0.0833 |                                -14.1986 |                                -0.0001 |                                -0.0134 |                                 0.0776 |                                -13.9863 |                                -0.0001 |                                -0.0132 |                                 0.0724 |                                -13.7917 |                                -0.0001 |                                -0.0129 |                                 0.0679 |                                -13.6129 |                                -0.0001 |                                -0.0127 |                                 0.0638 |                                -13.4486 |                                -0.0001 |                                -0.0125 |                                 0.06   |                                -13.2972 |                                -0.0001 |                                -0.0123 |                                 0.0567 |                                -13.1578 |                                -0.0001 |                                -0.0121 |                                 0.0536 |                                -13.0292 |                                -0.0001 |                                -0.0119 |                                 0.0508 |                                -12.9106 |                                -0.0001 |                                -0.0117 |                                 0.0483 |                                -12.8012 |                                -0.0001 |                                -0.0115 |                                 0.0459 |                                -12.7002 |
        |            280 |                                -0.0024 |                                -0.0372 |                                 4.2755 |                                -11.6654 |                                -0.001  |                                -0.0362 |                                 2.0747 |                                 -9.6792 |                                -0.0006 |                                -0.0325 |                                 1.2421 |                                 -8.3992 |                                -0.0004 |                                -0.0295 |                                 0.8448 |                                 -7.5475 |                                -0.0003 |                                -0.0271 |                                 0.6212 |                                 -6.9397 |                                -0.0003 |                                -0.0252 |                                 0.4812 |                                 -6.4835 |                                -0.0002 |                                -0.0236 |                                 0.3868 |                                 -6.1285 |                                -0.0002 |                                -0.0223 |                                 0.3197 |                                 -5.8449 |                                -0.0002 |                                -0.0212 |                                 0.2699 |                                 -5.6138 |                                -0.0002 |                                -0.0202 |                                 0.2319 |                                 -5.4228 |                                -0.0001 |                                -0.0194 |                                 0.202  |                                 -5.263  |                                -0.0001 |                                -0.0186 |                                 0.178  |                                 -5.1284 |                                -0.0001 |                                -0.0179 |                                 0.1584 |                                 -5.0142 |                                -0.0001 |                                -0.0173 |                                 0.1422 |                                 -4.9169 |                                -0.0001 |                                -0.0168 |                                 0.1286 |                                 -4.834  |                                -0.0001 |                                -0.0163 |                                 0.117  |                                 -4.7632 |                                -0.0001 |                                -0.0158 |                                 0.107  |                                 -4.7028 |                                -0.0001 |                                -0.0154 |                                 0.0984 |                                 -4.6515 |                                -0.0001 |                                -0.015  |                                 0.0909 |                                 -4.6082 |                                -0.0001 |                                -0.0147 |                                 0.0843 |                                 -4.5719 |                                -0.0001 |                                -0.0143 |                                 0.0785 |                                 -4.5419 |                                -0.0001 |                                -0.014  |                                 0.0733 |                                 -4.5174 |                                -0.0001 |                                -0.0137 |                                 0.0686 |                                 -4.498  |                                -0.0001 |                                -0.0135 |                                 0.0645 |                                 -4.4832 |                                -0.0001 |                                -0.0132 |                                 0.0607 |                                 -4.4725 |                                -0.0001 |                                -0.013  |                                 0.0573 |                                 -4.4656 |                                -0.0001 |                                -0.0127 |                                 0.0542 |                                 -4.4621 |                                -0.0001 |                                -0.0125 |                                 0.0514 |                                 -4.4619 |                                -0.0001 |                                -0.0123 |                                 0.0488 |                                 -4.4646 |
        |            285 |                                -0.0004 |                                -0.068  |                                 7.7676 |                                 -0.1059 |                                -0.0002 |                                -0.0482 |                                 2.7508 |                                 -0.094  |                                -0.0002 |                                -0.0394 |                                 1.4983 |                                 -0.0977 |                                -0.0001 |                                -0.0341 |                                 0.9736 |                                 -0.1072 |                                -0.0001 |                                -0.0305 |                                 0.6969 |                                 -0.1202 |                                -0.0001 |                                -0.0279 |                                 0.5303 |                                 -0.1357 |                                -0.0001 |                                -0.0258 |                                 0.4209 |                                 -0.153  |                                -0.0001 |                                -0.0241 |                                 0.3446 |                                 -0.172  |                                -0.0001 |                                -0.0228 |                                 0.2889 |                                 -0.1924 |                                -0.0001 |                                -0.0216 |                                 0.2467 |                                 -0.2141 |                                -0.0001 |                                -0.0206 |                                 0.2139 |                                 -0.2369 |                                -0.0001 |                                -0.0197 |                                 0.1878 |                                 -0.2609 |                                -0.0001 |                                -0.0189 |                                 0.1666 |                                 -0.2858 |                                -0.0001 |                                -0.0182 |                                 0.1491 |                                 -0.3118 |                                -0.0001 |                                -0.0176 |                                 0.1344 |                                 -0.3386 |                                -0.0001 |                                -0.0171 |                                 0.1221 |                                 -0.3663 |                                -0.0001 |                                -0.0166 |                                 0.1115 |                                 -0.3949 |                                -0.0001 |                                -0.0161 |                                 0.1023 |                                 -0.4242 |                                -0.0001 |                                -0.0157 |                                 0.0944 |                                 -0.4544 |                                -0.0001 |                                -0.0153 |                                 0.0874 |                                 -0.4853 |                                -0.0001 |                                -0.0149 |                                 0.0813 |                                 -0.517  |                                -0.0001 |                                -0.0146 |                                 0.0758 |                                 -0.5493 |                                -0.0001 |                                -0.0142 |                                 0.0709 |                                 -0.5824 |                                -0.0001 |                                -0.0139 |                                 0.0665 |                                 -0.6161 |                                -0      |                                -0.0137 |                                 0.0626 |                                 -0.6505 |                                -0      |                                -0.0134 |                                 0.059  |                                 -0.6856 |                                -0      |                                -0.0132 |                                 0.0558 |                                 -0.7213 |                                -0      |                                -0.0129 |                                 0.0529 |                                 -0.7576 |                                -0      |                                -0.0127 |                                 0.0502 |                                 -0.7945 |
        |            290 |                                 0.0018 |                                -0.0469 |                                 5.3291 |                                 -8.8672 |                                 0.0006 |                                -0.0407 |                                 2.3116 |                                 -6.8533 |                                 0.0003 |                                -0.0354 |                                 1.3414 |                                 -5.7179 |                                 0.0002 |                                -0.0316 |                                 0.8991 |                                 -4.9794 |                                 0.0001 |                                -0.0288 |                                 0.6554 |                                 -4.4521 |                                 0.0001 |                                -0.0266 |                                 0.5049 |                                 -4.0524 |                                 0      |                                -0.0249 |                                 0.4042 |                                 -3.7368 |                                 0      |                                -0.0234 |                                 0.3331 |                                 -3.4803 |                                 0      |                                -0.0222 |                                 0.2806 |                                 -3.2672 |                                -0      |                                -0.0211 |                                 0.2406 |                                 -3.0872 |                                -0      |                                -0.0202 |                                 0.2093 |                                 -2.9333 |                                -0      |                                -0.0194 |                                 0.1842 |                                 -2.8003 |                                -0      |                                -0.0187 |                                 0.1638 |                                 -2.6845 |                                -0      |                                -0.018  |                                 0.1469 |                                 -2.5831 |                                -0      |                                -0.0175 |                                 0.1327 |                                 -2.4937 |                                -0      |                                -0.0169 |                                 0.1206 |                                 -2.4148 |                                -0      |                                -0.0165 |                                 0.1103 |                                 -2.3449 |                                -0      |                                -0.016  |                                 0.1014 |                                 -2.2829 |                                -0      |                                -0.0156 |                                 0.0936 |                                 -2.2279 |                                -0      |                                -0.0152 |                                 0.0868 |                                 -2.1791 |                                -0      |                                -0.0149 |                                 0.0807 |                                 -2.1359 |                                -0      |                                -0.0145 |                                 0.0754 |                                 -2.0976 |                                -0      |                                -0.0142 |                                 0.0706 |                                 -2.064  |                                -0      |                                -0.0139 |                                 0.0663 |                                 -2.0345 |                                -0      |                                -0.0137 |                                 0.0624 |                                 -2.0087 |                                -0      |                                -0.0134 |                                 0.0589 |                                 -1.9865 |                                -0      |                                -0.0132 |                                 0.0557 |                                 -1.9676 |                                -0      |                                -0.0129 |                                 0.0527 |                                 -1.9516 |                                -0      |                                -0.0127 |                                 0.0501 |                                 -1.9385 |
        |            295 |                                 0.0027 |                                 0.0005 |                                -0.0932 |                                -18.8061 |                                 0.0012 |                                -0.019  |                                 1.0689 |                                -21.4283 |                                 0.0006 |                                -0.0226 |                                 0.8532 |                                -20.22   |                                 0.0004 |                                -0.0231 |                                 0.6528 |                                -18.7408 |                                 0.0003 |                                -0.0226 |                                 0.512  |                                -17.412  |                                 0.0002 |                                -0.0219 |                                 0.4132 |                                -16.2711 |                                 0.0002 |                                -0.0211 |                                 0.3417 |                                -15.2945 |                                 0.0001 |                                -0.0203 |                                 0.2883 |                                -14.4524 |                                 0.0001 |                                -0.0196 |                                 0.2473 |                                -13.7195 |                                 0.0001 |                                -0.019  |                                 0.2151 |                                -13.0756 |                                 0.0001 |                                -0.0184 |                                 0.1893 |                                -12.5051 |                                 0      |                                -0.0178 |                                 0.1682 |                                -11.9957 |                                 0      |                                -0.0173 |                                 0.1508 |                                -11.5377 |                                 0      |                                -0.0168 |                                 0.1361 |                                -11.1235 |                                 0      |                                -0.0163 |                                 0.1237 |                                -10.7469 |                                 0      |                                -0.0159 |                                 0.1131 |                                -10.4028 |                                 0      |                                -0.0155 |                                 0.1039 |                                -10.0872 |                                 0      |                                -0.0152 |                                 0.0959 |                                 -9.7967 |                                 0      |                                -0.0149 |                                 0.0888 |                                 -9.5283 |                                -0      |                                -0.0145 |                                 0.0826 |                                 -9.2797 |                                -0      |                                -0.0142 |                                 0.0771 |                                 -9.0487 |                                -0      |                                -0.014  |                                 0.0721 |                                 -8.8337 |                                -0      |                                -0.0137 |                                 0.0677 |                                 -8.6331 |                                -0      |                                -0.0134 |                                 0.0637 |                                 -8.4456 |                                -0      |                                -0.0132 |                                 0.0601 |                                 -8.27   |                                -0      |                                -0.013  |                                 0.0568 |                                 -8.1053 |                                -0      |                                -0.0128 |                                 0.0538 |                                 -7.9508 |                                -0      |                                -0.0126 |                                 0.0511 |                                 -7.8054 |                                -0      |                                -0.0124 |                                 0.0486 |                                 -7.6687 |
        |            300 |                                 0.0022 |                                 0.0289 |                                -3.3231 |                                 -7.8443 |                                 0.0013 |                                 0.0042 |                                -0.2538 |                                -26.4756 |                                 0.0008 |                                -0.0065 |                                 0.2387 |                                -31.5829 |                                 0.0006 |                                -0.0113 |                                 0.3162 |                                -32.6003 |                                 0.0004 |                                -0.0136 |                                 0.306  |                                -32.246  |                                 0.0003 |                                -0.0148 |                                 0.2769 |                                -31.4026 |                                 0.0002 |                                -0.0153 |                                 0.2463 |                                -30.3989 |                                 0.0002 |                                -0.0155 |                                 0.2186 |                                -29.3679 |                                 0.0002 |                                -0.0155 |                                 0.1947 |                                -28.3655 |                                 0.0001 |                                -0.0154 |                                 0.1742 |                                -27.4139 |                                 0.0001 |                                -0.0153 |                                 0.1568 |                                -26.5206 |                                 0.0001 |                                -0.0151 |                                 0.142  |                                -25.6863 |                                 0.0001 |                                -0.0149 |                                 0.1292 |                                -24.9085 |                                 0.0001 |                                -0.0146 |                                 0.1182 |                                -24.1834 |                                 0.0001 |                                -0.0144 |                                 0.1086 |                                -23.5069 |                                 0      |                                -0.0142 |                                 0.1002 |                                -22.8749 |                                 0      |                                -0.014  |                                 0.0928 |                                -22.2836 |                                 0      |                                -0.0137 |                                 0.0863 |                                -21.7293 |                                 0      |                                -0.0135 |                                 0.0805 |                                -21.2087 |                                 0      |                                -0.0133 |                                 0.0753 |                                -20.7191 |                                 0      |                                -0.0131 |                                 0.0706 |                                -20.2578 |                                 0      |                                -0.0129 |                                 0.0664 |                                -19.8223 |                                 0      |                                -0.0127 |                                 0.0626 |                                -19.4107 |                                 0      |                                -0.0125 |                                 0.0591 |                                -19.0211 |                                 0      |                                -0.0124 |                                 0.056  |                                -18.6518 |                                 0      |                                -0.0122 |                                 0.0531 |                                -18.3012 |                                 0      |                                -0.012  |                                 0.0504 |                                -17.968  |                                 0      |                                -0.0119 |                                 0.048  |                                -17.6509 |                                 0      |                                -0.0117 |                                 0.0458 |                                -17.349  |
        |            305 |                                 0.0012 |                                 0.0282 |                                -3.2278 |                                  9.1554 |                                 0.0012 |                                 0.0187 |                                -1.083  |                                -15.6876 |                                 0.0009 |                                 0.0074 |                                -0.2912 |                                -30.891  |                                 0.0006 |                                 0.0003 |                                -0.0165 |                                -38.2669 |                                 0.0005 |                                -0.004  |                                 0.0859 |                                -41.7383 |                                 0.0004 |                                -0.0068 |                                 0.1238 |                                -43.2294 |                                 0.0003 |                                -0.0085 |                                 0.1353 |                                -43.6696 |                                 0.0003 |                                -0.0097 |                                 0.1353 |                                -43.5322 |                                 0.0002 |                                -0.0105 |                                 0.1304 |                                -43.0719 |                                 0.0002 |                                -0.011  |                                 0.1235 |                                -42.4316 |                                 0.0002 |                                -0.0114 |                                 0.116  |                                -41.6947 |                                 0.0001 |                                -0.0116 |                                 0.1085 |                                -40.9108 |                                 0.0001 |                                -0.0118 |                                 0.1015 |                                -40.1103 |                                 0.0001 |                                -0.0118 |                                 0.0949 |                                -39.3114 |                                 0.0001 |                                -0.0119 |                                 0.0888 |                                -38.5255 |                                 0.0001 |                                -0.0118 |                                 0.0832 |                                -37.7593 |                                 0.0001 |                                -0.0118 |                                 0.0782 |                                -37.0167 |                                 0.0001 |                                -0.0118 |                                 0.0735 |                                -36.2998 |                                 0.0001 |                                -0.0117 |                                 0.0693 |                                -35.6092 |                                 0      |                                -0.0116 |                                 0.0654 |                                -34.9451 |                                 0      |                                -0.0115 |                                 0.0618 |                                -34.3071 |                                 0      |                                -0.0114 |                                 0.0586 |                                -33.6945 |                                 0      |                                -0.0113 |                                 0.0556 |                                -33.1064 |                                 0      |                                -0.0112 |                                 0.0528 |                                -32.5417 |                                 0      |                                -0.0111 |                                 0.0503 |                                -31.9994 |                                 0      |                                -0.011  |                                 0.0479 |                                -31.4785 |                                 0      |                                -0.0109 |                                 0.0457 |                                -30.9779 |                                 0      |                                -0.0108 |                                 0.0437 |                                -30.4967 |                                 0      |                                -0.0107 |                                 0.0418 |                                -30.0338 |
        |            310 |                                 0.0005 |                                 0.0159 |                                -1.8152 |                                 14.1155 |                                 0.0008 |                                 0.0224 |                                -1.2862 |                                  2.4964 |                                 0.0008 |                                 0.0159 |                                -0.6122 |                                -18.0399 |                                 0.0006 |                                 0.0093 |                                -0.2734 |                                -32.44   |                                 0.0005 |                                 0.0044 |                                -0.1059 |                                -41.551  |                                 0.0004 |                                 0.0008 |                                -0.0198 |                                -47.215  |                                 0.0004 |                                -0.0018 |                                 0.0257 |                                -50.6998 |                                 0.0003 |                                -0.0038 |                                 0.05   |                                -52.7901 |                                 0.0003 |                                -0.0052 |                                 0.0627 |                                -53.9704 |                                 0.0002 |                                -0.0063 |                                 0.0688 |                                -54.5457 |                                 0.0002 |                                -0.0071 |                                 0.0712 |                                -54.7126 |                                 0.0002 |                                -0.0077 |                                 0.0713 |                                -54.6007 |                                 0.0001 |                                -0.0082 |                                 0.0702 |                                -54.2976 |                                 0.0001 |                                -0.0086 |                                 0.0683 |                                -53.8632 |                                 0.0001 |                                -0.0089 |                                 0.066  |                                -53.3394 |                                 0.0001 |                                -0.0091 |                                 0.0635 |                                -52.7558 |                                 0.0001 |                                -0.0093 |                                 0.061  |                                -52.1335 |                                 0.0001 |                                -0.0094 |                                 0.0584 |                                -51.4877 |                                 0.0001 |                                -0.0095 |                                 0.056  |                                -50.8294 |                                 0.0001 |                                -0.0096 |                                 0.0536 |                                -50.1666 |                                 0.0001 |                                -0.0096 |                                 0.0513 |                                -49.5052 |                                 0.0001 |                                -0.0097 |                                 0.0492 |                                -48.8493 |                                 0      |                                -0.0097 |                                 0.0471 |                                -48.2021 |                                 0      |                                -0.0097 |                                 0.0452 |                                -47.5656 |                                 0      |                                -0.0097 |                                 0.0433 |                                -46.9414 |                                 0      |                                -0.0096 |                                 0.0416 |                                -46.3305 |                                 0      |                                -0.0096 |                                 0.04   |                                -45.7335 |                                 0      |                                -0.0096 |                                 0.0384 |                                -45.1509 |                                 0      |                                -0.0096 |                                 0.037  |                                -44.5828 |
        |            315 |                                 0.0002 |                                 0.0062 |                                -0.7113 |                                  9.4638 |                                 0.0005 |                                 0.0186 |                                -1.0681 |                                 16.2432 |                                 0.0006 |                                 0.0185 |                                -0.7095 |                                  0.1324 |                                 0.0005 |                                 0.0145 |                                -0.4214 |                                -17.5553 |                                 0.0005 |                                 0.0104 |                                -0.2429 |                                -31.5763 |                                 0.0004 |                                 0.0068 |                                -0.135  |                                -41.8849 |                                 0.0004 |                                 0.004  |                                -0.0688 |                                -49.3073 |                                 0.0003 |                                 0.0017 |                                -0.0274 |                                -54.6171 |                                 0.0003 |                                -0.0002 |                                -0.0011 |                                -58.3985 |                                 0.0002 |                                -0.0017 |                                 0.0158 |                                -61.0704 |                                 0.0002 |                                -0.0028 |                                 0.0267 |                                -62.9296 |                                 0.0002 |                                -0.0038 |                                 0.0336 |                                -64.1878 |                                 0.0002 |                                -0.0046 |                                 0.038  |                                -64.9971 |                                 0.0002 |                                -0.0052 |                                 0.0406 |                                -65.4684 |                                 0.0001 |                                -0.0058 |                                 0.042  |                                -65.6834 |                                 0.0001 |                                -0.0062 |                                 0.0425 |                                -65.7029 |                                 0.0001 |                                -0.0066 |                                 0.0425 |                                -65.5727 |                                 0.0001 |                                -0.0069 |                                 0.0421 |                                -65.3279 |                                 0.0001 |                                -0.0071 |                                 0.0415 |                                -64.9952 |                                 0.0001 |                                -0.0073 |                                 0.0406 |                                -64.5952 |                                 0.0001 |                                -0.0075 |                                 0.0397 |                                -64.1442 |                                 0.0001 |                                -0.0077 |                                 0.0387 |                                -63.6548 |                                 0.0001 |                                -0.0078 |                                 0.0376 |                                -63.1369 |                                 0.0001 |                                -0.0079 |                                 0.0366 |                                -62.5984 |                                 0.0001 |                                -0.008  |                                 0.0355 |                                -62.0457 |                                 0.0001 |                                -0.008  |                                 0.0344 |                                -61.4836 |                                 0      |                                -0.0081 |                                 0.0334 |                                -60.9162 |                                 0      |                                -0.0081 |                                 0.0324 |                                -60.3467 |                                 0      |                                -0.0082 |                                 0.0314 |                                -59.7775 |
        |            320 |                                 0      |                                 0.0018 |                                -0.208  |                                  4.0675 |                                 0.0003 |                                 0.0124 |                                -0.7105 |                                 20.4355 |                                 0.0004 |                                 0.0168 |                                -0.6437 |                                 15.6617 |                                 0.0004 |                                 0.0161 |                                -0.4649 |                                  0.6304 |                                 0.0004 |                                 0.0137 |                                -0.317  |                                -15.1561 |                                 0.0004 |                                 0.0109 |                                -0.212  |                                -28.7855 |                                 0.0003 |                                 0.0083 |                                -0.1397 |                                -39.819  |                                 0.0003 |                                 0.006  |                                -0.09   |                                -48.5361 |                                 0.0003 |                                 0.0041 |                                -0.0555 |                                -55.3575 |                                 0.0002 |                                 0.0025 |                                -0.0312 |                                -60.674  |                                 0.0002 |                                 0.0011 |                                -0.014  |                                -64.8075 |                                 0.0002 |                                -0.0001 |                                -0.0016 |                                -68.0117 |                                 0.0002 |                                -0.0011 |                                 0.0072 |                                -70.4835 |                                 0.0002 |                                -0.0019 |                                 0.0136 |                                -72.3756 |                                 0.0002 |                                -0.0027 |                                 0.0182 |                                -73.8059 |                                 0.0001 |                                -0.0033 |                                 0.0215 |                                -74.8667 |                                 0.0001 |                                -0.0038 |                                 0.0239 |                                -75.6302 |                                 0.0001 |                                -0.0043 |                                 0.0255 |                                -76.1533 |                                 0.0001 |                                -0.0047 |                                 0.0265 |                                -76.4813 |                                 0.0001 |                                -0.005  |                                 0.0272 |                                -76.6505 |                                 0.0001 |                                -0.0053 |                                 0.0275 |                                -76.6898 |                                 0.0001 |                                -0.0056 |                                 0.0277 |                                -76.6229 |                                 0.0001 |                                -0.0058 |                                 0.0276 |                                -76.469  |                                 0.0001 |                                -0.006  |                                 0.0274 |                                -76.2437 |                                 0.0001 |                                -0.0062 |                                 0.0271 |                                -75.96   |                                 0.0001 |                                -0.0063 |                                 0.0268 |                                -75.6285 |                                 0.0001 |                                -0.0065 |                                 0.0263 |                                -75.258  |                                 0.0001 |                                -0.0066 |                                 0.0259 |                                -74.8559 |                                 0      |                                -0.0067 |                                 0.0254 |                                -74.4282 |
        |            325 |                                 0      |                                 0.0004 |                                -0.0471 |                                  1.2505 |                                 0.0001 |                                 0.007  |                                -0.3988 |                                 17.2785 |                                 0.0003 |                                 0.013  |                                -0.4987 |                                 24.002  |                                 0.0003 |                                 0.015  |                                -0.4314 |                                 16.3756 |                                 0.0003 |                                 0.0145 |                                -0.3349 |                                  2.9042 |                                 0.0003 |                                 0.013  |                                -0.2502 |                                -11.3946 |                                 0.0003 |                                 0.0111 |                                -0.1841 |                                -24.5224 |                                 0.0003 |                                 0.0092 |                                -0.1342 |                                -35.8892 |                                 0.0003 |                                 0.0074 |                                -0.097  |                                -45.4716 |                                 0.0002 |                                 0.0058 |                                -0.0692 |                                -53.447  |                                 0.0002 |                                 0.0044 |                                -0.0482 |                                -60.0435 |                                 0.0002 |                                 0.0031 |                                -0.0323 |                                -65.4824 |                                 0.0002 |                                 0.002  |                                -0.0202 |                                -69.9586 |                                 0.0002 |                                 0.0011 |                                -0.011  |                                -73.6369 |                                 0.0002 |                                 0.0002 |                                -0.0038 |                                -76.6538 |                                 0.0001 |                                -0.0005 |                                 0.0017 |                                -79.1215 |                                 0.0001 |                                -0.0012 |                                 0.006  |                                -81.1321 |                                 0.0001 |                                -0.0017 |                                 0.0094 |                                -82.7608 |                                 0.0001 |                                -0.0023 |                                 0.0119 |                                -84.0696 |                                 0.0001 |                                -0.0027 |                                 0.0139 |                                -85.1093 |                                 0.0001 |                                -0.0031 |                                 0.0155 |                                -85.9222 |                                 0.0001 |                                -0.0035 |                                 0.0166 |                                -86.5433 |                                 0.0001 |                                -0.0038 |                                 0.0175 |                                -87.0017 |                                 0.0001 |                                -0.0041 |                                 0.0181 |                                -87.3218 |                                 0.0001 |                                -0.0043 |                                 0.0186 |                                -87.5242 |                                 0.0001 |                                -0.0045 |                                 0.0189 |                                -87.626  |                                 0.0001 |                                -0.0047 |                                 0.0191 |                                -87.6419 |                                 0.0001 |                                -0.0049 |                                 0.0191 |                                -87.5844 |                                 0.0001 |                                -0.0051 |                                 0.0191 |                                -87.464  |
        |            330 |                                 0      |                                 0.0001 |                                -0.0085 |                                  0.2902 |                                 0.0001 |                                 0.0034 |                                -0.194  |                                 11.4716 |                                 0.0002 |                                 0.009  |                                -0.3423 |                                 24.9764 |                                 0.0002 |                                 0.0124 |                                -0.3557 |                                 26.2308 |                                 0.0003 |                                 0.0135 |                                -0.3116 |                                 18.3287 |                                 0.0003 |                                 0.0133 |                                -0.2554 |                                  6.3319 |                                 0.0003 |                                 0.0123 |                                -0.2034 |                                 -6.674  |                                 0.0003 |                                 0.011  |                                -0.1598 |                                -19.1842 |                                 0.0002 |                                 0.0096 |                                -0.1247 |                                -30.5722 |                                 0.0002 |                                 0.0082 |                                -0.0967 |                                -40.6494 |                                 0.0002 |                                 0.0069 |                                -0.0746 |                                -49.4305 |                                 0.0002 |                                 0.0057 |                                -0.0571 |                                -57.017  |                                 0.0002 |                                 0.0047 |                                -0.0432 |                                -63.5398 |                                 0.0002 |                                 0.0037 |                                -0.0321 |                                -69.1325 |                                 0.0002 |                                 0.0028 |                                -0.0232 |                                -73.9198 |                                 0.0002 |                                 0.002  |                                -0.016  |                                -78.0129 |                                 0.0001 |                                 0.0013 |                                -0.0102 |                                -81.5088 |                                 0.0001 |                                 0.0006 |                                -0.0055 |                                -84.4911 |                                 0.0001 |                                 0      |                                -0.0017 |                                -87.0314 |                                 0.0001 |                                -0.0005 |                                 0.0014 |                                -89.1907 |                                 0.0001 |                                -0.001  |                                 0.0039 |                                -91.021  |                                 0.0001 |                                -0.0014 |                                 0.006  |                                -92.5666 |                                 0.0001 |                                -0.0018 |                                 0.0077 |                                -93.8654 |                                 0.0001 |                                -0.0022 |                                 0.0091 |                                -94.9496 |                                 0.0001 |                                -0.0025 |                                 0.0102 |                                -95.8471 |                                 0.0001 |                                -0.0028 |                                 0.0111 |                                -96.5817 |                                 0.0001 |                                -0.003  |                                 0.0118 |                                -97.1739 |                                 0.0001 |                                -0.0033 |                                 0.0124 |                                -97.6415 |                                 0.0001 |                                -0.0035 |                                 0.0128 |                                -97.9997 |
        |            335 |                                 0      |                                 0      |                                -0.0012 |                                  0.0526 |                                 0      |                                 0.0015 |                                -0.0833 |                                  6.3454 |                                 0.0001 |                                 0.0056 |                                -0.2125 |                                 21.1017 |                                 0.0001 |                                 0.0093 |                                -0.2675 |                                 29.5581 |                                 0.0002 |                                 0.0115 |                                -0.2647 |                                 28.571  |                                 0.0002 |                                 0.0123 |                                -0.2369 |                                 21.1681 |                                 0.0002 |                                 0.0122 |                                -0.2023 |                                 10.513  |                                 0.0002 |                                 0.0116 |                                -0.1687 |                                 -1.3055 |                                 0.0002 |                                 0.0108 |                                -0.1391 |                                -13.1029 |                                 0.0002 |                                 0.0098 |                                -0.1138 |                                -24.2715 |                                 0.0002 |                                 0.0087 |                                -0.0928 |                                -34.5374 |                                 0.0002 |                                 0.0077 |                                -0.0753 |                                -43.8114 |                                 0.0002 |                                 0.0067 |                                -0.0609 |                                -52.1013 |                                 0.0002 |                                 0.0058 |                                -0.049  |                                -59.4634 |                                 0.0002 |                                 0.0049 |                                -0.0392 |                                -65.9749 |                                 0.0002 |                                 0.0041 |                                -0.0311 |                                -71.7192 |                                 0.0001 |                                 0.0034 |                                -0.0243 |                                -76.7783 |                                 0.0001 |                                 0.0027 |                                -0.0187 |                                -81.229  |                                 0.0001 |                                 0.0021 |                                -0.014  |                                -85.1412 |                                 0.0001 |                                 0.0015 |                                -0.01   |                                -88.5775 |                                 0.0001 |                                 0.001  |                                -0.0067 |                                -91.5935 |                                 0.0001 |                                 0.0005 |                                -0.0039 |                                -94.2381 |                                 0.0001 |                                 0      |                                -0.0015 |                                -96.5544 |                                 0.0001 |                                -0.0004 |                                 0.0005 |                                -98.5802 |                                 0.0001 |                                -0.0007 |                                 0.0022 |                               -100.349  |                                 0.0001 |                                -0.0011 |                                 0.0036 |                               -101.888  |                                 0.0001 |                                -0.0014 |                                 0.0048 |                               -103.226  |                                 0.0001 |                                -0.0017 |                                 0.0058 |                               -104.382  |                                 0.0001 |                                -0.002  |                                 0.0067 |                               -105.378  |
        |            340 |                                 0      |                                 0      |                                -0.0001 |                                  0.0076 |                                 0      |                                 0.0006 |                                -0.0319 |                                  3.0177 |                                 0      |                                 0.0032 |                                -0.1209 |                                 15.3892 |                                 0.0001 |                                 0.0065 |                                -0.1863 |                                 27.7438 |                                 0.0001 |                                 0.0091 |                                -0.209  |                                 33.0367 |                                 0.0002 |                                 0.0106 |                                -0.2043 |                                 31.2557 |                                 0.0002 |                                 0.0113 |                                -0.1866 |                                 24.5979 |                                 0.0002 |                                 0.0114 |                                -0.1645 |                                 15.1681 |                                 0.0002 |                                 0.011  |                                -0.1421 |                                  4.468  |                                 0.0002 |                                 0.0104 |                                -0.1214 |                                 -6.5486 |                                 0.0002 |                                 0.0097 |                                -0.103  |                                -17.3174 |                                 0.0002 |                                 0.0089 |                                -0.087  |                                -27.528  |                                 0.0002 |                                 0.0081 |                                -0.0733 |                                -37.0276 |                                 0.0002 |                                 0.0073 |                                -0.0616 |                                -45.7585 |                                 0.0002 |                                 0.0065 |                                -0.0516 |                                -53.7187 |                                 0.0001 |                                 0.0058 |                                -0.0431 |                                -60.9374 |                                 0.0001 |                                 0.0051 |                                -0.0359 |                                -67.46   |                                 0.0001 |                                 0.0044 |                                -0.0297 |                                -73.3392 |                                 0.0001 |                                 0.0038 |                                -0.0244 |                                -78.6296 |                                 0.0001 |                                 0.0032 |                                -0.0199 |                                -83.3846 |                                 0.0001 |                                 0.0027 |                                -0.016  |                                -87.6547 |                                 0.0001 |                                 0.0022 |                                -0.0127 |                                -91.4869 |                                 0.0001 |                                 0.0017 |                                -0.0098 |                                -94.9242 |                                 0.0001 |                                 0.0013 |                                -0.0074 |                                -98.0055 |                                 0.0001 |                                 0.0009 |                                -0.0052 |                               -100.766  |                                 0.0001 |                                 0.0005 |                                -0.0034 |                               -103.238  |                                 0.0001 |                                 0.0001 |                                -0.0018 |                               -105.449  |                                 0.0001 |                                -0.0002 |                                -0.0004 |                               -107.425  |                                 0.0001 |                                -0.0005 |                                 0.0008 |                               -109.188  |
        |            345 |                                 0      |                                 0      |                                -0      |                                  0.0009 |                                 0      |                                 0.0002 |                                -0.011  |                                  1.2588 |                                 0      |                                 0.0017 |                                -0.0636 |                                  9.9868 |                                 0.0001 |                                 0.0042 |                                -0.1213 |                                 22.9769 |                                 0.0001 |                                 0.0068 |                                -0.1553 |                                 32.6056 |                                 0.0001 |                                 0.0087 |                                -0.1662 |                                 36.1616 |                                 0.0001 |                                 0.0099 |                                -0.1624 |                                 34.2435 |                                 0.0001 |                                 0.0105 |                                -0.151  |                                 28.3978 |                                 0.0002 |                                 0.0106 |                                -0.1364 |                                 20.0949 |                                 0.0002 |                                 0.0104 |                                -0.1211 |                                 10.4575 |                                 0.0002 |                                 0.0101 |                                -0.1064 |                                  0.2642 |                                 0.0002 |                                 0.0096 |                                -0.0928 |                                 -9.9746 |                                 0.0002 |                                 0.009  |                                -0.0806 |                                -19.9405 |                                 0.0001 |                                 0.0083 |                                -0.0698 |                                -29.4457 |                                 0.0001 |                                 0.0077 |                                -0.0603 |                                -38.3889 |                                 0.0001 |                                 0.0071 |                                -0.052  |                                -46.7248 |                                 0.0001 |                                 0.0064 |                                -0.0448 |                                -54.4442 |                                 0.0001 |                                 0.0058 |                                -0.0385 |                                -61.5597 |                                 0.0001 |                                 0.0052 |                                -0.033  |                                -68.0969 |                                 0.0001 |                                 0.0047 |                                -0.0282 |                                -74.0886 |                                 0.0001 |                                 0.0042 |                                -0.024  |                                -79.571  |                                 0.0001 |                                 0.0037 |                                -0.0203 |                                -84.5812 |                                 0.0001 |                                 0.0032 |                                -0.0171 |                                -89.1556 |                                 0.0001 |                                 0.0027 |                                -0.0143 |                                -93.3294 |                                 0.0001 |                                 0.0023 |                                -0.0118 |                                -97.1354 |                                 0.0001 |                                 0.0019 |                                -0.0097 |                               -100.605  |                                 0.0001 |                                 0.0016 |                                -0.0077 |                               -103.765  |                                 0.0001 |                                 0.0012 |                                -0.0061 |                               -106.644  |                                 0.0001 |                                 0.0009 |                                -0.0046 |                               -109.264  |
        |            350 |                                 0      |                                 0      |                                -0      |                                  0.0001 |                                 0      |                                 0.0001 |                                -0.0035 |                                  0.4671 |                                 0      |                                 0.0008 |                                -0.0312 |                                  5.8736 |                                 0      |                                 0.0026 |                                -0.0745 |                                 17.2697 |                                 0.0001 |                                 0.0048 |                                -0.1094 |                                 28.8861 |                                 0.0001 |                                 0.0067 |                                -0.1286 |                                 36.5375 |                                 0.0001 |                                 0.0082 |                                -0.1346 |                                 39.185  |                                 0.0001 |                                 0.0092 |                                -0.1321 |                                 37.4407 |                                 0.0001 |                                 0.0097 |                                -0.1247 |                                 32.4054 |                                 0.0001 |                                 0.0099 |                                -0.1148 |                                 25.144  |                                 0.0001 |                                 0.0099 |                                -0.1042 |                                 16.518  |                                 0.0001 |                                 0.0097 |                                -0.0936 |                                  7.1686 |                                 0.0001 |                                 0.0093 |                                -0.0835 |                                 -2.4505 |                                 0.0001 |                                 0.0089 |                                -0.0741 |                                -12.0295 |                                 0.0001 |                                 0.0084 |                                -0.0656 |                                -21.3647 |                                 0.0001 |                                 0.0079 |                                -0.058  |                                -30.3278 |                                 0.0001 |                                 0.0074 |                                -0.0511 |                                -38.8436 |                                 0.0001 |                                 0.0069 |                                -0.045  |                                -46.8732 |                                 0.0001 |                                 0.0063 |                                -0.0396 |                                -54.4023 |                                 0.0001 |                                 0.0058 |                                -0.0347 |                                -61.4333 |                                 0.0001 |                                 0.0053 |                                -0.0304 |                                -67.9791 |                                 0.0001 |                                 0.0049 |                                -0.0266 |                                -74.0592 |                                 0.0001 |                                 0.0044 |                                -0.0233 |                                -79.6971 |                                 0.0001 |                                 0.004  |                                -0.0202 |                                -84.9181 |                                 0.0001 |                                 0.0036 |                                -0.0176 |                                -89.7482 |                                 0.0001 |                                 0.0032 |                                -0.0152 |                                -94.2134 |                                 0.0001 |                                 0.0028 |                                -0.013  |                                -98.3388 |                                 0.0001 |                                 0.0025 |                                -0.0111 |                               -102.148  |                                 0.0001 |                                 0.0021 |                                -0.0094 |                               -105.665  |
        """
        greeks = {}

        greeks["Speed"] = self.get_speed(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Zomma"] = self.get_zomma(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Color"] = self.get_color(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=False,
            rounding=rounding,
            standardize=standardize,
        )

        greeks["Ultima"] = self.get_ultima(
            start_date=start_date,
            strike_price_range=strike_price_range,
            strike_step_size=strike_step_size,
            expiration_time_range=expiration_time_range,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            show_input_info=show_input_info,
            rounding=rounding,
            standardize=standardize,
        )

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    def get_speed(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the speed of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The speed is
        the rate of change of the gamma with respect to the price of the underlying asset.

        The speed calculation is the theoretical value of the speed. The actual speed can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - Speed = —e^(—q * t) * ((N'(d1) / (S ** 2 * σ * sqrt(t)))) * ((d1 / (σ * sqrt(t))) + 1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration and N'(d1) is the standard normal probability density at d1.

        The Speed can be interpreted as follows:

        - If Speed is positive, the option's Gamma rises as the underlying price rises, so the position's convexity
        builds up on the way up.
        - If Speed is negative, the option's Gamma falls as the underlying price rises, which is the usual case just
        below the strike where Gamma is already close to its peak.

        Note that the speed of a call option and put option are equal to each other.

        Also known as: gamma rate of change.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the speed values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["F"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_speed(
            strike_price_range=0.25, strike_step_size=1
        ).loc["F", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |              9 |       0      |       0      |      -0.0002 |
        |             10 |       0      |      -0.0003 |      -0.0098 |
        |             11 |      -0.0026 |      -0.0366 |      -0.0832 |
        |             12 |      -0.3232 |      -0.3013 |      -0.1804 |
        |             13 |      -0.2804 |      -0.124  |      -0.061  |
        |             14 |       0.5578 |       0.3088 |       0.1259 |
        |             15 |       0.0474 |       0.136  |       0.1319 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        speed: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            speed[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            speed[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_speed(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        speed_df = helpers.create_greek_dataframe(
            greek_dictionary=speed,
            start_date=start_date,
        )

        speed_df = speed_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            speed_df = calculate_standardization(
                dataset=speed_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return speed_df

    def get_zomma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the zomma of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The zomma is
        the rate of change of the gamma with respect to volatility.

        The zomma calculation is the theoretical value of the zomma. The actual zomma can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Zomma = e^(—q * t) * (N'(d1) * (d1 * d2 — 1)) / (S * σ **2 * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration and N'(d1) is the standard normal probability density at d1. This is
        equivalently Gamma * (d1 * d2 — 1) / σ, and it is reported unscaled, per 1.00 of volatility.

        The Zomma can be interpreted as follows:

        - If Zomma is positive, the option's Gamma rises as implied volatility rises, which is typical for strikes
        well away from the money.
        - If Zomma is negative, the option's Gamma falls as implied volatility rises, which is typical for strikes
        near the money where Gamma is already at its peak.

        Note that the zomma of a call option and put option are equal to each other.

        Also known as: gamma sensitivity to volatility.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the zomma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["CSCO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_zomma(
            strike_price_range=0.15, strike_step_size=2
        ).loc["CSCO", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |             64 |       0      |       0.0022 |       0.0379 |
        |             66 |       0.0002 |       0.0168 |       0.085  |
        |             68 |       0.0062 |       0.0729 |       0.1319 |
        |             70 |       0.0671 |       0.1714 |       0.1246 |
        |             72 |       0.2478 |       0.1771 |       0.0182 |
        |             74 |       0.156  |      -0.0713 |      -0.1589 |
        |             76 |      -0.4947 |      -0.4015 |      -0.3028 |
        |             78 |      -0.5373 |      -0.4265 |      -0.3184 |
        |             80 |       0.1031 |      -0.1274 |      -0.2014 |
        |             82 |       0.2943 |       0.1514 |      -0.032  |
        |             84 |       0.1311 |       0.2138 |       0.0997 |
        |             86 |       0.0273 |       0.1413 |       0.1536 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        zomma: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            zomma[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            zomma[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_zomma(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        zomma_df = helpers.create_greek_dataframe(
            greek_dictionary=zomma,
            start_date=start_date,
        )

        zomma_df = zomma_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            zomma_df = calculate_standardization(
                dataset=zomma_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return zomma_df

    def get_color(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the color of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The color is
        the rate of change of the gamma with respect to time to expiration.

        The color calculation is the theoretical value of the color. The actual color can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Color = e^(—q * t) * (N'(d1) / (2 * S * t * σ * sqrt(t))) * (2 * q * t + 1 + ((2 * (r — q) * t — d2 * σ * sqrt(t)) / (σ * sqrt(t))) * d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration and N'(d1) is the standard normal probability density at d1.

        This expression equals dGamma/dt, the change in Gamma as calendar time passes (the negative of the derivative
        with respect to the remaining time to maturity t), so Color, like Theta, Charm and Veta, measures the change
        per unit of time that passes. Tables that print the expression with a leading minus sign give the derivative
        with respect to the time to maturity instead. The result is per year, matching Charm; divide by 365 for gamma
        decay per calendar day.

        The Color can be interpreted as follows:

        - If Color is positive, the option's Gamma builds up with each day that passes, which is what happens to a
        near-the-money option as expiration approaches.
        - If Color is negative, the option's Gamma bleeds away with each day that passes, which is what happens to a
        strike far from the money that is running out of time to reach it.

        Note that the color of a call option and put option are equal to each other.

        Also known as: gamma time decay.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the color values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["PEP"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_color(
            strike_price_range=0.15, strike_step_size=5
        ).loc["PEP", ["2026-01-07", "2026-01-14", "2026-01-28"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-07 |   2026-01-14 |   2026-01-28 |
        |---------------:|-------------:|-------------:|-------------:|
        |            120 |       0      |      -0.0035 |      -0.0323 |
        |            125 |      -0.0026 |      -0.0527 |      -0.0911 |
        |            130 |      -0.1387 |      -0.2563 |      -0.1154 |
        |            135 |      -0.9333 |      -0.2703 |       0.0108 |
        |            140 |       0.6313 |       0.475  |       0.2228 |
        |            145 |       1.9641 |       0.7561 |       0.279  |
        |            150 |      -0.8506 |       0.0102 |       0.1184 |
        |            155 |      -0.6021 |      -0.3776 |      -0.0689 |
        |            160 |      -0.0689 |      -0.2173 |      -0.1354 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        color: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            color[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            color[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_color(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        color_df = helpers.create_greek_dataframe(
            greek_dictionary=color,
            start_date=start_date,
        )

        color_df = color_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            color_df = calculate_standardization(
                dataset=color_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return color_df

    def get_ultima(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
    ):
        """
        Calculate the ultima of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The ultima is
        the rate of change of the vomma with respect to volatility.

        The ultima calculation is the theoretical value of the ultima. The actual gamma can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Ultima = (—vega / σ ** 2) * (d1 * d2 * (1 — d1 * d2) + d1 ** 2 + d2 ** 2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration and vega is the unscaled S * e^(—q * t) * N'(d1) * sqrt(t), i.e. the
        Vega before the division by 100 that `get_vega` applies. Ultima itself is likewise reported unscaled, per
        1.00 of volatility.

        The Ultima can be interpreted as follows:

        - If Ultima is positive, the option's Vomma rises as implied volatility rises, so the volatility convexity
        of the position builds up in a rising volatility regime.
        - If Ultima is negative, the option's Vomma falls as implied volatility rises, which is the usual case for
        strikes near the money.

        Note that the ultima of a call option and put option are equal to each other.

        Also known as: third-order vega.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: the ultima values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NKE"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_ultima(
            strike_price_range=0.15, strike_step_size=2, expiration_time_range=91
        ).loc["NKE", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             54 |     -30.4566 |     -65.1764 |     -73.0765 |
        |             56 |     -46.1558 |     -58.2569 |     -57.796  |
        |             58 |     -42.7252 |     -40.9146 |     -37.735  |
        |             60 |     -25.0168 |     -20.763  |     -18.6404 |
        |             62 |      -6.6538 |      -5.5465 |      -5.3442 |
        |             64 |      -0.2364 |      -0.4387 |      -0.7623 |
        |             66 |      -9.4832 |      -6.7211 |      -5.5464 |
        |             68 |     -28.3077 |     -22.0046 |     -18.3164 |
        |             70 |     -45.8895 |     -41.5496 |     -36.2787 |
        """
        if start_date is not None and start_date not in self._prices.index:
            raise ValueError(f"The start date {start_date} is not a valid date.")

        start_date = start_date if start_date else self._daily_historical.index[-1]
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
        )

        # This creates a list of time to expiration values from 0 to time_range, with a step size of 1
        time_to_expiration_list = [
            time / 365 for time in range(0, expiration_time_range)
        ]

        ultima: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            ultima[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            ultima[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: greeks_model.get_ultima(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
            )

        ultima_df = helpers.create_greek_dataframe(
            greek_dictionary=ultima,
            start_date=start_date,
        )

        ultima_df = ultima_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            ultima_df = calculate_standardization(
                dataset=ultima_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return ultima_df

    def _expiries(
        self,
        ticker: str,
        expiration_dates: list[str] | None,
        number_of_expirations: int,
    ) -> list[str]:
        """The expiries of a ticker to use: as given, otherwise the nearest ones."""
        available = options_model.get_option_expiry_dates(ticker)
        if expiration_dates is not None:
            invalid = [date for date in expiration_dates if date not in available]
            if invalid:
                raise ValueError(
                    f"The expiration date(s) {', '.join(invalid)} are not available for "
                    f"{ticker}. Choose from {', '.join(available)}"
                )
            return list(expiration_dates)
        upcoming = [date for date in available if _days_to_expiration(date) > 0]
        return upcoming[:number_of_expirations]

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

    def _warn_if_history_is_stale(self, purpose: str) -> None:
        """
        Warn when the historical data ends more than 5 days before today. The option chains
        are always live, so history that ends earlier (an end_date in the past) mixes old
        prices, rates and dividends with today's option quotes.
        """
        last_date = to_period_index(self._daily_historical.index)[-1].to_timestamp()
        lag = (pd.Timestamp.today().normalize() - last_date).days
        if lag > 5:  # noqa: PLR2004
            logger.warning(
                "The historical data ends on %s, %d days before today, while the option "
                "chains are live: %s. Leave end_date at today for these metrics.",
                last_date.strftime("%Y-%m-%d"),
                lag,
                purpose,
            )

    def _expiry_inputs(
        self, ticker: str, expiration_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
        """
        The calls, puts, stock price, time to expiration, risk free rate and dividend yield.
        The stock price is the one quoted with the chain when available (see _spot_price).
        """
        calls = options_model.get_option_chains([ticker], expiration_date, False)
        puts = options_model.get_option_chains([ticker], expiration_date, True)
        calls = calls.loc[ticker] if not calls.empty else calls
        puts = puts.loc[ticker] if not puts.empty else puts

        # Outside market hours Yahoo Finance often returns chains without bids, while Cboe's
        # delayed quotes keep the last ones; the better quoted chain is used.
        def quoted(chain: pd.DataFrame) -> int:
            return int((pd.to_numeric(chain.get("Bid"), errors="coerce") > 0).sum())

        if quoted(calls) + quoted(puts) < (len(calls) + len(puts)) / 2:
            cboe_calls = options_model._get_cboe_option_chain(  # noqa: SLF001
                ticker, expiration_date, False
            )
            cboe_puts = options_model._get_cboe_option_chain(  # noqa: SLF001
                ticker, expiration_date, True
            )
            if quoted(cboe_calls) + quoted(cboe_puts) > quoted(calls) + quoted(puts):
                calls, puts = cboe_calls, cboe_puts
        stock_price = self._spot_price(ticker, calls, puts)
        time_to_expiration = max(_days_to_expiration(expiration_date), 0.5) / 365
        risk_free_rate = float(self._risk_free_rate.dropna().iloc[-1])
        dividend_yield = (
            float(self._dividend_yield[ticker].iloc[-1])
            if not self._dividend_yield[ticker].empty
            else 0.0
        )
        if np.isnan(dividend_yield):
            dividend_yield = 0.0
        return (
            calls,
            puts,
            stock_price,
            time_to_expiration,
            risk_free_rate,
            dividend_yield,
        )

    def _per_expiry(
        self,
        calculate,
        expiration_dates: list[str] | dict[str, list[str]] | None,
        number_of_expirations: int,
        rounding: int | None,
    ) -> pd.DataFrame:
        """
        Run a calculation on every expiry of every ticker, rows (ticker, expiration). The
        expiries are the same for every ticker, or given per ticker as a dictionary, which
        then also selects the tickers.
        """
        self._warn_if_history_is_stale(_CHAIN_INPUTS_FROM_HISTORY)
        per_ticker = (
            expiration_dates
            if isinstance(expiration_dates, dict)
            else dict.fromkeys(self._tickers, expiration_dates)
        )
        rows = {}
        for ticker, ticker_expiration_dates in per_ticker.items():
            for expiration_date in self._expiries(
                ticker, ticker_expiration_dates, number_of_expirations
            ):
                inputs = self._expiry_inputs(ticker, expiration_date)
                if inputs[0].empty or inputs[1].empty:
                    continue
                result = calculate(*inputs)
                rows[(ticker, expiration_date)] = pd.concat(
                    [
                        pd.Series(
                            {"Days to Expiration": _days_to_expiration(expiration_date)}
                        ),
                        result,
                    ]
                )

        if not rows:
            raise ValueError("No option chains are available for these tickers.")

        table = pd.DataFrame(rows).T.astype(float)
        table.index.names = ["Ticker", "Expiration"]
        return apply_rounding(
            table, rounding if rounding is not None else self._rounding
        )

    def get_put_call_ratio(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the put-call ratio of every expiry, by volume and by open interest. The
        ratio is the number of puts divided by the number of calls: a ratio above one means
        more puts are traded or held than calls, which is usually read as hedging demand or
        bearish sentiment, and an unusually low ratio as complacency. The volume ratio
        reflects today's trading; the open interest ratio the positions that remain.

        Also known as: PCR, put/call ratio, options sentiment.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and expiry, and over all of them ("All"), the Days to
            Expiration, Call Volume, Put Volume, Volume Put-Call Ratio, Call Open Interest,
            Put Open Interest and Open Interest Put-Call Ratio.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_put_call_ratio(number_of_expirations=3)
        ```

        Which returns:

        |                        |   Days to Expiration |   Call Volume |   Put Volume |   Volume Put-Call Ratio |   Call Open Interest |   Put Open Interest |   Open Interest Put-Call Ratio |
        |:-----------------------|---------------------:|--------------:|-------------:|------------------------:|---------------------:|--------------------:|-------------------------------:|
        | ('AAPL', '2026-10-26') |                    2 |         90053 |        85187 |                  0.946  |                14688 |               22097 |                         1.5044 |
        | ('AAPL', '2026-10-28') |                    4 |         27330 |        21152 |                  0.7739 |                17505 |               10137 |                         0.5791 |
        | ('AAPL', '2026-10-30') |                    6 |         76540 |       134401 |                  1.756  |                81432 |               86739 |                         1.0652 |
        | ('AAPL', 'All')        |                  nan |        193923 |       240740 |                  1.2414 |               113625 |              118973 |                         1.0471 |
        | ('TSLA', '2026-10-26') |                    2 |        149535 |       129459 |                  0.8657 |                68890 |               34714 |                         0.5039 |
        | ('TSLA', '2026-10-28') |                    4 |         32602 |        24491 |                  0.7512 |                14542 |               15028 |                         1.0334 |
        | ('TSLA', '2026-10-30') |                    6 |        121542 |       101610 |                  0.836  |               104417 |               97818 |                         0.9368 |
        | ('TSLA', 'All')        |                  nan |        303679 |       255560 |                  0.8415 |               187849 |              147560 |                         0.7855 |
        """
        table = self._per_expiry(
            lambda calls, puts, *_: market_metrics_model.get_put_call_ratio(
                calls, puts
            ),
            expiration_dates,
            number_of_expirations,
            None,
        )
        # Each ticker's expiries are followed by an "All" row with the summed volume and
        # open interest and the ratios of those sums.
        parts = []
        for ticker in table.index.get_level_values(0).unique():
            total = table.loc[ticker].sum()
            total["Days to Expiration"] = np.nan
            for ratio, side in (
                ("Volume Put-Call Ratio", "Volume"),
                ("Open Interest Put-Call Ratio", "Open Interest"),
            ):
                calls = total[f"Call {side}"]
                total[ratio] = total[f"Put {side}"] / calls if calls else np.nan
            parts += [
                table.loc[[ticker]],
                pd.DataFrame(
                    [total], index=pd.MultiIndex.from_tuples([(ticker, "All")])
                ),
            ]
        combined = pd.concat(parts)
        combined.index.names = ["Ticker", "Expiration"]
        return apply_rounding(
            combined, rounding if rounding is not None else self._rounding
        )

    def get_max_pain(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the max pain strike of every expiry: the expiry price at which the
        options outstanding would pay their holders the least, the sum over every call of
        open interest times max(price - strike, 0) and over every put of open interest
        times max(strike - price, 0) being smallest. Option writers, often market makers,
        profit most there, and some traders expect the price to drift towards it as the
        expiry nears; the evidence for that is mixed, so it is best read as a description
        of where open interest is concentrated.

        Also known as: max pain theory, option pain, maximum pain.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Max Pain strike,
            Stock Price and the Distance to Max Pain (as a fraction of the price).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_max_pain(number_of_expirations=4)
        ```

        Which returns:

        |                        |   Days to Expiration |   Max Pain |   Stock Price |   Distance to Max Pain |
        |:-----------------------|---------------------:|-----------:|--------------:|-----------------------:|
        | ('AAPL', '2026-10-26') |                    2 |      345   |        347.73 |                -0.0079 |
        | ('AAPL', '2026-10-28') |                    4 |      345   |        347.73 |                -0.0079 |
        | ('AAPL', '2026-10-30') |                    6 |      342.5 |        347.73 |                -0.015  |
        | ('AAPL', '2026-11-06') |                   13 |      340   |        347.73 |                -0.0222 |
        """

        def calculate(calls, puts, stock_price, *_):
            max_pain = market_metrics_model.get_max_pain(calls, puts)
            return pd.Series(
                {
                    "Max Pain": max_pain,
                    "Stock Price": stock_price,
                    "Distance to Max Pain": max_pain / stock_price - 1,
                }
            )

        return self._per_expiry(
            calculate, expiration_dates, number_of_expirations, rounding
        )

    def get_gamma_exposure(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        by_strike: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the dealer gamma exposure (GEX): the dollar value of shares market makers
        would have to buy or sell for every 1% move in the underlying to keep their option
        books delta-neutral. Following the common convention, dealers are assumed long the
        calls and short the puts that investors trade:

            - Gamma Exposure = Gamma x Open Interest x 100 x Price² x 0.01
            - positive for calls, negative for puts, summed over strikes and expiries

        With positive exposure dealers buy dips and sell rallies, which dampens volatility;
        with negative exposure they trade in the direction of the move and amplify it. The
        gamma flip point is the price at which the total exposure changes sign.

        For more information, see the following sources:

        - SqueezeMetrics (2017). "Gamma Exposure (GEX)." White paper.
        - Perfiliev, S. (2021). "How to Calculate Gamma Exposure (GEX) and Zero Gamma
          Level." https://perfiliev.com/blog/how-to-calculate-gamma-exposure-and-zero-gamma-level/

        Also known as: GEX, dealer gamma, gamma positioning, net gamma.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to include. Defaults
                to None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to include
                when no dates are given. Defaults to 6.
            by_strike (bool, optional): Whether to return the exposure per strike instead of
                the totals. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Stock Price, Call, Put and Net Gamma Exposure (in
            dollars per 1% move) and the Gamma Flip Point; with by_strike=True the Call, Put
            and Net Gamma Exposure per ticker and strike.

        Notes:
        - The assumption about who holds which side is the main uncertainty: when
          investors sell calls (covered calls) the sign of that part flips.
        - Gamma uses each option's own implied volatility (Black-Scholes).
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_gamma_exposure()
        ```

        Which returns:

        |      |   Stock Price |   Call Gamma Exposure |   Put Gamma Exposure |   Net Gamma Exposure |   Gamma Flip Point |
        |:-----|--------------:|----------------------:|---------------------:|---------------------:|-------------------:|
        | QQQ  |        704.4  |           1.83881e+09 |         -6.41458e+09 |         -4.57577e+09 |            727.993 |
        | AAPL |        347.73 |           7.48337e+08 |         -4.80606e+08 |          2.67731e+08 |            342.412 |
        """
        self._warn_if_history_is_stale(_CHAIN_INPUTS_FROM_HISTORY)
        summary, strikes = {}, {}
        for ticker in self._tickers:
            expiries = []
            per_strike = []
            stock_price = risk_free_rate = dividend_yield = np.nan
            for expiration_date in self._expiries(
                ticker, expiration_dates, number_of_expirations
            ):
                (
                    calls,
                    puts,
                    stock_price,
                    time_to_expiration,
                    risk_free_rate,
                    dividend_yield,
                ) = self._expiry_inputs(ticker, expiration_date)
                if calls.empty or puts.empty:
                    continue
                expiries.append((calls, puts, time_to_expiration))
                per_strike.append(
                    market_metrics_model.get_gamma_exposure(
                        calls,
                        puts,
                        stock_price,
                        time_to_expiration,
                        risk_free_rate,
                        dividend_yield,
                    )
                )
            if not per_strike:
                continue
            combined = pd.concat(per_strike).groupby(level=0).sum()
            strikes[ticker] = combined
            summary[ticker] = pd.Series(
                {
                    "Stock Price": stock_price,
                    **combined.sum(),
                    "Gamma Flip Point": market_metrics_model.get_gamma_flip(
                        expiries, stock_price, risk_free_rate, dividend_yield
                    ),
                }
            )

        if not summary:
            raise ValueError("No option chains are available for these tickers.")

        result = (
            pd.concat(strikes, names=["Ticker", "Strike Price"])
            if by_strike
            else pd.DataFrame(summary).T
        )
        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    def get_implied_volatility_term_structure(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 10,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the term structure of implied volatility: the at-the-money implied
        volatility of every expiry, interpolated at the current price and averaged over
        calls and puts. Normally longer expiries carry a higher volatility (contango); an
        inverted curve, with short-dated options dearer than long-dated ones, shows that
        the market expects turbulence soon, as around earnings or in a sell-off.

        Also known as: volatility term structure, IV curve, volatility curve.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 10.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration and the At-the-Money
            Volatility (annualized, as a decimal).

        Notes:
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_volatility_term_structure(number_of_expirations=5)
        ```

        Which returns:

        |                        |   Days to Expiration |   At-the-Money Volatility |
        |:-----------------------|---------------------:|--------------------------:|
        | ('AAPL', '2026-10-26') |                    2 |                    0.1637 |
        | ('AAPL', '2026-10-28') |                    4 |                    0.2136 |
        | ('AAPL', '2026-10-30') |                    6 |                    0.2273 |
        | ('AAPL', '2026-11-06') |                   13 |                    0.319  |
        | ('AAPL', '2026-11-13') |                   20 |                    0.2967 |
        | ('TSLA', '2026-10-26') |                    2 |                    0.297  |
        | ('TSLA', '2026-10-28') |                    4 |                    0.3781 |
        | ('TSLA', '2026-10-30') |                    6 |                    0.4003 |
        | ('TSLA', '2026-11-06') |                   13 |                    0.4124 |
        | ('TSLA', '2026-11-13') |                   20 |                    0.4058 |
        """
        return self._per_expiry(
            lambda calls, puts, stock_price, *_: pd.Series(
                {
                    "At-the-Money Volatility": market_metrics_model.get_at_the_money_volatility(
                        calls, puts, stock_price
                    )
                }
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
        )

    def get_volatility_skew(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 6,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the volatility skew of every expiry from the 25-delta options, the
        standard way to compare skew across underlyings and dates:

            - 25-Delta Risk Reversal = IV(25-delta call) - IV(25-delta put)
            - 25-Delta Butterfly = (IV(25-delta call) + IV(25-delta put)) / 2 - ATM IV

        For equities the risk reversal is usually negative: out-of-the-money puts are
        dearer than calls because investors pay for crash protection. A more negative risk
        reversal means more fear of a fall; the butterfly measures how much both tails cost
        relative to the centre. The deltas follow from each option's implied volatility, and
        the volatility at exactly 25 delta is interpolated.

        For more information, see the following book:

        - Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide." Wiley.

        Also known as: risk reversal, volatility smile, skew, butterfly spread volatility.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 6.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, the 25-Delta Put,
            At-the-Money and 25-Delta Call Volatility, the 25-Delta Risk Reversal and the
            25-Delta Butterfly.

        Notes:
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_volatility_skew(number_of_expirations=3)
        ```

        Which returns:

        |                        |   Days to Expiration |   25-Delta Put Volatility |   At-the-Money Volatility |   25-Delta Call Volatility |   25-Delta Risk Reversal |   25-Delta Butterfly |
        |:-----------------------|---------------------:|--------------------------:|--------------------------:|---------------------------:|-------------------------:|---------------------:|
        | ('QQQ', '2026-10-26')  |                    2 |                    0.1908 |                    0.1938 |                     0.1855 |                  -0.0052 |              -0.0056 |
        | ('QQQ', '2026-10-27')  |                    3 |                    0.2287 |                    0.2253 |                     0.213  |                  -0.0157 |              -0.0044 |
        | ('QQQ', '2026-10-28')  |                    4 |                    0.2629 |                    0.2545 |                     0.2396 |                  -0.0233 |              -0.0033 |
        | ('AAPL', '2026-10-26') |                    2 |                    0.1716 |                    0.1637 |                     0.1555 |                  -0.0161 |              -0.0002 |
        | ('AAPL', '2026-10-28') |                    4 |                    0.2238 |                    0.2136 |                     0.2031 |                  -0.0207 |              -0.0002 |
        | ('AAPL', '2026-10-30') |                    6 |                    0.2349 |                    0.2273 |                     0.215  |                  -0.0199 |              -0.0023 |
        """
        return self._per_expiry(
            market_metrics_model.get_volatility_skew,
            expiration_dates,
            number_of_expirations,
            rounding,
        )

    def get_implied_move(
        self,
        expiration_dates: list[str] | None = None,
        number_of_expirations: int = 4,
        around_earnings: bool = False,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the move the options market expects up to every expiry, from the price of
        the at-the-money straddle (buying the call and the put at the strike closest to the
        current price):

            - Implied Move = (Call Price + Put Price) / Stock Price
            - Expected Range = Stock Price ± Straddle Price

        The straddle pays the absolute move, so its price is the market's expected size of
        the move, regardless of direction. With around_earnings=True the first expiry after
        the next earnings date is used, which gives the move the market prices for the
        earnings announcement (an expiry on the announcement day itself settles at that
        day's close, before the usual after-the-close announcement).

        Also known as: expected move, straddle move, earnings move, implied earnings move.

        Args:
            expiration_dates (list[str] | None, optional): The expiries to use. Defaults to
                None, the nearest `number_of_expirations` expiries.
            number_of_expirations (int, optional): The number of nearest expiries to use when
                no dates are given. Defaults to 4.
            around_earnings (bool, optional): Whether to use the first expiry after the next
                earnings date instead. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker and expiry the Days to Expiration, Straddle Strike,
            Straddle Price, Implied Move (as a fraction of the price) and the Lower and Upper
            Price of the expected range.

        Notes:
        - The move up to an expiry after earnings also includes the ordinary daily moves
          until then; the closer the expiry is to the announcement, the purer the measure.
        - The stock price is the price of the underlying quoted with the live option chain
          (Yahoo Finance's regular market price or Cboe's current price), so both come from
          the same moment; when the chain quotes none, the traded Close of the last
          historical date is used. The risk free rate and dividend yield come from the
          historical data as well, so leave end_date at today: a warning is logged when the
          history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "NVDA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_implied_move(around_earnings=True)
        ```

        Which returns:

        |                        | Earnings Date   |   Days to Expiration |   Straddle Strike |   Straddle Price |   Implied Move |   Lower Price |   Upper Price |
        |:-----------------------|:----------------|---------------------:|------------------:|-----------------:|---------------:|--------------:|--------------:|
        | ('AAPL', '2026-11-06') | 2026-11-02      |                   13 |               350 |            16.85 |         0.0485 |        330.88 |        364.58 |
        | ('NVDA', '2026-11-20') | 2026-11-18      |                   27 |               210 |            20.5  |         0.0972 |        190.46 |        231.46 |
        """
        earnings_dates = None
        if around_earnings:
            if self._earnings_function is None:
                raise ValueError("Earnings dates are not available for this Toolkit.")
            earnings_dates = self._earnings_function()
            # Each ticker uses the first expiry after its own earnings date.
            expiration_dates = {}
            for ticker in self._tickers:
                earnings_date = earnings_dates.get(ticker)
                if not earnings_date:
                    logger.warning("No upcoming earnings date is known for %s.", ticker)
                    continue
                after = [
                    date
                    for date in options_model.get_option_expiry_dates(ticker)
                    if date > earnings_date
                ]
                if after:
                    expiration_dates[ticker] = [after[0]]
            if not expiration_dates:
                raise ValueError("No expiry after an upcoming earnings date was found.")

        table = self._per_expiry(
            lambda calls, puts, stock_price, *_: market_metrics_model.get_implied_move(
                calls, puts, stock_price
            ),
            expiration_dates,
            number_of_expirations,
            rounding,
        )
        if earnings_dates is not None:
            table.insert(
                0,
                "Earnings Date",
                table.index.get_level_values("Ticker").map(earnings_dates),
            )
        return table

    def _model_free_volatility(self, ticker: str, days: int) -> pd.Series:
        """The constant-maturity model-free implied volatility of one ticker."""
        usable = [
            date
            for date in options_model.get_option_expiry_dates(ticker)
            if _days_to_expiration(date) >= market_metrics_model.MINIMUM_DAYS
        ]
        if len(usable) < 2:  # noqa: PLR2004
            raise ValueError(
                f"{ticker} has too few expiries for a model-free volatility."
            )

        # Beyond the furthest expiry there is nothing to interpolate towards, and holding
        # the last expiry's variance would present it as the volatility of a longer span.
        if days > _days_to_expiration(usable[-1]):
            raise ValueError(
                f"The longest usable expiry of {ticker} is {usable[-1]} "
                f"({_days_to_expiration(usable[-1])} days), shorter than days={days}."
            )

        # A target shorter than the nearest usable expiry is extrapolated from the two
        # nearest terms, as the Cboe does when no expiry lies before the 30 days.
        near_candidates = [date for date in usable if _days_to_expiration(date) <= days]
        near = near_candidates[-1] if near_candidates else usable[0]
        following = [date for date in usable if date > near]
        next_term = following[0] if following else near

        terms = {}
        for label, date in (("Near-Term", near), ("Next-Term", next_term)):
            calls, puts, _, time, rate, _ = self._expiry_inputs(ticker, date)
            # A ValueError reaches the per-ticker warning, so the other tickers still answer.
            if calls.empty or puts.empty:
                raise ValueError(f"{ticker} has no option chain for {date}.")
            variance, forward = market_metrics_model.get_model_free_variance(
                calls, puts, time, rate
            )
            terms[label] = (date, variance, forward, _days_to_expiration(date))

        near_date, near_variance, forward, near_days = terms["Near-Term"]
        next_date, next_variance, _, next_days = terms["Next-Term"]
        volatility = market_metrics_model.get_constant_maturity_volatility(
            near_variance, near_days, next_variance, next_days, days
        )
        return pd.Series(
            {
                "Near-Term Expiration": near_date,
                "Next-Term Expiration": next_date,
                "Near-Term Volatility": (
                    np.sqrt(near_variance) if near_variance > 0 else np.nan
                ),
                "Next-Term Volatility": (
                    np.sqrt(next_variance) if next_variance > 0 else np.nan
                ),
                "Forward Price": forward,
                "Model-Free Implied Volatility": volatility,
            }
        )

    def get_model_free_implied_volatility(
        self,
        days: int = 30,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the model-free implied volatility with the Cboe VIX methodology, for any
        ticker with listed options. Rather than inverting a pricing model, it reads the
        expected variance directly from the prices of out-of-the-money options across all
        strikes, weighted by 1/K², for the two expiries around the target maturity, and
        interpolates their variances to a constant maturity:

            - σ² = (2/T) Σ (ΔK_i / K_i²) e^(RT) Q(K_i) - (1/T) (F/K_0 - 1)²
            - Volatility = sqrt(interpolated variance at `days`)

        Applied to the S&P 500 index options ("^SPX") with days=30 it approximates the VIX,
        within about a tenth of a volatility point: the Cboe counts the time to expiry in
        minutes, uses Treasury rates for each expiry and the Friday expiries between 23 and
        37 days away. For a single stock it gives the equivalent volatility index.

        For more information, see the following papers:

        - Cboe Global Markets (2023). "Cboe Volatility Index (VIX) White Paper."
        - Britten-Jones, M. and Neuberger, A. (2000). "Option Prices, Implied Price
          Processes, and Stochastic Volatility." The Journal of Finance, 55(2), 839-866.

        Also known as: VIX methodology, model-free volatility, variance swap volatility.

        Args:
            days (int, optional): The constant maturity in calendar days. Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Near-Term and Next-Term Expiration and Volatility,
            the Forward Price and the Model-Free Implied Volatility (annualized, as a
            decimal).

        Notes:
        - Options without a bid are left out of the strip, as in the Cboe method. Outside
          market hours, when bids are missing, an expiry has no strip and the result is
          NaN.
        - A target shorter than the nearest usable expiry (at least 7 days away) is
          extrapolated from the two nearest terms, as in the Cboe method. A target beyond
          the longest usable expiry raises a ValueError naming the ticker and that expiry,
          since there is no later term to interpolate towards.
        - The risk free rate comes from the historical data, so leave end_date at today: a
          warning is logged when the history ends more than 5 days before today.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["^SPX", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_model_free_implied_volatility()
        ```

        Which returns:

        |      | Near-Term Expiration   | Next-Term Expiration   |   Near-Term Volatility |   Next-Term Volatility |   Forward Price |   Model-Free Implied Volatility |
        |:-----|:-----------------------|:-----------------------|-----------------------:|-----------------------:|----------------:|--------------------------------:|
        | ^SPX | 2026-11-23             | 2026-11-24             |                 0.1955 |                 0.1965 |        7649.3   |                          0.1955 |
        | AAPL | 2026-11-20             | 2026-11-27             |                 0.297  |                 0.2817 |         348.495 |                          0.2897 |
        """
        self._warn_if_history_is_stale("the risk free rate comes from that date")
        rows, errors = {}, []
        for ticker in self._tickers:
            try:
                rows[ticker] = self._model_free_volatility(ticker, days)
            except ValueError as error:
                logger.warning("%s", error)
                errors.append(str(error))

        if not rows:
            raise ValueError(
                "No model-free implied volatility could be calculated. "
                + " ".join(errors)
            )

        table = pd.DataFrame(rows).T
        numeric = table.columns.drop(["Near-Term Expiration", "Next-Term Expiration"])
        table[numeric] = apply_rounding(
            table[numeric].astype(float),
            rounding if rounding is not None else self._rounding,
        )
        return table

    def get_volatility_risk_premium(
        self,
        days: int = 30,
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the volatility risk premium: the model-free implied volatility over the
        coming `days` minus the volatility the stock actually realized over the same span
        in the past, and the same in variance units. Implied volatility usually exceeds realized volatility, because
        investors pay a premium for protection, which is the return option sellers earn. A
        premium that turns negative means the market prices less movement than it has just
        seen, often after a shock.

            - Realized Volatility = standard deviation of the daily returns over the last
              `days` calendar days, annualized
            - Volatility Risk Premium = Implied Volatility - Realized Volatility
            - Variance Risk Premium = Implied Volatility² - Realized Volatility²

        Both compare the implied figure with the realized one of the past, which is known
        today, as Bollerslev, Tauchen and Zhou (2009) do. Carr and Wu (2009) measure the
        premium after the fact instead: the variance realized over the coming `days` minus
        the variance swap rate, which is negative on average.

        For more information, see the following papers:

        - Bollerslev, T., Tauchen, G. and Zhou, H. (2009). "Expected Stock Returns and
          Variance Risk Premia." The Review of Financial Studies, 22(11), 4463-4492.
        - Carr, P. and Wu, L. (2009). "Variance Risk Premiums." The Review of Financial
          Studies, 22(3), 1311-1341.

        Also known as: VRP, implied minus realized volatility, variance risk premium.

        Args:
            days (int, optional): The horizon in calendar days of both volatilities.
                Defaults to 30.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the Toolkit's rounding.

        Returns:
            pd.DataFrame: Per ticker the Implied Volatility, Realized Volatility, Volatility
            Risk Premium, Variance Risk Premium and the ratio of implied to realized
            volatility.

        Notes:
        - The realized volatility covers the last `days` calendar days of the historical
          data, while the implied volatility comes from the live option chains. Leave
          end_date at today so both describe the same moment: a warning is logged when the
          history ends more than 5 days before today.
        - A `days` beyond the longest usable expiry raises a ValueError, see
          get_model_free_implied_volatility.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["QQQ", "AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_volatility_risk_premium()
        ```

        Which returns:

        |      |   Implied Volatility |   Realized Volatility |   Volatility Risk Premium |   Variance Risk Premium |   Implied to Realized Ratio |
        |:-----|---------------------:|----------------------:|--------------------------:|------------------------:|----------------------------:|
        | QQQ  |               0.2846 |                0.1769 |                    0.1077 |                  0.0497 |                      1.609  |
        | AAPL |               0.2897 |                0.1641 |                    0.1256 |                  0.057  |                      1.7653 |
        | TSLA |               0.4357 |                0.3516 |                    0.0841 |                  0.0662 |                      1.2392 |
        """
        self._warn_if_history_is_stale(
            "the realized volatility covers the days up to that date instead of the days "
            "up to today, and the risk free rate comes from that date"
        )
        rows, errors = {}, []
        for ticker in self._tickers:
            try:
                implied = float(
                    self._model_free_volatility(ticker, days)[
                        "Model-Free Implied Volatility"
                    ]
                )
            except ValueError as error:
                logger.warning("%s", error)
                errors.append(str(error))
                continue
            returns = self._daily_historical["Return"][ticker].dropna()
            dates = to_period_index(returns.index).to_timestamp()
            recent = returns[dates > dates.max() - pd.Timedelta(days=days)]
            realized = float(recent.std() * np.sqrt(252))
            rows[ticker] = {
                "Implied Volatility": implied,
                "Realized Volatility": realized,
                "Volatility Risk Premium": implied - realized,
                "Variance Risk Premium": implied**2 - realized**2,
                "Implied to Realized Ratio": (
                    implied / realized if realized > 0 else np.nan
                ),
            }

        if not rows:
            raise ValueError(
                "No volatility risk premium could be calculated. " + " ".join(errors)
            )

        return apply_rounding(
            pd.DataFrame(rows).T, rounding if rounding is not None else self._rounding
        )
