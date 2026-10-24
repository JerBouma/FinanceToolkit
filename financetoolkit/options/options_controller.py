"""Options Module"""

__docformat__ = "google"


import numpy as np
import pandas as pd

from financetoolkit.options import (
    binomial_trees_model,
    black_scholes_model,
    exotics_model,
    greeks_model,
    helpers,
    options_model,
    risk_neutral_density_model,
    svi_model,
)
from financetoolkit.ratios import valuation_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_standardization,
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
MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK = 2


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
        daily_historical: pd.DataFrame = pd.DataFrame(),
        annual_historical: pd.DataFrame = pd.DataFrame(),
        risk_free_rate: pd.DataFrame = pd.DataFrame(),
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Options Controller Class. The Options module is meant to calculate important options
        metrics such as the First, Second and Third Order Greeks, the Black Scholes Model and the Option
        Chains as well as Implied Volatilities, Breeden—Litzenberger and more.

        Args:
            tickers (str | list[str]): The tickers to use.
            daily_historical (pd.DataFrame, optional): The daily historical data. Defaults to pd.DataFrame().
            annual_historical (pd.DataFrame, optional): The annual historical data. Defaults to pd.DataFrame().
            risk_free_rate (pd.DataFrame, optional): The risk free rate. Defaults to pd.DataFrame().
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

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
        self._daily_historical = daily_historical
        self._quarterly = quarterly
        self._rounding: int | None = rounding
        self._start_date: str | None = start_date
        self._end_date: str | None = end_date

        # The spot price is the traded close. On a past pricing date the Adj Close is lowered by
        # every dividend paid after that date, which was not known yet and which the dividend
        # yield q already accounts for. Returns (and so the volatility) do use the Adj Close.
        self._prices = self._daily_historical[
            "Close" if "Close" in self._daily_historical else "Adj Close"
        ]

        # The volatility on a date is that of the year of daily returns up to it, so a past
        # pricing date uses only what was known on that date.
        self._volatility = self._daily_historical["Return"].rolling(
            TRAILING_DAYS, min_periods=MINIMUM_VOLATILITY_DAYS
        ).std() * np.sqrt(TRAILING_DAYS)

        self._risk_free_rate = risk_free_rate["Adj Close"]
        self._annual_historical = annual_historical

        # Calculate Dividend Yield, relevant for Black Scholes formula, relative to the traded
        # close like the spot price.
        dividend_yield = valuation_model.get_dividend_yield(
            dividends=self._annual_historical.loc[:, "Dividends"].T,
            stock_price=self._annual_historical.loc[
                :, "Close" if "Close" in self._annual_historical else "Adj Close"
            ].T,
        )

        self._dividend_yield = {ticker: pd.Series() for ticker in self._tickers}

        for ticker in self._tickers:
            dividend_yield_cleaned = dividend_yield.loc[
                ticker, dividend_yield.loc[ticker] != 0
            ]

            if dividend_yield_cleaned.empty:
                # An empty value means the company pays no dividends, so 0 is correct.
                dividend_yield_cleaned = dividend_yield.loc[ticker]

            self._dividend_yield[ticker] = dividend_yield_cleaned

        # The dividend yield on a date: the dividends of the 365 days up to it over the close of
        # that date, see options_model.get_trailing_dividend_yield.
        self._daily_dividend_yield = pd.DataFrame()
        if "Close" in self._daily_historical:
            self._daily_dividend_yield = options_model.get_trailing_dividend_yield(
                close=self._daily_historical["Close"],
                adjusted_close=self._daily_historical["Adj Close"],
            )

    def _dividend_yield_on(self, ticker: str, date) -> float:
        """
        The dividend yield of a ticker on a pricing date: the dividends of the 365 days up to
        that date over its close, or the latest annual yield when there are no daily prices.

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
                return float(value)
        return float(self._dividend_yield[ticker].iloc[-1])

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

        toolkit = Toolkit(["AAPL", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        option_chains = toolkit.options.get_option_chains()

        option_chains.loc[('AAPL', option_chains['In The Money'] == True), :]
        ```

        Which returns:

        |                 | Contract Symbol     | Last Trade Date   |   Last Price |   Bid |   Ask |   Change |   Percent Change |   Volume |   Open Interest |   Implied Volatility | In The Money   | Currency   |
        |:----------------|:--------------------|:------------------|-------------:|------:|------:|---------:|-----------------:|---------:|----------------:|---------------------:|:---------------|:-----------|
        | ('AAPL', 110.0) | AAPL261009C00110000 | 2026-10-07        |       224.9  |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 120.0) | AAPL261009C00120000 | 2026-10-07        |       216.17 |     0 |     0 |        0 |                0 |        3 |               0 |                    0 | True           | USD        |
        | ('AAPL', 175.0) | AAPL261009C00175000 | 2026-10-07        |       161.94 |     0 |     0 |        0 |                0 |        2 |               0 |                    0 | True           | USD        |
        | ('AAPL', 185.0) | AAPL261009C00185000 | 2026-10-07        |       151.75 |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 195.0) | AAPL261009C00195000 | 2026-09-18        |       141.39 |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 210.0) | AAPL261009C00210000 | 2026-09-29        |       121.37 |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 215.0) | AAPL261009C00215000 | 2026-09-29        |       114.62 |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 220.0) | AAPL261009C00220000 | 2026-10-07        |       114.26 |     0 |     0 |        0 |                0 |        3 |               0 |                    0 | True           | USD        |
        | ('AAPL', 225.0) | AAPL261009C00225000 | 2026-10-05        |       109.87 |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 230.0) | AAPL261009C00230000 | 2026-10-05        |       103.5  |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 235.0) | AAPL261009C00235000 | 2026-10-01        |        92.78 |     0 |     0 |        0 |                0 |       16 |               0 |                    0 | True           | USD        |
        | ('AAPL', 240.0) | AAPL261009C00240000 | 2026-10-02        |        92.35 |     0 |     0 |        0 |                0 |       30 |               0 |                    0 | True           | USD        |
        | ('AAPL', 245.0) | AAPL261009C00245000 | 2026-09-30        |        93.88 |     0 |     0 |        0 |                0 |       32 |               0 |                    0 | True           | USD        |
        | ('AAPL', 250.0) | AAPL261009C00250000 | 2026-10-06        |        82.91 |     0 |     0 |        0 |                0 |        4 |               0 |                    0 | True           | USD        |
        | ('AAPL', 255.0) | AAPL261009C00255000 | 2026-10-02        |        78.8  |     0 |     0 |        0 |                0 |      345 |               0 |                    0 | True           | USD        |
        | ('AAPL', 260.0) | AAPL261009C00260000 | 2026-10-05        |        73.8  |     0 |     0 |        0 |                0 |        1 |               0 |                    0 | True           | USD        |
        | ('AAPL', 265.0) | AAPL261009C00265000 | 2026-10-06        |        68.46 |     0 |     0 |        0 |                0 |        3 |               0 |                    0 | True           | USD        |
        | ('AAPL', 270.0) | AAPL261009C00270000 | 2026-10-07        |        66.84 |     0 |     0 |        0 |                0 |        6 |               0 |                    0 | True           | USD        |
        | ('AAPL', 275.0) | AAPL261009C00275000 | 2026-10-07        |        62.17 |     0 |     0 |        0 |                0 |        7 |               0 |                    0 | True           | USD        |
        | ('AAPL', 280.0) | AAPL261009C00280000 | 2026-10-07        |        56.73 |     0 |     0 |        0 |                0 |      227 |               0 |                    0 | True           | USD        |
        | ('AAPL', 285.0) | AAPL261009C00285000 | 2026-10-07        |        52.33 |     0 |     0 |        0 |                0 |       60 |               0 |                    0 | True           | USD        |
        | ('AAPL', 290.0) | AAPL261009C00290000 | 2026-10-07        |        47.45 |     0 |     0 |        0 |                0 |       11 |               0 |                    0 | True           | USD        |
        | ('AAPL', 295.0) | AAPL261009C00295000 | 2026-10-07        |        41.9  |     0 |     0 |        0 |                0 |      144 |               0 |                    0 | True           | USD        |
        | ('AAPL', 300.0) | AAPL261009C00300000 | 2026-10-07        |        36.47 |     0 |     0 |        0 |                0 |      129 |               0 |                    0 | True           | USD        |
        | ('AAPL', 305.0) | AAPL261009C00305000 | 2026-10-07        |        31.46 |     0 |     0 |        0 |                0 |       27 |               0 |                    0 | True           | USD        |
        | ('AAPL', 310.0) | AAPL261009C00310000 | 2026-10-07        |        27.35 |     0 |     0 |        0 |                0 |      205 |               0 |                    0 | True           | USD        |
        | ('AAPL', 312.5) | AAPL261009C00312500 | 2026-10-07        |        24.7  |     0 |     0 |        0 |                0 |      254 |               0 |                    0 | True           | USD        |
        | ('AAPL', 315.0) | AAPL261009C00315000 | 2026-10-07        |        22.3  |     0 |     0 |        0 |                0 |       79 |               0 |                    0 | True           | USD        |
        | ('AAPL', 317.5) | AAPL261009C00317500 | 2026-10-07        |        18.75 |     0 |     0 |        0 |                0 |      142 |               0 |                    0 | True           | USD        |
        | ('AAPL', 320.0) | AAPL261009C00320000 | 2026-10-07        |        16.6  |     0 |     0 |        0 |                0 |       62 |               0 |                    0 | True           | USD        |
        | ('AAPL', 322.5) | AAPL261009C00322500 | 2026-10-07        |        14.21 |     0 |     0 |        0 |                0 |       71 |               0 |                    0 | True           | USD        |
        | ('AAPL', 325.0) | AAPL261009C00325000 | 2026-10-07        |        12    |     0 |     0 |        0 |                0 |      292 |               0 |                    0 | True           | USD        |
        | ('AAPL', 327.5) | AAPL261009C00327500 | 2026-10-07        |         9.7  |     0 |     0 |        0 |                0 |      469 |               0 |                    0 | True           | USD        |
        | ('AAPL', 330.0) | AAPL261009C00330000 | 2026-10-07        |         7.5  |     0 |     0 |        0 |                0 |     2122 |               0 |                    0 | True           | USD        |
        | ('AAPL', 332.5) | AAPL261009C00332500 | 2026-10-07        |         5.25 |     0 |     0 |        0 |                0 |     7506 |               0 |                    0 | True           | USD        |
        | ('AAPL', 335.0) | AAPL261009C00335000 | 2026-10-07        |         3.55 |     0 |     0 |        0 |                0 |    24852 |               0 |                    0 | True           | USD        |
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
            current_prices = self._prices.iloc[-1]
            spot = option_chains.index.get_level_values(0).map(current_prices)
            strikes = option_chains.index.get_level_values(1)
            option_chains = option_chains[
                (strikes >= spot * (1 - strike_price_range))
                & (strikes <= spot * (1 + strike_price_range))
            ]

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AMZN", "AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        black_scholes = toolkit.options.get_black_scholes_model()

        black_scholes.loc['AMZN']
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            175 |      55.8399 |      55.8599 |      55.8798 |      55.8998 |      55.9197 |      55.9396 |      55.9596 |      55.9795 |      55.9994 |      56.0193 |      56.0393 |      56.0592 |      56.0791 |      56.0991 |      56.119  |      56.139  |      56.159  |      56.1791 |      56.1993 |      56.2195 |      56.2398 |      56.2603 |      56.2808 |      56.3016 |      56.3225 |      56.3436 |      56.3648 |      56.3863 |      56.4081 |
        |            180 |      50.8405 |      50.861  |      50.8815 |      50.902  |      50.9225 |      50.943  |      50.9635 |      50.984  |      51.0045 |      51.025  |      51.0456 |      51.0661 |      51.0867 |      51.1073 |      51.128  |      51.1488 |      51.1698 |      51.1909 |      51.2123 |      51.2338 |      51.2556 |      51.2777 |      51.3001 |      51.3229 |      51.346  |      51.3695 |      51.3933 |      51.4176 |      51.4422 |
        |            185 |      45.8411 |      45.8622 |      45.8832 |      45.9043 |      45.9254 |      45.9465 |      45.9675 |      45.9886 |      46.0097 |      46.0309 |      46.0521 |      46.0734 |      46.0949 |      46.1166 |      46.1386 |      46.1609 |      46.1836 |      46.2068 |      46.2304 |      46.2545 |      46.2792 |      46.3045 |      46.3303 |      46.3568 |      46.3839 |      46.4117 |      46.4401 |      46.4691 |      46.4988 |
        |            190 |      40.8417 |      40.8633 |      40.885  |      40.9066 |      40.9282 |      40.9499 |      40.9716 |      40.9933 |      41.0152 |      41.0373 |      41.0597 |      41.0826 |      41.106  |      41.13   |      41.1548 |      41.1804 |      41.2068 |      41.2341 |      41.2624 |      41.2917 |      41.3219 |      41.3531 |      41.3852 |      41.4184 |      41.4525 |      41.4876 |      41.5236 |      41.5604 |      41.5982 |
        |            195 |      35.8422 |      35.8644 |      35.8867 |      35.9089 |      35.9311 |      35.9534 |      35.976  |      35.9989 |      36.0225 |      36.0469 |      36.0724 |      36.0991 |      36.1271 |      36.1566 |      36.1876 |      36.2202 |      36.2543 |      36.2899 |      36.3271 |      36.3657 |      36.4058 |      36.4473 |      36.4902 |      36.5343 |      36.5796 |      36.6262 |      36.6738 |      36.7225 |      36.7722 |
        |            200 |      30.8428 |      30.8656 |      30.8884 |      30.9112 |      30.9343 |      30.958  |      30.983  |      31.0095 |      31.0382 |      31.0691 |      31.1026 |      31.1387 |      31.1773 |      31.2185 |      31.2622 |      31.3082 |      31.3565 |      31.4068 |      31.4592 |      31.5134 |      31.5693 |      31.6268 |      31.6858 |      31.7462 |      31.8079 |      31.8707 |      31.9346 |      31.9995 |      32.0653 |
        |            205 |      25.8434 |      25.8667 |      25.8902 |      25.9144 |      25.9405 |      25.9699 |      26.0033 |      26.0414 |      26.0841 |      26.1314 |      26.183  |      26.2388 |      26.2982 |      26.361  |      26.4268 |      26.4954 |      26.5664 |      26.6396 |      26.7147 |      26.7915 |      26.8699 |      26.9496 |      27.0305 |      27.1125 |      27.1954 |      27.2792 |      27.3636 |      27.4486 |      27.5342 |
        |            210 |      20.8439 |      20.868  |      20.894  |      20.9257 |      20.9661 |      21.0159 |      21.0747 |      21.1418 |      21.216  |      21.2963 |      21.3818 |      21.4715 |      21.5648 |      21.6611 |      21.7599 |      21.8607 |      21.9631 |      22.0669 |      22.1718 |      22.2775 |      22.3839 |      22.4908 |      22.598  |      22.7055 |      22.8132 |      22.9209 |      23.0286 |      23.1362 |      23.2437 |
        |            215 |      15.8445 |      15.8732 |      15.9193 |      15.9878 |      16.0759 |      16.1792 |      16.2937 |      16.4164 |      16.5449 |      16.6777 |      16.8133 |      16.9509 |      17.0897 |      17.2291 |      17.3689 |      17.5086 |      17.648  |      17.7869 |      17.9253 |      18.0629 |      18.1997 |      18.3356 |      18.4707 |      18.6047 |      18.7378 |      18.87   |      19.0011 |      19.1313 |      19.2604 |
        |            220 |      10.8497 |      10.9335 |      11.0764 |      11.2516 |      11.4427 |      11.6409 |      11.8417 |      12.0423 |      12.2413 |      12.4381 |      12.632  |      12.8228 |      13.0106 |      13.1952 |      13.3767 |      13.5553 |      13.7308 |      13.9036 |      14.0736 |      14.241  |      14.4059 |      14.5683 |      14.7284 |      14.8863 |      15.042  |      15.1956 |      15.3473 |      15.497  |      15.6449 |
        |            225 |       5.9866 |       6.3402 |       6.7018 |       7.0454 |       7.3691 |       7.6746 |       7.9643 |       8.2401 |       8.5038 |       8.7569 |       9.0005 |       9.2356 |       9.4631 |       9.6836 |       9.8979 |      10.1064 |      10.3095 |      10.5078 |      10.7015 |      10.891  |      11.0765 |      11.2583 |      11.4366 |      11.6117 |      11.7836 |      11.9526 |      12.1189 |      12.2825 |      12.4436 |
        |            230 |       2.1081 |       2.7963 |       3.3303 |       3.7836 |       4.1849 |       4.5491 |       4.8853 |       5.1992 |       5.4949 |       5.7752 |       6.0426 |       6.2986 |       6.5446 |       6.7819 |       7.0113 |       7.2335 |       7.4493 |       7.6592 |       7.8637 |       8.0632 |       8.2581 |       8.4487 |       8.6352 |       8.8181 |       8.9973 |       9.1733 |       9.3462 |       9.5161 |       9.6832 |
        |            235 |       0.3531 |       0.8483 |       1.2906 |       1.6878 |       2.0507 |       2.3865 |       2.7006 |       2.9968 |       3.2779 |       3.546  |       3.8029 |       4.0499 |       4.288  |       4.5183 |       4.7415 |       4.9583 |       5.1691 |       5.3745 |       5.5749 |       5.7706 |       5.9621 |       6.1495 |       6.3332 |       6.5133 |       6.6901 |       6.8638 |       7.0345 |       7.2024 |       7.3677 |
        |            240 |       0.0228 |       0.1648 |       0.3758 |       0.6113 |       0.8537 |       1.0956 |       1.334  |       1.5674 |       1.7956 |       2.0182 |       2.2355 |       2.4476 |       2.6548 |       2.8574 |       3.0555 |       3.2495 |       3.4396 |       3.6259 |       3.8088 |       3.9883 |       4.1647 |       4.3381 |       4.5086 |       4.6765 |       4.8418 |       5.0046 |       5.1651 |       5.3234 |       5.4795 |
        |            245 |       0.0005 |       0.0197 |       0.0805 |       0.1774 |       0.2991 |       0.4371 |       0.5857 |       0.7409 |       0.9002 |       1.0619 |       1.2247 |       1.3879 |       1.5507 |       1.7129 |       1.8742 |       2.0342 |       2.193  |       2.3503 |       2.5062 |       2.6605 |       2.8134 |       2.9647 |       3.1146 |       3.2629 |       3.4097 |       3.5551 |       3.699  |       3.8416 |       3.9828 |
        |            250 |       0      |       0.0014 |       0.0126 |       0.041  |       0.0878 |       0.1511 |       0.228  |       0.3159 |       0.4125 |       0.5159 |       0.6248 |       0.7378 |       0.8541 |       0.9729 |       1.0937 |       1.216  |       1.3393 |       1.4634 |       1.588  |       1.7129 |       1.8379 |       1.9628 |       2.0877 |       2.2123 |       2.3366 |       2.4605 |       2.5839 |       2.7069 |       2.8294 |
        |            255 |       0      |       0.0001 |       0.0014 |       0.0075 |       0.0216 |       0.0453 |       0.0787 |       0.1215 |       0.1728 |       0.2315 |       0.2968 |       0.3677 |       0.4436 |       0.5237 |       0.6074 |       0.6942 |       0.7837 |       0.8754 |       0.9691 |       1.0645 |       1.1612 |       1.2592 |       1.3581 |       1.4579 |       1.5584 |       1.6595 |       1.761  |       1.8629 |       1.9651 |
        |            260 |       0      |       0      |       0.0001 |       0.0011 |       0.0045 |       0.0118 |       0.0241 |       0.0422 |       0.0662 |       0.096  |       0.1314 |       0.172  |       0.2174 |       0.2673 |       0.3212 |       0.3787 |       0.4396 |       0.5034 |       0.57   |       0.6389 |       0.71   |       0.7831 |       0.8579 |       0.9343 |       1.0121 |       1.0911 |       1.1713 |       1.2526 |       1.3347 |
        |            265 |       0      |       0      |       0      |       0.0001 |       0.0008 |       0.0027 |       0.0066 |       0.0133 |       0.0233 |       0.0369 |       0.0543 |       0.0756 |       0.1007 |       0.1295 |       0.1619 |       0.1976 |       0.2366 |       0.2785 |       0.3233 |       0.3706 |       0.4204 |       0.4724 |       0.5265 |       0.5825 |       0.6403 |       0.6998 |       0.7608 |       0.8233 |       0.887  |
        |            270 |       0      |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0016 |       0.0038 |       0.0075 |       0.0131 |       0.021  |       0.0313 |       0.0441 |       0.0596 |       0.0779 |       0.0988 |       0.1223 |       0.1484 |       0.177  |       0.2079 |       0.2412 |       0.2766 |       0.3141 |       0.3536 |       0.3949 |       0.4381 |       0.4829 |       0.5292 |       0.5771 |
        |            275 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.001  |       0.0022 |       0.0044 |       0.0076 |       0.0122 |       0.0183 |       0.0262 |       0.0358 |       0.0473 |       0.0608 |       0.0762 |       0.0936 |       0.113  |       0.1342 |       0.1574 |       0.1824 |       0.2091 |       0.2377 |       0.2678 |       0.2996 |       0.3329 |       0.3678 |
        |            280 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0006 |       0.0013 |       0.0026 |       0.0045 |       0.0072 |       0.0109 |       0.0157 |       0.0218 |       0.0291 |       0.0378 |       0.0479 |       0.0595 |       0.0725 |       0.0871 |       0.1031 |       0.1206 |       0.1396 |       0.1601 |       0.1819 |       0.2051 |       0.2297 |
        |            285 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0008 |       0.0016 |       0.0027 |       0.0044 |       0.0066 |       0.0096 |       0.0134 |       0.0181 |       0.0237 |       0.0304 |       0.0381 |       0.0469 |       0.0568 |       0.0679 |       0.0802 |       0.0936 |       0.1081 |       0.1239 |       0.1408 |
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

        - Implied Volatility = MINIMIZE(Black Scholes Theoretical Price — Actual Option Price)

        To determine the Implied Volatility, the Black Scholes Model is used to calculate the theoretical option price in
        which sigma (σ) is the only unknown variable. The actual option price is then used to determine the implied
        volatility by minimizing the difference between the theoretical and actual option price.

        The implied volatility is solved from each option's last traded price, which can be hours or days old for
        illiquid strikes, while the stock price is the latest close. Deep in-the-money and far out-of-the-money
        strikes therefore often show extreme values (above 100%) that say more about stale quotes than about the
        market's view. Practitioners read the smile around the money, where trading is liquid, and use
        `get_volatility_surface`, which filters outliers and fits a smooth curve, when they need values across all
        strikes.

        Also known as: IV, option-implied volatility.

        Args:
            expiration_date (str | None, optional): The expiration date to use for the calculation. Defaults to None
            which means it will use the most recent expiration date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["MSFT", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        implied_volatility = toolkit.options.get_implied_volatility()

        implied_volatility.loc['AAPL']
        ```

        Which returns:

        |   Strike Price |   2026-10-26 |
        |---------------:|-------------:|
        |          337.5 |       0.1856 |
        |          342.5 |       0.1964 |
        |          347.5 |       0.1877 |
        |          352.5 |       0.179  |
        |          357.5 |       0.1914 |
        |          362.5 |       0.2381 |
        |          367.5 |       0.2829 |
        |          372.5 |       0.3441 |
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
        SVI fit for that whole expiry.

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
            risk_free_rate (float, optional): The risk free rate to use for the
                calculation. Defaults to None which means it will use the current
                risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the
                calculation. Defaults to None, which uses the dividend yield on the
                pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        volatility_surface = toolkit.options.get_volatility_surface(number_of_expirations=3)

        volatility_surface.loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-10-26 |   2026-10-28 |   2026-10-30 |
        |---------------:|-------------:|-------------:|-------------:|
        |          327.5 |     nan      |     nan      |       0.2768 |
        |          332.5 |     nan      |     nan      |       0.2664 |
        |          337.5 |       0.2191 |       0.2362 |       0.2562 |
        |          342.5 |       0.1752 |       0.2238 |       0.2463 |
        |          347.5 |       0.1538 |       0.2143 |       0.2372 |
        |          352.5 |       0.1637 |       0.209  |       0.2292 |
        |          357.5 |       0.1989 |       0.209  |       0.2231 |
        |          362.5 |       0.2473 |       0.2157 |       0.2201 |
        |          367.5 |       0.3009 |       0.2299 |       0.2217 |
        |          372.5 |       0.3559 |       0.2516 |     nan      |
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
        numerically via a central finite difference on a fine, evenly-spaced strike
        grid, since the smile only gives implied volatility at a sparse set of
        traded strikes.

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
            risk_free_rate (float, optional): The risk free rate to use for the
                calculation. Defaults to None which means it will use the current
                risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the
                calculation. Defaults to None, which uses the dividend yield on the
                pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        risk_neutral_density = toolkit.options.get_risk_neutral_density()
        ```

        Which returns:

        |   Strike Price |   AAPL |
        |---------------:|-------:|
        |        478.033 |      0 |
        |        479.781 |      0 |
        |        481.529 |      0 |
        |        483.277 |      0 |
        |        485.024 |      0 |
        |        486.772 |      0 |
        |        488.52  |      0 |
        |        490.268 |      0 |
        |        492.016 |      0 |
        |        493.764 |      0 |
        |        495.511 |      0 |
        |        497.259 |      0 |
        |        499.007 |      0 |
        |        500.755 |      0 |
        |        502.503 |      0 |
        |        504.251 |      0 |
        |        505.998 |      0 |
        |        507.746 |      0 |
        |        509.494 |      0 |
        |        511.242 |      0 |
        |        512.99  |      0 |
        |        514.738 |      0 |
        |        516.485 |      0 |
        |        518.233 |      0 |
        |        519.981 |      0 |
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
        time_to_expiration: int = 1,
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

        - up movement (u) = e^(σ * sqrt(t))
        - down movement (d) = 1 / u
        - risk neutral probability (p) = (e^((r — q) * t) — d) / (u — d)
        - stock price at each node = S * u^j * d^(n — j)
        - call option price at expiration date = max(S — K, 0)
        - put option price at expiration date = max(K — S, 0)

        For European Style options:

        - call option price at each node = (p * C_u + (1 — p) * C_d) * e^(—r * t)
        - put option price at each node = (p * P_u + (1 — p) * P_d) * e^(—r * t)

        For American Style options:

        - call option price at each node = max(S — K, (p * C_u + (1 — p) * C_d) * e^(—r * t))
        - put option price at each node = max(K — S, (p * P_u + (1 — p) * P_d) * e^(—r * t))

        Where S is the stock price, K is the strike price, r is the risk free rate, σ is the volatility, t is the time to
        expiration, j is the number of up movements, n is the number of time steps, C_u is the call option price at the up
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
            time_to_expiration (int): The number of year to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        binomial_trees_model = toolkit.options.get_binomial_model(
            start_date='2024-02-02'
        )

        binomial_trees_model.loc['AAPL', 140]
        ```

        Which returns:

        | Movement   |   2024-02-02 |   2024-03-09 |   2024-04-15 |   2024-05-21 |   2024-06-27 |   2024-08-02 |   2024-09-08 |   2024-10-14 |   2024-11-20 |   2024-12-26 |   2025-02-01 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUUUUUUUU |      51.1633 |      62.3755 |      74.7064 |      88.0506 |     102.357  |     117.646  |     133.983  |     151.436  |     170.082  |     189.997  |     211.268  |
        | DUUUUUUUUU |     nan      |      39.8343 |      49.9666 |      61.3374 |      73.7833 |      87.1785 |     101.496  |     116.796  |     133.146  |     150.613  |     169.274  |
        | DDUUUUUUUU |     nan      |     nan      |      29.5435 |      38.4613 |      48.7994 |      60.3533 |      72.8927 |      86.2976 |     100.626  |     115.938  |     132.3    |
        | DDDUUUUUUU |     nan      |     nan      |     nan      |      20.4404 |      27.9435 |      37.0927 |      47.7091 |      59.445  |      71.9933 |      85.408  |      99.7467 |
        | DDDDUUUUUU |     nan      |     nan      |     nan      |     nan      |      12.7451 |      18.5848 |      26.2717 |      35.8026 |      46.784  |      58.5279 |      71.0851 |
        | DDDDDUUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |       6.7296 |      10.6813 |      16.4991 |      24.5884 |      34.8613 |      45.85   |
        | DDDDDDUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       2.6432 |       4.6716 |       8.1567 |      14.024  |      23.6317 |
        | DDDDDDDUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0.5389 |       1.0573 |       2.0743 |       4.0696 |
        | DDDDDDDDUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |       0      |
        | DDDDDDDDDU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |
        | DDDDDDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |
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
        time_to_expiration: int = 1,
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
        For example for below's example use `stock_price_simulation.loc['AMZN'].T.plot(legend=False)`

        Also known as: binomial tree, stock price tree, stock price path, lattice.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            time_to_expiration (int): The number of year to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate shown with show_input_info; it does not affect the
            simulated prices. Defaults to None which means it will use the current risk free rate.
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
            ["AMZN", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        stock_price_simulation = toolkit.options.get_stock_price_simulation(
            start_date='2025-06-20', timesteps=4
        )

        stock_price_simulation.loc['AMZN']
        ```

        Which returns:

        | Movement   |   2025-06-20 |   2025-09-19 |   2025-12-19 |   2026-03-20 |   2026-06-20 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUU       |       209.69 |      249.401 |      296.632 |      352.808 |      419.623 |
        | UUUD       |       209.69 |      249.401 |      296.632 |      352.808 |      296.632 |
        | UUDU       |       209.69 |      249.401 |      296.632 |      249.401 |      296.632 |
        | UUDD       |       209.69 |      249.401 |      296.632 |      249.401 |      209.69  |
        | UDUU       |       209.69 |      249.401 |      209.69  |      249.401 |      296.632 |
        | UDUD       |       209.69 |      249.401 |      209.69  |      249.401 |      209.69  |
        | UDDU       |       209.69 |      249.401 |      209.69  |      176.302 |      209.69  |
        | UDDD       |       209.69 |      249.401 |      209.69  |      176.302 |      148.23  |
        | DUUU       |       209.69 |      176.302 |      209.69  |      249.401 |      296.632 |
        | DUUD       |       209.69 |      176.302 |      209.69  |      249.401 |      209.69  |
        | DUDU       |       209.69 |      176.302 |      209.69  |      176.302 |      209.69  |
        | DUDD       |       209.69 |      176.302 |      209.69  |      176.302 |      148.23  |
        | DDUU       |       209.69 |      176.302 |      148.23  |      176.302 |      209.69  |
        | DDUD       |       209.69 |      176.302 |      148.23  |      176.302 |      148.23  |
        | DDDU       |       209.69 |      176.302 |      148.23  |      124.628 |      148.23  |
        | DDDD       |       209.69 |      176.302 |      148.23  |      124.628 |      104.784 |
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
        arbitrage:

        - C - P = S * e^(-q * t) - K * e^(-r * t)

        Where C is the call option price, P is the put option price, S is the stock
        price, K is the strike price, r is the risk-free rate, q is the dividend yield
        and t is the time to expiration.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        parity_gap = toolkit.options.get_put_call_parity()

        parity_gap.loc['AMZN']
        ```
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The domestic risk free rate to use for the calculation. Defaults to
            None which means it will use the current risk free rate.
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        garman_kohlhagen = toolkit.options.get_garman_kohlhagen(foreign_risk_free_rate=0.02)

        garman_kohlhagen.loc['AMZN']
        ```
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
        parameter:

        - "cash-or-nothing": pays a fixed cash amount if the option expires
          in-the-money.

            - Call = cash_payout * e^(—r * t) * N(d2)
            - Put = cash_payout * e^(—r * t) * N(—d2)

        - "asset-or-nothing": pays the value of the underlying asset if the option
          expires in-the-money.

            - Call = S * e^(—q * t) * N(d1)
            - Put = S * e^(—q * t) * N(—d1)

        Where S is the stock price, r is the risk-free rate, q is the dividend yield, t
        is the time to expiration, N(d1) is the cumulative normal distribution of d1 and
        N(d2) is the cumulative normal distribution of d2.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        binary_option = toolkit.options.get_binary_option()

        binary_option.loc['AMZN']
        ```
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
        Calculate American option prices using the Bjerksund-Stensland (1993)
        closed-form analytical approximation.

        Unlike European options, American options can be exercised at any time up to
        and including expiration, which normally requires a numerical approach such as
        the Binomial Tree model (see ``get_binomial_model``). The Bjerksund-Stensland
        model instead derives a closed-form approximation by assuming the early-exercise
        boundary is a flat trigger price: once the stock price crosses this level,
        immediate exercise is assumed optimal.

        The approximation (call, cost of carry b = r - q smaller than r) is:

        - β = (0.5 — b / σ²) + sqrt((b / σ² — 0.5)² + 2r / σ²)
        - B∞ = β / (β — 1) * K
        - B0 = max(K, r / (r — b) * K)
        - h(T) = —(b * T + 2σ√T) * (B0 / (B∞ — B0))
        - I = B0 + (B∞ — B0) * (1 — e^h(T))

        If S ≥ I, immediate exercise is optimal and the value is S — K. American puts
        are priced through the put-call transformation AmericanPut(S, K, T, r, b, σ) =
        AmericanCall(K, S, T, r — b, —b, σ).

        Also known as: BS93, Bjerksund-Stensland approximation, American option
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        bjerksund_stensland = toolkit.options.get_bjerksund_stensland()

        bjerksund_stensland.loc['AMZN']
        ```
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
        discounting the average payoff back to the present:

        - S(t + Δt) = S(t) * e^((r — q — σ²/2) * Δt + σ * √Δt * Z)
        - Call Price = e^(—r * T) * mean(max(S(T) — K, 0))
        - Put Price = e^(—r * T) * mean(max(K — S(T), 0))

        Where S(t) is the stock price at time t, r is the risk-free rate, q is the
        dividend yield, σ is the volatility, Δt is the length of a single time step and
        Z is a standard normal random variable.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        monte_carlo = toolkit.options.get_monte_carlo_option_price(seed=42)

        monte_carlo.loc['AMZN']
        ```
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

        A useful identity is in-out parity: for identical parameters, a knock-in option
        plus its corresponding knock-out option (same direction) always equals the price
        of the equivalent vanilla Black-Scholes option, since the underlying either does
        or does not touch the barrier.

        Also known as: knock-in option, knock-out option, down-and-out, down-and-in,
        up-and-out, up-and-in option.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            barrier_percentage (float, optional): The barrier level as a percentage of the current stock price.
            Defaults to 0.9 which equals 90% of the current stock price.
            barrier_direction (str, optional): Either "down" or "up". Defaults to "down".
            knock_type (str, optional): Either "in" or "out". Defaults to "out".
            rebate (float, optional): The fixed cash amount paid out if the option knocks out (or fails to knock
            in). Defaults to 0.0.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.DataFrame: Barrier option values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        barrier_option = toolkit.options.get_barrier_option()

        barrier_option.loc['AMZN']
        ```
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
        reduces variance). The geometric-average version has a closed-form solution
        based on an adjusted volatility and cost of carry:

        - σ_A = σ / √3
        - b_A = 0.5 * (b — σ²/6), where b = r — q is the cost of carry
        - d1 = (ln(S / K) + (b_A + σ_A²/2) * t) / (σ_A * √t)
        - d2 = d1 — σ_A * √t
        - Call Price = S * e^((b_A — r) * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Price = K * e^(—r * t) * N(—d2) — S * e^((b_A — r) * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk-free rate, q is
        the dividend yield, σ is the volatility, t is the time to expiration, N(d1) is
        the cumulative normal distribution of d1 and N(d2) is the cumulative normal
        distribution of d2.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        asian_option = toolkit.options.get_asian_option()

        asian_option.loc['AMZN']
        ```
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

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        straddle_legs = [
            {"strike_price": 150, "put_option": False, "position": "long", "premium": 8},
            {"strike_price": 150, "put_option": True, "position": "long", "premium": 6},
        ]

        strategy_payoff = toolkit.options.get_strategy_payoff(legs=straddle_legs)

        strategy_payoff["AMZN"]
        ```
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
        - Epsilon: measures the percentage change in option value per percentage change in the underlying dividend yield,
        a measure of the dividend risk.
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
        |            180 |  1      |      -0.9999 | 0      | -0.0193 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0198 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0204 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0209 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0      |
        |            200 |  1      |      -0.9999 | 0      | -0.0214 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      | -0      |   0      |  0      | -0      | -0      | 0      | -0      |  0      | -0      |   0.0001 |
        |            205 |  1      |      -0.9999 | 0      | -0.022  | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      | -0      |   0.0002 |  0.0001 | -0      | -0      | 0      | -0      |  0      | -0.0002 |   0.0071 |
        |            210 |  1      |      -0.9999 | 0      | -0.0225 | 0.5753 |   -0.6533 |   8.3747 |  0      |       0      | -0.0001 |   0.0124 |  0.0071 | -0.0001 | -0      | 0      | -0      |  0.0001 | -0.0079 |   0.2261 |
        |            215 |  0.9999 |      -0.9998 | 0.0001 | -0.0246 | 0.5889 |   -0.6532 |  10.1572 |  0.0001 |       0.0001 | -0.0031 |   0.3048 |  0.1423 | -0.002  | -0.0004 | 0.0001 | -0      |  0.0016 | -0.1539 |   2.7481 |
        |            220 |  0.9979 |      -0.9976 | 0.0008 | -0.0456 | 0.6013 |   -0.6519 |  12.8779 |  0.001  |       0.0011 | -0.0345 |   3.3887 |  1.2331 | -0.0227 | -0.0037 | 0.0011 | -0.0004 |  0.0129 | -1.2631 |  11.7208 |
        |            225 |  0.9808 |      -0.9794 | 0.0058 | -0.181  | 0.6037 |   -0.6408 |  17.2967 |  0.0069 |       0.0078 | -0.1766 |  17.3428 |  4.5653 | -0.117  | -0.0152 | 0.0078 | -0.0022 |  0.0414 | -4.0649 |  10.4133 |
        |            230 |  0.9022 |      -0.8971 | 0.0216 | -0.6042 | 0.5653 |   -0.5894 |  24.4761 |  0.0256 |       0.0275 | -0.4047 |  39.6546 |  6.5361 | -0.2703 | -0.0283 | 0.0275 | -0.005  |  0.0303 | -2.9367 | -16.4867 |
        |            235 |  0.7031 |      -0.6932 | 0.0432 | -1.1839 | 0.4463 |   -0.4593 |  35.2262 |  0.0513 |       0.0528 | -0.3236 |  31.4215 |  2.1543 | -0.2232 | -0.0273 | 0.0528 | -0.0043 | -0.0694 |  6.8814 | -10.9022 |
        |            240 |  0.4163 |      -0.4053 | 0.0487 | -1.3255 | 0.2665 |   -0.272  |  49.5363 |  0.0579 |       0.0571 |  0.173  | -17.5931 |  0.4561 |  0.0997 | -0.0256 | 0.0571 |  0.0016 | -0.1017 | 10.0099 |  -2.5038 |
        |            245 |  0.1735 |      -0.1663 | 0.032  | -0.8685 | 0.1116 |   -0.1133 |  66.7002 |  0.038  |       0.036  |  0.4598 | -45.6777 |  5.3971 |  0.2916 | -0.0307 | 0.036  |  0.0051 | -0.0063 |  0.5668 | -20.8795 |
        |            250 |  0.049  |      -0.0461 | 0.0127 | -0.3431 | 0.0316 |   -0.032  |  85.8267 |  0.015  |       0.0137 |  0.316  | -31.2945 |  6.5284 |  0.203  | -0.024  | 0.0137 |  0.0036 |  0.0497 | -4.9376 |  -2.5888 |
        |            255 |  0.0092 |      -0.0086 | 0.0031 | -0.0842 | 0.006  |   -0.006  | 106.168  |  0.0037 |       0.0032 |  0.1099 | -10.8646 |  3.2298 |  0.0709 | -0.0103 | 0.0032 |  0.0013 |  0.0315 | -3.1221 |  15.6348 |
        |            260 |  0.0012 |      -0.0011 | 0.0005 | -0.0132 | 0.0008 |   -0.0008 | 127.196  |  0.0006 |       0.0005 |  0.0222 |  -2.1911 |  0.8419 |  0.0144 | -0.0025 | 0.0005 |  0.0003 |  0.0089 | -0.8832 |   9.8836 |
        |            265 |  0.0001 |      -0.0001 | 0      | -0.0014 | 0.0001 |   -0.0001 | 148.56   |  0.0001 |       0      |  0.0028 |  -0.2741 |  0.1287 |  0.0018 | -0.0004 | 0      |  0      |  0.0014 | -0.1402 |   2.6002 |
        |            270 |  0      |      -0      | 0      | -0.0001 | 0      |   -0      | 170.036  |  0      |       0      |  0.0002 |  -0.0222 |  0.0123 |  0.0001 | -0      | 0      |  0      |  0.0001 | -0.0137 |   0.3704 |
        |            275 |  0      |      -0      | 0      | -0      | 0      |   -0      | 191.482  |  0      |       0      |  0      |  -0.0012 |  0.0008 |  0      | -0      | 0      |  0      |  0      | -0.0009 |   0.0318 |
        |            280 |  0      |      -0      | 0      | -0      | 0      |   -0      | 212.806  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0017 |
        |            285 |  0      |      -0      | 0      | -0      | 0      |   -0      | 233.948  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0.0001 |
        |            290 |  0      |      -0      | 0      | -0      | 0      |   -0      | 254.871  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
        |            295 |  0      |      -0      | 0      | -0      | 0      |   -0      | 275.551  |  0      |       0      |  0      |  -0      |  0      |  0      | -0      | 0      |  0      |  0      | -0      |   0      |
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
        - Epsilon: measures the percentage change in option value per percentage change in the underlying dividend yield,
        a measure of the dividend risk.
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
        |            290 |                                 0.0001 |                                     -0.0001 |                                     0 |                                -0.0007 |                               0.0001 |                                  -0.0001 |                                 250.947 |                                 0.0038 |                                     -0.0035 |                                0.0023 |                                -0.0185 |                               0.0056 |                                  -0.0056 |                                 136.159 |                                 0.0149 |                                     -0.0138 |                                0.0093 |                                -0.0506 |                               0.033  |                                  -0.0333 |                                 96.9271 |                                 0.0305 |                                     -0.0282 |                                0.0196 |                                -0.0803 |                               0.0896 |                                  -0.0908 |                                 76.827  |                                 0.0475 |                                     -0.0438 |                                0.0315 |                                -0.1034 |                               0.1741 |                                  -0.1769 |                                 64.4841 |                                 0.0645 |                                     -0.0594 |                                0.0439 |                                -0.1204 |                               0.283  |                                  -0.2881 |                                 56.0733 |                                 0.0807 |                                     -0.0742 |                                0.0564 |                                -0.1327 |                               0.4125 |                                  -0.4209 |                                 49.9401 |                                 0.096  |                                     -0.088  |                                0.0686 |                                -0.1416 |                               0.5595 |                                  -0.5721 |                                 45.2492 |                                 0.1103 |                                     -0.1009 |                                0.0804 |                                -0.1479 |                               0.7214 |                                  -0.7392 |                                 41.5325 |                                 0.1235 |                                     -0.1128 |                                0.0919 |                                -0.1523 |                               0.896  |                                  -0.9199 |                                 38.5062 |                                 0.1358 |                                     -0.1238 |                                0.1029 |                                -0.1553 |                               1.0817 |                                  -1.1126 |                                 35.9882 |                                 0.1472 |                                     -0.1339 |                                0.1135 |                                -0.1573 |                               1.2771 |                                  -1.316  |                                 33.8559 |                                 0.1579 |                                     -0.1434 |                                0.1237 |                                -0.1586 |                               1.481  |                                  -1.5287 |                                 32.0237 |                                 0.1678 |                                     -0.1522 |                                0.1336 |                                -0.1593 |                               1.6924 |                                  -1.7499 |                                 30.43   |                                 0.1771 |                                     -0.1603 |                                0.1431 |                                -0.1595 |                               1.9106 |                                  -1.9788 |                                 29.0291 |                                 0.1858 |                                     -0.1679 |                                0.1523 |                                -0.1594 |                               2.1349 |                                  -2.2146 |                                 27.7865 |                                 0.194  |                                     -0.1751 |                                0.1613 |                                -0.159  |                               2.3646 |                                  -2.4567 |                                 26.6756 |                                 0.2017 |                                     -0.1818 |                                0.1699 |                                -0.1584 |                               2.5993 |                                  -2.7047 |                                 25.6755 |                                 0.209  |                                     -0.188  |                                0.1783 |                                -0.1577 |                               2.8386 |                                  -2.9581 |                                 24.7697 |                                 0.2159 |                                     -0.194  |                                0.1864 |                                -0.1569 |                               3.0821 |                                  -3.2164 |                                 23.9447 |                                 0.2225 |                                     -0.1995 |                                0.1943 |                                -0.1559 |                               3.3294 |                                  -3.4794 |                                 23.1897 |                                 0.2287 |                                     -0.2048 |                                0.202  |                                -0.1549 |                               3.5802 |                                  -3.7468 |                                 22.4956 |                                 0.2346 |                                     -0.2098 |                                0.2095 |                                -0.1539 |                               3.8344 |                                  -4.0182 |                                 21.855  |                                 0.2402 |                                     -0.2146 |                                0.2168 |                                -0.1528 |                               4.0916 |                                  -4.2935 |                                 21.2615 |                                 0.2456 |                                     -0.2191 |                                0.2239 |                                -0.1517 |                               4.3516 |                                  -4.5724 |                                 20.7099 |                                 0.2507 |                                     -0.2234 |                                0.2309 |                                -0.1506 |                               4.6143 |                                  -4.8547 |                                 20.1956 |                                 0.2556 |                                     -0.2275 |                                0.2377 |                                -0.1495 |                               4.8795 |                                  -5.1402 |                                 19.7148 |                                 0.2603 |                                     -0.2314 |                                0.2444 |                                -0.1484 |                               5.147  |                                  -5.4288 |                                 19.2641 |                                 0.2648 |                                     -0.2351 |                                0.2509 |                                -0.1473 |                               5.4167 |                                  -5.7204 |                                 18.8406 |
        |            295 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 305.834 |                                 0.0004 |                                     -0.0003 |                                0.0003 |                                -0.0021 |                               0.0005 |                                  -0.0005 |                                 162.329 |                                 0.0029 |                                     -0.0027 |                                0.0022 |                                -0.0121 |                               0.0065 |                                  -0.0066 |                                113.794  |                                 0.0087 |                                     -0.0079 |                                0.0067 |                                -0.0275 |                               0.0257 |                                  -0.0259 |                                 89.1531 |                                 0.017  |                                     -0.0155 |                                0.0134 |                                -0.044  |                               0.0624 |                                  -0.0633 |                                 74.1409 |                                 0.0268 |                                     -0.0243 |                                0.0216 |                                -0.0591 |                               0.118  |                                  -0.1199 |                                 63.9817 |                                 0.0375 |                                     -0.0339 |                                0.0308 |                                -0.0723 |                               0.1919 |                                  -0.1953 |                                 56.6188 |                                 0.0484 |                                     -0.0437 |                                0.0404 |                                -0.0833 |                               0.2827 |                                  -0.2883 |                                 51.0184 |                                 0.0593 |                                     -0.0535 |                                0.0504 |                                -0.0924 |                               0.3888 |                                  -0.3974 |                                 46.6028 |                                 0.0699 |                                     -0.063  |                                0.0604 |                                -0.0998 |                               0.5089 |                                  -0.521  |                                 43.0236 |                                 0.0803 |                                     -0.0721 |                                0.0704 |                                -0.1059 |                               0.6414 |                                  -0.6579 |                                 40.0578 |                                 0.0903 |                                     -0.081  |                                0.0802 |                                -0.1108 |                               0.7853 |                                  -0.8067 |                                 37.5558 |                                 0.0998 |                                     -0.0894 |                                0.0899 |                                -0.1148 |                               0.9392 |                                  -0.9665 |                                 35.4135 |                                 0.109  |                                     -0.0974 |                                0.0994 |                                -0.1181 |                               1.1024 |                                  -1.1363 |                                 33.556  |                                 0.1177 |                                     -0.1051 |                                0.1088 |                                -0.1207 |                               1.2739 |                                  -1.3151 |                                 31.9281 |                                 0.1261 |                                     -0.1124 |                                0.1179 |                                -0.1228 |                               1.4531 |                                  -1.5023 |                                 30.4883 |                                 0.134  |                                     -0.1193 |                                0.1268 |                                -0.1245 |                               1.6392 |                                  -1.6973 |                                 29.2045 |                                 0.1417 |                                     -0.1259 |                                0.1355 |                                -0.1258 |                               1.8316 |                                  -1.8994 |                                 28.0517 |                                 0.149  |                                     -0.1322 |                                0.1439 |                                -0.1268 |                               2.03   |                                  -2.1081 |                                 27.01   |                                 0.1559 |                                     -0.1382 |                                0.1522 |                                -0.1276 |                               2.2338 |                                  -2.3229 |                                 26.0633 |                                 0.1626 |                                     -0.1439 |                                0.1603 |                                -0.1281 |                               2.4427 |                                  -2.5436 |                                 25.1988 |                                 0.169  |                                     -0.1494 |                                0.1683 |                                -0.1285 |                               2.6562 |                                  -2.7697 |                                 24.4056 |                                 0.1752 |                                     -0.1546 |                                0.176  |                                -0.1288 |                               2.874  |                                  -3.0008 |                                 23.675  |                                 0.1811 |                                     -0.1596 |                                0.1836 |                                -0.1289 |                               3.096  |                                  -3.2367 |                                 22.9993 |                                 0.1867 |                                     -0.1644 |                                0.191  |                                -0.1288 |                               3.3217 |                                  -3.4771 |                                 22.3724 |                                 0.1922 |                                     -0.169  |                                0.1983 |                                -0.1288 |                               3.551  |                                  -3.7218 |                                 21.7889 |                                 0.1974 |                                     -0.1734 |                                0.2054 |                                -0.1286 |                               3.7836 |                                  -3.9705 |                                 21.2442 |                                 0.2025 |                                     -0.1776 |                                0.2124 |                                -0.1283 |                               4.0194 |                                  -4.2231 |                                 20.7344 |                                 0.2074 |                                     -0.1817 |                                0.2192 |                                -0.128  |                               4.2582 |                                  -4.4793 |                                 20.2561 |
        |            300 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 360.985 |                                 0      |                                     -0      |                                0      |                                -0.0002 |                               0      |                                  -0      |                                 188.855 |                                 0.0004 |                                     -0.0004 |                                0.0004 |                                -0.0021 |                               0.001  |                                  -0.001  |                                130.965  |                                 0.002  |                                     -0.0018 |                                0.0018 |                                -0.0075 |                               0.006  |                                  -0.0061 |                                101.731  |                                 0.0052 |                                     -0.0046 |                                0.0048 |                                -0.0156 |                               0.0191 |                                  -0.0193 |                                 84.0084 |                                 0.0098 |                                     -0.0088 |                                0.0091 |                                -0.0249 |                               0.0432 |                                  -0.0438 |                                 72.069  |                                 0.0156 |                                     -0.0139 |                                0.0147 |                                -0.0345 |                               0.0799 |                                  -0.0812 |                                 63.4514 |                                 0.0222 |                                     -0.0197 |                                0.0213 |                                -0.0437 |                               0.1298 |                                  -0.1321 |                                 56.9214 |                                 0.0293 |                                     -0.026  |                                0.0285 |                                -0.0521 |                               0.1927 |                                  -0.1965 |                                 51.7909 |                                 0.0368 |                                     -0.0326 |                                0.0362 |                                -0.0597 |                               0.2682 |                                  -0.2739 |                                 47.6456 |                                 0.0444 |                                     -0.0393 |                                0.0443 |                                -0.0664 |                               0.3556 |                                  -0.3638 |                                 44.221  |                                 0.0521 |                                     -0.046  |                                0.0525 |                                -0.0724 |                               0.4542 |                                  -0.4654 |                                 41.3399 |                                 0.0597 |                                     -0.0527 |                                0.0609 |                                -0.0775 |                               0.5632 |                                  -0.5781 |                                 38.8795 |                                 0.0672 |                                     -0.0593 |                                0.0693 |                                -0.0821 |                               0.682  |                                  -0.7011 |                                 36.7515 |                                 0.0746 |                                     -0.0657 |                                0.0777 |                                -0.086  |                               0.8098 |                                  -0.8337 |                                 34.8908 |                                 0.0818 |                                     -0.0719 |                                0.0861 |                                -0.0894 |                               0.946  |                                  -0.9753 |                                 33.2486 |                                 0.0889 |                                     -0.078  |                                0.0944 |                                -0.0924 |                               1.09   |                                  -1.1254 |                                 31.7874 |                                 0.0957 |                                     -0.0839 |                                0.1026 |                                -0.095  |                               1.2413 |                                  -1.2834 |                                 30.4779 |                                 0.1024 |                                     -0.0896 |                                0.1107 |                                -0.0972 |                               1.3994 |                                  -1.4489 |                                 29.2967 |                                 0.1088 |                                     -0.0951 |                                0.1187 |                                -0.0992 |                               1.5638 |                                  -1.6213 |                                 28.2252 |                                 0.1151 |                                     -0.1005 |                                0.1266 |                                -0.1009 |                               1.7342 |                                  -1.8003 |                                 27.2483 |                                 0.1212 |                                     -0.1056 |                                0.1344 |                                -0.1023 |                               1.9101 |                                  -1.9855 |                                 26.3535 |                                 0.1271 |                                     -0.1106 |                                0.1421 |                                -0.1036 |                               2.0913 |                                  -2.1765 |                                 25.5304 |                                 0.1328 |                                     -0.1154 |                                0.1496 |                                -0.1046 |                               2.2773 |                                  -2.3731 |                                 24.7705 |                                 0.1383 |                                     -0.1201 |                                0.157  |                                -0.1055 |                               2.4679 |                                  -2.5749 |                                 24.0664 |                                 0.1436 |                                     -0.1246 |                                0.1643 |                                -0.1063 |                               2.6629 |                                  -2.7818 |                                 23.4119 |                                 0.1488 |                                     -0.129  |                                0.1715 |                                -0.107  |                               2.8621 |                                  -2.9933 |                                 22.8017 |                                 0.1539 |                                     -0.1332 |                                0.1786 |                                -0.1075 |                               3.0651 |                                  -3.2094 |                                 22.2314 |                                 0.1588 |                                     -0.1373 |                                0.1855 |                                -0.1079 |                               3.2717 |                                  -3.4298 |                                 21.6968 |
        |            305 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 415.986 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 215.493 |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0003 |                               0.0001 |                                  -0.0001 |                                148.279  |                                 0.0004 |                                     -0.0003 |                                0.0004 |                                -0.0016 |                               0.0012 |                                  -0.0012 |                                114.447  |                                 0.0014 |                                     -0.0012 |                                0.0014 |                                -0.0046 |                               0.005  |                                  -0.005  |                                 94.0014 |                                 0.0032 |                                     -0.0028 |                                0.0033 |                                -0.0091 |                               0.0139 |                                  -0.0141 |                                 80.2685 |                                 0.0058 |                                     -0.0051 |                                0.0062 |                                -0.0146 |                               0.0299 |                                  -0.0303 |                                 70.3844 |                                 0.0093 |                                     -0.0081 |                                0.01   |                                -0.0206 |                               0.0543 |                                  -0.0551 |                                 62.9144 |                                 0.0134 |                                     -0.0117 |                                0.0146 |                                -0.0267 |                               0.088  |                                  -0.0895 |                                 57.0599 |                                 0.018  |                                     -0.0157 |                                0.0199 |                                -0.0328 |                               0.1314 |                                  -0.1339 |                                 52.3407 |                                 0.023  |                                     -0.0201 |                                0.0257 |                                -0.0385 |                               0.1846 |                                  -0.1884 |                                 48.4506 |                                 0.0283 |                                     -0.0247 |                                0.032  |                                -0.044  |                               0.2474 |                                  -0.253  |                                 45.1847 |                                 0.0338 |                                     -0.0294 |                                0.0385 |                                -0.049  |                               0.3197 |                                  -0.3275 |                                 42.4012 |                                 0.0395 |                                     -0.0343 |                                0.0454 |                                -0.0536 |                               0.4011 |                                  -0.4114 |                                 39.9981 |                                 0.0452 |                                     -0.0392 |                                0.0524 |                                -0.0578 |                               0.4912 |                                  -0.5045 |                                 37.9007 |                                 0.0509 |                                     -0.0441 |                                0.0595 |                                -0.0617 |                               0.5896 |                                  -0.6065 |                                 36.0527 |                                 0.0566 |                                     -0.049  |                                0.0668 |                                -0.0652 |                               0.6959 |                                  -0.7168 |                                 34.411  |                                 0.0623 |                                     -0.0538 |                                0.074  |                                -0.0684 |                               0.8097 |                                  -0.8351 |                                 32.9419 |                                 0.0679 |                                     -0.0586 |                                0.0814 |                                -0.0713 |                               0.9306 |                                  -0.961  |                                 31.6188 |                                 0.0735 |                                     -0.0633 |                                0.0887 |                                -0.0739 |                               1.0583 |                                  -1.0942 |                                 30.4202 |                                 0.0789 |                                     -0.0679 |                                0.0959 |                                -0.0762 |                               1.1923 |                                  -1.2343 |                                 29.3289 |                                 0.0843 |                                     -0.0725 |                                0.1032 |                                -0.0783 |                               1.3323 |                                  -1.3811 |                                 28.3306 |                                 0.0895 |                                     -0.0769 |                                0.1104 |                                -0.0802 |                               1.4781 |                                  -1.5341 |                                 27.4135 |                                 0.0947 |                                     -0.0812 |                                0.1176 |                                -0.082  |                               1.6293 |                                  -1.6931 |                                 26.5677 |                                 0.0998 |                                     -0.0855 |                                0.1246 |                                -0.0835 |                               1.7857 |                                  -1.8578 |                                 25.7849 |                                 0.1047 |                                     -0.0896 |                                0.1317 |                                -0.0849 |                               1.9471 |                                  -2.028  |                                 25.0582 |                                 0.1096 |                                     -0.0937 |                                0.1386 |                                -0.0862 |                               2.113  |                                  -2.2034 |                                 24.3813 |                                 0.1143 |                                     -0.0976 |                                0.1455 |                                -0.0873 |                               2.2835 |                                  -2.3838 |                                 23.7493 |                                 0.1189 |                                     -0.1014 |                                0.1523 |                                -0.0883 |                               2.4581 |                                  -2.5691 |                                 23.1575 |
        |            310 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 470.601 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 242.09  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                165.628  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                127.22   |                                 0.0003 |                                     -0.0003 |                                0.0004 |                                -0.0012 |                               0.0011 |                                  -0.0011 |                                104.056  |                                 0.0009 |                                     -0.0008 |                                0.0011 |                                -0.0029 |                               0.004  |                                  -0.004  |                                 88.5295 |                                 0.002  |                                     -0.0017 |                                0.0023 |                                -0.0055 |                               0.0101 |                                  -0.0102 |                                 77.376  |                                 0.0035 |                                     -0.0031 |                                0.0043 |                                -0.0087 |                               0.0207 |                                  -0.021  |                                 68.9624 |                                 0.0056 |                                     -0.0049 |                                0.0068 |                                -0.0125 |                               0.0371 |                                  -0.0377 |                                 62.3802 |                                 0.0082 |                                     -0.0071 |                                0.0101 |                                -0.0165 |                               0.0599 |                                  -0.061  |                                 57.0835 |                                 0.0112 |                                     -0.0096 |                                0.0139 |                                -0.0207 |                               0.0898 |                                  -0.0916 |                                 52.7245 |                                 0.0145 |                                     -0.0125 |                                0.0182 |                                -0.0249 |                               0.1272 |                                  -0.1298 |                                 49.0707 |                                 0.0182 |                                     -0.0156 |                                0.0229 |                                -0.029  |                               0.1721 |                                  -0.1759 |                                 45.9611 |                                 0.0221 |                                     -0.0189 |                                0.028  |                                -0.033  |                               0.2246 |                                  -0.2299 |                                 43.2805 |                                 0.0261 |                                     -0.0223 |                                0.0334 |                                -0.0368 |                               0.2847 |                                  -0.2918 |                                 40.944  |                                 0.0303 |                                     -0.0259 |                                0.0391 |                                -0.0404 |                               0.3523 |                                  -0.3616 |                                 38.888  |                                 0.0347 |                                     -0.0296 |                                0.045  |                                -0.0438 |                               0.427  |                                  -0.4389 |                                 37.0638 |                                 0.0391 |                                     -0.0333 |                                0.051  |                                -0.047  |                               0.5089 |                                  -0.5237 |                                 35.4334 |                                 0.0435 |                                     -0.037  |                                0.0572 |                                -0.05   |                               0.5976 |                                  -0.6157 |                                 33.9667 |                                 0.048  |                                     -0.0408 |                                0.0635 |                                -0.0528 |                               0.6928 |                                  -0.7147 |                                 32.6396 |                                 0.0525 |                                     -0.0445 |                                0.0699 |                                -0.0554 |                               0.7944 |                                  -0.8205 |                                 31.4325 |                                 0.0569 |                                     -0.0483 |                                0.0763 |                                -0.0578 |                               0.902  |                                  -0.9328 |                                 30.3294 |                                 0.0614 |                                     -0.052  |                                0.0827 |                                -0.06   |                               1.0156 |                                  -1.0514 |                                 29.3171 |                                 0.0658 |                                     -0.0557 |                                0.0892 |                                -0.062  |                               1.1347 |                                  -1.1761 |                                 28.3844 |                                 0.0702 |                                     -0.0593 |                                0.0957 |                                -0.064  |                               1.2592 |                                  -1.3067 |                                 27.522  |                                 0.0745 |                                     -0.0629 |                                0.1022 |                                -0.0657 |                               1.3889 |                                  -1.4429 |                                 26.722  |                                 0.0788 |                                     -0.0664 |                                0.1087 |                                -0.0674 |                               1.5236 |                                  -1.5846 |                                 25.9777 |                                 0.083  |                                     -0.0699 |                                0.1151 |                                -0.0689 |                               1.663  |                                  -1.7315 |                                 25.2831 |                                 0.0872 |                                     -0.0734 |                                0.1215 |                                -0.0703 |                               1.807  |                                  -1.8834 |                                 24.6335 |
        |            315 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 524.698 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 268.547 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                182.938  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                139.992  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                114.127  |                                 0.0002 |                                     -0.0002 |                                0.0003 |                                -0.0008 |                               0.001  |                                  -0.001  |                                 96.8137 |                                 0.0006 |                                     -0.0005 |                                0.0008 |                                -0.0018 |                               0.0031 |                                  -0.0031 |                                 84.3941 |                                 0.0012 |                                     -0.0011 |                                0.0016 |                                -0.0034 |                               0.0073 |                                  -0.0073 |                                 75.0381 |                                 0.0022 |                                     -0.0019 |                                0.0029 |                                -0.0054 |                               0.0145 |                                  -0.0147 |                                 67.7283 |                                 0.0035 |                                     -0.003  |                                0.0047 |                                -0.0077 |                               0.0255 |                                  -0.0259 |                                 61.8535 |                                 0.0051 |                                     -0.0043 |                                0.007  |                                -0.0104 |                               0.0411 |                                  -0.0418 |                                 57.0247 |                                 0.007  |                                     -0.006  |                                0.0097 |                                -0.0132 |                               0.0618 |                                  -0.063  |                                 52.9819 |                                 0.0093 |                                     -0.0078 |                                0.0128 |                                -0.0162 |                               0.0879 |                                  -0.0898 |                                 49.5451 |                                 0.0118 |                                     -0.0099 |                                0.0163 |                                -0.0192 |                               0.1199 |                                  -0.1225 |                                 46.5857 |                                 0.0145 |                                     -0.0122 |                                0.0202 |                                -0.0223 |                               0.1579 |                                  -0.1616 |                                 44.009  |                                 0.0174 |                                     -0.0146 |                                0.0244 |                                -0.0252 |                               0.202  |                                  -0.2069 |                                 41.744  |                                 0.0204 |                                     -0.0172 |                                0.0289 |                                -0.0281 |                               0.2522 |                                  -0.2587 |                                 39.7363 |                                 0.0236 |                                     -0.0199 |                                0.0337 |                                -0.031  |                               0.3085 |                                  -0.3168 |                                 37.9436 |                                 0.0269 |                                     -0.0226 |                                0.0386 |                                -0.0337 |                               0.3709 |                                  -0.3814 |                                 36.3324 |                                 0.0304 |                                     -0.0254 |                                0.0437 |                                -0.0363 |                               0.4392 |                                  -0.4522 |                                 34.8758 |                                 0.0338 |                                     -0.0283 |                                0.049  |                                -0.0387 |                               0.5134 |                                  -0.5292 |                                 33.5522 |                                 0.0374 |                                     -0.0313 |                                0.0544 |                                -0.0411 |                               0.5933 |                                  -0.6123 |                                 32.3436 |                                 0.0409 |                                     -0.0342 |                                0.0599 |                                -0.0433 |                               0.6788 |                                  -0.7013 |                                 31.2353 |                                 0.0445 |                                     -0.0372 |                                0.0655 |                                -0.0455 |                               0.7698 |                                  -0.7961 |                                 30.215  |                                 0.0482 |                                     -0.0401 |                                0.0712 |                                -0.0475 |                               0.866  |                                  -0.8966 |                                 29.2723 |                                 0.0518 |                                     -0.0431 |                                0.0769 |                                -0.0494 |                               0.9673 |                                  -1.0026 |                                 28.3985 |                                 0.0554 |                                     -0.0461 |                                0.0827 |                                -0.0512 |                               1.0736 |                                  -1.114  |                                 27.5861 |                                 0.059  |                                     -0.049  |                                0.0885 |                                -0.0528 |                               1.1847 |                                  -1.2305 |                                 26.8286 |                                 0.0626 |                                     -0.052  |                                0.0944 |                                -0.0544 |                               1.3004 |                                  -1.3522 |                                 26.1204 |
        |            320 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 578.198 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 294.801 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                200.157  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                152.72   |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                124.179  |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                105.092  |                                 0.0002 |                                     -0.0001 |                                0.0002 |                                -0.0006 |                               0.0008 |                                  -0.0009 |                                 91.4142 |                                 0.0004 |                                     -0.0003 |                                0.0006 |                                -0.0012 |                               0.0023 |                                  -0.0024 |                                 81.1204 |                                 0.0008 |                                     -0.0007 |                                0.0012 |                                -0.0021 |                               0.0052 |                                  -0.0053 |                                 73.0856 |                                 0.0014 |                                     -0.0012 |                                0.002  |                                -0.0033 |                               0.0102 |                                  -0.0103 |                                 66.6343 |                                 0.0022 |                                     -0.0018 |                                0.0033 |                                -0.0049 |                               0.0177 |                                  -0.018  |                                 61.3365 |                                 0.0032 |                                     -0.0027 |                                0.0048 |                                -0.0066 |                               0.0284 |                                  -0.0289 |                                 56.9052 |                                 0.0045 |                                     -0.0038 |                                0.0068 |                                -0.0085 |                               0.0428 |                                  -0.0436 |                                 53.1415 |                                 0.006  |                                     -0.005  |                                0.009  |                                -0.0106 |                               0.0611 |                                  -0.0624 |                                 49.9032 |                                 0.0077 |                                     -0.0064 |                                0.0116 |                                -0.0128 |                               0.0839 |                                  -0.0857 |                                 47.0861 |                                 0.0096 |                                     -0.0079 |                                0.0146 |                                -0.015  |                               0.1113 |                                  -0.1138 |                                 44.6117 |                                 0.0116 |                                     -0.0096 |                                0.0178 |                                -0.0173 |                               0.1434 |                                  -0.1469 |                                 42.4203 |                                 0.0138 |                                     -0.0114 |                                0.0213 |                                -0.0196 |                               0.1806 |                                  -0.1851 |                                 40.4649 |                                 0.0162 |                                     -0.0134 |                                0.025  |                                -0.0218 |                               0.2227 |                                  -0.2286 |                                 38.7089 |                                 0.0186 |                                     -0.0154 |                                0.029  |                                -0.024  |                               0.2699 |                                  -0.2774 |                                 37.1225 |                                 0.0212 |                                     -0.0175 |                                0.0332 |                                -0.0262 |                               0.3222 |                                  -0.3315 |                                 35.6819 |                                 0.0239 |                                     -0.0197 |                                0.0375 |                                -0.0283 |                               0.3795 |                                  -0.3909 |                                 34.3674 |                                 0.0266 |                                     -0.0219 |                                0.042  |                                -0.0303 |                               0.4418 |                                  -0.4556 |                                 33.1628 |                                 0.0294 |                                     -0.0242 |                                0.0466 |                                -0.0323 |                               0.5091 |                                  -0.5255 |                                 32.0546 |                                 0.0323 |                                     -0.0265 |                                0.0514 |                                -0.0342 |                               0.5812 |                                  -0.6006 |                                 31.0313 |                                 0.0352 |                                     -0.0289 |                                0.0563 |                                -0.036  |                               0.6582 |                                  -0.6808 |                                 30.0834 |                                 0.0381 |                                     -0.0313 |                                0.0612 |                                -0.0378 |                               0.7398 |                                  -0.766  |                                 29.2025 |                                 0.0411 |                                     -0.0337 |                                0.0663 |                                -0.0395 |                               0.826  |                                  -0.8562 |                                 28.3817 |                                 0.044  |                                     -0.0361 |                                0.0714 |                                -0.0411 |                               0.9168 |                                  -0.9512 |                                 27.6149 |
        |            325 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 631.057 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 320.809 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                217.25   |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                165.377  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                134.186  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                113.344  |                                 0      |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                 98.4175 |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0004 |                               0.0007 |                                  -0.0007 |                                 87.1926 |                                 0.0003 |                                     -0.0002 |                                0.0004 |                                -0.0008 |                               0.0018 |                                  -0.0018 |                                 78.4375 |                                 0.0005 |                                     -0.0004 |                                0.0008 |                                -0.0014 |                               0.0038 |                                  -0.0039 |                                 71.413  |                                 0.0009 |                                     -0.0007 |                                0.0014 |                                -0.0021 |                               0.0072 |                                  -0.0073 |                                 65.6485 |                                 0.0014 |                                     -0.0012 |                                0.0023 |                                -0.0031 |                               0.0124 |                                  -0.0126 |                                 60.8301 |                                 0.0021 |                                     -0.0017 |                                0.0034 |                                -0.0043 |                               0.0198 |                                  -0.0202 |                                 56.7405 |                                 0.0029 |                                     -0.0024 |                                0.0047 |                                -0.0056 |                               0.0298 |                                  -0.0304 |                                 53.2243 |                                 0.0039 |                                     -0.0032 |                                0.0064 |                                -0.007  |                               0.0428 |                                  -0.0436 |                                 50.1673 |                                 0.0051 |                                     -0.0041 |                                0.0083 |                                -0.0086 |                               0.059  |                                  -0.0603 |                                 47.484  |                                 0.0064 |                                     -0.0052 |                                0.0105 |                                -0.0102 |                               0.0787 |                                  -0.0805 |                                 45.109  |                                 0.0078 |                                     -0.0064 |                                0.0129 |                                -0.0119 |                               0.1022 |                                  -0.1046 |                                 42.9912 |                                 0.0094 |                                     -0.0077 |                                0.0156 |                                -0.0136 |                               0.1295 |                                  -0.1328 |                                 41.0904 |                                 0.0111 |                                     -0.009  |                                0.0186 |                                -0.0153 |                               0.1609 |                                  -0.1651 |                                 39.3742 |                                 0.0129 |                                     -0.0105 |                                0.0217 |                                -0.0171 |                               0.1965 |                                  -0.2018 |                                 37.8166 |                                 0.0148 |                                     -0.0121 |                                0.025  |                                -0.0188 |                               0.2362 |                                  -0.2429 |                                 36.3962 |                                 0.0168 |                                     -0.0137 |                                0.0285 |                                -0.0206 |                               0.2802 |                                  -0.2885 |                                 35.0952 |                                 0.0189 |                                     -0.0154 |                                0.0322 |                                -0.0223 |                               0.3285 |                                  -0.3385 |                                 33.899  |                                 0.0211 |                                     -0.0171 |                                0.0361 |                                -0.024  |                               0.3811 |                                  -0.3931 |                                 32.795  |                                 0.0233 |                                     -0.0189 |                                0.04   |                                -0.0256 |                               0.4379 |                                  -0.4521 |                                 31.7729 |                                 0.0256 |                                     -0.0208 |                                0.0441 |                                -0.0272 |                               0.4989 |                                  -0.5157 |                                 30.8235 |                                 0.028  |                                     -0.0226 |                                0.0484 |                                -0.0288 |                               0.5642 |                                  -0.5837 |                                 29.9393 |                                 0.0304 |                                     -0.0245 |                                0.0527 |                                -0.0303 |                               0.6336 |                                  -0.6561 |                                 29.1136 |
        |            330 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 683.252 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 346.546 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                234.193  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                177.94   |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                144.132  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                121.551  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                105.39   |                                 0      |                                     -0      |                                0.0001 |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 93.2421 |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0006 |                                  -0.0006 |                                 83.7726 |                                 0.0002 |                                     -0.0001 |                                0.0003 |                                -0.0005 |                               0.0013 |                                  -0.0014 |                                 76.1792 |                                 0.0003 |                                     -0.0003 |                                0.0006 |                                -0.0009 |                               0.0028 |                                  -0.0028 |                                 69.9512 |                                 0.0006 |                                     -0.0005 |                                0.001  |                                -0.0014 |                               0.0051 |                                  -0.0052 |                                 64.7483 |                                 0.0009 |                                     -0.0007 |                                0.0016 |                                -0.002  |                               0.0088 |                                  -0.0089 |                                 60.3347 |                                 0.0014 |                                     -0.0011 |                                0.0024 |                                -0.0028 |                               0.0139 |                                  -0.0142 |                                 56.5418 |                                 0.0019 |                                     -0.0015 |                                0.0034 |                                -0.0037 |                               0.021  |                                  -0.0214 |                                 53.2461 |                                 0.0026 |                                     -0.0021 |                                0.0045 |                                -0.0047 |                               0.0301 |                                  -0.0307 |                                 50.3548 |                                 0.0034 |                                     -0.0027 |                                0.006  |                                -0.0058 |                               0.0417 |                                  -0.0426 |                                 47.7969 |                                 0.0043 |                                     -0.0034 |                                0.0076 |                                -0.0069 |                               0.056  |                                  -0.0572 |                                 45.5172 |                                 0.0053 |                                     -0.0043 |                                0.0094 |                                -0.0082 |                               0.0731 |                                  -0.0748 |                                 43.472  |                                 0.0064 |                                     -0.0052 |                                0.0115 |                                -0.0095 |                               0.0932 |                                  -0.0955 |                                 41.6264 |                                 0.0076 |                                     -0.0061 |                                0.0137 |                                -0.0108 |                               0.1166 |                                  -0.1196 |                                 39.9521 |                                 0.009  |                                     -0.0072 |                                0.0162 |                                -0.0122 |                               0.1433 |                                  -0.1471 |                                 38.426  |                                 0.0104 |                                     -0.0083 |                                0.0188 |                                -0.0136 |                               0.1734 |                                  -0.1782 |                                 37.0289 |                                 0.0119 |                                     -0.0095 |                                0.0216 |                                -0.0149 |                               0.207  |                                  -0.213  |                                 35.7447 |                                 0.0135 |                                     -0.0108 |                                0.0246 |                                -0.0163 |                               0.2442 |                                  -0.2515 |                                 34.5602 |                                 0.0152 |                                     -0.0121 |                                0.0278 |                                -0.0177 |                               0.2851 |                                  -0.2939 |                                 33.4639 |                                 0.0169 |                                     -0.0135 |                                0.031  |                                -0.0191 |                               0.3296 |                                  -0.34   |                                 32.4461 |                                 0.0187 |                                     -0.0149 |                                0.0344 |                                -0.0204 |                               0.3777 |                                  -0.3901 |                                 31.4985 |                                 0.0206 |                                     -0.0164 |                                0.038  |                                -0.0218 |                               0.4296 |                                  -0.4441 |                                 30.6139 |
        |            335 |                                 0      |                                     -0      |                                     0 |                                -0      |                               0      |                                  -0      |                                 734.773 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 371.994 |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                250.971  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                190.394  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                154.001  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                129.704  |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                112.32   |                                 0      |                                     -0      |                                0      |                                -0      |                               0      |                                  -0      |                                 99.2592 |                                 0      |                                     -0      |                                0      |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 89.0822 |                                 0.0001 |                                     -0      |                                0.0001 |                                -0.0002 |                               0.0004 |                                  -0.0004 |                                 80.9248 |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0003 |                               0.001  |                                  -0.001  |                                 74.2373 |                                 0.0002 |                                     -0.0002 |                                0.0004 |                                -0.0006 |                               0.002  |                                  -0.0021 |                                 68.6529 |                                 0.0004 |                                     -0.0003 |                                0.0007 |                                -0.0009 |                               0.0037 |                                  -0.0038 |                                 63.9176 |                                 0.0006 |                                     -0.0005 |                                0.0011 |                                -0.0013 |                               0.0062 |                                  -0.0064 |                                 59.8501 |                                 0.0009 |                                     -0.0007 |                                0.0017 |                                -0.0018 |                               0.0099 |                                  -0.0101 |                                 56.3172 |                                 0.0013 |                                     -0.001  |                                0.0024 |                                -0.0025 |                               0.0149 |                                  -0.0151 |                                 53.2191 |                                 0.0017 |                                     -0.0014 |                                0.0032 |                                -0.0031 |                               0.0214 |                                  -0.0218 |                                 50.4794 |                                 0.0023 |                                     -0.0018 |                                0.0043 |                                -0.0039 |                               0.0297 |                                  -0.0303 |                                 48.0386 |                                 0.0029 |                                     -0.0023 |                                0.0055 |                                -0.0048 |                               0.04   |                                  -0.0409 |                                 45.8497 |                                 0.0036 |                                     -0.0029 |                                0.0069 |                                -0.0057 |                               0.0525 |                                  -0.0537 |                                 43.8753 |                                 0.0044 |                                     -0.0035 |                                0.0084 |                                -0.0066 |                               0.0674 |                                  -0.069  |                                 42.0849 |                                 0.0053 |                                     -0.0042 |                                0.0102 |                                -0.0076 |                               0.0847 |                                  -0.0869 |                                 40.4534 |                                 0.0063 |                                     -0.005  |                                0.0121 |                                -0.0087 |                               0.1047 |                                  -0.1075 |                                 38.9604 |                                 0.0073 |                                     -0.0058 |                                0.0141 |                                -0.0098 |                               0.1275 |                                  -0.131  |                                 37.5887 |                                 0.0085 |                                     -0.0067 |                                0.0164 |                                -0.0108 |                               0.1532 |                                  -0.1575 |                                 36.3239 |                                 0.0097 |                                     -0.0076 |                                0.0188 |                                -0.012  |                               0.1818 |                                  -0.1871 |                                 35.1536 |                                 0.0109 |                                     -0.0086 |                                0.0213 |                                -0.0131 |                               0.2134 |                                  -0.2198 |                                 34.0676 |                                 0.0123 |                                     -0.0097 |                                0.024  |                                -0.0142 |                               0.2481 |                                  -0.2558 |                                 33.0568 |                                 0.0137 |                                     -0.0107 |                                0.0268 |                                -0.0153 |                               0.2859 |                                  -0.2951 |                                 32.1135 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_delta().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0038 |       0.0149 |       0.0305 |       0.0475 |       0.0645 |       0.0807 |       0.096  |       0.1103 |       0.1235 |       0.1358 |       0.1472 |       0.1579 |       0.1678 |       0.1771 |       0.1858 |       0.194  |       0.2017 |       0.209  |       0.2159 |       0.2225 |       0.2287 |       0.2346 |       0.2402 |       0.2456 |       0.2507 |       0.2556 |       0.2603 |       0.2648 |
        |            295 |       0      |       0.0004 |       0.0029 |       0.0087 |       0.017  |       0.0268 |       0.0375 |       0.0484 |       0.0593 |       0.0699 |       0.0803 |       0.0903 |       0.0998 |       0.109  |       0.1177 |       0.1261 |       0.134  |       0.1417 |       0.149  |       0.1559 |       0.1626 |       0.169  |       0.1752 |       0.1811 |       0.1867 |       0.1922 |       0.1974 |       0.2025 |       0.2074 |
        |            300 |       0      |       0      |       0.0004 |       0.002  |       0.0052 |       0.0098 |       0.0156 |       0.0222 |       0.0293 |       0.0368 |       0.0444 |       0.0521 |       0.0597 |       0.0672 |       0.0746 |       0.0818 |       0.0889 |       0.0957 |       0.1024 |       0.1088 |       0.1151 |       0.1212 |       0.1271 |       0.1328 |       0.1383 |       0.1436 |       0.1488 |       0.1539 |       0.1588 |
        |            305 |       0      |       0      |       0.0001 |       0.0004 |       0.0014 |       0.0032 |       0.0058 |       0.0093 |       0.0134 |       0.018  |       0.023  |       0.0283 |       0.0338 |       0.0395 |       0.0452 |       0.0509 |       0.0566 |       0.0623 |       0.0679 |       0.0735 |       0.0789 |       0.0843 |       0.0895 |       0.0947 |       0.0998 |       0.1047 |       0.1096 |       0.1143 |       0.1189 |
        |            310 |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0009 |       0.002  |       0.0035 |       0.0056 |       0.0082 |       0.0112 |       0.0145 |       0.0182 |       0.0221 |       0.0261 |       0.0303 |       0.0347 |       0.0391 |       0.0435 |       0.048  |       0.0525 |       0.0569 |       0.0614 |       0.0658 |       0.0702 |       0.0745 |       0.0788 |       0.083  |       0.0872 |
        |            315 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0006 |       0.0012 |       0.0022 |       0.0035 |       0.0051 |       0.007  |       0.0093 |       0.0118 |       0.0145 |       0.0174 |       0.0204 |       0.0236 |       0.0269 |       0.0304 |       0.0338 |       0.0374 |       0.0409 |       0.0445 |       0.0482 |       0.0518 |       0.0554 |       0.059  |       0.0626 |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0008 |       0.0014 |       0.0022 |       0.0032 |       0.0045 |       0.006  |       0.0077 |       0.0096 |       0.0116 |       0.0138 |       0.0162 |       0.0186 |       0.0212 |       0.0239 |       0.0266 |       0.0294 |       0.0323 |       0.0352 |       0.0381 |       0.0411 |       0.044  |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0005 |       0.0009 |       0.0014 |       0.0021 |       0.0029 |       0.0039 |       0.0051 |       0.0064 |       0.0078 |       0.0094 |       0.0111 |       0.0129 |       0.0148 |       0.0168 |       0.0189 |       0.0211 |       0.0233 |       0.0256 |       0.028  |       0.0304 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0006 |       0.0009 |       0.0014 |       0.0019 |       0.0026 |       0.0034 |       0.0043 |       0.0053 |       0.0064 |       0.0076 |       0.009  |       0.0104 |       0.0119 |       0.0135 |       0.0152 |       0.0169 |       0.0187 |       0.0206 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0004 |       0.0006 |       0.0009 |       0.0013 |       0.0017 |       0.0023 |       0.0029 |       0.0036 |       0.0044 |       0.0053 |       0.0063 |       0.0073 |       0.0085 |       0.0097 |       0.0109 |       0.0123 |       0.0137 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_dual_delta().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0001 |      -0.0035 |      -0.0138 |      -0.0282 |      -0.0438 |      -0.0594 |      -0.0742 |      -0.088  |      -0.1009 |      -0.1128 |      -0.1238 |      -0.1339 |      -0.1434 |      -0.1522 |      -0.1603 |      -0.1679 |      -0.1751 |      -0.1818 |      -0.188  |      -0.194  |      -0.1995 |      -0.2048 |      -0.2098 |      -0.2146 |      -0.2191 |      -0.2234 |      -0.2275 |      -0.2314 |      -0.2351 |
        |            295 |      -0      |      -0.0003 |      -0.0027 |      -0.0079 |      -0.0155 |      -0.0243 |      -0.0339 |      -0.0437 |      -0.0535 |      -0.063  |      -0.0721 |      -0.081  |      -0.0894 |      -0.0974 |      -0.1051 |      -0.1124 |      -0.1193 |      -0.1259 |      -0.1322 |      -0.1382 |      -0.1439 |      -0.1494 |      -0.1546 |      -0.1596 |      -0.1644 |      -0.169  |      -0.1734 |      -0.1776 |      -0.1817 |
        |            300 |      -0      |      -0      |      -0.0004 |      -0.0018 |      -0.0046 |      -0.0088 |      -0.0139 |      -0.0197 |      -0.026  |      -0.0326 |      -0.0393 |      -0.046  |      -0.0527 |      -0.0593 |      -0.0657 |      -0.0719 |      -0.078  |      -0.0839 |      -0.0896 |      -0.0951 |      -0.1005 |      -0.1056 |      -0.1106 |      -0.1154 |      -0.1201 |      -0.1246 |      -0.129  |      -0.1332 |      -0.1373 |
        |            305 |      -0      |      -0      |      -0      |      -0.0003 |      -0.0012 |      -0.0028 |      -0.0051 |      -0.0081 |      -0.0117 |      -0.0157 |      -0.0201 |      -0.0247 |      -0.0294 |      -0.0343 |      -0.0392 |      -0.0441 |      -0.049  |      -0.0538 |      -0.0586 |      -0.0633 |      -0.0679 |      -0.0725 |      -0.0769 |      -0.0812 |      -0.0855 |      -0.0896 |      -0.0937 |      -0.0976 |      -0.1014 |
        |            310 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0008 |      -0.0017 |      -0.0031 |      -0.0049 |      -0.0071 |      -0.0096 |      -0.0125 |      -0.0156 |      -0.0189 |      -0.0223 |      -0.0259 |      -0.0296 |      -0.0333 |      -0.037  |      -0.0408 |      -0.0445 |      -0.0483 |      -0.052  |      -0.0557 |      -0.0593 |      -0.0629 |      -0.0664 |      -0.0699 |      -0.0734 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0005 |      -0.0011 |      -0.0019 |      -0.003  |      -0.0043 |      -0.006  |      -0.0078 |      -0.0099 |      -0.0122 |      -0.0146 |      -0.0172 |      -0.0199 |      -0.0226 |      -0.0254 |      -0.0283 |      -0.0313 |      -0.0342 |      -0.0372 |      -0.0401 |      -0.0431 |      -0.0461 |      -0.049  |      -0.052  |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0007 |      -0.0012 |      -0.0018 |      -0.0027 |      -0.0038 |      -0.005  |      -0.0064 |      -0.0079 |      -0.0096 |      -0.0114 |      -0.0134 |      -0.0154 |      -0.0175 |      -0.0197 |      -0.0219 |      -0.0242 |      -0.0265 |      -0.0289 |      -0.0313 |      -0.0337 |      -0.0361 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0007 |      -0.0012 |      -0.0017 |      -0.0024 |      -0.0032 |      -0.0041 |      -0.0052 |      -0.0064 |      -0.0077 |      -0.009  |      -0.0105 |      -0.0121 |      -0.0137 |      -0.0154 |      -0.0171 |      -0.0189 |      -0.0208 |      -0.0226 |      -0.0245 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0003 |      -0.0005 |      -0.0007 |      -0.0011 |      -0.0015 |      -0.0021 |      -0.0027 |      -0.0034 |      -0.0043 |      -0.0052 |      -0.0061 |      -0.0072 |      -0.0083 |      -0.0095 |      -0.0108 |      -0.0121 |      -0.0135 |      -0.0149 |      -0.0164 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0005 |      -0.0007 |      -0.001  |      -0.0014 |      -0.0018 |      -0.0023 |      -0.0029 |      -0.0035 |      -0.0042 |      -0.005  |      -0.0058 |      -0.0067 |      -0.0076 |      -0.0086 |      -0.0097 |      -0.0107 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vega().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0023 |       0.0093 |       0.0196 |       0.0315 |       0.0439 |       0.0564 |       0.0686 |       0.0804 |       0.0919 |       0.1029 |       0.1135 |       0.1237 |       0.1336 |       0.1431 |       0.1523 |       0.1613 |       0.1699 |       0.1783 |       0.1864 |       0.1943 |       0.202  |       0.2095 |       0.2168 |       0.2239 |       0.2309 |       0.2377 |       0.2444 |       0.2509 |
        |            295 |            0 |       0.0003 |       0.0022 |       0.0067 |       0.0134 |       0.0216 |       0.0308 |       0.0404 |       0.0504 |       0.0604 |       0.0704 |       0.0802 |       0.0899 |       0.0994 |       0.1088 |       0.1179 |       0.1268 |       0.1355 |       0.1439 |       0.1522 |       0.1603 |       0.1683 |       0.176  |       0.1836 |       0.191  |       0.1983 |       0.2054 |       0.2124 |       0.2192 |
        |            300 |            0 |       0      |       0.0004 |       0.0018 |       0.0048 |       0.0091 |       0.0147 |       0.0213 |       0.0285 |       0.0362 |       0.0443 |       0.0525 |       0.0609 |       0.0693 |       0.0777 |       0.0861 |       0.0944 |       0.1026 |       0.1107 |       0.1187 |       0.1266 |       0.1344 |       0.1421 |       0.1496 |       0.157  |       0.1643 |       0.1715 |       0.1786 |       0.1855 |
        |            305 |            0 |       0      |       0.0001 |       0.0004 |       0.0014 |       0.0033 |       0.0062 |       0.01   |       0.0146 |       0.0199 |       0.0257 |       0.032  |       0.0385 |       0.0454 |       0.0524 |       0.0595 |       0.0668 |       0.074  |       0.0814 |       0.0887 |       0.0959 |       0.1032 |       0.1104 |       0.1176 |       0.1246 |       0.1317 |       0.1386 |       0.1455 |       0.1523 |
        |            310 |            0 |       0      |       0      |       0.0001 |       0.0004 |       0.0011 |       0.0023 |       0.0043 |       0.0068 |       0.0101 |       0.0139 |       0.0182 |       0.0229 |       0.028  |       0.0334 |       0.0391 |       0.045  |       0.051  |       0.0572 |       0.0635 |       0.0699 |       0.0763 |       0.0827 |       0.0892 |       0.0957 |       0.1022 |       0.1087 |       0.1151 |       0.1215 |
        |            315 |            0 |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0008 |       0.0016 |       0.0029 |       0.0047 |       0.007  |       0.0097 |       0.0128 |       0.0163 |       0.0202 |       0.0244 |       0.0289 |       0.0337 |       0.0386 |       0.0437 |       0.049  |       0.0544 |       0.0599 |       0.0655 |       0.0712 |       0.0769 |       0.0827 |       0.0885 |       0.0944 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0006 |       0.0012 |       0.002  |       0.0033 |       0.0048 |       0.0068 |       0.009  |       0.0116 |       0.0146 |       0.0178 |       0.0213 |       0.025  |       0.029  |       0.0332 |       0.0375 |       0.042  |       0.0466 |       0.0514 |       0.0563 |       0.0612 |       0.0663 |       0.0714 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0008 |       0.0014 |       0.0023 |       0.0034 |       0.0047 |       0.0064 |       0.0083 |       0.0105 |       0.0129 |       0.0156 |       0.0186 |       0.0217 |       0.025  |       0.0285 |       0.0322 |       0.0361 |       0.04   |       0.0441 |       0.0484 |       0.0527 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0003 |       0.0006 |       0.001  |       0.0016 |       0.0024 |       0.0034 |       0.0045 |       0.006  |       0.0076 |       0.0094 |       0.0115 |       0.0137 |       0.0162 |       0.0188 |       0.0216 |       0.0246 |       0.0278 |       0.031  |       0.0344 |       0.038  |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0007 |       0.0011 |       0.0017 |       0.0024 |       0.0032 |       0.0043 |       0.0055 |       0.0069 |       0.0084 |       0.0102 |       0.0121 |       0.0141 |       0.0164 |       0.0188 |       0.0213 |       0.024  |       0.0268 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_theta().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0007 |      -0.0185 |      -0.0506 |      -0.0803 |      -0.1034 |      -0.1204 |      -0.1327 |      -0.1416 |      -0.1479 |      -0.1523 |      -0.1553 |      -0.1573 |      -0.1586 |      -0.1593 |      -0.1595 |      -0.1594 |      -0.159  |      -0.1584 |      -0.1577 |      -0.1569 |      -0.1559 |      -0.1549 |      -0.1539 |      -0.1528 |      -0.1517 |      -0.1506 |      -0.1495 |      -0.1484 |      -0.1473 |
        |            295 |      -0      |      -0.0021 |      -0.0121 |      -0.0275 |      -0.044  |      -0.0591 |      -0.0723 |      -0.0833 |      -0.0924 |      -0.0998 |      -0.1059 |      -0.1108 |      -0.1148 |      -0.1181 |      -0.1207 |      -0.1228 |      -0.1245 |      -0.1258 |      -0.1268 |      -0.1276 |      -0.1281 |      -0.1285 |      -0.1288 |      -0.1289 |      -0.1288 |      -0.1288 |      -0.1286 |      -0.1283 |      -0.128  |
        |            300 |      -0      |      -0.0002 |      -0.0021 |      -0.0075 |      -0.0156 |      -0.0249 |      -0.0345 |      -0.0437 |      -0.0521 |      -0.0597 |      -0.0664 |      -0.0724 |      -0.0775 |      -0.0821 |      -0.086  |      -0.0894 |      -0.0924 |      -0.095  |      -0.0972 |      -0.0992 |      -0.1009 |      -0.1023 |      -0.1036 |      -0.1046 |      -0.1055 |      -0.1063 |      -0.107  |      -0.1075 |      -0.1079 |
        |            305 |      -0      |      -0      |      -0.0003 |      -0.0016 |      -0.0046 |      -0.0091 |      -0.0146 |      -0.0206 |      -0.0267 |      -0.0328 |      -0.0385 |      -0.044  |      -0.049  |      -0.0536 |      -0.0578 |      -0.0617 |      -0.0652 |      -0.0684 |      -0.0713 |      -0.0739 |      -0.0762 |      -0.0783 |      -0.0802 |      -0.082  |      -0.0835 |      -0.0849 |      -0.0862 |      -0.0873 |      -0.0883 |
        |            310 |      -0      |      -0      |      -0      |      -0.0003 |      -0.0012 |      -0.0029 |      -0.0055 |      -0.0087 |      -0.0125 |      -0.0165 |      -0.0207 |      -0.0249 |      -0.029  |      -0.033  |      -0.0368 |      -0.0404 |      -0.0438 |      -0.047  |      -0.05   |      -0.0528 |      -0.0554 |      -0.0578 |      -0.06   |      -0.062  |      -0.064  |      -0.0657 |      -0.0674 |      -0.0689 |      -0.0703 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0003 |      -0.0008 |      -0.0018 |      -0.0034 |      -0.0054 |      -0.0077 |      -0.0104 |      -0.0132 |      -0.0162 |      -0.0192 |      -0.0223 |      -0.0252 |      -0.0281 |      -0.031  |      -0.0337 |      -0.0363 |      -0.0387 |      -0.0411 |      -0.0433 |      -0.0455 |      -0.0475 |      -0.0494 |      -0.0512 |      -0.0528 |      -0.0544 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0012 |      -0.0021 |      -0.0033 |      -0.0049 |      -0.0066 |      -0.0085 |      -0.0106 |      -0.0128 |      -0.015  |      -0.0173 |      -0.0196 |      -0.0218 |      -0.024  |      -0.0262 |      -0.0283 |      -0.0303 |      -0.0323 |      -0.0342 |      -0.036  |      -0.0378 |      -0.0395 |      -0.0411 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0004 |      -0.0008 |      -0.0014 |      -0.0021 |      -0.0031 |      -0.0043 |      -0.0056 |      -0.007  |      -0.0086 |      -0.0102 |      -0.0119 |      -0.0136 |      -0.0153 |      -0.0171 |      -0.0188 |      -0.0206 |      -0.0223 |      -0.024  |      -0.0256 |      -0.0272 |      -0.0288 |      -0.0303 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0005 |      -0.0009 |      -0.0014 |      -0.002  |      -0.0028 |      -0.0037 |      -0.0047 |      -0.0058 |      -0.0069 |      -0.0082 |      -0.0095 |      -0.0108 |      -0.0122 |      -0.0136 |      -0.0149 |      -0.0163 |      -0.0177 |      -0.0191 |      -0.0204 |      -0.0218 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0006 |      -0.0009 |      -0.0013 |      -0.0018 |      -0.0025 |      -0.0031 |      -0.0039 |      -0.0048 |      -0.0057 |      -0.0066 |      -0.0076 |      -0.0087 |      -0.0098 |      -0.0108 |      -0.012  |      -0.0131 |      -0.0142 |      -0.0153 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_rho().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0056 |       0.033  |       0.0896 |       0.1741 |       0.283  |       0.4125 |       0.5595 |       0.7214 |       0.896  |       1.0817 |       1.2771 |       1.481  |       1.6924 |       1.9106 |       2.1349 |       2.3646 |       2.5993 |       2.8386 |       3.0821 |       3.3294 |       3.5802 |       3.8344 |       4.0916 |       4.3516 |       4.6143 |       4.8795 |       5.147  |       5.4167 |
        |            295 |       0      |       0.0005 |       0.0065 |       0.0257 |       0.0624 |       0.118  |       0.1919 |       0.2827 |       0.3888 |       0.5089 |       0.6414 |       0.7853 |       0.9392 |       1.1024 |       1.2739 |       1.4531 |       1.6392 |       1.8316 |       2.03   |       2.2338 |       2.4427 |       2.6562 |       2.874  |       3.096  |       3.3217 |       3.551  |       3.7836 |       4.0194 |       4.2582 |
        |            300 |       0      |       0      |       0.001  |       0.006  |       0.0191 |       0.0432 |       0.0799 |       0.1298 |       0.1927 |       0.2682 |       0.3556 |       0.4542 |       0.5632 |       0.682  |       0.8098 |       0.946  |       1.09   |       1.2413 |       1.3994 |       1.5638 |       1.7342 |       1.9101 |       2.0913 |       2.2773 |       2.4679 |       2.6629 |       2.8621 |       3.0651 |       3.2717 |
        |            305 |       0      |       0      |       0.0001 |       0.0012 |       0.005  |       0.0139 |       0.0299 |       0.0543 |       0.088  |       0.1314 |       0.1846 |       0.2474 |       0.3197 |       0.4011 |       0.4912 |       0.5896 |       0.6959 |       0.8097 |       0.9306 |       1.0583 |       1.1923 |       1.3323 |       1.4781 |       1.6293 |       1.7857 |       1.9471 |       2.113  |       2.2835 |       2.4581 |
        |            310 |       0      |       0      |       0      |       0.0002 |       0.0011 |       0.004  |       0.0101 |       0.0207 |       0.0371 |       0.0599 |       0.0898 |       0.1272 |       0.1721 |       0.2246 |       0.2847 |       0.3523 |       0.427  |       0.5089 |       0.5976 |       0.6928 |       0.7944 |       0.902  |       1.0156 |       1.1347 |       1.2592 |       1.3889 |       1.5236 |       1.663  |       1.807  |
        |            315 |       0      |       0      |       0      |       0      |       0.0002 |       0.001  |       0.0031 |       0.0073 |       0.0145 |       0.0255 |       0.0411 |       0.0618 |       0.0879 |       0.1199 |       0.1579 |       0.202  |       0.2522 |       0.3085 |       0.3709 |       0.4392 |       0.5134 |       0.5933 |       0.6788 |       0.7698 |       0.866  |       0.9673 |       1.0736 |       1.1847 |       1.3004 |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0023 |       0.0052 |       0.0102 |       0.0177 |       0.0284 |       0.0428 |       0.0611 |       0.0839 |       0.1113 |       0.1434 |       0.1806 |       0.2227 |       0.2699 |       0.3222 |       0.3795 |       0.4418 |       0.5091 |       0.5812 |       0.6582 |       0.7398 |       0.826  |       0.9168 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0018 |       0.0038 |       0.0072 |       0.0124 |       0.0198 |       0.0298 |       0.0428 |       0.059  |       0.0787 |       0.1022 |       0.1295 |       0.1609 |       0.1965 |       0.2362 |       0.2802 |       0.3285 |       0.3811 |       0.4379 |       0.4989 |       0.5642 |       0.6336 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0006 |       0.0013 |       0.0028 |       0.0051 |       0.0088 |       0.0139 |       0.021  |       0.0301 |       0.0417 |       0.056  |       0.0731 |       0.0932 |       0.1166 |       0.1433 |       0.1734 |       0.207  |       0.2442 |       0.2851 |       0.3296 |       0.3777 |       0.4296 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0004 |       0.001  |       0.002  |       0.0037 |       0.0062 |       0.0099 |       0.0149 |       0.0214 |       0.0297 |       0.04   |       0.0525 |       0.0674 |       0.0847 |       0.1047 |       0.1275 |       0.1532 |       0.1818 |       0.2134 |       0.2481 |       0.2859 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_epsilon().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0001 |      -0.0056 |      -0.0333 |      -0.0908 |      -0.1769 |      -0.2881 |      -0.4209 |      -0.5721 |      -0.7392 |      -0.9199 |      -1.1126 |      -1.316  |      -1.5287 |      -1.7499 |      -1.9788 |      -2.2146 |      -2.4567 |      -2.7047 |      -2.9581 |      -3.2164 |      -3.4794 |      -3.7468 |      -4.0182 |      -4.2935 |      -4.5724 |      -4.8547 |      -5.1402 |      -5.4288 |      -5.7204 |
        |            295 |      -0      |      -0.0005 |      -0.0066 |      -0.0259 |      -0.0633 |      -0.1199 |      -0.1953 |      -0.2883 |      -0.3974 |      -0.521  |      -0.6579 |      -0.8067 |      -0.9665 |      -1.1363 |      -1.3151 |      -1.5023 |      -1.6973 |      -1.8994 |      -2.1081 |      -2.3229 |      -2.5436 |      -2.7697 |      -3.0008 |      -3.2367 |      -3.4771 |      -3.7218 |      -3.9705 |      -4.2231 |      -4.4793 |
        |            300 |      -0      |      -0      |      -0.001  |      -0.0061 |      -0.0193 |      -0.0438 |      -0.0812 |      -0.1321 |      -0.1965 |      -0.2739 |      -0.3638 |      -0.4654 |      -0.5781 |      -0.7011 |      -0.8337 |      -0.9753 |      -1.1254 |      -1.2834 |      -1.4489 |      -1.6213 |      -1.8003 |      -1.9855 |      -2.1765 |      -2.3731 |      -2.5749 |      -2.7818 |      -2.9933 |      -3.2094 |      -3.4298 |
        |            305 |      -0      |      -0      |      -0.0001 |      -0.0012 |      -0.005  |      -0.0141 |      -0.0303 |      -0.0551 |      -0.0895 |      -0.1339 |      -0.1884 |      -0.253  |      -0.3275 |      -0.4114 |      -0.5045 |      -0.6065 |      -0.7168 |      -0.8351 |      -0.961  |      -1.0942 |      -1.2343 |      -1.3811 |      -1.5341 |      -1.6931 |      -1.8578 |      -2.028  |      -2.2034 |      -2.3838 |      -2.5691 |
        |            310 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0011 |      -0.004  |      -0.0102 |      -0.021  |      -0.0377 |      -0.061  |      -0.0916 |      -0.1298 |      -0.1759 |      -0.2299 |      -0.2918 |      -0.3616 |      -0.4389 |      -0.5237 |      -0.6157 |      -0.7147 |      -0.8205 |      -0.9328 |      -1.0514 |      -1.1761 |      -1.3067 |      -1.4429 |      -1.5846 |      -1.7315 |      -1.8834 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.001  |      -0.0031 |      -0.0073 |      -0.0147 |      -0.0259 |      -0.0418 |      -0.063  |      -0.0898 |      -0.1225 |      -0.1616 |      -0.2069 |      -0.2587 |      -0.3168 |      -0.3814 |      -0.4522 |      -0.5292 |      -0.6123 |      -0.7013 |      -0.7961 |      -0.8966 |      -1.0026 |      -1.114  |      -1.2305 |      -1.3522 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0009 |      -0.0024 |      -0.0053 |      -0.0103 |      -0.018  |      -0.0289 |      -0.0436 |      -0.0624 |      -0.0857 |      -0.1138 |      -0.1469 |      -0.1851 |      -0.2286 |      -0.2774 |      -0.3315 |      -0.3909 |      -0.4556 |      -0.5255 |      -0.6006 |      -0.6808 |      -0.766  |      -0.8562 |      -0.9512 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0007 |      -0.0018 |      -0.0039 |      -0.0073 |      -0.0126 |      -0.0202 |      -0.0304 |      -0.0436 |      -0.0603 |      -0.0805 |      -0.1046 |      -0.1328 |      -0.1651 |      -0.2018 |      -0.2429 |      -0.2885 |      -0.3385 |      -0.3931 |      -0.4521 |      -0.5157 |      -0.5837 |      -0.6561 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0014 |      -0.0028 |      -0.0052 |      -0.0089 |      -0.0142 |      -0.0214 |      -0.0307 |      -0.0426 |      -0.0572 |      -0.0748 |      -0.0955 |      -0.1196 |      -0.1471 |      -0.1782 |      -0.213  |      -0.2515 |      -0.2939 |      -0.34   |      -0.3901 |      -0.4441 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0004 |      -0.001  |      -0.0021 |      -0.0038 |      -0.0064 |      -0.0101 |      -0.0151 |      -0.0218 |      -0.0303 |      -0.0409 |      -0.0537 |      -0.069  |      -0.0869 |      -0.1075 |      -0.131  |      -0.1575 |      -0.1871 |      -0.2198 |      -0.2558 |      -0.2951 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close).
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_lambda().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      250.947 |      136.159 |      96.9271 |      76.827  |      64.4841 |      56.0733 |      49.9401 |      45.2492 |      41.5325 |      38.5062 |      35.9882 |      33.8559 |      32.0237 |      30.43   |      29.0291 |      27.7865 |      26.6756 |      25.6755 |      24.7697 |      23.9447 |      23.1897 |      22.4956 |      21.855  |      21.2615 |      20.7099 |      20.1956 |      19.7148 |      19.2641 |      18.8406 |
        |            295 |      305.834 |      162.329 |     113.794  |      89.1531 |      74.1409 |      63.9817 |      56.6188 |      51.0184 |      46.6028 |      43.0236 |      40.0578 |      37.5558 |      35.4135 |      33.556  |      31.9281 |      30.4883 |      29.2045 |      28.0517 |      27.01   |      26.0633 |      25.1988 |      24.4056 |      23.675  |      22.9993 |      22.3724 |      21.7889 |      21.2442 |      20.7344 |      20.2561 |
        |            300 |      360.985 |      188.855 |     130.965  |     101.731  |      84.0084 |      72.069  |      63.4514 |      56.9214 |      51.7909 |      47.6456 |      44.221  |      41.3399 |      38.8795 |      36.7515 |      34.8908 |      33.2486 |      31.7874 |      30.4779 |      29.2967 |      28.2252 |      27.2483 |      26.3535 |      25.5304 |      24.7705 |      24.0664 |      23.4119 |      22.8017 |      22.2314 |      21.6968 |
        |            305 |      415.986 |      215.493 |     148.279  |     114.447  |      94.0014 |      80.2685 |      70.3844 |      62.9144 |      57.0599 |      52.3407 |      48.4506 |      45.1847 |      42.4012 |      39.9981 |      37.9007 |      36.0527 |      34.411  |      32.9419 |      31.6188 |      30.4202 |      29.3289 |      28.3306 |      27.4135 |      26.5677 |      25.7849 |      25.0582 |      24.3813 |      23.7493 |      23.1575 |
        |            310 |      470.601 |      242.09  |     165.628  |     127.22   |     104.056  |      88.5295 |      77.376  |      68.9624 |      62.3802 |      57.0835 |      52.7245 |      49.0707 |      45.9611 |      43.2805 |      40.944  |      38.888  |      37.0638 |      35.4334 |      33.9667 |      32.6396 |      31.4325 |      30.3294 |      29.3171 |      28.3844 |      27.522  |      26.722  |      25.9777 |      25.2831 |      24.6335 |
        |            315 |      524.698 |      268.547 |     182.938  |     139.992  |     114.127  |      96.8137 |      84.3941 |      75.0381 |      67.7283 |      61.8535 |      57.0247 |      52.9819 |      49.5451 |      46.5857 |      44.009  |      41.744  |      39.7363 |      37.9436 |      36.3324 |      34.8758 |      33.5522 |      32.3436 |      31.2353 |      30.215  |      29.2723 |      28.3985 |      27.5861 |      26.8286 |      26.1204 |
        |            320 |      578.198 |      294.801 |     200.157  |     152.72   |     124.179  |     105.092  |      91.4142 |      81.1204 |      73.0856 |      66.6343 |      61.3365 |      56.9052 |      53.1415 |      49.9032 |      47.0861 |      44.6117 |      42.4203 |      40.4649 |      38.7089 |      37.1225 |      35.6819 |      34.3674 |      33.1628 |      32.0546 |      31.0313 |      30.0834 |      29.2025 |      28.3817 |      27.6149 |
        |            325 |      631.057 |      320.809 |     217.25   |     165.377  |     134.186  |     113.344  |      98.4175 |      87.1926 |      78.4375 |      71.413  |      65.6485 |      60.8301 |      56.7405 |      53.2243 |      50.1673 |      47.484  |      45.109  |      42.9912 |      41.0904 |      39.3742 |      37.8166 |      36.3962 |      35.0952 |      33.899  |      32.795  |      31.7729 |      30.8235 |      29.9393 |      29.1136 |
        |            330 |      683.252 |      346.546 |     234.193  |     177.94   |     144.132  |     121.551  |     105.39   |      93.2421 |      83.7726 |      76.1792 |      69.9512 |      64.7483 |      60.3347 |      56.5418 |      53.2461 |      50.3548 |      47.7969 |      45.5172 |      43.472  |      41.6264 |      39.9521 |      38.426  |      37.0289 |      35.7447 |      34.5602 |      33.4639 |      32.4461 |      31.4985 |      30.6139 |
        |            335 |      734.773 |      371.994 |     250.971  |     190.394  |     154.001  |     129.704  |     112.32   |      99.2592 |      89.0822 |      80.9248 |      74.2373 |      68.6529 |      63.9176 |      59.8501 |      56.3172 |      53.2191 |      50.4794 |      48.0386 |      45.8497 |      43.8753 |      42.0849 |      40.4534 |      38.9604 |      37.5887 |      36.3239 |      35.1536 |      34.0676 |      33.0568 |      32.1135 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
        |            290 |                                 0.0001 |                                      0.0001 |                                 0.0035 |                                -0.21   |                                 0.1909 |                                0.0026 |                               -0.0003 |                              0.0001 |                                 0.0017 |                                      0.0015 |                                 0.0937 |                                -2.7884 |                                 5.034  |                                0.1383 |                               -0.0047 |                              0.0015 |                                 0.0047 |                                      0.0041 |                                 0.2559 |                                -5.0955 |                                13.7018 |                                0.5642 |                               -0.009  |                              0.0041 |                                 0.0075 |                                      0.0066 |                                 0.4055 |                                -6.0751 |                                21.6286 |                                1.1867 |                               -0.0113 |                              0.0066 |                                 0.0096 |                                      0.0084 |                                 0.5213 |                                -6.2674 |                                27.6957 |                                1.8983 |                               -0.0123 |                              0.0084 |                                 0.0111 |                                      0.0098 |                                 0.6062 |                                -6.092  |                                32.0786 |                                2.6368 |                               -0.0125 |                              0.0098 |                                 0.0123 |                                      0.0108 |                                 0.6672 |                                -5.7653 |                                35.1694 |                                3.3706 |                               -0.0124 |                              0.0108 |                                 0.0131 |                                      0.0115 |                                 0.7107 |                                -5.3898 |                                37.3128 |                                4.0843 |                               -0.012  |                              0.0115 |                                 0.0136 |                                      0.012  |                                 0.7413 |                                -5.013  |                                38.7684 |                                4.7711 |                               -0.0116 |                              0.012  |                                 0.014  |                                      0.0123 |                                 0.7626 |                                -4.6555 |                                39.7237 |                                5.4285 |                               -0.0112 |                              0.0123 |                                 0.0142 |                                      0.0125 |                                 0.777  |                                -4.3254 |                                40.3125 |                                6.0561 |                               -0.0108 |                              0.0125 |                                 0.0144 |                                      0.0127 |                                 0.7863 |                                -4.0245 |                                40.6311 |                                6.6547 |                               -0.0104 |                              0.0127 |                                 0.0145 |                                      0.0127 |                                 0.7918 |                                -3.752  |                                40.7489 |                                7.2257 |                               -0.01   |                              0.0127 |                                 0.0145 |                                      0.0128 |                                 0.7943 |                                -3.5059 |                                40.7168 |                                7.7705 |                               -0.0097 |                              0.0128 |                                 0.0145 |                                      0.0128 |                                 0.7947 |                                -3.2836 |                                40.5723 |                                8.2909 |                               -0.0094 |                              0.0128 |                                 0.0145 |                                      0.0127 |                                 0.7935 |                                -3.0827 |                                40.3435 |                                8.7883 |                               -0.0091 |                              0.0127 |                                 0.0144 |                                      0.0127 |                                 0.791  |                                -2.9007 |                                40.0514 |                                9.2642 |                               -0.0088 |                              0.0127 |                                 0.0144 |                                      0.0126 |                                 0.7875 |                                -2.7356 |                                39.7119 |                                9.72   |                               -0.0085 |                              0.0126 |                                 0.0143 |                                      0.0126 |                                 0.7833 |                                -2.5853 |                                39.3374 |                               10.157  |                               -0.0082 |                              0.0126 |                                 0.0142 |                                      0.0125 |                                 0.7785 |                                -2.4483 |                                38.9373 |                               10.5763 |                               -0.008  |                              0.0125 |                                 0.0141 |                                      0.0124 |                                 0.7734 |                                -2.323  |                                38.5188 |                               10.979  |                               -0.0078 |                              0.0124 |                                 0.014  |                                      0.0123 |                                 0.7679 |                                -2.2082 |                                38.0877 |                               11.366  |                               -0.0076 |                              0.0123 |                                 0.0139 |                                      0.0122 |                                 0.7623 |                                -2.1026 |                                37.6483 |                               11.7383 |                               -0.0074 |                              0.0122 |                                 0.0138 |                                      0.0121 |                                 0.7565 |                                -2.0054 |                                37.2041 |                               12.0967 |                               -0.0072 |                              0.0121 |                                 0.0136 |                                      0.012  |                                 0.7505 |                                -1.9156 |                                36.7578 |                               12.4419 |                               -0.007  |                              0.012  |                                 0.0135 |                                      0.0119 |                                 0.7446 |                                -1.8325 |                                36.3116 |                               12.7747 |                               -0.0069 |                              0.0119 |                                 0.0134 |                                      0.0118 |                                 0.7386 |                                -1.7555 |                                35.867  |                               13.0955 |                               -0.0067 |                              0.0118 |                                 0.0133 |                                      0.0117 |                                 0.7327 |                                -1.6839 |                                35.4255 |                               13.4051 |                               -0.0066 |                              0.0117 |                                 0.0132 |                                      0.0116 |                                 0.7267 |                                -1.6172 |                                34.9879 |                               13.7039 |                               -0.0065 |                              0.0116 |
        |            295 |                                 0      |                                      0      |                                 0.0001 |                                -0.0035 |                                 0.004  |                                0      |                               -0      |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0137 |                                -0.4076 |                                 0.9333 |                                0.0203 |                               -0.0008 |                              0.0002 |                                 0.0011 |                                      0.001  |                                 0.0772 |                                -1.5345 |                                 5.2413 |                                0.1707 |                               -0.0032 |                              0.001  |                                 0.0026 |                                      0.0022 |                                 0.1757 |                                -2.625  |                                11.8885 |                                0.5161 |                               -0.0057 |                              0.0022 |                                 0.0041 |                                      0.0035 |                                 0.2806 |                                -3.362  |                                18.9276 |                                1.0265 |                               -0.0076 |                              0.0035 |                                 0.0055 |                                      0.0047 |                                 0.377  |                                -3.774  |                                25.3547 |                                1.6493 |                               -0.0088 |                              0.0047 |                                 0.0067 |                                      0.0057 |                                 0.4601 |                                -3.9575 |                                30.8469 |                                2.3398 |                               -0.0095 |                              0.0057 |                                 0.0077 |                                      0.0065 |                                 0.5295 |                                -3.995  |                                35.3905 |                                3.0664 |                               -0.0098 |                              0.0065 |                                 0.0085 |                                      0.0072 |                                 0.5866 |                                -3.9437 |                                39.085  |                                3.8079 |                               -0.01   |                              0.0072 |                                 0.0092 |                                      0.0078 |                                 0.6332 |                                -3.8406 |                                42.058  |                                4.5506 |                               -0.01   |                              0.0078 |                                 0.0097 |                                      0.0083 |                                 0.671  |                                -3.7092 |                                44.4324 |                                5.2857 |                               -0.0099 |                              0.0083 |                                 0.0102 |                                      0.0086 |                                 0.7017 |                                -3.5639 |                                46.3154 |                                6.0076 |                               -0.0098 |                              0.0086 |                                 0.0105 |                                      0.0089 |                                 0.7264 |                                -3.4139 |                                47.7963 |                                6.7131 |                               -0.0096 |                              0.0089 |                                 0.0108 |                                      0.0092 |                                 0.7462 |                                -3.2646 |                                48.9485 |                                7.4001 |                               -0.0094 |                              0.0092 |                                 0.011  |                                      0.0094 |                                 0.7621 |                                -3.1193 |                                49.8316 |                                8.0678 |                               -0.0092 |                              0.0094 |                                 0.0112 |                                      0.0095 |                                 0.7747 |                                -2.9797 |                                50.4938 |                                8.7157 |                               -0.009  |                              0.0095 |                                 0.0114 |                                      0.0096 |                                 0.7846 |                                -2.8469 |                                50.9741 |                                9.3439 |                               -0.0088 |                              0.0096 |                                 0.0115 |                                      0.0097 |                                 0.7922 |                                -2.7213 |                                51.3044 |                                9.9528 |                               -0.0086 |                              0.0097 |                                 0.0115 |                                      0.0098 |                                 0.7979 |                                -2.6029 |                                51.5103 |                               10.5427 |                               -0.0084 |                              0.0098 |                                 0.0116 |                                      0.0098 |                                 0.8021 |                                -2.4915 |                                51.6132 |                               11.1143 |                               -0.0082 |                              0.0098 |                                 0.0116 |                                      0.0099 |                                 0.805  |                                -2.3869 |                                51.6304 |                               11.6682 |                               -0.008  |                              0.0099 |                                 0.0117 |                                      0.0099 |                                 0.8067 |                                -2.2888 |                                51.5762 |                               12.205  |                               -0.0078 |                              0.0099 |                                 0.0117 |                                      0.0099 |                                 0.8076 |                                -2.1966 |                                51.4624 |                               12.7254 |                               -0.0077 |                              0.0099 |                                 0.0117 |                                      0.0099 |                                 0.8076 |                                -2.1102 |                                51.2989 |                               13.2301 |                               -0.0075 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8071 |                                -2.029  |                                51.094  |                               13.7196 |                               -0.0073 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8059 |                                -1.9527 |                                50.8545 |                               14.1945 |                               -0.0072 |                              0.0099 |                                 0.0116 |                                      0.0098 |                                 0.8043 |                                -1.8809 |                                50.5862 |                               14.6556 |                               -0.007  |                              0.0098 |                                 0.0116 |                                      0.0098 |                                 0.8023 |                                -1.8134 |                                50.294  |                               15.1032 |                               -0.0069 |                              0.0098 |                                 0.0115 |                                      0.0098 |                                 0.8    |                                -1.7497 |                                49.982  |                               15.538  |                               -0.0068 |                              0.0098 |
        |            300 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0012 |                                -0.0359 |                                 0.0994 |                                0.0018 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0164 |                                -0.3248 |                                 1.3415 |                                0.0362 |                               -0.0008 |                              0.0002 |                                 0.0007 |                                      0.0006 |                                 0.0577 |                                -0.8607 |                                 4.7182 |                                0.1699 |                               -0.0022 |                              0.0006 |                                 0.0014 |                                      0.0012 |                                 0.1199 |                                -1.4332 |                                 9.7751 |                                0.4399 |                               -0.0037 |                              0.0012 |                                 0.0023 |                                      0.0019 |                                 0.1919 |                                -1.9161 |                                15.6098 |                                0.8425 |                               -0.005  |                              0.0019 |                                 0.0032 |                                      0.0026 |                                 0.2654 |                                -2.2761 |                                21.5337 |                                1.3554 |                               -0.0061 |                              0.0026 |                                 0.004  |                                      0.0033 |                                 0.3355 |                                -2.5226 |                                27.1501 |                                1.9523 |                               -0.0069 |                              0.0033 |                                 0.0048 |                                      0.004  |                                 0.3998 |                                -2.6776 |                                32.2708 |                                2.6095 |                               -0.0075 |                              0.004  |                                 0.0055 |                                      0.0045 |                                 0.4574 |                                -2.7631 |                                36.8318 |                                3.3079 |                               -0.0079 |                              0.0045 |                                 0.0061 |                                      0.005  |                                 0.5085 |                                -2.7978 |                                40.8353 |                                4.0326 |                               -0.0082 |                              0.005  |                                 0.0067 |                                      0.0055 |                                 0.5533 |                                -2.7962 |                                44.3163 |                                4.7722 |                               -0.0083 |                              0.0055 |                                 0.0071 |                                      0.0059 |                                 0.5923 |                                -2.769  |                                47.3234 |                                5.5185 |                               -0.0084 |                              0.0059 |                                 0.0075 |                                      0.0062 |                                 0.6263 |                                -2.7242 |                                49.9086 |                                6.2651 |                               -0.0084 |                              0.0062 |                                 0.0079 |                                      0.0065 |                                 0.6558 |                                -2.6676 |                                52.1222 |                                7.0075 |                               -0.0084 |                              0.0065 |                                 0.0082 |                                      0.0067 |                                 0.6813 |                                -2.6035 |                                54.0107 |                                7.7423 |                               -0.0083 |                              0.0067 |                                 0.0085 |                                      0.0069 |                                 0.7034 |                                -2.5348 |                                55.6157 |                                8.4672 |                               -0.0083 |                              0.0069 |                                 0.0087 |                                      0.0071 |                                 0.7225 |                                -2.4638 |                                56.974  |                                9.1805 |                               -0.0082 |                              0.0071 |                                 0.0089 |                                      0.0073 |                                 0.739  |                                -2.392  |                                58.1176 |                                9.881  |                               -0.0081 |                              0.0073 |                                 0.009  |                                      0.0074 |                                 0.7531 |                                -2.3205 |                                59.0743 |                               10.568  |                               -0.008  |                              0.0074 |                                 0.0092 |                                      0.0075 |                                 0.7653 |                                -2.2501 |                                59.8683 |                               11.2409 |                               -0.0078 |                              0.0075 |                                 0.0093 |                                      0.0076 |                                 0.7756 |                                -2.1813 |                                60.5203 |                               11.8996 |                               -0.0077 |                              0.0076 |                                 0.0094 |                                      0.0077 |                                 0.7845 |                                -2.1144 |                                61.0485 |                               12.544  |                               -0.0076 |                              0.0077 |                                 0.0095 |                                      0.0078 |                                 0.792  |                                -2.0497 |                                61.4683 |                               13.174  |                               -0.0075 |                              0.0078 |                                 0.0096 |                                      0.0079 |                                 0.7983 |                                -1.9872 |                                61.7933 |                               13.7899 |                               -0.0074 |                              0.0079 |                                 0.0096 |                                      0.0079 |                                 0.8036 |                                -1.9272 |                                62.0353 |                               14.3918 |                               -0.0072 |                              0.0079 |                                 0.0097 |                                      0.0079 |                                 0.808  |                                -1.8695 |                                62.2043 |                               14.98   |                               -0.0071 |                              0.0079 |                                 0.0097 |                                      0.008  |                                 0.8115 |                                -1.8141 |                                62.3092 |                               15.5547 |                               -0.007  |                              0.008  |                                 0.0097 |                                      0.008  |                                 0.8144 |                                -1.7611 |                                62.3578 |                               16.1163 |                               -0.0069 |                              0.008  |
        |            305 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.002  |                                 0.0064 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0498 |                                 0.2409 |                                0.0056 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0001 |                                 0.0148 |                                -0.22   |                                 1.4122 |                                0.0436 |                               -0.0006 |                              0.0001 |                                 0.0004 |                                      0.0003 |                                 0.0417 |                                -0.4975 |                                 3.9759 |                                0.1532 |                               -0.0014 |                              0.0003 |                                 0.0008 |                                      0.0007 |                                 0.0818 |                                -0.8156 |                                 7.7904 |                                0.3602 |                               -0.0024 |                              0.0007 |                                 0.0014 |                                      0.0011 |                                 0.1309 |                                -1.1206 |                                12.439  |                                0.6707 |                               -0.0034 |                              0.0011 |                                 0.0019 |                                      0.0015 |                                 0.1846 |                                -1.3852 |                                17.5024 |                                1.0782 |                               -0.0042 |                              0.0015 |                                 0.0025 |                                      0.002  |                                 0.2396 |                                -1.6003 |                                22.6588 |                                1.5698 |                               -0.005  |                              0.002  |                                 0.003  |                                      0.0024 |                                 0.2934 |                                -1.7672 |                                27.6923 |                                2.1309 |                               -0.0056 |                              0.0024 |                                 0.0036 |                                      0.0028 |                                 0.3448 |                                -1.8913 |                                32.4718 |                                2.7476 |                               -0.006  |                              0.0028 |                                 0.0041 |                                      0.0032 |                                 0.393  |                                -1.9793 |                                36.9264 |                                3.4074 |                               -0.0064 |                              0.0032 |                                 0.0045 |                                      0.0036 |                                 0.4376 |                                -2.0379 |                                41.0252 |                                4.0997 |                               -0.0067 |                              0.0036 |                                 0.0049 |                                      0.0039 |                                 0.4785 |                                -2.0729 |                                44.7625 |                                4.8156 |                               -0.0069 |                              0.0039 |                                 0.0053 |                                      0.0042 |                                 0.5158 |                                -2.0892 |                                48.1479 |                                5.5478 |                               -0.0071 |                              0.0042 |                                 0.0057 |                                      0.0045 |                                 0.5498 |                                -2.091  |                                51.1997 |                                6.2905 |                               -0.0072 |                              0.0045 |                                 0.006  |                                      0.0048 |                                 0.5805 |                                -2.0816 |                                53.9404 |                                7.039  |                               -0.0073 |                              0.0048 |                                 0.0063 |                                      0.005  |                                 0.6082 |                                -2.0635 |                                56.3943 |                                7.7894 |                               -0.0073 |                              0.005  |                                 0.0065 |                                      0.0052 |                                 0.6333 |                                -2.0389 |                                58.586  |                                8.5388 |                               -0.0073 |                              0.0052 |                                 0.0068 |                                      0.0054 |                                 0.6559 |                                -2.0095 |                                60.5391 |                                9.2846 |                               -0.0073 |                              0.0054 |                                 0.007  |                                      0.0055 |                                 0.6762 |                                -1.9765 |                                62.2757 |                               10.0249 |                               -0.0073 |                              0.0055 |                                 0.0071 |                                      0.0057 |                                 0.6945 |                                -1.941  |                                63.8164 |                               10.7584 |                               -0.0072 |                              0.0057 |                                 0.0073 |                                      0.0058 |                                 0.711  |                                -1.9038 |                                65.1801 |                               11.4837 |                               -0.0072 |                              0.0058 |                                 0.0075 |                                      0.0059 |                                 0.7257 |                                -1.8655 |                                66.3839 |                               12.2001 |                               -0.0071 |                              0.0059 |                                 0.0076 |                                      0.006  |                                 0.739  |                                -1.8267 |                                67.4435 |                               12.9068 |                               -0.0071 |                              0.006  |                                 0.0077 |                                      0.0061 |                                 0.7509 |                                -1.7877 |                                68.3729 |                               13.6033 |                               -0.007  |                              0.0061 |                                 0.0078 |                                      0.0062 |                                 0.7615 |                                -1.7488 |                                69.1847 |                               14.2893 |                               -0.0069 |                              0.0062 |                                 0.0079 |                                      0.0063 |                                 0.7711 |                                -1.7103 |                                69.8902 |                               14.9644 |                               -0.0068 |                              0.0063 |                                 0.008  |                                      0.0064 |                                 0.7796 |                                -1.6723 |                                70.4997 |                               15.6286 |                               -0.0068 |                              0.0064 |
        |            310 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0003 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0003 |                                -0.0057 |                                 0.0314 |                                0.0006 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.003  |                                -0.0447 |                                 0.3282 |                                0.0089 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.012  |                                -0.1431 |                                 1.3086 |                                0.0442 |                               -0.0005 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0297 |                                -0.2957 |                                 3.2338 |                                0.131  |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0561 |                                -0.4794 |                                 6.0946 |                                0.288  |                               -0.0016 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.0896 |                                -0.6708 |                                 9.7116 |                                0.5244 |                               -0.0023 |                              0.0006 |                                 0.0012 |                                      0.0009 |                                 0.128  |                                -0.8533 |                                13.8509 |                                0.8411 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0012 |                                 0.1693 |                                -1.0177 |                                18.2919 |                                1.2338 |                               -0.0035 |                              0.0012 |                                 0.0019 |                                      0.0015 |                                 0.212  |                                -1.16   |                                22.8541 |                                1.6951 |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0018 |                                 0.2547 |                                -1.2793 |                                27.4017 |                                2.2165 |                               -0.0045 |                              0.0018 |                                 0.0027 |                                      0.0021 |                                 0.2965 |                                -1.3769 |                                31.8386 |                                2.7892 |                               -0.0049 |                              0.0021 |                                 0.003  |                                      0.0023 |                                 0.3368 |                                -1.4547 |                                36.1007 |                                3.4048 |                               -0.0053 |                              0.0023 |                                 0.0034 |                                      0.0026 |                                 0.3753 |                                -1.5152 |                                40.1481 |                                4.0557 |                               -0.0056 |                              0.0026 |                                 0.0037 |                                      0.0029 |                                 0.4117 |                                -1.5607 |                                43.9585 |                                4.7352 |                               -0.0058 |                              0.0029 |                                 0.004  |                                      0.0031 |                                 0.446  |                                -1.5935 |                                47.5224 |                                5.4374 |                               -0.006  |                              0.0031 |                                 0.0043 |                                      0.0033 |                                 0.478  |                                -1.6155 |                                50.8388 |                                6.1571 |                               -0.0061 |                              0.0033 |                                 0.0046 |                                      0.0035 |                                 0.5079 |                                -1.6287 |                                53.9127 |                                6.8901 |                               -0.0062 |                              0.0035 |                                 0.0048 |                                      0.0037 |                                 0.5357 |                                -1.6344 |                                56.7528 |                                7.6324 |                               -0.0063 |                              0.0037 |                                 0.0051 |                                      0.0039 |                                 0.5615 |                                -1.634  |                                59.3701 |                                8.3811 |                               -0.0064 |                              0.0039 |                                 0.0053 |                                      0.0041 |                                 0.5854 |                                -1.6286 |                                61.777  |                                9.1333 |                               -0.0064 |                              0.0041 |                                 0.0055 |                                      0.0042 |                                 0.6076 |                                -1.6191 |                                63.9862 |                                9.8869 |                               -0.0065 |                              0.0042 |                                 0.0057 |                                      0.0044 |                                 0.628  |                                -1.6063 |                                66.0108 |                               10.64   |                               -0.0065 |                              0.0044 |                                 0.0058 |                                      0.0045 |                                 0.6469 |                                -1.5908 |                                67.8634 |                               11.3909 |                               -0.0065 |                              0.0045 |                                 0.006  |                                      0.0046 |                                 0.6644 |                                -1.5732 |                                69.5562 |                               12.1383 |                               -0.0065 |                              0.0046 |                                 0.0061 |                                      0.0047 |                                 0.6805 |                                -1.5539 |                                71.1008 |                               12.8812 |                               -0.0065 |                              0.0047 |                                 0.0063 |                                      0.0048 |                                 0.6954 |                                -1.5334 |                                72.5082 |                               13.6185 |                               -0.0064 |                              0.0048 |                                 0.0064 |                                      0.0049 |                                 0.709  |                                -1.5119 |                                73.7886 |                               14.3496 |                               -0.0064 |                              0.0049 |
        |            315 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0005 |                                 0.003  |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0073 |                                 0.0605 |                                0.0015 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0346 |                                 0.3555 |                                0.0107 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0093 |                                -0.0925 |                                 1.1372 |                                0.0411 |                               -0.0003 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0211 |                                -0.1802 |                                 2.5774 |                                0.1086 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0387 |                                -0.2895 |                                 4.7166 |                                0.2271 |                               -0.0011 |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0616 |                                -0.41   |                                 7.4916 |                                0.4056 |                               -0.0015 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.0888 |                                -0.5328 |                                10.784  |                                0.6486 |                               -0.002  |                              0.0005 |                                 0.001  |                                      0.0007 |                                 0.1193 |                                -0.6514 |                                14.4583 |                                0.9562 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1519 |                                -0.7617 |                                18.3853 |                                1.3261 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0011 |                                 0.1859 |                                -0.8613 |                                22.4528 |                                1.754  |                               -0.0033 |                              0.0011 |                                 0.0018 |                                      0.0013 |                                 0.2203 |                                -0.9493 |                                26.5692 |                                2.2346 |                               -0.0037 |                              0.0013 |                                 0.0021 |                                      0.0015 |                                 0.2547 |                                -1.0257 |                                30.6628 |                                2.7624 |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0017 |                                 0.2886 |                                -1.0909 |                                34.6796 |                                3.3316 |                               -0.0044 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3216 |                                -1.1457 |                                38.5804 |                                3.937  |                               -0.0046 |                              0.0019 |                                 0.0028 |                                      0.0021 |                                 0.3535 |                                -1.1911 |                                42.3377 |                                4.5733 |                               -0.0048 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.3842 |                                -1.2281 |                                45.9338 |                                5.2359 |                               -0.005  |                              0.0023 |                                 0.0033 |                                      0.0025 |                                 0.4135 |                                -1.2575 |                                49.3576 |                                5.9207 |                               -0.0052 |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4415 |                                -1.2804 |                                52.604  |                                6.6238 |                               -0.0053 |                              0.0026 |                                 0.0038 |                                      0.0028 |                                 0.4681 |                                -1.2974 |                                55.6717 |                                7.3419 |                               -0.0055 |                              0.0028 |                                 0.004  |                                      0.003  |                                 0.4932 |                                -1.3095 |                                58.5624 |                                8.072  |                               -0.0056 |                              0.003  |                                 0.0042 |                                      0.0031 |                                 0.517  |                                -1.3172 |                                61.28   |                                8.8114 |                               -0.0056 |                              0.0031 |                                 0.0043 |                                      0.0032 |                                 0.5395 |                                -1.3212 |                                63.8299 |                                9.5579 |                               -0.0057 |                              0.0032 |                                 0.0045 |                                      0.0034 |                                 0.5607 |                                -1.322  |                                66.2184 |                               10.3094 |                               -0.0058 |                              0.0034 |                                 0.0047 |                                      0.0035 |                                 0.5806 |                                -1.3201 |                                68.4524 |                               11.064  |                               -0.0058 |                              0.0035 |                                 0.0048 |                                      0.0036 |                                 0.5994 |                                -1.3158 |                                70.5393 |                               11.8204 |                               -0.0058 |                              0.0036 |                                 0.005  |                                      0.0037 |                                 0.617  |                                -1.3095 |                                72.4865 |                               12.5771 |                               -0.0058 |                              0.0037 |
        |            320 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0002 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.001  |                                 0.009  |                                0.0002 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0071 |                                 0.081  |                                0.0022 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0252 |                                 0.3437 |                                0.0112 |                               -0.0001 |                              0      |                                 0.0001 |                                      0      |                                 0.007  |                                -0.0601 |                                 0.9534 |                                0.0363 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.015  |                                -0.1123 |                                 2.0304 |                                0.0883 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0269 |                                -0.1789 |                                 3.6289 |                                0.1775 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0426 |                                -0.2555 |                                 5.7409 |                                0.312  |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0003 |                                 0.0619 |                                -0.3374 |                                 8.3154 |                                0.497  |                               -0.0014 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.084  |                                -0.4206 |                                11.2776 |                                0.7351 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.1085 |                                -0.5021 |                                14.5445 |                                1.0268 |                               -0.0021 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1348 |                                -0.5797 |                                18.0351 |                                1.3709 |                               -0.0024 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1622 |                                -0.6521 |                                21.6752 |                                1.7648 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.1904 |                                -0.7184 |                                25.4005 |                                2.2054 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0012 |                                 0.2189 |                                -0.7783 |                                29.1572 |                                2.6892 |                               -0.0034 |                              0.0012 |                                 0.0018 |                                      0.0013 |                                 0.2474 |                                -0.8318 |                                32.9013 |                                3.2122 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.2757 |                                -0.879  |                                36.5979 |                                3.7707 |                               -0.0039 |                              0.0014 |                                 0.0022 |                                      0.0016 |                                 0.3034 |                                -0.9202 |                                40.2199 |                                4.3609 |                               -0.0041 |                              0.0016 |                                 0.0024 |                                      0.0017 |                                 0.3305 |                                -0.9559 |                                43.7465 |                                4.9792 |                               -0.0043 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3569 |                                -0.9865 |                                47.1628 |                                5.6222 |                               -0.0044 |                              0.0019 |                                 0.0028 |                                      0.002  |                                 0.3824 |                                -1.0123 |                                50.4582 |                                6.2869 |                               -0.0046 |                              0.002  |                                 0.003  |                                      0.0021 |                                 0.4071 |                                -1.0339 |                                53.6256 |                                6.9704 |                               -0.0047 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.4308 |                                -1.0517 |                                56.661  |                                7.6699 |                               -0.0048 |                              0.0023 |                                 0.0033 |                                      0.0024 |                                 0.4536 |                                -1.066  |                                59.5626 |                                8.3831 |                               -0.0049 |                              0.0024 |                                 0.0035 |                                      0.0025 |                                 0.4754 |                                -1.0772 |                                62.3302 |                                9.1078 |                               -0.005  |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4963 |                                -1.0856 |                                64.9653 |                                9.842  |                               -0.0051 |                              0.0026 |                                 0.0037 |                                      0.0027 |                                 0.5163 |                                -1.0917 |                                67.4702 |                               10.584  |                               -0.0051 |                              0.0027 |
        |            325 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0011 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0156 |                                0.0004 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.006  |                                 0.0903 |                                0.0027 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0021 |                                -0.0179 |                                 0.3118 |                                0.0108 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0053 |                                -0.0395 |                                 0.7825 |                                0.0311 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0107 |                                -0.0714 |                                 1.5893 |                                0.071  |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0189 |                                -0.1129 |                                 2.785  |                                0.1382 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0298 |                                -0.1621 |                                 4.3861 |                                0.2394 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0433 |                                -0.2166 |                                 6.3784 |                                0.3798 |                               -0.001  |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0594 |                                -0.2742 |                                 8.7267 |                                0.5627 |                               -0.0012 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.0775 |                                -0.333  |                                11.3831 |                                0.7903 |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0005 |                                 0.0975 |                                -0.3913 |                                14.2945 |                                1.0631 |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0006 |                                 0.1189 |                                -0.4478 |                                17.4076 |                                1.3807 |                               -0.0021 |                              0.0006 |                                 0.0009 |                                      0.0007 |                                 0.1414 |                                -0.5018 |                                20.6717 |                                1.7416 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.1647 |                                -0.5526 |                                24.0406 |                                2.1441 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.1885 |                                -0.5998 |                                27.4736 |                                2.5858 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.2125 |                                -0.6432 |                                30.9357 |                                3.0642 |                               -0.003  |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2366 |                                -0.6829 |                                34.3971 |                                3.5767 |                               -0.0032 |                              0.0011 |                                 0.0017 |                                      0.0012 |                                 0.2607 |                                -0.7188 |                                37.8333 |                                4.1204 |                               -0.0034 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.2844 |                                -0.751  |                                41.2239 |                                4.6926 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.3078 |                                -0.7799 |                                44.553  |                                5.2909 |                               -0.0038 |                              0.0014 |                                 0.0022 |                                      0.0015 |                                 0.3308 |                                -0.8054 |                                47.8076 |                                5.9126 |                               -0.0039 |                              0.0015 |                                 0.0023 |                                      0.0016 |                                 0.3532 |                                -0.8279 |                                50.9779 |                                6.5555 |                               -0.004  |                              0.0016 |                                 0.0025 |                                      0.0017 |                                 0.3751 |                                -0.8475 |                                54.0566 |                                7.2171 |                               -0.0042 |                              0.0017 |                                 0.0026 |                                      0.0018 |                                 0.3964 |                                -0.8645 |                                57.0384 |                                7.8955 |                               -0.0043 |                              0.0018 |                                 0.0028 |                                      0.0019 |                                 0.417  |                                -0.8791 |                                59.9196 |                                8.5886 |                               -0.0044 |                              0.0019 |
        |            330 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0001 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0026 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0208 |                                0.0006 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0048 |                                 0.0909 |                                0.0029 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0017 |                                -0.0126 |                                 0.2722 |                                0.01   |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0039 |                                -0.0262 |                                 0.6342 |                                0.0261 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0077 |                                -0.0462 |                                 1.2403 |                                0.0567 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0133 |                                -0.0726 |                                 2.137  |                                0.1075 |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0209 |                                -0.1045 |                                 3.3492 |                                0.1837 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0305 |                                -0.141  |                                 4.882  |                                0.2901 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0421 |                                -0.1807 |                                 6.7237 |                                0.4302 |                               -0.0009 |                              0.0002 |                                 0.0003 |                                      0.0002 |                                 0.0555 |                                -0.2225 |                                 8.8506 |                                0.6066 |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0706 |                                -0.2653 |                                11.2317 |                                0.8209 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.087  |                                -0.3083 |                                13.8319 |                                1.0739 |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0004 |                                 0.1046 |                                -0.3505 |                                16.6151 |                                1.3656 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0005 |                                 0.1233 |                                -0.3916 |                                19.5456 |                                1.6953 |                               -0.0019 |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.1426 |                                -0.431  |                                22.59   |                                2.0621 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1626 |                                -0.4684 |                                25.7175 |                                2.4644 |                               -0.0024 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.183  |                                -0.5036 |                                28.9002 |                                2.9007 |                               -0.0025 |                              0.0008 |                                 0.0012 |                                      0.0008 |                                 0.2036 |                                -0.5365 |                                32.1138 |                                3.369  |                               -0.0027 |                              0.0008 |                                 0.0014 |                                      0.0009 |                                 0.2243 |                                -0.5671 |                                35.3368 |                                3.8675 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.001  |                                 0.245  |                                -0.5953 |                                38.551  |                                4.3942 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2657 |                                -0.6212 |                                41.7406 |                                4.947  |                               -0.0032 |                              0.0011 |                                 0.0018 |                                      0.0012 |                                 0.2861 |                                -0.6449 |                                44.8926 |                                5.5241 |                               -0.0033 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.3063 |                                -0.6664 |                                47.9962 |                                6.1235 |                               -0.0035 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.3262 |                                -0.6859 |                                51.0424 |                                6.7433 |                               -0.0036 |                              0.0014 |
        |            335 |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0      |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0      |                                 0.0004 |                                0      |                               -0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0042 |                                0.0001 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0012 |                                 0.0238 |                                0.0007 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0037 |                                 0.0861 |                                0.0029 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0013 |                                -0.0089 |                                 0.2321 |                                0.0089 |                               -0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0176 |                                 0.5103 |                                0.0217 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0056 |                                -0.0304 |                                 0.9674 |                                0.0452 |                               -0.0002 |                              0      |                                 0.0001 |                                      0      |                                 0.0095 |                                -0.0475 |                                 1.6421 |                                0.0836 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0149 |                                -0.0685 |                                 2.5606 |                                0.1412 |                               -0.0004 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0217 |                                -0.093  |                                 3.7366 |                                0.2219 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0301 |                                -0.1203 |                                 5.1715 |                                0.329  |                               -0.0006 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0399 |                                -0.1499 |                                 6.8572 |                                0.4652 |                               -0.0008 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0512 |                                -0.181  |                                 8.7783 |                                0.6326 |                               -0.0009 |                              0.0002 |                                 0.0004 |                                      0.0002 |                                 0.0637 |                                -0.213  |                                10.914  |                                0.8326 |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0774 |                                -0.2454 |                                13.2409 |                                1.0661 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0003 |                                 0.092  |                                -0.2776 |                                15.7339 |                                1.3332 |                               -0.0015 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.1076 |                                -0.3093 |                                18.3677 |                                1.6339 |                               -0.0016 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.1238 |                                -0.3402 |                                21.1177 |                                1.9676 |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0005 |                                 0.1407 |                                -0.37   |                                23.9605 |                                2.3335 |                               -0.002  |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.158  |                                -0.3986 |                                26.8744 |                                2.7305 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1756 |                                -0.4258 |                                29.8396 |                                3.1575 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0007 |                                 0.1935 |                                -0.4515 |                                32.838  |                                3.6131 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0008 |                                 0.2115 |                                -0.4758 |                                35.8536 |                                4.0958 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.2296 |                                -0.4985 |                                38.8723 |                                4.6042 |                               -0.0027 |                              0.0009 |                                 0.0014 |                                      0.0009 |                                 0.2477 |                                -0.5197 |                                41.8817 |                                5.1369 |                               -0.0029 |                              0.0009 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
                means it will use the current dividend yield.
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_gamma().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0017 |       0.0047 |       0.0075 |       0.0096 |       0.0111 |       0.0123 |       0.0131 |       0.0136 |       0.014  |       0.0142 |       0.0144 |       0.0145 |       0.0145 |       0.0145 |       0.0145 |       0.0144 |       0.0144 |       0.0143 |       0.0142 |       0.0141 |       0.014  |       0.0139 |       0.0138 |       0.0136 |       0.0135 |       0.0134 |       0.0133 |       0.0132 |
        |            295 |       0      |       0.0002 |       0.0011 |       0.0026 |       0.0041 |       0.0055 |       0.0067 |       0.0077 |       0.0085 |       0.0092 |       0.0097 |       0.0102 |       0.0105 |       0.0108 |       0.011  |       0.0112 |       0.0114 |       0.0115 |       0.0115 |       0.0116 |       0.0116 |       0.0117 |       0.0117 |       0.0117 |       0.0116 |       0.0116 |       0.0116 |       0.0116 |       0.0115 |
        |            300 |       0      |       0      |       0.0002 |       0.0007 |       0.0014 |       0.0023 |       0.0032 |       0.004  |       0.0048 |       0.0055 |       0.0061 |       0.0067 |       0.0071 |       0.0075 |       0.0079 |       0.0082 |       0.0085 |       0.0087 |       0.0089 |       0.009  |       0.0092 |       0.0093 |       0.0094 |       0.0095 |       0.0096 |       0.0096 |       0.0097 |       0.0097 |       0.0097 |
        |            305 |       0      |       0      |       0      |       0.0002 |       0.0004 |       0.0008 |       0.0014 |       0.0019 |       0.0025 |       0.003  |       0.0036 |       0.0041 |       0.0045 |       0.0049 |       0.0053 |       0.0057 |       0.006  |       0.0063 |       0.0065 |       0.0068 |       0.007  |       0.0071 |       0.0073 |       0.0075 |       0.0076 |       0.0077 |       0.0078 |       0.0079 |       0.008  |
        |            310 |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0005 |       0.0008 |       0.0012 |       0.0015 |       0.0019 |       0.0023 |       0.0027 |       0.003  |       0.0034 |       0.0037 |       0.004  |       0.0043 |       0.0046 |       0.0048 |       0.0051 |       0.0053 |       0.0055 |       0.0057 |       0.0058 |       0.006  |       0.0061 |       0.0063 |       0.0064 |
        |            315 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0007 |       0.001  |       0.0012 |       0.0015 |       0.0018 |       0.0021 |       0.0023 |       0.0026 |       0.0028 |       0.0031 |       0.0033 |       0.0036 |       0.0038 |       0.004  |       0.0042 |       0.0043 |       0.0045 |       0.0047 |       0.0048 |       0.005  |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0006 |       0.0008 |       0.001  |       0.0012 |       0.0014 |       0.0016 |       0.0018 |       0.002  |       0.0022 |       0.0024 |       0.0026 |       0.0028 |       0.003  |       0.0031 |       0.0033 |       0.0035 |       0.0036 |       0.0037 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0008 |       0.0009 |       0.0011 |       0.0013 |       0.0014 |       0.0016 |       0.0017 |       0.0019 |       0.002  |       0.0022 |       0.0023 |       0.0025 |       0.0026 |       0.0028 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0014 |       0.0015 |       0.0016 |       0.0018 |       0.0019 |       0.002  |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_dual_gamma().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0015 |       0.0041 |       0.0066 |       0.0084 |       0.0098 |       0.0108 |       0.0115 |       0.012  |       0.0123 |       0.0125 |       0.0127 |       0.0127 |       0.0128 |       0.0128 |       0.0127 |       0.0127 |       0.0126 |       0.0126 |       0.0125 |       0.0124 |       0.0123 |       0.0122 |       0.0121 |       0.012  |       0.0119 |       0.0118 |       0.0117 |       0.0116 |
        |            295 |       0      |       0.0002 |       0.001  |       0.0022 |       0.0035 |       0.0047 |       0.0057 |       0.0065 |       0.0072 |       0.0078 |       0.0083 |       0.0086 |       0.0089 |       0.0092 |       0.0094 |       0.0095 |       0.0096 |       0.0097 |       0.0098 |       0.0098 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0098 |       0.0098 |       0.0098 |
        |            300 |       0      |       0      |       0.0002 |       0.0006 |       0.0012 |       0.0019 |       0.0026 |       0.0033 |       0.004  |       0.0045 |       0.005  |       0.0055 |       0.0059 |       0.0062 |       0.0065 |       0.0067 |       0.0069 |       0.0071 |       0.0073 |       0.0074 |       0.0075 |       0.0076 |       0.0077 |       0.0078 |       0.0079 |       0.0079 |       0.0079 |       0.008  |       0.008  |
        |            305 |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0007 |       0.0011 |       0.0015 |       0.002  |       0.0024 |       0.0028 |       0.0032 |       0.0036 |       0.0039 |       0.0042 |       0.0045 |       0.0048 |       0.005  |       0.0052 |       0.0054 |       0.0055 |       0.0057 |       0.0058 |       0.0059 |       0.006  |       0.0061 |       0.0062 |       0.0063 |       0.0064 |
        |            310 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0006 |       0.0009 |       0.0012 |       0.0015 |       0.0018 |       0.0021 |       0.0023 |       0.0026 |       0.0029 |       0.0031 |       0.0033 |       0.0035 |       0.0037 |       0.0039 |       0.0041 |       0.0042 |       0.0044 |       0.0045 |       0.0046 |       0.0047 |       0.0048 |       0.0049 |
        |            315 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0004 |       0.0005 |       0.0007 |       0.0009 |       0.0011 |       0.0013 |       0.0015 |       0.0017 |       0.0019 |       0.0021 |       0.0023 |       0.0025 |       0.0026 |       0.0028 |       0.003  |       0.0031 |       0.0032 |       0.0034 |       0.0035 |       0.0036 |       0.0037 |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0006 |       0.0007 |       0.0009 |       0.001  |       0.0012 |       0.0013 |       0.0014 |       0.0016 |       0.0017 |       0.0019 |       0.002  |       0.0021 |       0.0023 |       0.0024 |       0.0025 |       0.0026 |       0.0027 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |       0.0015 |       0.0016 |       0.0017 |       0.0018 |       0.0019 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vanna().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0035 |       0.0937 |       0.2559 |       0.4055 |       0.5213 |       0.6062 |       0.6672 |       0.7107 |       0.7413 |       0.7626 |       0.777  |       0.7863 |       0.7918 |       0.7943 |       0.7947 |       0.7935 |       0.791  |       0.7875 |       0.7833 |       0.7785 |       0.7734 |       0.7679 |       0.7623 |       0.7565 |       0.7505 |       0.7446 |       0.7386 |       0.7327 |       0.7267 |
        |            295 |       0.0001 |       0.0137 |       0.0772 |       0.1757 |       0.2806 |       0.377  |       0.4601 |       0.5295 |       0.5866 |       0.6332 |       0.671  |       0.7017 |       0.7264 |       0.7462 |       0.7621 |       0.7747 |       0.7846 |       0.7922 |       0.7979 |       0.8021 |       0.805  |       0.8067 |       0.8076 |       0.8076 |       0.8071 |       0.8059 |       0.8043 |       0.8023 |       0.8    |
        |            300 |       0      |       0.0012 |       0.0164 |       0.0577 |       0.1199 |       0.1919 |       0.2654 |       0.3355 |       0.3998 |       0.4574 |       0.5085 |       0.5533 |       0.5923 |       0.6263 |       0.6558 |       0.6813 |       0.7034 |       0.7225 |       0.739  |       0.7531 |       0.7653 |       0.7756 |       0.7845 |       0.792  |       0.7983 |       0.8036 |       0.808  |       0.8115 |       0.8144 |
        |            305 |       0      |       0.0001 |       0.0025 |       0.0148 |       0.0417 |       0.0818 |       0.1309 |       0.1846 |       0.2396 |       0.2934 |       0.3448 |       0.393  |       0.4376 |       0.4785 |       0.5158 |       0.5498 |       0.5805 |       0.6082 |       0.6333 |       0.6559 |       0.6762 |       0.6945 |       0.711  |       0.7257 |       0.739  |       0.7509 |       0.7615 |       0.7711 |       0.7796 |
        |            310 |       0      |       0      |       0.0003 |       0.003  |       0.012  |       0.0297 |       0.0561 |       0.0896 |       0.128  |       0.1693 |       0.212  |       0.2547 |       0.2965 |       0.3368 |       0.3753 |       0.4117 |       0.446  |       0.478  |       0.5079 |       0.5357 |       0.5615 |       0.5854 |       0.6076 |       0.628  |       0.6469 |       0.6644 |       0.6805 |       0.6954 |       0.709  |
        |            315 |       0      |       0      |       0      |       0.0005 |       0.0029 |       0.0093 |       0.0211 |       0.0387 |       0.0616 |       0.0888 |       0.1193 |       0.1519 |       0.1859 |       0.2203 |       0.2547 |       0.2886 |       0.3216 |       0.3535 |       0.3842 |       0.4135 |       0.4415 |       0.4681 |       0.4932 |       0.517  |       0.5395 |       0.5607 |       0.5806 |       0.5994 |       0.617  |
        |            320 |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0025 |       0.007  |       0.015  |       0.0269 |       0.0426 |       0.0619 |       0.084  |       0.1085 |       0.1348 |       0.1622 |       0.1904 |       0.2189 |       0.2474 |       0.2757 |       0.3034 |       0.3305 |       0.3569 |       0.3824 |       0.4071 |       0.4308 |       0.4536 |       0.4754 |       0.4963 |       0.5163 |
        |            325 |       0      |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0021 |       0.0053 |       0.0107 |       0.0189 |       0.0298 |       0.0433 |       0.0594 |       0.0775 |       0.0975 |       0.1189 |       0.1414 |       0.1647 |       0.1885 |       0.2125 |       0.2366 |       0.2607 |       0.2844 |       0.3078 |       0.3308 |       0.3532 |       0.3751 |       0.3964 |       0.417  |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0017 |       0.0039 |       0.0077 |       0.0133 |       0.0209 |       0.0305 |       0.0421 |       0.0555 |       0.0706 |       0.087  |       0.1046 |       0.1233 |       0.1426 |       0.1626 |       0.183  |       0.2036 |       0.2243 |       0.245  |       0.2657 |       0.2861 |       0.3063 |       0.3262 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0013 |       0.0029 |       0.0056 |       0.0095 |       0.0149 |       0.0217 |       0.0301 |       0.0399 |       0.0512 |       0.0637 |       0.0774 |       0.092  |       0.1076 |       0.1238 |       0.1407 |       0.158  |       0.1756 |       0.1935 |       0.2115 |       0.2296 |       0.2477 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_charm().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.21   |      -2.7884 |      -5.0955 |      -6.0751 |      -6.2674 |      -6.092  |      -5.7653 |      -5.3898 |      -5.013  |      -4.6555 |      -4.3254 |      -4.0245 |      -3.752  |      -3.5059 |      -3.2836 |      -3.0827 |      -2.9007 |      -2.7356 |      -2.5853 |      -2.4483 |      -2.323  |      -2.2082 |      -2.1026 |      -2.0054 |      -1.9156 |      -1.8325 |      -1.7555 |      -1.6839 |      -1.6172 |
        |            295 |      -0.0035 |      -0.4076 |      -1.5345 |      -2.625  |      -3.362  |      -3.774  |      -3.9575 |      -3.995  |      -3.9437 |      -3.8406 |      -3.7092 |      -3.5639 |      -3.4139 |      -3.2646 |      -3.1193 |      -2.9797 |      -2.8469 |      -2.7213 |      -2.6029 |      -2.4915 |      -2.3869 |      -2.2888 |      -2.1966 |      -2.1102 |      -2.029  |      -1.9527 |      -1.8809 |      -1.8134 |      -1.7497 |
        |            300 |      -0      |      -0.0359 |      -0.3248 |      -0.8607 |      -1.4332 |      -1.9161 |      -2.2761 |      -2.5226 |      -2.6776 |      -2.7631 |      -2.7978 |      -2.7962 |      -2.769  |      -2.7242 |      -2.6676 |      -2.6035 |      -2.5348 |      -2.4638 |      -2.392  |      -2.3205 |      -2.2501 |      -2.1813 |      -2.1144 |      -2.0497 |      -1.9872 |      -1.9272 |      -1.8695 |      -1.8141 |      -1.7611 |
        |            305 |      -0      |      -0.002  |      -0.0498 |      -0.22   |      -0.4975 |      -0.8156 |      -1.1206 |      -1.3852 |      -1.6003 |      -1.7672 |      -1.8913 |      -1.9793 |      -2.0379 |      -2.0729 |      -2.0892 |      -2.091  |      -2.0816 |      -2.0635 |      -2.0389 |      -2.0095 |      -1.9765 |      -1.941  |      -1.9038 |      -1.8655 |      -1.8267 |      -1.7877 |      -1.7488 |      -1.7103 |      -1.6723 |
        |            310 |      -0      |      -0.0001 |      -0.0057 |      -0.0447 |      -0.1431 |      -0.2957 |      -0.4794 |      -0.6708 |      -0.8533 |      -1.0177 |      -1.16   |      -1.2793 |      -1.3769 |      -1.4547 |      -1.5152 |      -1.5607 |      -1.5935 |      -1.6155 |      -1.6287 |      -1.6344 |      -1.634  |      -1.6286 |      -1.6191 |      -1.6063 |      -1.5908 |      -1.5732 |      -1.5539 |      -1.5334 |      -1.5119 |
        |            315 |      -0      |      -0      |      -0.0005 |      -0.0073 |      -0.0346 |      -0.0925 |      -0.1802 |      -0.2895 |      -0.41   |      -0.5328 |      -0.6514 |      -0.7617 |      -0.8613 |      -0.9493 |      -1.0257 |      -1.0909 |      -1.1457 |      -1.1911 |      -1.2281 |      -1.2575 |      -1.2804 |      -1.2974 |      -1.3095 |      -1.3172 |      -1.3212 |      -1.322  |      -1.3201 |      -1.3158 |      -1.3095 |
        |            320 |      -0      |      -0      |      -0      |      -0.001  |      -0.0071 |      -0.0252 |      -0.0601 |      -0.1123 |      -0.1789 |      -0.2555 |      -0.3374 |      -0.4206 |      -0.5021 |      -0.5797 |      -0.6521 |      -0.7184 |      -0.7783 |      -0.8318 |      -0.879  |      -0.9202 |      -0.9559 |      -0.9865 |      -1.0123 |      -1.0339 |      -1.0517 |      -1.066  |      -1.0772 |      -1.0856 |      -1.0917 |
        |            325 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0013 |      -0.006  |      -0.0179 |      -0.0395 |      -0.0714 |      -0.1129 |      -0.1621 |      -0.2166 |      -0.2742 |      -0.333  |      -0.3913 |      -0.4478 |      -0.5018 |      -0.5526 |      -0.5998 |      -0.6432 |      -0.6829 |      -0.7188 |      -0.751  |      -0.7799 |      -0.8054 |      -0.8279 |      -0.8475 |      -0.8645 |      -0.8791 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0013 |      -0.0048 |      -0.0126 |      -0.0262 |      -0.0462 |      -0.0726 |      -0.1045 |      -0.141  |      -0.1807 |      -0.2225 |      -0.2653 |      -0.3083 |      -0.3505 |      -0.3916 |      -0.431  |      -0.4684 |      -0.5036 |      -0.5365 |      -0.5671 |      -0.5953 |      -0.6212 |      -0.6449 |      -0.6664 |      -0.6859 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0012 |      -0.0037 |      -0.0089 |      -0.0176 |      -0.0304 |      -0.0475 |      -0.0685 |      -0.093  |      -0.1203 |      -0.1499 |      -0.181  |      -0.213  |      -0.2454 |      -0.2776 |      -0.3093 |      -0.3402 |      -0.37   |      -0.3986 |      -0.4258 |      -0.4515 |      -0.4758 |      -0.4985 |      -0.5197 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vomma().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.1909 |       5.034  |      13.7018 |      21.6286 |      27.6957 |      32.0786 |      35.1694 |      37.3128 |      38.7684 |      39.7237 |      40.3125 |      40.6311 |      40.7489 |      40.7168 |      40.5723 |      40.3435 |      40.0514 |      39.7119 |      39.3374 |      38.9373 |      38.5188 |      38.0877 |      37.6483 |      37.2041 |      36.7578 |      36.3116 |      35.867  |      35.4255 |      34.9879 |
        |            295 |       0.004  |       0.9333 |       5.2413 |      11.8885 |      18.9276 |      25.3547 |      30.8469 |      35.3905 |      39.085  |      42.058  |      44.4324 |      46.3154 |      47.7963 |      48.9485 |      49.8316 |      50.4938 |      50.9741 |      51.3044 |      51.5103 |      51.6132 |      51.6304 |      51.5762 |      51.4624 |      51.2989 |      51.094  |      50.8545 |      50.5862 |      50.294  |      49.982  |
        |            300 |       0      |       0.0994 |       1.3415 |       4.7182 |       9.7751 |      15.6098 |      21.5337 |      27.1501 |      32.2708 |      36.8318 |      40.8353 |      44.3163 |      47.3234 |      49.9086 |      52.1222 |      54.0107 |      55.6157 |      56.974  |      58.1176 |      59.0743 |      59.8683 |      60.5203 |      61.0485 |      61.4683 |      61.7933 |      62.0353 |      62.2043 |      62.3092 |      62.3578 |
        |            305 |       0      |       0.0064 |       0.2409 |       1.4122 |       3.9759 |       7.7904 |      12.439  |      17.5024 |      22.6588 |      27.6923 |      32.4718 |      36.9264 |      41.0252 |      44.7625 |      48.1479 |      51.1997 |      53.9404 |      56.3943 |      58.586  |      60.5391 |      62.2757 |      63.8164 |      65.1801 |      66.3839 |      67.4435 |      68.3729 |      69.1847 |      69.8902 |      70.4997 |
        |            310 |       0      |       0.0003 |       0.0314 |       0.3282 |       1.3086 |       3.2338 |       6.0946 |       9.7116 |      13.8509 |      18.2919 |      22.8541 |      27.4017 |      31.8386 |      36.1007 |      40.1481 |      43.9585 |      47.5224 |      50.8388 |      53.9127 |      56.7528 |      59.3701 |      61.777  |      63.9862 |      66.0108 |      67.8634 |      69.5562 |      71.1008 |      72.5082 |      73.7886 |
        |            315 |       0      |       0      |       0.003  |       0.0605 |       0.3555 |       1.1372 |       2.5774 |       4.7166 |       7.4916 |      10.784  |      14.4583 |      18.3853 |      22.4528 |      26.5692 |      30.6628 |      34.6796 |      38.5804 |      42.3377 |      45.9338 |      49.3576 |      52.604  |      55.6717 |      58.5624 |      61.28   |      63.8299 |      66.2184 |      68.4524 |      70.5393 |      72.4865 |
        |            320 |       0      |       0      |       0.0002 |       0.009  |       0.081  |       0.3437 |       0.9534 |       2.0304 |       3.6289 |       5.7409 |       8.3154 |      11.2776 |      14.5445 |      18.0351 |      21.6752 |      25.4005 |      29.1572 |      32.9013 |      36.5979 |      40.2199 |      43.7465 |      47.1628 |      50.4582 |      53.6256 |      56.661  |      59.5626 |      62.3302 |      64.9653 |      67.4702 |
        |            325 |       0      |       0      |       0      |       0.0011 |       0.0156 |       0.0903 |       0.3118 |       0.7825 |       1.5893 |       2.785  |       4.3861 |       6.3784 |       8.7267 |      11.3831 |      14.2945 |      17.4076 |      20.6717 |      24.0406 |      27.4736 |      30.9357 |      34.3971 |      37.8333 |      41.2239 |      44.553  |      47.8076 |      50.9779 |      54.0566 |      57.0384 |      59.9196 |
        |            330 |       0      |       0      |       0      |       0.0001 |       0.0026 |       0.0208 |       0.0909 |       0.2722 |       0.6342 |       1.2403 |       2.137  |       3.3492 |       4.882  |       6.7237 |       8.8506 |      11.2317 |      13.8319 |      16.6151 |      19.5456 |      22.59   |      25.7175 |      28.9002 |      32.1138 |      35.3368 |      38.551  |      41.7406 |      44.8926 |      47.9962 |      51.0424 |
        |            335 |       0      |       0      |       0      |       0      |       0.0004 |       0.0042 |       0.0238 |       0.0861 |       0.2321 |       0.5103 |       0.9674 |       1.6421 |       2.5606 |       3.7366 |       5.1715 |       6.8572 |       8.7783 |      10.914  |      13.2409 |      15.7339 |      18.3677 |      21.1177 |      23.9605 |      26.8744 |      29.8396 |      32.838  |      35.8536 |      38.8723 |      41.8817 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_vera().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0026 |       0.1383 |       0.5642 |       1.1867 |       1.8983 |       2.6368 |       3.3706 |       4.0843 |       4.7711 |       5.4285 |       6.0561 |       6.6547 |       7.2257 |       7.7705 |       8.2909 |       8.7883 |       9.2642 |       9.72   |      10.157  |      10.5763 |      10.979  |      11.366  |      11.7383 |      12.0967 |      12.4419 |      12.7747 |      13.0955 |      13.4051 |      13.7039 |
        |            295 |       0      |       0.0203 |       0.1707 |       0.5161 |       1.0265 |       1.6493 |       2.3398 |       3.0664 |       3.8079 |       4.5506 |       5.2857 |       6.0076 |       6.7131 |       7.4001 |       8.0678 |       8.7157 |       9.3439 |       9.9528 |      10.5427 |      11.1143 |      11.6682 |      12.205  |      12.7254 |      13.2301 |      13.7196 |      14.1945 |      14.6556 |      15.1032 |      15.538  |
        |            300 |       0      |       0.0018 |       0.0362 |       0.1699 |       0.4399 |       0.8425 |       1.3554 |       1.9523 |       2.6095 |       3.3079 |       4.0326 |       4.7722 |       5.5185 |       6.2651 |       7.0075 |       7.7423 |       8.4672 |       9.1805 |       9.881  |      10.568  |      11.2409 |      11.8996 |      12.544  |      13.174  |      13.7899 |      14.3918 |      14.98   |      15.5547 |      16.1163 |
        |            305 |       0      |       0.0001 |       0.0056 |       0.0436 |       0.1532 |       0.3602 |       0.6707 |       1.0782 |       1.5698 |       2.1309 |       2.7476 |       3.4074 |       4.0997 |       4.8156 |       5.5478 |       6.2905 |       7.039  |       7.7894 |       8.5388 |       9.2846 |      10.0249 |      10.7584 |      11.4837 |      12.2001 |      12.9068 |      13.6033 |      14.2893 |      14.9644 |      15.6286 |
        |            310 |       0      |       0      |       0.0006 |       0.0089 |       0.0442 |       0.131  |       0.288  |       0.5244 |       0.8411 |       1.2338 |       1.6951 |       2.2165 |       2.7892 |       3.4048 |       4.0557 |       4.7352 |       5.4374 |       6.1571 |       6.8901 |       7.6324 |       8.3811 |       9.1333 |       9.8869 |      10.64   |      11.3909 |      12.1383 |      12.8812 |      13.6185 |      14.3496 |
        |            315 |       0      |       0      |       0.0001 |       0.0015 |       0.0107 |       0.0411 |       0.1086 |       0.2271 |       0.4056 |       0.6486 |       0.9562 |       1.3261 |       1.754  |       2.2346 |       2.7624 |       3.3316 |       3.937  |       4.5733 |       5.2359 |       5.9207 |       6.6238 |       7.3419 |       8.072  |       8.8114 |       9.5579 |      10.3094 |      11.064  |      11.8204 |      12.5771 |
        |            320 |       0      |       0      |       0      |       0.0002 |       0.0022 |       0.0112 |       0.0363 |       0.0883 |       0.1775 |       0.312  |       0.497  |       0.7351 |       1.0268 |       1.3709 |       1.7648 |       2.2054 |       2.6892 |       3.2122 |       3.7707 |       4.3609 |       4.9792 |       5.6222 |       6.2869 |       6.9704 |       7.6699 |       8.3831 |       9.1078 |       9.842  |      10.584  |
        |            325 |       0      |       0      |       0      |       0      |       0.0004 |       0.0027 |       0.0108 |       0.0311 |       0.071  |       0.1382 |       0.2394 |       0.3798 |       0.5627 |       0.7903 |       1.0631 |       1.3807 |       1.7416 |       2.1441 |       2.5858 |       3.0642 |       3.5767 |       4.1204 |       4.6926 |       5.2909 |       5.9126 |       6.5555 |       7.2171 |       7.8955 |       8.5886 |
        |            330 |       0      |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0029 |       0.01   |       0.0261 |       0.0567 |       0.1075 |       0.1837 |       0.2901 |       0.4302 |       0.6066 |       0.8209 |       1.0739 |       1.3656 |       1.6953 |       2.0621 |       2.4644 |       2.9007 |       3.369  |       3.8675 |       4.3942 |       4.947  |       5.5241 |       6.1235 |       6.7433 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0007 |       0.0029 |       0.0089 |       0.0217 |       0.0452 |       0.0836 |       0.1412 |       0.2219 |       0.329  |       0.4652 |       0.6326 |       0.8326 |       1.0661 |       1.3332 |       1.6339 |       1.9676 |       2.3335 |       2.7305 |       3.1575 |       3.6131 |       4.0958 |       4.6042 |       5.1369 |
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

        The formula as usually published carries a leading minus sign because it differentiates with respect to the
        time to maturity, which runs opposite to elapsed calendar time. That sign is absorbed here so that Veta,
        like Theta, Charm and Color, measures the change per unit of time that passes: a long option loses Vega as
        expiry approaches, so its Veta is negative.

        It is common practice to divide the mathematical result of veta by 100 times the number of days per year to
        reduce the value to the percentage change in vega per one day. This is also done here.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_veta().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0003 |      -0.0047 |      -0.009  |      -0.0113 |      -0.0123 |      -0.0125 |      -0.0124 |      -0.012  |      -0.0116 |      -0.0112 |      -0.0108 |      -0.0104 |      -0.01   |      -0.0097 |      -0.0094 |      -0.0091 |      -0.0088 |      -0.0085 |      -0.0082 |      -0.008  |      -0.0078 |      -0.0076 |      -0.0074 |      -0.0072 |      -0.007  |      -0.0069 |      -0.0067 |      -0.0066 |      -0.0065 |
        |            295 |      -0      |      -0.0008 |      -0.0032 |      -0.0057 |      -0.0076 |      -0.0088 |      -0.0095 |      -0.0098 |      -0.01   |      -0.01   |      -0.0099 |      -0.0098 |      -0.0096 |      -0.0094 |      -0.0092 |      -0.009  |      -0.0088 |      -0.0086 |      -0.0084 |      -0.0082 |      -0.008  |      -0.0078 |      -0.0077 |      -0.0075 |      -0.0073 |      -0.0072 |      -0.007  |      -0.0069 |      -0.0068 |
        |            300 |      -0      |      -0.0001 |      -0.0008 |      -0.0022 |      -0.0037 |      -0.005  |      -0.0061 |      -0.0069 |      -0.0075 |      -0.0079 |      -0.0082 |      -0.0083 |      -0.0084 |      -0.0084 |      -0.0084 |      -0.0083 |      -0.0083 |      -0.0082 |      -0.0081 |      -0.008  |      -0.0078 |      -0.0077 |      -0.0076 |      -0.0075 |      -0.0074 |      -0.0072 |      -0.0071 |      -0.007  |      -0.0069 |
        |            305 |      -0      |      -0      |      -0.0001 |      -0.0006 |      -0.0014 |      -0.0024 |      -0.0034 |      -0.0042 |      -0.005  |      -0.0056 |      -0.006  |      -0.0064 |      -0.0067 |      -0.0069 |      -0.0071 |      -0.0072 |      -0.0073 |      -0.0073 |      -0.0073 |      -0.0073 |      -0.0073 |      -0.0072 |      -0.0072 |      -0.0071 |      -0.0071 |      -0.007  |      -0.0069 |      -0.0068 |      -0.0068 |
        |            310 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0005 |      -0.001  |      -0.0016 |      -0.0023 |      -0.0029 |      -0.0035 |      -0.0041 |      -0.0045 |      -0.0049 |      -0.0053 |      -0.0056 |      -0.0058 |      -0.006  |      -0.0061 |      -0.0062 |      -0.0063 |      -0.0064 |      -0.0064 |      -0.0065 |      -0.0065 |      -0.0065 |      -0.0065 |      -0.0065 |      -0.0064 |      -0.0064 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0007 |      -0.0011 |      -0.0015 |      -0.002  |      -0.0025 |      -0.0029 |      -0.0033 |      -0.0037 |      -0.0041 |      -0.0044 |      -0.0046 |      -0.0048 |      -0.005  |      -0.0052 |      -0.0053 |      -0.0055 |      -0.0056 |      -0.0056 |      -0.0057 |      -0.0058 |      -0.0058 |      -0.0058 |      -0.0058 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0005 |      -0.0007 |      -0.001  |      -0.0014 |      -0.0017 |      -0.0021 |      -0.0024 |      -0.0028 |      -0.0031 |      -0.0034 |      -0.0036 |      -0.0039 |      -0.0041 |      -0.0043 |      -0.0044 |      -0.0046 |      -0.0047 |      -0.0048 |      -0.0049 |      -0.005  |      -0.0051 |      -0.0051 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0005 |      -0.0007 |      -0.001  |      -0.0012 |      -0.0015 |      -0.0018 |      -0.0021 |      -0.0023 |      -0.0026 |      -0.0028 |      -0.003  |      -0.0032 |      -0.0034 |      -0.0036 |      -0.0038 |      -0.0039 |      -0.004  |      -0.0042 |      -0.0043 |      -0.0044 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0005 |      -0.0007 |      -0.0009 |      -0.0011 |      -0.0013 |      -0.0015 |      -0.0017 |      -0.0019 |      -0.0022 |      -0.0024 |      -0.0025 |      -0.0027 |      -0.0029 |      -0.0031 |      -0.0032 |      -0.0033 |      -0.0035 |      -0.0036 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0002 |      -0.0004 |      -0.0005 |      -0.0006 |      -0.0008 |      -0.0009 |      -0.0011 |      -0.0013 |      -0.0015 |      -0.0016 |      -0.0018 |      -0.002  |      -0.0022 |      -0.0023 |      -0.0025 |      -0.0026 |      -0.0027 |      -0.0029 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_partial_derivative().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0015 |       0.0041 |       0.0066 |       0.0084 |       0.0098 |       0.0108 |       0.0115 |       0.012  |       0.0123 |       0.0125 |       0.0127 |       0.0127 |       0.0128 |       0.0128 |       0.0127 |       0.0127 |       0.0126 |       0.0126 |       0.0125 |       0.0124 |       0.0123 |       0.0122 |       0.0121 |       0.012  |       0.0119 |       0.0118 |       0.0117 |       0.0116 |
        |            295 |       0      |       0.0002 |       0.001  |       0.0022 |       0.0035 |       0.0047 |       0.0057 |       0.0065 |       0.0072 |       0.0078 |       0.0083 |       0.0086 |       0.0089 |       0.0092 |       0.0094 |       0.0095 |       0.0096 |       0.0097 |       0.0098 |       0.0098 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0099 |       0.0098 |       0.0098 |       0.0098 |
        |            300 |       0      |       0      |       0.0002 |       0.0006 |       0.0012 |       0.0019 |       0.0026 |       0.0033 |       0.004  |       0.0045 |       0.005  |       0.0055 |       0.0059 |       0.0062 |       0.0065 |       0.0067 |       0.0069 |       0.0071 |       0.0073 |       0.0074 |       0.0075 |       0.0076 |       0.0077 |       0.0078 |       0.0079 |       0.0079 |       0.0079 |       0.008  |       0.008  |
        |            305 |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0007 |       0.0011 |       0.0015 |       0.002  |       0.0024 |       0.0028 |       0.0032 |       0.0036 |       0.0039 |       0.0042 |       0.0045 |       0.0048 |       0.005  |       0.0052 |       0.0054 |       0.0055 |       0.0057 |       0.0058 |       0.0059 |       0.006  |       0.0061 |       0.0062 |       0.0063 |       0.0064 |
        |            310 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0006 |       0.0009 |       0.0012 |       0.0015 |       0.0018 |       0.0021 |       0.0023 |       0.0026 |       0.0029 |       0.0031 |       0.0033 |       0.0035 |       0.0037 |       0.0039 |       0.0041 |       0.0042 |       0.0044 |       0.0045 |       0.0046 |       0.0047 |       0.0048 |       0.0049 |
        |            315 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0004 |       0.0005 |       0.0007 |       0.0009 |       0.0011 |       0.0013 |       0.0015 |       0.0017 |       0.0019 |       0.0021 |       0.0023 |       0.0025 |       0.0026 |       0.0028 |       0.003  |       0.0031 |       0.0032 |       0.0034 |       0.0035 |       0.0036 |       0.0037 |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0006 |       0.0007 |       0.0009 |       0.001  |       0.0012 |       0.0013 |       0.0014 |       0.0016 |       0.0017 |       0.0019 |       0.002  |       0.0021 |       0.0023 |       0.0024 |       0.0025 |       0.0026 |       0.0027 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |       0.0015 |       0.0016 |       0.0017 |       0.0018 |       0.0019 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
        |            215 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0      |                                  0.0002 |                                -0      |                                 0      |                                -0.0001 |                                  0.0462 |                                -0      |                                 0      |                                -0.0012 |                                  0.6515 |                                -0      |                                 0.0002 |                                -0.0046 |                                  2.8954 |                                -0      |                                 0.0005 |                                -0.01   |                                  7.3084 |                                -0      |                                 0.001  |                                -0.0163 |                                 13.432  |                                -0      |                                 0.0016 |                                -0.0224 |                                 20.3112 |                                -0      |                                 0.0022 |                                -0.0277 |                                 27.0127 |                                -0      |                                 0.0028 |                                -0.0318 |                                 32.8595 |                                -0      |                                 0.0034 |                                -0.0346 |                                 37.4608 |                                -0      |                                 0.0039 |                                -0.0364 |                                 40.6562 |                                -0      |                                 0.0043 |                                -0.0373 |                                 42.4427 |                                -0      |                                 0.0047 |                                -0.0374 |                                 42.9123 |                                -0.0001 |                                 0.0049 |                                -0.037  |                                 42.2066 |                                -0.0001 |                                 0.0052 |                                -0.0362 |                                 40.4868 |                                -0.0001 |                                 0.0053 |                                -0.0351 |                                 37.9149 |                                -0.0001 |                                 0.0054 |                                -0.0338 |                                 34.6439 |                                -0.0001 |                                 0.0055 |                                -0.0323 |                                 30.812  |                                -0.0001 |                                 0.0055 |                                -0.0308 |                                 26.5409 |                                -0.0001 |                                 0.0055 |                                -0.0292 |                                 21.9357 |                                -0.0001 |                                 0.0055 |                                -0.0277 |                                 17.0858 |                                -0.0001 |                                 0.0054 |                                -0.0262 |                                 12.0663 |                                -0.0001 |                                 0.0053 |                                -0.0247 |                                  6.9398 |                                -0.0001 |                                 0.0052 |                                -0.0232 |                                  1.7581 |                                -0.0001 |                                 0.0051 |                                -0.0218 |                                 -3.4366 |                                -0.0001 |                                 0.005  |                                -0.0205 |                                 -8.61   |                                -0.0001 |                                 0.0049 |                                -0.0193 |                                -13.7345 |                                -0.0001 |                                 0.0048 |                                -0.0181 |                                -18.7881 |
        |            220 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0      |                                  0.0023 |                                -0      |                                 0      |                                -0.0007 |                                  0.2278 |                                -0      |                                 0.0002 |                                -0.0045 |                                  1.9552 |                                -0      |                                 0.0005 |                                -0.0124 |                                  6.418  |                                -0      |                                 0.0012 |                                -0.0222 |                                 13.1705 |                                -0      |                                 0.0019 |                                -0.0315 |                                 20.7717 |                                -0      |                                 0.0028 |                                -0.0389 |                                 27.8541 |                                -0      |                                 0.0035 |                                -0.0442 |                                 33.5438 |                                -0      |                                 0.0042 |                                -0.0473 |                                 37.4473 |                                -0.0001 |                                 0.0048 |                                -0.0488 |                                 39.505  |                                -0.0001 |                                 0.0052 |                                -0.0489 |                                 39.8494 |                                -0.0001 |                                 0.0056 |                                -0.048  |                                 38.7036 |                                -0.0001 |                                 0.0058 |                                -0.0465 |                                 36.3188 |                                -0.0001 |                                 0.006  |                                -0.0446 |                                 32.94   |                                -0.0001 |                                 0.0061 |                                -0.0424 |                                 28.7889 |                                -0.0001 |                                 0.0061 |                                -0.04   |                                 24.0575 |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                 18.9074 |                                -0.0001 |                                 0.006  |                                -0.0352 |                                 13.4718 |                                -0.0001 |                                 0.0059 |                                -0.0329 |                                  7.8588 |                                -0.0001 |                                 0.0058 |                                -0.0306 |                                  2.1549 |                                -0.0001 |                                 0.0057 |                                -0.0285 |                                 -3.5714 |                                -0.0001 |                                 0.0055 |                                -0.0265 |                                 -9.2666 |                                -0.0001 |                                 0.0053 |                                -0.0246 |                                -14.8895 |                                -0.0001 |                                 0.0052 |                                -0.0228 |                                -20.4085 |                                -0.0001 |                                 0.005  |                                -0.0211 |                                -25.8005 |                                -0.0001 |                                 0.0048 |                                -0.0195 |                                -31.0484 |                                -0.0001 |                                 0.0046 |                                -0.018  |                                -36.1406 |                                -0.0001 |                                 0.0044 |                                -0.0167 |                                -41.0692 |
        |            225 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0.0001 |                                  0.0214 |                                -0      |                                 0.0001 |                                -0.0032 |                                  0.8893 |                                -0      |                                 0.0005 |                                -0.014  |                                  4.8693 |                                -0      |                                 0.0013 |                                -0.0293 |                                 12.1163 |                                -0      |                                 0.0023 |                                -0.0439 |                                 20.515  |                                -0      |                                 0.0034 |                                -0.0548 |                                 27.9756 |                                -0      |                                 0.0043 |                                -0.0615 |                                 33.3376 |                                -0.0001 |                                 0.0051 |                                -0.0646 |                                 36.2509 |                                -0.0001 |                                 0.0058 |                                -0.065  |                                 36.8411 |                                -0.0001 |                                 0.0062 |                                -0.0636 |                                 35.4473 |                                -0.0001 |                                 0.0065 |                                -0.061  |                                 32.4677 |                                -0.0001 |                                 0.0067 |                                -0.0576 |                                 28.2829 |                                -0.0001 |                                 0.0067 |                                -0.054  |                                 23.2238 |                                -0.0001 |                                 0.0067 |                                -0.0501 |                                 17.5634 |                                -0.0001 |                                 0.0066 |                                -0.0463 |                                 11.5195 |                                -0.0001 |                                 0.0065 |                                -0.0427 |                                  5.2618 |                                -0.0001 |                                 0.0063 |                                -0.0392 |                                 -1.08   |                                -0.0001 |                                 0.0061 |                                -0.0359 |                                 -7.4084 |                                -0.0001 |                                 0.0059 |                                -0.0328 |                                -13.6515 |                                -0.0001 |                                 0.0057 |                                -0.0299 |                                -19.7572 |                                -0.0001 |                                 0.0055 |                                -0.0273 |                                -25.689  |                                -0.0001 |                                 0.0052 |                                -0.0249 |                                -31.4224 |                                -0.0001 |                                 0.005  |                                -0.0227 |                                -36.9416 |                                -0.0001 |                                 0.0047 |                                -0.0206 |                                -42.238  |                                -0.0001 |                                 0.0045 |                                -0.0187 |                                -47.308  |                                -0.0001 |                                 0.0043 |                                -0.017  |                                -52.1516 |                                -0.0001 |                                 0.004  |                                -0.0155 |                                -56.772  |                                -0.0001 |                                 0.0038 |                                -0.014  |                                -61.1741 |
        |            230 |                                -0      |                                 0      |                                -0      |                                  0      |                                -0      |                                 0      |                                -0.0009 |                                  0.1465 |                                -0      |                                 0.0003 |                                -0.0126 |                                  2.7612 |                                -0      |                                 0.0013 |                                -0.0367 |                                 10.058  |                                -0      |                                 0.0027 |                                -0.0607 |                                 19.3748 |                                -0.0001 |                                 0.0041 |                                -0.0774 |                                 27.3172 |                                -0.0001 |                                 0.0053 |                                -0.086  |                                 32.226  |                                -0.0001 |                                 0.0063 |                                -0.0883 |                                 33.864  |                                -0.0001 |                                 0.0069 |                                -0.0865 |                                 32.6723 |                                -0.0001 |                                 0.0073 |                                -0.0821 |                                 29.2934 |                                -0.0001 |                                 0.0075 |                                -0.0765 |                                 24.3486 |                                -0.0001 |                                 0.0075 |                                -0.0703 |                                 18.3587 |                                -0.0001 |                                 0.0074 |                                -0.064  |                                 11.7303 |                                -0.0001 |                                 0.0073 |                                -0.0579 |                                  4.7675 |                                -0.0001 |                                 0.007  |                                -0.0522 |                                 -2.3091 |                                -0.0001 |                                 0.0068 |                                -0.0469 |                                 -9.3435 |                                -0.0001 |                                 0.0064 |                                -0.042  |                                -16.2285 |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                -22.8927 |                                -0.0001 |                                 0.0058 |                                -0.0336 |                                -29.2909 |                                -0.0001 |                                 0.0055 |                                -0.03   |                                -35.397  |                                -0.0001 |                                 0.0051 |                                -0.0267 |                                -41.1981 |                                -0.0001 |                                 0.0048 |                                -0.0238 |                                -46.691  |                                -0.0001 |                                 0.0045 |                                -0.0212 |                                -51.879  |                                -0.0001 |                                 0.0042 |                                -0.0188 |                                -56.7698 |                                -0.0001 |                                 0.0039 |                                -0.0167 |                                -61.3742 |                                -0.0001 |                                 0.0036 |                                -0.0148 |                                -65.7045 |                                -0.0001 |                                 0.0033 |                                -0.013  |                                -69.7741 |                                -0.0001 |                                 0.0031 |                                -0.0115 |                                -73.597  |                                -0.0001 |                                 0.0028 |                                -0.0101 |                                -77.187  |
        |            235 |                                -0      |                                 0      |                                -0      |                                  0.0003 |                                -0      |                                 0.0001 |                                -0.006  |                                  0.7368 |                                -0      |                                 0.0011 |                                -0.0402 |                                  6.82   |                                -0      |                                 0.0029 |                                -0.0824 |                                 17.1089 |                                -0.0001 |                                 0.0049 |                                -0.1102 |                                 25.8451 |                                -0.0001 |                                 0.0064 |                                -0.1216 |                                 30.2852 |                                -0.0001 |                                 0.0075 |                                -0.1215 |                                 30.3929 |                                -0.0001 |                                 0.0082 |                                -0.115  |                                 27.1375 |                                -0.0001 |                                 0.0084 |                                -0.1053 |                                 21.6003 |                                -0.0001 |                                 0.0084 |                                -0.0947 |                                 14.6748 |                                -0.0001 |                                 0.0083 |                                -0.084  |                                  7.0175 |                                -0.0001 |                                 0.008  |                                -0.074  |                                 -0.9182 |                                -0.0001 |                                 0.0076 |                                -0.0649 |                                 -8.832  |                                -0.0001 |                                 0.0071 |                                -0.0567 |                                -16.5336 |                                -0.0001 |                                 0.0067 |                                -0.0494 |                                -23.9078 |                                -0.0001 |                                 0.0062 |                                -0.043  |                                -30.8909 |                                -0.0001 |                                 0.0058 |                                -0.0373 |                                -37.453  |                                -0.0001 |                                 0.0053 |                                -0.0323 |                                -43.5864 |                                -0.0001 |                                 0.0049 |                                -0.028  |                                -49.2974 |                                -0.0001 |                                 0.0045 |                                -0.0242 |                                -54.6012 |                                -0.0001 |                                 0.0041 |                                -0.0208 |                                -59.5181 |                                -0.0001 |                                 0.0037 |                                -0.0178 |                                -64.0708 |                                -0.0001 |                                 0.0033 |                                -0.0152 |                                -68.283  |                                -0.0001 |                                 0.003  |                                -0.0129 |                                -72.1788 |                                -0.0001 |                                 0.0027 |                                -0.0109 |                                -75.7811 |                                -0.0001 |                                 0.0023 |                                -0.0091 |                                -79.1121 |                                -0.0001 |                                 0.0021 |                                -0.0076 |                                -82.1926 |                                -0.0001 |                                 0.0018 |                                -0.0062 |                                -85.0422 |                                -0.0001 |                                 0.0015 |                                -0.0049 |                                -87.6788 |
        |            240 |                                -0      |                                 0      |                                -0.0002 |                                  0.0079 |                                -0      |                                 0.0005 |                                -0.0292 |                                  2.7235 |                                -0      |                                 0.0028 |                                -0.1053 |                                 13.2951 |                                -0.0001 |                                 0.0056 |                                -0.1583 |                                 23.4951 |                                -0.0001 |                                 0.0077 |                                -0.175  |                                 27.6958 |                                -0.0002 |                                 0.009  |                                -0.1693 |                                 26.083  |                                -0.0002 |                                 0.0095 |                                -0.1534 |                                 20.527  |                                -0.0002 |                                 0.0096 |                                -0.1344 |                                 12.754  |                                -0.0002 |                                 0.0093 |                                -0.1155 |                                  3.975  |                                -0.0002 |                                 0.0088 |                                -0.0982 |                                 -5.0487 |                                -0.0002 |                                 0.0082 |                                -0.0829 |                                -13.8707 |                                -0.0002 |                                 0.0075 |                                -0.0698 |                                -22.2468 |                                -0.0002 |                                 0.0069 |                                -0.0585 |                                -30.058  |                                -0.0002 |                                 0.0062 |                                -0.049  |                                -37.2602 |                                -0.0002 |                                 0.0056 |                                -0.0409 |                                -43.8526 |                                -0.0002 |                                 0.005  |                                -0.034  |                                -49.859  |                                -0.0002 |                                 0.0044 |                                -0.0281 |                                -55.3154 |                                -0.0002 |                                 0.0039 |                                -0.0232 |                                -60.2636 |                                -0.0001 |                                 0.0034 |                                -0.0189 |                                -64.7467 |                                -0.0001 |                                 0.0029 |                                -0.0153 |                                -68.8066 |                                -0.0001 |                                 0.0025 |                                -0.0122 |                                -72.4831 |                                -0.0001 |                                 0.0021 |                                -0.0095 |                                -75.8131 |                                -0.0001 |                                 0.0017 |                                -0.0072 |                                -78.8304 |                                -0.0001 |                                 0.0014 |                                -0.0053 |                                -81.5654 |                                -0.0001 |                                 0.0011 |                                -0.0036 |                                -84.0459 |                                -0.0001 |                                 0.0008 |                                -0.0021 |                                -86.2967 |                                -0.0001 |                                 0.0005 |                                -0.0008 |                                -88.3401 |                                -0.0001 |                                 0.0002 |                                 0.0003 |                                -90.196  |                                -0.0001 |                                -0      |                                 0.0012 |                                -91.8823 |
        |            245 |                                -0      |                                 0      |                                -0.0028 |                                  0.107  |                                -0      |                                 0.0019 |                                -0.1076 |                                  7.3588 |                                -0.0001 |                                 0.006  |                                -0.2274 |                                 19.979  |                                -0.0002 |                                 0.0092 |                                -0.2596 |                                 24.7349 |                                -0.0002 |                                 0.0107 |                                -0.2415 |                                 21.3428 |                                -0.0002 |                                 0.011  |                                -0.2068 |                                 13.2976 |                                -0.0002 |                                 0.0106 |                                -0.1706 |                                  3.3211 |                                -0.0002 |                                 0.0099 |                                -0.1383 |                                 -7.0203 |                                -0.0002 |                                 0.009  |                                -0.111  |                                -16.9349 |                                -0.0002 |                                 0.008  |                                -0.0886 |                                -26.0709 |                                -0.0002 |                                 0.007  |                                -0.0704 |                                -34.3091 |                                -0.0002 |                                 0.0061 |                                -0.0557 |                                -41.648  |                                -0.0002 |                                 0.0052 |                                -0.0438 |                                -48.141  |                                -0.0002 |                                 0.0044 |                                -0.0342 |                                -53.8645 |                                -0.0002 |                                 0.0037 |                                -0.0263 |                                -58.9004 |                                -0.0002 |                                 0.003  |                                -0.0199 |                                -63.3285 |                                -0.0002 |                                 0.0024 |                                -0.0146 |                                -67.2222 |                                -0.0002 |                                 0.0019 |                                -0.0102 |                                -70.6474 |                                -0.0002 |                                 0.0014 |                                -0.0067 |                                -73.6624 |                                -0.0002 |                                 0.0009 |                                -0.0037 |                                -76.3182 |                                -0.0001 |                                 0.0005 |                                -0.0012 |                                -78.6593 |                                -0.0001 |                                 0.0001 |                                 0.0008 |                                -80.7241 |                                -0.0001 |                                -0.0002 |                                 0.0025 |                                -82.5461 |                                -0.0001 |                                -0.0005 |                                 0.0039 |                                -84.1545 |                                -0.0001 |                                -0.0008 |                                 0.0051 |                                -85.5744 |                                -0.0001 |                                -0.0011 |                                 0.0061 |                                -86.8277 |                                -0.0001 |                                -0.0014 |                                 0.0069 |                                -87.9336 |                                -0.0001 |                                -0.0016 |                                 0.0076 |                                -88.9087 |                                -0.0001 |                                -0.0018 |                                 0.0081 |                                -89.7677 |
        |            250 |                                -0      |                                 0.0003 |                                -0.0306 |                                  0.8571 |                                -0.0001 |                                 0.0054 |                                -0.3043 |                                 14.1998 |                                -0.0002 |                                 0.0106 |                                -0.4024 |                                 21.6786 |                                -0.0003 |                                 0.0127 |                                -0.3588 |                                 16.7741 |                                -0.0003 |                                 0.0126 |                                -0.2842 |                                  6.08   |                                -0.0003 |                                 0.0115 |                                -0.2158 |                                 -5.9579 |                                -0.0003 |                                 0.0101 |                                -0.1611 |                                -17.3945 |                                -0.0003 |                                 0.0086 |                                -0.1192 |                                -27.5542 |                                -0.0003 |                                 0.0071 |                                -0.0875 |                                -36.3102 |                                -0.0003 |                                 0.0058 |                                -0.0635 |                                -43.7501 |                                -0.0003 |                                 0.0046 |                                -0.0454 |                                -50.0316 |                                -0.0002 |                                 0.0036 |                                -0.0315 |                                -55.3222 |                                -0.0002 |                                 0.0026 |                                -0.0208 |                                -59.7766 |                                -0.0002 |                                 0.0018 |                                -0.0126 |                                -63.5291 |                                -0.0002 |                                 0.0011 |                                -0.0062 |                                -66.6934 |                                -0.0002 |                                 0.0004 |                                -0.0012 |                                -69.3643 |                                -0.0002 |                                -0.0001 |                                 0.0027 |                                -71.6204 |                                -0.0002 |                                -0.0006 |                                 0.0058 |                                -73.5269 |                                -0.0002 |                                -0.0011 |                                 0.0082 |                                -75.1378 |                                -0.0002 |                                -0.0015 |                                 0.01   |                                -76.4979 |                                -0.0002 |                                -0.0018 |                                 0.0115 |                                -77.6445 |                                -0.0001 |                                -0.0022 |                                 0.0126 |                                -78.6089 |                                -0.0001 |                                -0.0024 |                                 0.0135 |                                -79.4171 |                                -0.0001 |                                -0.0027 |                                 0.0142 |                                -80.0914 |                                -0.0001 |                                -0.0029 |                                 0.0146 |                                -80.6502 |                                -0.0001 |                                -0.0031 |                                 0.015  |                                -81.1094 |                                -0.0001 |                                -0.0033 |                                 0.0152 |                                -81.4824 |                                -0.0001 |                                -0.0035 |                                 0.0154 |                                -81.7808 |                                -0.0001 |                                -0.0036 |                                 0.0154 |                                -82.0143 |
        |            255 |                                -0      |                                 0.0019 |                                -0.2109 |                                  4.0058 |                                -0.0003 |                                 0.0115 |                                -0.6549 |                                 18.2481 |                                -0.0004 |                                 0.0152 |                                -0.5732 |                                 13.2152 |                                -0.0004 |                                 0.0144 |                                -0.4056 |                                 -0.3135 |                                -0.0004 |                                 0.0121 |                                -0.2723 |                                -14.1944 |                                -0.0004 |                                 0.0097 |                                -0.1794 |                                -26.0853 |                                -0.0004 |                                 0.0074 |                                -0.1163 |                                -35.7002 |                                -0.0003 |                                 0.0054 |                                -0.0733 |                                -43.32   |                                -0.0003 |                                 0.0037 |                                -0.0437 |                                -49.322  |                                -0.0003 |                                 0.0023 |                                -0.023  |                                -54.0467 |                                -0.0003 |                                 0.0011 |                                -0.0085 |                                -57.7704 |                                -0.0003 |                                 0.0001 |                                 0.0018 |                                -60.7093 |                                -0.0002 |                                -0.0008 |                                 0.0092 |                                -63.03   |                                -0.0002 |                                -0.0015 |                                 0.0144 |                                -64.8612 |                                -0.0002 |                                -0.0021 |                                 0.0182 |                                -66.3022 |                                -0.0002 |                                -0.0026 |                                 0.0208 |                                -67.4305 |                                -0.0002 |                                -0.0031 |                                 0.0226 |                                -68.3065 |                                -0.0002 |                                -0.0035 |                                 0.0238 |                                -68.9781 |                                -0.0002 |                                -0.0038 |                                 0.0246 |                                -69.483  |                                -0.0002 |                                -0.0041 |                                 0.025  |                                -69.8515 |                                -0.0002 |                                -0.0044 |                                 0.0252 |                                -70.1082 |                                -0.0001 |                                -0.0046 |                                 0.0252 |                                -70.2727 |                                -0.0001 |                                -0.0048 |                                 0.0251 |                                -70.3612 |                                -0.0001 |                                -0.005  |                                 0.0248 |                                -70.3867 |                                -0.0001 |                                -0.0051 |                                 0.0245 |                                -70.3602 |                                -0.0001 |                                -0.0052 |                                 0.0241 |                                -70.2907 |                                -0.0001 |                                -0.0053 |                                 0.0237 |                                -70.1855 |                                -0.0001 |                                -0.0054 |                                 0.0233 |                                -70.0508 |                                -0.0001 |                                -0.0055 |                                 0.0228 |                                -69.892  |
        |            260 |                                -0.0002 |                                 0.008  |                                -0.9051 |                                 10.3985 |                                -0.0006 |                                 0.0185 |                                -1.0465 |                                 11.7635 |                                -0.0006 |                                 0.0166 |                                -0.6246 |                                 -4.858  |                                -0.0006 |                                 0.0123 |                                -0.343  |                                -20.403  |                                -0.0005 |                                 0.0082 |                                -0.1818 |                                -31.9586 |                                -0.0005 |                                 0.005  |                                -0.0893 |                                -40.1528 |                                -0.0004 |                                 0.0024 |                                -0.035  |                                -45.9182 |                                -0.0004 |                                 0.0004 |                                -0.0023 |                                -49.9798 |                                -0.0003 |                                -0.0011 |                                 0.0176 |                                -52.8442 |                                -0.0003 |                                -0.0023 |                                 0.0298 |                                -54.8582 |                                -0.0003 |                                -0.0033 |                                 0.0371 |                                -56.2602 |                                -0.0003 |                                -0.0041 |                                 0.0414 |                                -57.2161 |                                -0.0002 |                                -0.0047 |                                 0.0437 |                                -57.8432 |                                -0.0002 |                                -0.0052 |                                 0.0447 |                                -58.2258 |                                -0.0002 |                                -0.0056 |                                 0.0448 |                                -58.4248 |                                -0.0002 |                                -0.006  |                                 0.0444 |                                -58.4856 |                                -0.0002 |                                -0.0062 |                                 0.0437 |                                -58.4416 |                                -0.0002 |                                -0.0065 |                                 0.0427 |                                -58.3183 |                                -0.0002 |                                -0.0067 |                                 0.0415 |                                -58.1349 |                                -0.0002 |                                -0.0068 |                                 0.0403 |                                -57.9062 |                                -0.0001 |                                -0.0069 |                                 0.0391 |                                -57.6437 |                                -0.0001 |                                -0.007  |                                 0.0378 |                                -57.3562 |                                -0.0001 |                                -0.0071 |                                 0.0366 |                                -57.0507 |                                -0.0001 |                                -0.0072 |                                 0.0354 |                                -56.7326 |                                -0.0001 |                                -0.0072 |                                 0.0342 |                                -56.4063 |                                -0.0001 |                                -0.0073 |                                 0.033  |                                -56.075  |                                -0.0001 |                                -0.0073 |                                 0.0319 |                                -55.7416 |                                -0.0001 |                                -0.0073 |                                 0.0309 |                                -55.408  |                                -0.0001 |                                -0.0073 |                                 0.0298 |                                -55.076  |
        |            265 |                                -0.0008 |                                 0.0208 |                                -2.3609 |                                 12.3945 |                                -0.001  |                                 0.0202 |                                -1.1438 |                                 -6.128  |                                -0.0009 |                                 0.0117 |                                -0.4336 |                                -23.586  |                                -0.0007 |                                 0.0051 |                                -0.1389 |                                -33.9296 |                                -0.0006 |                                 0.0007 |                                -0.0104 |                                -39.8482 |                                -0.0005 |                                -0.0022 |                                 0.0481 |                                -43.2339 |                                -0.0004 |                                -0.0043 |                                 0.0747 |                                -45.1411 |                                -0.0004 |                                -0.0058 |                                 0.086  |                                -46.1579 |                                -0.0003 |                                -0.0068 |                                 0.0896 |                                -46.623  |                                -0.0003 |                                -0.0075 |                                 0.0891 |                                -46.7393 |                                -0.0003 |                                -0.0081 |                                 0.0867 |                                -46.6319 |                                -0.0002 |                                -0.0085 |                                 0.0832 |                                -46.3806 |                                -0.0002 |                                -0.0088 |                                 0.0794 |                                -46.0373 |                                -0.0002 |                                -0.009  |                                 0.0754 |                                -45.6362 |                                -0.0002 |                                -0.0092 |                                 0.0715 |                                -45.2007 |                                -0.0002 |                                -0.0093 |                                 0.0677 |                                -44.7464 |                                -0.0002 |                                -0.0093 |                                 0.0642 |                                -44.2842 |                                -0.0002 |                                -0.0094 |                                 0.0608 |                                -43.8216 |                                -0.0001 |                                -0.0094 |                                 0.0577 |                                -43.3636 |                                -0.0001 |                                -0.0094 |                                 0.0548 |                                -42.9138 |                                -0.0001 |                                -0.0094 |                                 0.0521 |                                -42.4744 |                                -0.0001 |                                -0.0093 |                                 0.0496 |                                -42.0469 |                                -0.0001 |                                -0.0093 |                                 0.0473 |                                -41.6324 |                                -0.0001 |                                -0.0093 |                                 0.0451 |                                -41.2313 |                                -0.0001 |                                -0.0092 |                                 0.0431 |                                -40.8438 |                                -0.0001 |                                -0.0092 |                                 0.0412 |                                -40.47   |                                -0.0001 |                                -0.0091 |                                 0.0394 |                                -40.1096 |                                -0.0001 |                                -0.0091 |                                 0.0378 |                                -39.7625 |                                -0.0001 |                                -0.009  |                                 0.0362 |                                -39.4283 |
        |            270 |                                -0.0018 |                                 0.0296 |                                -3.3563 |                                 -1.136  |                                -0.0014 |                                 0.0102 |                                -0.568  |                                -22.8438 |                                -0.001  |                                -0.0007 |                                 0.0379 |                                -31.161  |                                -0.0007 |                                -0.0062 |                                 0.1856 |                                -34.1563 |                                -0.0006 |                                -0.0092 |                                 0.2155 |                                -35.0552 |                                -0.0005 |                                -0.0108 |                                 0.2108 |                                -35.0546 |                                -0.0004 |                                -0.0118 |                                 0.1959 |                                -34.6499 |                                -0.0003 |                                -0.0123 |                                 0.1789 |                                -34.0654 |                                -0.0003 |                                -0.0126 |                                 0.1625 |                                -33.4093 |                                -0.0003 |                                -0.0127 |                                 0.1476 |                                -32.7358 |                                -0.0002 |                                -0.0127 |                                 0.1344 |                                -32.0727 |                                -0.0002 |                                -0.0127 |                                 0.1228 |                                -31.4336 |                                -0.0002 |                                -0.0126 |                                 0.1126 |                                -30.8251 |                                -0.0002 |                                -0.0125 |                                 0.1036 |                                -30.2497 |                                -0.0002 |                                -0.0124 |                                 0.0957 |                                -29.7076 |                                -0.0002 |                                -0.0122 |                                 0.0887 |                                -29.198  |                                -0.0001 |                                -0.0121 |                                 0.0825 |                                -28.7194 |                                -0.0001 |                                -0.0119 |                                 0.0769 |                                -28.2701 |                                -0.0001 |                                -0.0118 |                                 0.072  |                                -27.8481 |                                -0.0001 |                                -0.0116 |                                 0.0675 |                                -27.4517 |                                -0.0001 |                                -0.0115 |                                 0.0635 |                                -27.079  |                                -0.0001 |                                -0.0113 |                                 0.0598 |                                -26.7284 |                                -0.0001 |                                -0.0112 |                                 0.0565 |                                -26.3984 |                                -0.0001 |                                -0.0111 |                                 0.0535 |                                -26.0875 |                                -0.0001 |                                -0.0109 |                                 0.0507 |                                -25.7944 |                                -0.0001 |                                -0.0108 |                                 0.0482 |                                -25.5178 |                                -0.0001 |                                -0.0107 |                                 0.0458 |                                -25.2567 |                                -0.0001 |                                -0.0105 |                                 0.0437 |                                -25.01   |                                -0.0001 |                                -0.0104 |                                 0.0417 |                                -24.7768 |
        |            275 |                                -0.0028 |                                 0.0104 |                                -1.1551 |                                -17.2894 |                                -0.0014 |                                -0.0119 |                                 0.6953 |                                -23.4014 |                                -0.0009 |                                -0.0175 |                                 0.6759 |                                -23.3652 |                                -0.0006 |                                -0.0191 |                                 0.5505 |                                -22.3872 |                                -0.0005 |                                -0.0193 |                                 0.4456 |                                -21.3152 |                                -0.0004 |                                -0.0191 |                                 0.3664 |                                -20.3252 |                                -0.0003 |                                -0.0186 |                                 0.3069 |                                -19.4483 |                                -0.0003 |                                -0.0181 |                                 0.2613 |                                -18.6791 |                                -0.0002 |                                -0.0176 |                                 0.2257 |                                -18.0042 |                                -0.0002 |                                -0.0171 |                                 0.1974 |                                -17.4099 |                                -0.0002 |                                -0.0166 |                                 0.1744 |                                -16.8839 |                                -0.0002 |                                -0.0162 |                                 0.1556 |                                -16.4159 |                                -0.0002 |                                -0.0158 |                                 0.1399 |                                -15.9976 |                                -0.0002 |                                -0.0154 |                                 0.1266 |                                -15.6219 |                                -0.0001 |                                -0.015  |                                 0.1153 |                                -15.2831 |                                -0.0001 |                                -0.0146 |                                 0.1056 |                                -14.9765 |                                -0.0001 |                                -0.0143 |                                 0.0971 |                                -14.6981 |                                -0.0001 |                                -0.014  |                                 0.0898 |                                -14.4446 |                                -0.0001 |                                -0.0137 |                                 0.0833 |                                -14.2131 |                                -0.0001 |                                -0.0134 |                                 0.0776 |                                -14.0013 |                                -0.0001 |                                -0.0132 |                                 0.0724 |                                -13.8072 |                                -0.0001 |                                -0.0129 |                                 0.0679 |                                -13.629  |                                -0.0001 |                                -0.0127 |                                 0.0638 |                                -13.4651 |                                -0.0001 |                                -0.0125 |                                 0.06   |                                -13.3142 |                                -0.0001 |                                -0.0123 |                                 0.0567 |                                -13.1752 |                                -0.0001 |                                -0.0121 |                                 0.0536 |                                -13.0471 |                                -0.0001 |                                -0.0119 |                                 0.0508 |                                -12.929  |                                -0.0001 |                                -0.0117 |                                 0.0483 |                                -12.82   |                                -0.0001 |                                -0.0115 |                                 0.0459 |                                -12.7194 |
        |            280 |                                -0.0024 |                                -0.0372 |                                 4.2754 |                                -11.6664 |                                -0.001  |                                -0.0362 |                                 2.0747 |                                 -9.6812 |                                -0.0006 |                                -0.0325 |                                 1.2421 |                                 -8.4019 |                                -0.0004 |                                -0.0295 |                                 0.8448 |                                 -7.5508 |                                -0.0003 |                                -0.0271 |                                 0.6212 |                                 -6.9435 |                                -0.0003 |                                -0.0252 |                                 0.4812 |                                 -6.4878 |                                -0.0002 |                                -0.0236 |                                 0.3868 |                                 -6.1332 |                                -0.0002 |                                -0.0223 |                                 0.3197 |                                 -5.85   |                                -0.0002 |                                -0.0212 |                                 0.2699 |                                 -5.6193 |                                -0.0002 |                                -0.0202 |                                 0.2319 |                                 -5.4287 |                                -0.0001 |                                -0.0194 |                                 0.202  |                                 -5.2693 |                                -0.0001 |                                -0.0186 |                                 0.178  |                                 -5.135  |                                -0.0001 |                                -0.0179 |                                 0.1584 |                                 -5.0211 |                                -0.0001 |                                -0.0173 |                                 0.1422 |                                 -4.9242 |                                -0.0001 |                                -0.0168 |                                 0.1285 |                                 -4.8415 |                                -0.0001 |                                -0.0163 |                                 0.117  |                                 -4.771  |                                -0.0001 |                                -0.0158 |                                 0.107  |                                 -4.711  |                                -0.0001 |                                -0.0154 |                                 0.0984 |                                 -4.66   |                                -0.0001 |                                -0.015  |                                 0.0909 |                                 -4.6169 |                                -0.0001 |                                -0.0147 |                                 0.0843 |                                 -4.5809 |                                -0.0001 |                                -0.0143 |                                 0.0785 |                                 -4.5512 |                                -0.0001 |                                -0.014  |                                 0.0733 |                                 -4.527  |                                -0.0001 |                                -0.0137 |                                 0.0686 |                                 -4.5079 |                                -0.0001 |                                -0.0135 |                                 0.0645 |                                 -4.4933 |                                -0.0001 |                                -0.0132 |                                 0.0607 |                                 -4.4829 |                                -0.0001 |                                -0.013  |                                 0.0573 |                                 -4.4762 |                                -0.0001 |                                -0.0127 |                                 0.0542 |                                 -4.4731 |                                -0.0001 |                                -0.0125 |                                 0.0514 |                                 -4.4731 |                                -0.0001 |                                -0.0123 |                                 0.0488 |                                 -4.4761 |
        |            285 |                                -0.0004 |                                -0.068  |                                 7.7675 |                                 -0.106  |                                -0.0002 |                                -0.0482 |                                 2.7508 |                                 -0.0943 |                                -0.0002 |                                -0.0394 |                                 1.4983 |                                 -0.098  |                                -0.0001 |                                -0.0341 |                                 0.9736 |                                 -0.1076 |                                -0.0001 |                                -0.0305 |                                 0.6969 |                                 -0.1207 |                                -0.0001 |                                -0.0279 |                                 0.5303 |                                 -0.1362 |                                -0.0001 |                                -0.0258 |                                 0.421  |                                 -0.1536 |                                -0.0001 |                                -0.0241 |                                 0.3446 |                                 -0.1727 |                                -0.0001 |                                -0.0228 |                                 0.2889 |                                 -0.1931 |                                -0.0001 |                                -0.0216 |                                 0.2467 |                                 -0.2149 |                                -0.0001 |                                -0.0206 |                                 0.2139 |                                 -0.2378 |                                -0.0001 |                                -0.0197 |                                 0.1878 |                                 -0.2618 |                                -0.0001 |                                -0.0189 |                                 0.1666 |                                 -0.2869 |                                -0.0001 |                                -0.0182 |                                 0.1491 |                                 -0.3129 |                                -0.0001 |                                -0.0176 |                                 0.1344 |                                 -0.3398 |                                -0.0001 |                                -0.0171 |                                 0.1221 |                                 -0.3676 |                                -0.0001 |                                -0.0166 |                                 0.1115 |                                 -0.3962 |                                -0.0001 |                                -0.0161 |                                 0.1023 |                                 -0.4257 |                                -0.0001 |                                -0.0157 |                                 0.0944 |                                 -0.4559 |                                -0.0001 |                                -0.0153 |                                 0.0874 |                                 -0.4869 |                                -0.0001 |                                -0.0149 |                                 0.0813 |                                 -0.5187 |                                -0.0001 |                                -0.0146 |                                 0.0758 |                                 -0.5511 |                                -0.0001 |                                -0.0142 |                                 0.0709 |                                 -0.5843 |                                -0.0001 |                                -0.0139 |                                 0.0665 |                                 -0.6181 |                                -0      |                                -0.0137 |                                 0.0626 |                                 -0.6526 |                                -0      |                                -0.0134 |                                 0.059  |                                 -0.6877 |                                -0      |                                -0.0132 |                                 0.0558 |                                 -0.7235 |                                -0      |                                -0.0129 |                                 0.0529 |                                 -0.7599 |                                -0      |                                -0.0127 |                                 0.0502 |                                 -0.7969 |
        |            290 |                                 0.0018 |                                -0.0469 |                                 5.3292 |                                 -8.8662 |                                 0.0006 |                                -0.0407 |                                 2.3117 |                                 -6.8514 |                                 0.0003 |                                -0.0354 |                                 1.3414 |                                 -5.7155 |                                 0.0002 |                                -0.0316 |                                 0.8991 |                                 -4.9766 |                                 0.0001 |                                -0.0288 |                                 0.6555 |                                 -4.4489 |                                 0.0001 |                                -0.0266 |                                 0.5049 |                                 -4.0488 |                                 0      |                                -0.0249 |                                 0.4042 |                                 -3.7329 |                                 0      |                                -0.0234 |                                 0.3331 |                                 -3.4762 |                                 0      |                                -0.0222 |                                 0.2806 |                                 -3.2628 |                                -0      |                                -0.0211 |                                 0.2406 |                                 -3.0827 |                                -0      |                                -0.0202 |                                 0.2093 |                                 -2.9285 |                                -0      |                                -0.0194 |                                 0.1842 |                                 -2.7953 |                                -0      |                                -0.0187 |                                 0.1638 |                                 -2.6794 |                                -0      |                                -0.018  |                                 0.1469 |                                 -2.5777 |                                -0      |                                -0.0175 |                                 0.1327 |                                 -2.4882 |                                -0      |                                -0.0169 |                                 0.1206 |                                 -2.4092 |                                -0      |                                -0.0165 |                                 0.1103 |                                 -2.3391 |                                -0      |                                -0.016  |                                 0.1014 |                                 -2.277  |                                -0      |                                -0.0156 |                                 0.0936 |                                 -2.2219 |                                -0      |                                -0.0152 |                                 0.0868 |                                 -2.173  |                                -0      |                                -0.0149 |                                 0.0807 |                                 -2.1296 |                                -0      |                                -0.0145 |                                 0.0754 |                                 -2.0913 |                                -0      |                                -0.0142 |                                 0.0706 |                                 -2.0575 |                                -0      |                                -0.0139 |                                 0.0663 |                                 -2.0279 |                                -0      |                                -0.0137 |                                 0.0624 |                                 -2.0021 |                                -0      |                                -0.0134 |                                 0.0589 |                                 -1.9799 |                                -0      |                                -0.0132 |                                 0.0557 |                                 -1.9608 |                                -0      |                                -0.0129 |                                 0.0527 |                                 -1.9448 |                                -0      |                                -0.0127 |                                 0.0501 |                                 -1.9316 |
        |            295 |                                 0.0027 |                                 0.0005 |                                -0.0932 |                                -18.8061 |                                 0.0012 |                                -0.019  |                                 1.069  |                                -21.4266 |                                 0.0006 |                                -0.0226 |                                 0.8532 |                                -20.2169 |                                 0.0004 |                                -0.0231 |                                 0.6528 |                                -18.7366 |                                 0.0003 |                                -0.0226 |                                 0.512  |                                -17.4068 |                                 0.0002 |                                -0.0219 |                                 0.4132 |                                -16.2651 |                                 0.0002 |                                -0.0211 |                                 0.3417 |                                -15.2876 |                                 0.0001 |                                -0.0203 |                                 0.2883 |                                -14.4448 |                                 0.0001 |                                -0.0196 |                                 0.2473 |                                -13.7113 |                                 0.0001 |                                -0.019  |                                 0.2151 |                                -13.0669 |                                 0.0001 |                                -0.0184 |                                 0.1893 |                                -12.4958 |                                 0      |                                -0.0178 |                                 0.1682 |                                -11.9858 |                                 0      |                                -0.0173 |                                 0.1508 |                                -11.5274 |                                 0      |                                -0.0168 |                                 0.1361 |                                -11.1127 |                                 0      |                                -0.0163 |                                 0.1237 |                                -10.7357 |                                 0      |                                -0.0159 |                                 0.1131 |                                -10.3912 |                                 0      |                                -0.0155 |                                 0.1039 |                                -10.0752 |                                 0      |                                -0.0152 |                                 0.0959 |                                 -9.7843 |                                 0      |                                -0.0149 |                                 0.0888 |                                 -9.5156 |                                -0      |                                -0.0145 |                                 0.0826 |                                 -9.2666 |                                -0      |                                -0.0142 |                                 0.0771 |                                 -9.0353 |                                -0      |                                -0.014  |                                 0.0721 |                                 -8.8199 |                                -0      |                                -0.0137 |                                 0.0677 |                                 -8.619  |                                -0      |                                -0.0135 |                                 0.0637 |                                 -8.4312 |                                -0      |                                -0.0132 |                                 0.0601 |                                 -8.2554 |                                -0      |                                -0.013  |                                 0.0568 |                                 -8.0904 |                                -0      |                                -0.0128 |                                 0.0538 |                                 -7.9356 |                                -0      |                                -0.0126 |                                 0.0511 |                                 -7.79   |                                -0      |                                -0.0124 |                                 0.0486 |                                 -7.653  |
        |            300 |                                 0.0022 |                                 0.0289 |                                -3.3233 |                                 -7.8456 |                                 0.0013 |                                 0.0042 |                                -0.2538 |                                -26.476  |                                 0.0008 |                                -0.0065 |                                 0.2387 |                                -31.5817 |                                 0.0006 |                                -0.0113 |                                 0.3163 |                                -32.5972 |                                 0.0004 |                                -0.0136 |                                 0.306  |                                -32.2413 |                                 0.0003 |                                -0.0148 |                                 0.2769 |                                -31.3965 |                                 0.0002 |                                -0.0153 |                                 0.2463 |                                -30.3915 |                                 0.0002 |                                -0.0155 |                                 0.2186 |                                -29.3593 |                                 0.0002 |                                -0.0155 |                                 0.1947 |                                -28.3558 |                                 0.0001 |                                -0.0154 |                                 0.1742 |                                -27.4031 |                                 0.0001 |                                -0.0153 |                                 0.1569 |                                -26.5089 |                                 0.0001 |                                -0.0151 |                                 0.142  |                                -25.6737 |                                 0.0001 |                                -0.0149 |                                 0.1292 |                                -24.895  |                                 0.0001 |                                -0.0146 |                                 0.1182 |                                -24.1691 |                                 0.0001 |                                -0.0144 |                                 0.1086 |                                -23.4919 |                                 0      |                                -0.0142 |                                 0.1002 |                                -22.8592 |                                 0      |                                -0.014  |                                 0.0928 |                                -22.2671 |                                 0      |                                -0.0137 |                                 0.0863 |                                -21.7121 |                                 0      |                                -0.0135 |                                 0.0805 |                                -21.191  |                                 0      |                                -0.0133 |                                 0.0753 |                                -20.7007 |                                 0      |                                -0.0131 |                                 0.0706 |                                -20.2388 |                                 0      |                                -0.0129 |                                 0.0664 |                                -19.8028 |                                 0      |                                -0.0127 |                                 0.0626 |                                -19.3906 |                                 0      |                                -0.0125 |                                 0.0591 |                                -19.0004 |                                 0      |                                -0.0124 |                                 0.056  |                                -18.6306 |                                 0      |                                -0.0122 |                                 0.0531 |                                -18.2795 |                                 0      |                                -0.012  |                                 0.0504 |                                -17.9458 |                                 0      |                                -0.0119 |                                 0.048  |                                -17.6283 |                                 0      |                                -0.0117 |                                 0.0458 |                                -17.3259 |
        |            305 |                                 0.0012 |                                 0.0282 |                                -3.228  |                                  9.1545 |                                 0.0012 |                                 0.0187 |                                -1.0831 |                                -15.69   |                                 0.0009 |                                 0.0074 |                                -0.2912 |                                -30.8926 |                                 0.0006 |                                 0.0003 |                                -0.0165 |                                -38.2669 |                                 0.0005 |                                -0.004  |                                 0.086  |                                -41.7365 |                                 0.0004 |                                -0.0068 |                                 0.1239 |                                -43.2258 |                                 0.0003 |                                -0.0086 |                                 0.1353 |                                -43.6642 |                                 0.0003 |                                -0.0097 |                                 0.1353 |                                -43.5252 |                                 0.0002 |                                -0.0105 |                                 0.1304 |                                -43.0633 |                                 0.0002 |                                -0.011  |                                 0.1235 |                                -42.4215 |                                 0.0002 |                                -0.0114 |                                 0.116  |                                -41.6832 |                                 0.0001 |                                -0.0116 |                                 0.1086 |                                -40.898  |                                 0.0001 |                                -0.0118 |                                 0.1015 |                                -40.0962 |                                 0.0001 |                                -0.0118 |                                 0.0949 |                                -39.2961 |                                 0.0001 |                                -0.0119 |                                 0.0888 |                                -38.5091 |                                 0.0001 |                                -0.0118 |                                 0.0832 |                                -37.7419 |                                 0.0001 |                                -0.0118 |                                 0.0782 |                                -36.9982 |                                 0.0001 |                                -0.0118 |                                 0.0735 |                                -36.2802 |                                 0.0001 |                                -0.0117 |                                 0.0693 |                                -35.5887 |                                 0      |                                -0.0116 |                                 0.0654 |                                -34.9237 |                                 0      |                                -0.0115 |                                 0.0618 |                                -34.2848 |                                 0      |                                -0.0114 |                                 0.0586 |                                -33.6713 |                                 0      |                                -0.0113 |                                 0.0556 |                                -33.0823 |                                 0      |                                -0.0112 |                                 0.0528 |                                -32.5168 |                                 0      |                                -0.0111 |                                 0.0503 |                                -31.9738 |                                 0      |                                -0.011  |                                 0.0479 |                                -31.4521 |                                 0      |                                -0.0109 |                                 0.0457 |                                -30.9508 |                                 0      |                                -0.0108 |                                 0.0437 |                                -30.4688 |                                 0      |                                -0.0108 |                                 0.0418 |                                -30.0052 |
        |            310 |                                 0.0005 |                                 0.0159 |                                -1.8154 |                                 14.1156 |                                 0.0008 |                                 0.0224 |                                -1.2863 |                                  2.4939 |                                 0.0008 |                                 0.0159 |                                -0.6123 |                                -18.0435 |                                 0.0006 |                                 0.0093 |                                -0.2734 |                                -32.4433 |                                 0.0005 |                                 0.0044 |                                -0.1059 |                                -41.5529 |                                 0.0004 |                                 0.0008 |                                -0.0198 |                                -47.2153 |                                 0.0004 |                                -0.0018 |                                 0.0257 |                                -50.6983 |                                 0.0003 |                                -0.0038 |                                 0.05   |                                -52.7867 |                                 0.0003 |                                -0.0052 |                                 0.0627 |                                -53.9652 |                                 0.0002 |                                -0.0063 |                                 0.0688 |                                -54.5387 |                                 0.0002 |                                -0.0071 |                                 0.0712 |                                -54.7038 |                                 0.0002 |                                -0.0077 |                                 0.0713 |                                -54.5903 |                                 0.0001 |                                -0.0082 |                                 0.0702 |                                -54.2855 |                                 0.0001 |                                -0.0086 |                                 0.0683 |                                -53.8496 |                                 0.0001 |                                -0.0089 |                                 0.066  |                                -53.3243 |                                 0.0001 |                                -0.0091 |                                 0.0635 |                                -52.7392 |                                 0.0001 |                                -0.0093 |                                 0.061  |                                -52.1155 |                                 0.0001 |                                -0.0094 |                                 0.0585 |                                -51.4684 |                                 0.0001 |                                -0.0095 |                                 0.056  |                                -50.8088 |                                 0.0001 |                                -0.0096 |                                 0.0536 |                                -50.1447 |                                 0.0001 |                                -0.0096 |                                 0.0513 |                                -49.4821 |                                 0.0001 |                                -0.0097 |                                 0.0492 |                                -48.825  |                                 0      |                                -0.0097 |                                 0.0471 |                                -48.1766 |                                 0      |                                -0.0097 |                                 0.0452 |                                -47.539  |                                 0      |                                -0.0097 |                                 0.0433 |                                -46.9137 |                                 0      |                                -0.0096 |                                 0.0416 |                                -46.3018 |                                 0      |                                -0.0096 |                                 0.04   |                                -45.7038 |                                 0      |                                -0.0096 |                                 0.0384 |                                -45.1202 |                                 0      |                                -0.0096 |                                 0.037  |                                -44.5512 |
        |            315 |                                 0.0002 |                                 0.0062 |                                -0.7114 |                                  9.4643 |                                 0.0005 |                                 0.0186 |                                -1.0682 |                                 16.2419 |                                 0.0006 |                                 0.0185 |                                -0.7095 |                                  0.1284 |                                 0.0005 |                                 0.0145 |                                -0.4215 |                                -17.5605 |                                 0.0005 |                                 0.0104 |                                -0.2429 |                                -31.5815 |                                 0.0004 |                                 0.0068 |                                -0.135  |                                -41.8892 |                                 0.0004 |                                 0.0039 |                                -0.0688 |                                -49.3103 |                                 0.0003 |                                 0.0016 |                                -0.0274 |                                -54.6184 |                                 0.0003 |                                -0.0002 |                                -0.0011 |                                -58.398  |                                 0.0002 |                                -0.0017 |                                 0.0158 |                                -61.068  |                                 0.0002 |                                -0.0028 |                                 0.0267 |                                -62.9254 |                                 0.0002 |                                -0.0038 |                                 0.0337 |                                -64.1817 |                                 0.0002 |                                -0.0046 |                                 0.038  |                                -64.9892 |                                 0.0002 |                                -0.0052 |                                 0.0406 |                                -65.4586 |                                 0.0001 |                                -0.0058 |                                 0.042  |                                -65.6718 |                                 0.0001 |                                -0.0062 |                                 0.0425 |                                -65.6895 |                                 0.0001 |                                -0.0066 |                                 0.0425 |                                -65.5577 |                                 0.0001 |                                -0.0069 |                                 0.0421 |                                -65.3112 |                                 0.0001 |                                -0.0071 |                                 0.0415 |                                -64.9769 |                                 0.0001 |                                -0.0073 |                                 0.0406 |                                -64.5754 |                                 0.0001 |                                -0.0075 |                                 0.0397 |                                -64.1228 |                                 0.0001 |                                -0.0077 |                                 0.0387 |                                -63.6319 |                                 0.0001 |                                -0.0078 |                                 0.0376 |                                -63.1125 |                                 0.0001 |                                -0.0079 |                                 0.0366 |                                -62.5727 |                                 0.0001 |                                -0.008  |                                 0.0355 |                                -62.0186 |                                 0.0001 |                                -0.008  |                                 0.0344 |                                -61.4551 |                                 0      |                                -0.0081 |                                 0.0334 |                                -60.8864 |                                 0      |                                -0.0081 |                                 0.0324 |                                -60.3156 |                                 0      |                                -0.0082 |                                 0.0314 |                                -59.7452 |
        |            320 |                                 0      |                                 0.0018 |                                -0.208  |                                  4.0678 |                                 0.0003 |                                 0.0124 |                                -0.7106 |                                 20.4356 |                                 0.0004 |                                 0.0168 |                                -0.6438 |                                 15.659  |                                 0.0004 |                                 0.0161 |                                -0.465  |                                  0.6251 |                                 0.0004 |                                 0.0137 |                                -0.317  |                                -15.1628 |                                 0.0004 |                                 0.0109 |                                -0.212  |                                -28.7926 |                                 0.0003 |                                 0.0083 |                                -0.1398 |                                -39.8257 |                                 0.0003 |                                 0.006  |                                -0.09   |                                -48.5419 |                                 0.0003 |                                 0.0041 |                                -0.0555 |                                -55.362  |                                 0.0002 |                                 0.0025 |                                -0.0312 |                                -60.677  |                                 0.0002 |                                 0.0011 |                                -0.014  |                                -64.8087 |                                 0.0002 |                                -0.0001 |                                -0.0016 |                                -68.0111 |                                 0.0002 |                                -0.0011 |                                 0.0072 |                                -70.4811 |                                 0.0002 |                                -0.0019 |                                 0.0136 |                                -72.3712 |                                 0.0002 |                                -0.0027 |                                 0.0182 |                                -73.7996 |                                 0.0001 |                                -0.0033 |                                 0.0215 |                                -74.8585 |                                 0.0001 |                                -0.0038 |                                 0.0239 |                                -75.6201 |                                 0.0001 |                                -0.0043 |                                 0.0255 |                                -76.1414 |                                 0.0001 |                                -0.0047 |                                 0.0265 |                                -76.4675 |                                 0.0001 |                                -0.005  |                                 0.0272 |                                -76.6348 |                                 0.0001 |                                -0.0053 |                                 0.0276 |                                -76.6724 |                                 0.0001 |                                -0.0056 |                                 0.0277 |                                -76.6038 |                                 0.0001 |                                -0.0058 |                                 0.0276 |                                -76.4481 |                                 0.0001 |                                -0.006  |                                 0.0274 |                                -76.2212 |                                 0.0001 |                                -0.0062 |                                 0.0271 |                                -75.9359 |                                 0.0001 |                                -0.0063 |                                 0.0268 |                                -75.6028 |                                 0.0001 |                                -0.0065 |                                 0.0264 |                                -75.2307 |                                 0.0001 |                                -0.0066 |                                 0.0259 |                                -74.827  |                                 0      |                                -0.0067 |                                 0.0254 |                                -74.3978 |
        |            325 |                                 0      |                                 0.0004 |                                -0.0471 |                                  1.2506 |                                 0.0001 |                                 0.007  |                                -0.3988 |                                 17.2793 |                                 0.0003 |                                 0.013  |                                -0.4987 |                                 24.001  |                                 0.0003 |                                 0.015  |                                -0.4315 |                                 16.3716 |                                 0.0003 |                                 0.0145 |                                -0.3349 |                                  2.8976 |                                 0.0003 |                                 0.0129 |                                -0.2502 |                                -11.4029 |                                 0.0003 |                                 0.0111 |                                -0.1841 |                                -24.5315 |                                 0.0003 |                                 0.0092 |                                -0.1343 |                                -35.8984 |                                 0.0003 |                                 0.0074 |                                -0.097  |                                -45.4804 |                                 0.0002 |                                 0.0058 |                                -0.0692 |                                -53.4549 |                                 0.0002 |                                 0.0044 |                                -0.0482 |                                -60.0501 |                                 0.0002 |                                 0.0031 |                                -0.0323 |                                -65.4876 |                                 0.0002 |                                 0.002  |                                -0.0202 |                                -69.9621 |                                 0.0002 |                                 0.0011 |                                -0.011  |                                -73.6387 |                                 0.0002 |                                 0.0002 |                                -0.0038 |                                -76.6537 |                                 0.0001 |                                -0.0005 |                                 0.0017 |                                -79.1196 |                                 0.0001 |                                -0.0012 |                                 0.006  |                                -81.1282 |                                 0.0001 |                                -0.0017 |                                 0.0094 |                                -82.755  |                                 0.0001 |                                -0.0023 |                                 0.0119 |                                -84.0618 |                                 0.0001 |                                -0.0027 |                                 0.0139 |                                -85.0996 |                                 0.0001 |                                -0.0031 |                                 0.0155 |                                -85.9106 |                                 0.0001 |                                -0.0035 |                                 0.0166 |                                -86.5298 |                                 0.0001 |                                -0.0038 |                                 0.0175 |                                -86.9863 |                                 0.0001 |                                -0.0041 |                                 0.0181 |                                -87.3046 |                                 0.0001 |                                -0.0043 |                                 0.0186 |                                -87.505  |                                 0.0001 |                                -0.0045 |                                 0.0189 |                                -87.605  |                                 0.0001 |                                -0.0048 |                                 0.0191 |                                -87.6192 |                                 0.0001 |                                -0.0049 |                                 0.0191 |                                -87.5599 |                                 0.0001 |                                -0.0051 |                                 0.0191 |                                -87.4378 |
        |            330 |                                 0      |                                 0.0001 |                                -0.0085 |                                  0.2902 |                                 0.0001 |                                 0.0034 |                                -0.1941 |                                 11.4725 |                                 0.0002 |                                 0.009  |                                -0.3423 |                                 24.9768 |                                 0.0002 |                                 0.0124 |                                -0.3558 |                                 26.2288 |                                 0.0003 |                                 0.0135 |                                -0.3117 |                                 18.3237 |                                 0.0003 |                                 0.0133 |                                -0.2554 |                                  6.3242 |                                 0.0003 |                                 0.0123 |                                -0.2034 |                                 -6.6837 |                                 0.0003 |                                 0.011  |                                -0.1598 |                                -19.1951 |                                 0.0002 |                                 0.0096 |                                -0.1247 |                                -30.5838 |                                 0.0002 |                                 0.0082 |                                -0.0967 |                                -40.6609 |                                 0.0002 |                                 0.0069 |                                -0.0746 |                                -49.4416 |                                 0.0002 |                                 0.0057 |                                -0.0571 |                                -57.0273 |                                 0.0002 |                                 0.0047 |                                -0.0432 |                                -63.549  |                                 0.0002 |                                 0.0037 |                                -0.0321 |                                -69.1404 |                                 0.0002 |                                 0.0028 |                                -0.0232 |                                -73.9261 |                                 0.0002 |                                 0.002  |                                -0.016  |                                -78.0175 |                                 0.0001 |                                 0.0012 |                                -0.0102 |                                -81.5117 |                                 0.0001 |                                 0.0006 |                                -0.0055 |                                -84.4922 |                                 0.0001 |                                 0      |                                -0.0017 |                                -87.0307 |                                 0.0001 |                                -0.0005 |                                 0.0014 |                                -89.188  |                                 0.0001 |                                -0.001  |                                 0.0039 |                                -91.0164 |                                 0.0001 |                                -0.0014 |                                 0.006  |                                -92.56   |                                 0.0001 |                                -0.0018 |                                 0.0077 |                                -93.8568 |                                 0.0001 |                                -0.0022 |                                 0.0091 |                                -94.9391 |                                 0.0001 |                                -0.0025 |                                 0.0102 |                                -95.8346 |                                 0.0001 |                                -0.0028 |                                 0.0111 |                                -96.5673 |                                 0.0001 |                                -0.0031 |                                 0.0118 |                                -97.1575 |                                 0.0001 |                                -0.0033 |                                 0.0124 |                                -97.6231 |                                 0.0001 |                                -0.0035 |                                 0.0128 |                                -97.9795 |
        |            335 |                                 0      |                                 0      |                                -0.0012 |                                  0.0526 |                                 0      |                                 0.0015 |                                -0.0833 |                                  6.3461 |                                 0.0001 |                                 0.0056 |                                -0.2125 |                                 21.103  |                                 0.0001 |                                 0.0093 |                                -0.2675 |                                 29.5579 |                                 0.0002 |                                 0.0115 |                                -0.2647 |                                 28.5681 |                                 0.0002 |                                 0.0123 |                                -0.2369 |                                 21.1621 |                                 0.0002 |                                 0.0122 |                                -0.2023 |                                 10.5042 |                                 0.0002 |                                 0.0116 |                                -0.1688 |                                 -1.3165 |                                 0.0002 |                                 0.0108 |                                -0.1391 |                                -13.1155 |                                 0.0002 |                                 0.0098 |                                -0.1138 |                                -24.2851 |                                 0.0002 |                                 0.0087 |                                -0.0928 |                                -34.5515 |                                 0.0002 |                                 0.0077 |                                -0.0753 |                                -43.8255 |                                 0.0002 |                                 0.0067 |                                -0.0609 |                                -52.1151 |                                 0.0002 |                                 0.0058 |                                -0.049  |                                -59.4764 |                                 0.0002 |                                 0.0049 |                                -0.0392 |                                -65.9869 |                                 0.0002 |                                 0.0041 |                                -0.0311 |                                -71.73   |                                 0.0001 |                                 0.0033 |                                -0.0243 |                                -76.7878 |                                 0.0001 |                                 0.0027 |                                -0.0187 |                                -81.237  |                                 0.0001 |                                 0.002  |                                -0.014  |                                -85.1476 |                                 0.0001 |                                 0.0015 |                                -0.01   |                                -88.5821 |                                 0.0001 |                                 0.001  |                                -0.0067 |                                -91.5963 |                                 0.0001 |                                 0.0005 |                                -0.0039 |                                -94.239  |                                 0.0001 |                                 0      |                                -0.0015 |                                -96.5534 |                                 0.0001 |                                -0.0004 |                                 0.0005 |                                -98.5773 |                                 0.0001 |                                -0.0007 |                                 0.0022 |                               -100.344  |                                 0.0001 |                                -0.0011 |                                 0.0036 |                               -101.882  |                                 0.0001 |                                -0.0014 |                                 0.0048 |                               -103.217  |                                 0.0001 |                                -0.0017 |                                 0.0058 |                               -104.371  |                                 0.0001 |                                -0.002  |                                 0.0067 |                               -105.365  |
        |            340 |                                 0      |                                 0      |                                -0.0001 |                                  0.0076 |                                 0      |                                 0.0006 |                                -0.0319 |                                  3.0181 |                                 0      |                                 0.0032 |                                -0.1209 |                                 15.3905 |                                 0.0001 |                                 0.0065 |                                -0.1863 |                                 27.7449 |                                 0.0001 |                                 0.0091 |                                -0.2091 |                                 33.036  |                                 0.0002 |                                 0.0106 |                                -0.2044 |                                 31.2521 |                                 0.0002 |                                 0.0113 |                                -0.1867 |                                 24.5913 |                                 0.0002 |                                 0.0114 |                                -0.1645 |                                 15.1585 |                                 0.0002 |                                 0.011  |                                -0.1421 |                                  4.4559 |                                 0.0002 |                                 0.0104 |                                -0.1214 |                                 -6.5626 |                                 0.0002 |                                 0.0097 |                                -0.103  |                                -17.3329 |                                 0.0002 |                                 0.0089 |                                -0.087  |                                -27.5444 |                                 0.0002 |                                 0.0081 |                                -0.0733 |                                -37.0445 |                                 0.0002 |                                 0.0073 |                                -0.0616 |                                -45.7754 |                                 0.0002 |                                 0.0065 |                                -0.0516 |                                -53.7354 |                                 0.0001 |                                 0.0058 |                                -0.0431 |                                -60.9535 |                                 0.0001 |                                 0.0051 |                                -0.0359 |                                -67.4753 |                                 0.0001 |                                 0.0044 |                                -0.0297 |                                -73.3535 |                                 0.0001 |                                 0.0038 |                                -0.0244 |                                -78.6427 |                                 0.0001 |                                 0.0032 |                                -0.0199 |                                -83.3963 |                                 0.0001 |                                 0.0027 |                                -0.016  |                                -87.6649 |                                 0.0001 |                                 0.0022 |                                -0.0127 |                                -91.4955 |                                 0.0001 |                                 0.0017 |                                -0.0098 |                                -94.931  |                                 0.0001 |                                 0.0013 |                                -0.0074 |                                -98.0106 |                                 0.0001 |                                 0.0009 |                                -0.0052 |                               -100.769  |                                 0.0001 |                                 0.0005 |                                -0.0034 |                               -103.239  |                                 0.0001 |                                 0.0001 |                                -0.0018 |                               -105.448  |                                 0.0001 |                                -0.0002 |                                -0.0004 |                               -107.422  |                                 0.0001 |                                -0.0005 |                                 0.0008 |                               -109.184  |
        |            345 |                                 0      |                                 0      |                                -0      |                                  0.0009 |                                 0      |                                 0.0002 |                                -0.011  |                                  1.2589 |                                 0      |                                 0.0017 |                                -0.0636 |                                  9.988  |                                 0.0001 |                                 0.0042 |                                -0.1214 |                                 22.9787 |                                 0.0001 |                                 0.0068 |                                -0.1553 |                                 32.6066 |                                 0.0001 |                                 0.0087 |                                -0.1662 |                                 36.1604 |                                 0.0001 |                                 0.0099 |                                -0.1624 |                                 34.2394 |                                 0.0001 |                                 0.0105 |                                -0.151  |                                 28.3906 |                                 0.0002 |                                 0.0106 |                                -0.1364 |                                 20.0847 |                                 0.0002 |                                 0.0104 |                                -0.1211 |                                 10.4445 |                                 0.0002 |                                 0.0101 |                                -0.1064 |                                  0.249  |                                 0.0002 |                                 0.0096 |                                -0.0928 |                                 -9.9916 |                                 0.0002 |                                 0.009  |                                -0.0806 |                                -19.9589 |                                 0.0001 |                                 0.0083 |                                -0.0698 |                                -29.4651 |                                 0.0001 |                                 0.0077 |                                -0.0604 |                                -38.4087 |                                 0.0001 |                                 0.0071 |                                -0.0521 |                                -46.7449 |                                 0.0001 |                                 0.0064 |                                -0.0448 |                                -54.4641 |                                 0.0001 |                                 0.0058 |                                -0.0385 |                                -61.5792 |                                 0.0001 |                                 0.0052 |                                -0.033  |                                -68.1157 |                                 0.0001 |                                 0.0047 |                                -0.0282 |                                -74.1065 |                                 0.0001 |                                 0.0042 |                                -0.024  |                                -79.5879 |                                 0.0001 |                                 0.0037 |                                -0.0203 |                                -84.5968 |                                 0.0001 |                                 0.0032 |                                -0.0171 |                                -89.1699 |                                 0.0001 |                                 0.0027 |                                -0.0143 |                                -93.3422 |                                 0.0001 |                                 0.0023 |                                -0.0118 |                                -97.1467 |                                 0.0001 |                                 0.0019 |                                -0.0097 |                               -100.614  |                                 0.0001 |                                 0.0016 |                                -0.0077 |                               -103.773  |                                 0.0001 |                                 0.0012 |                                -0.0061 |                               -106.65   |                                 0.0001 |                                 0.0009 |                                -0.0046 |                               -109.269  |
        |            350 |                                 0      |                                 0      |                                -0      |                                  0.0001 |                                 0      |                                 0.0001 |                                -0.0035 |                                  0.4672 |                                 0      |                                 0.0008 |                                -0.0312 |                                  5.8744 |                                 0      |                                 0.0026 |                                -0.0745 |                                 17.2715 |                                 0.0001 |                                 0.0048 |                                -0.1094 |                                 28.8881 |                                 0.0001 |                                 0.0067 |                                -0.1286 |                                 36.5383 |                                 0.0001 |                                 0.0082 |                                -0.1347 |                                 39.1834 |                                 0.0001 |                                 0.0092 |                                -0.1321 |                                 37.4362 |                                 0.0001 |                                 0.0097 |                                -0.1247 |                                 32.3978 |                                 0.0001 |                                 0.0099 |                                -0.1149 |                                 25.1333 |                                 0.0001 |                                 0.0099 |                                -0.1042 |                                 16.5044 |                                 0.0001 |                                 0.0097 |                                -0.0936 |                                  7.1524 |                                 0.0001 |                                 0.0093 |                                -0.0835 |                                 -2.4688 |                                 0.0001 |                                 0.0089 |                                -0.0741 |                                -12.0495 |                                 0.0001 |                                 0.0084 |                                -0.0656 |                                -21.386  |                                 0.0001 |                                 0.0079 |                                -0.058  |                                -30.3501 |                                 0.0001 |                                 0.0074 |                                -0.0511 |                                -38.8666 |                                 0.0001 |                                 0.0069 |                                -0.045  |                                -46.8964 |                                 0.0001 |                                 0.0063 |                                -0.0396 |                                -54.4256 |                                 0.0001 |                                 0.0058 |                                -0.0347 |                                -61.4563 |                                 0.0001 |                                 0.0053 |                                -0.0305 |                                -68.0016 |                                 0.0001 |                                 0.0049 |                                -0.0266 |                                -74.081  |                                 0.0001 |                                 0.0044 |                                -0.0233 |                                -79.718  |                                 0.0001 |                                 0.004  |                                -0.0202 |                                -84.9379 |                                 0.0001 |                                 0.0036 |                                -0.0176 |                                -89.7669 |                                 0.0001 |                                 0.0032 |                                -0.0152 |                                -94.2308 |                                 0.0001 |                                 0.0028 |                                -0.013  |                                -98.3547 |                                 0.0001 |                                 0.0025 |                                -0.0111 |                               -102.163  |                                 0.0001 |                                 0.0021 |                                -0.0094 |                               -105.678  |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_speed().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0001 |       0.0007 |       0.0013 |       0.0015 |       0.0015 |       0.0015 |       0.0014 |       0.0013 |       0.0012 |       0.0011 |       0.001  |       0.0009 |       0.0008 |       0.0008 |       0.0007 |       0.0006 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            295 |       0      |       0.0001 |       0.0004 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.001  |       0.0009 |       0.0009 |       0.0009 |       0.0008 |       0.0008 |       0.0007 |       0.0007 |       0.0007 |       0.0006 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |
        |            300 |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0005 |       0.0006 |       0.0006 |       0.0006 |       0.0007 |       0.0007 |       0.0007 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |
        |            305 |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |
        |            310 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            315 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            320 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_zomma().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0027 |       0.033  |       0.055  |       0.0593 |       0.0548 |       0.0471 |       0.0387 |       0.0308 |       0.0236 |       0.0174 |       0.0119 |       0.0071 |       0.003  |      -0.0005 |      -0.0036 |      -0.0063 |      -0.0087 |      -0.0107 |      -0.0125 |      -0.0141 |      -0.0155 |      -0.0168 |      -0.0179 |      -0.0188 |      -0.0197 |      -0.0204 |      -0.0211 |      -0.0217 |      -0.0223 |
        |            295 |       0.0001 |       0.0065 |       0.0231 |       0.0374 |       0.0451 |       0.0474 |       0.0465 |       0.0436 |       0.0399 |       0.0357 |       0.0315 |       0.0274 |       0.0235 |       0.0199 |       0.0165 |       0.0135 |       0.0106 |       0.0081 |       0.0057 |       0.0035 |       0.0016 |      -0.0002 |      -0.0019 |      -0.0034 |      -0.0048 |      -0.006  |      -0.0072 |      -0.0083 |      -0.0093 |
        |            300 |       0      |       0.0007 |       0.0062 |       0.0158 |       0.0253 |       0.0325 |       0.037  |       0.0392 |       0.0397 |       0.0391 |       0.0376 |       0.0357 |       0.0334 |       0.031  |       0.0286 |       0.0261 |       0.0237 |       0.0214 |       0.0192 |       0.0171 |       0.0151 |       0.0132 |       0.0114 |       0.0097 |       0.0081 |       0.0067 |       0.0053 |       0.0039 |       0.0027 |
        |            305 |       0      |       0      |       0.0011 |       0.0049 |       0.0108 |       0.0172 |       0.0229 |       0.0274 |       0.0307 |       0.0328 |       0.034  |       0.0344 |       0.0341 |       0.0335 |       0.0325 |       0.0313 |       0.0299 |       0.0284 |       0.0269 |       0.0253 |       0.0237 |       0.0221 |       0.0206 |       0.0191 |       0.0177 |       0.0163 |       0.0149 |       0.0136 |       0.0124 |
        |            310 |       0      |       0      |       0.0002 |       0.0012 |       0.0037 |       0.0074 |       0.0117 |       0.016  |       0.0199 |       0.0231 |       0.0257 |       0.0277 |       0.029  |       0.0299 |       0.0303 |       0.0304 |       0.0302 |       0.0297 |       0.0291 |       0.0283 |       0.0274 |       0.0265 |       0.0255 |       0.0244 |       0.0234 |       0.0223 |       0.0212 |       0.0201 |       0.0191 |
        |            315 |       0      |       0      |       0      |       0.0002 |       0.001  |       0.0027 |       0.0051 |       0.008  |       0.0111 |       0.0142 |       0.0171 |       0.0196 |       0.0217 |       0.0234 |       0.0248 |       0.0258 |       0.0266 |       0.027  |       0.0273 |       0.0273 |       0.0272 |       0.0269 |       0.0265 |       0.0261 |       0.0255 |       0.0249 |       0.0242 |       0.0235 |       0.0228 |
        |            320 |       0      |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0019 |       0.0035 |       0.0055 |       0.0078 |       0.0101 |       0.0124 |       0.0146 |       0.0166 |       0.0184 |       0.0199 |       0.0212 |       0.0223 |       0.0232 |       0.0238 |       0.0243 |       0.0246 |       0.0248 |       0.0249 |       0.0249 |       0.0247 |       0.0245 |       0.0242 |       0.0239 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0006 |       0.0014 |       0.0025 |       0.0039 |       0.0055 |       0.0072 |       0.009  |       0.0108 |       0.0125 |       0.0141 |       0.0156 |       0.017  |       0.0182 |       0.0192 |       0.0201 |       0.0209 |       0.0215 |       0.022  |       0.0224 |       0.0226 |       0.0228 |       0.0229 |       0.0229 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0005 |       0.001  |       0.0017 |       0.0027 |       0.0039 |       0.0051 |       0.0065 |       0.0079 |       0.0094 |       0.0107 |       0.0121 |       0.0133 |       0.0145 |       0.0156 |       0.0166 |       0.0174 |       0.0182 |       0.0189 |       0.0194 |       0.0199 |       0.0203 |       0.0207 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0004 |       0.0007 |       0.0012 |       0.0019 |       0.0027 |       0.0037 |       0.0047 |       0.0058 |       0.007  |       0.0081 |       0.0093 |       0.0104 |       0.0114 |       0.0125 |       0.0134 |       0.0143 |       0.0151 |       0.0158 |       0.0165 |       0.0171 |       0.0177 |
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

        The formula as usually published carries a leading minus sign because it differentiates with respect to the
        time to maturity, which runs opposite to elapsed calendar time. That sign is absorbed here so that Color,
        like Theta, Charm and Veta, measures the change per unit of time that passes. The result is per year, matching
        Charm; divide by 365 for gamma decay per calendar day.

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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_color().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.1607 |      -0.9839 |      -1.0984 |      -0.8932 |      -0.6642 |      -0.4793 |      -0.3416 |      -0.241  |      -0.1677 |      -0.114  |      -0.0744 |      -0.0448 |      -0.0227 |      -0.006  |       0.0066 |       0.0162 |       0.0235 |       0.029  |       0.0332 |       0.0363 |       0.0386 |       0.0402 |       0.0413 |       0.042  |       0.0425 |       0.0426 |       0.0426 |       0.0424 |       0.0421 |
        |            295 |      -0.0035 |      -0.1931 |      -0.4604 |      -0.5597 |      -0.5419 |      -0.4775 |      -0.4029 |      -0.3328 |      -0.2719 |      -0.2207 |      -0.1785 |      -0.1438 |      -0.1153 |      -0.0919 |      -0.0727 |      -0.0569 |      -0.0438 |      -0.0329 |      -0.0239 |      -0.0163 |      -0.01   |      -0.0047 |      -0.0002 |       0.0036 |       0.0067 |       0.0094 |       0.0116 |       0.0135 |       0.0151 |
        |            300 |      -0      |      -0.0212 |      -0.1232 |      -0.2362 |      -0.3032 |      -0.3252 |      -0.3184 |      -0.2964 |      -0.268  |      -0.2382 |      -0.2095 |      -0.1829 |      -0.1591 |      -0.1379 |      -0.1193 |      -0.103  |      -0.0888 |      -0.0764 |      -0.0655 |      -0.0561 |      -0.0478 |      -0.0406 |      -0.0343 |      -0.0287 |      -0.0239 |      -0.0196 |      -0.0158 |      -0.0125 |      -0.0095 |
        |            305 |      -0      |      -0.0014 |      -0.0227 |      -0.0731 |      -0.1289 |      -0.1714 |      -0.1964 |      -0.2065 |      -0.206  |      -0.1987 |      -0.1875 |      -0.1744 |      -0.1605 |      -0.1466 |      -0.1333 |      -0.1208 |      -0.1091 |      -0.0984 |      -0.0886 |      -0.0796 |      -0.0715 |      -0.0642 |      -0.0576 |      -0.0516 |      -0.0461 |      -0.0412 |      -0.0368 |      -0.0327 |      -0.0291 |
        |            310 |      -0      |      -0.0001 |      -0.003  |      -0.0174 |      -0.0436 |      -0.0735 |      -0.1001 |      -0.12   |      -0.1328 |      -0.1395 |      -0.1413 |      -0.1397 |      -0.1356 |      -0.1299 |      -0.1233 |      -0.1162 |      -0.1089 |      -0.1016 |      -0.0945 |      -0.0877 |      -0.0813 |      -0.0752 |      -0.0694 |      -0.0641 |      -0.0591 |      -0.0545 |      -0.0502 |      -0.0462 |      -0.0425 |
        |            315 |      -0      |      -0      |      -0.0003 |      -0.0032 |      -0.012  |      -0.0264 |      -0.0434 |      -0.06   |      -0.0743 |      -0.0855 |      -0.0934 |      -0.0984 |      -0.1009 |      -0.1014 |      -0.1004 |      -0.0983 |      -0.0954 |      -0.0919 |      -0.088  |      -0.0839 |      -0.0798 |      -0.0756 |      -0.0715 |      -0.0675 |      -0.0636 |      -0.0598 |      -0.0563 |      -0.0529 |      -0.0496 |
        |            320 |      -0      |      -0      |      -0      |      -0.0005 |      -0.0028 |      -0.0081 |      -0.0164 |      -0.0264 |      -0.0369 |      -0.0467 |      -0.0553 |      -0.0623 |      -0.0677 |      -0.0716 |      -0.0741 |      -0.0754 |      -0.0758 |      -0.0753 |      -0.0743 |      -0.0728 |      -0.0709 |      -0.0687 |      -0.0664 |      -0.0639 |      -0.0614 |      -0.0589 |      -0.0563 |      -0.0538 |      -0.0513 |
        |            325 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0005 |      -0.0022 |      -0.0054 |      -0.0103 |      -0.0164 |      -0.0231 |      -0.0298 |      -0.0361 |      -0.0417 |      -0.0465 |      -0.0504 |      -0.0534 |      -0.0556 |      -0.0572 |      -0.0581 |      -0.0584 |      -0.0583 |      -0.0579 |      -0.0571 |      -0.0561 |      -0.0549 |      -0.0535 |      -0.0521 |      -0.0505 |      -0.0489 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0005 |      -0.0016 |      -0.0036 |      -0.0066 |      -0.0104 |      -0.0147 |      -0.0193 |      -0.0238 |      -0.028  |      -0.0319 |      -0.0353 |      -0.0382 |      -0.0406 |      -0.0425 |      -0.044  |      -0.0451 |      -0.0458 |      -0.0462 |      -0.0463 |      -0.0461 |      -0.0458 |      -0.0453 |      -0.0446 |      -0.0438 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0012 |      -0.0025 |      -0.0043 |      -0.0068 |      -0.0096 |      -0.0126 |      -0.0158 |      -0.0189 |      -0.0219 |      -0.0247 |      -0.0272 |      -0.0295 |      -0.0314 |      -0.033  |      -0.0343 |      -0.0354 |      -0.0362 |      -0.0368 |      -0.0372 |      -0.0374 |      -0.0374 |      -0.0373 |
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
            expiration_time_range (int): The number of days to use for the time to expiration. Defaults to 30 which equals
            30 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
            ["AAPL", "ASML"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2021-01-01",
            end_date="2025-12-31",
        )

        toolkit.options.get_ultima().loc["AAPL"]
        ```

        Which returns:

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |   2026-01-10 |   2026-01-11 |   2026-01-12 |   2026-01-13 |   2026-01-14 |   2026-01-15 |   2026-01-16 |   2026-01-17 |   2026-01-18 |   2026-01-19 |   2026-01-20 |   2026-01-21 |   2026-01-22 |   2026-01-23 |   2026-01-24 |   2026-01-25 |   2026-01-26 |   2026-01-27 |   2026-01-28 |   2026-01-29 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       6.7369 |      65.1811 |      75.3506 |      38.3805 |     -12.7593 |     -62.5916 |    -106.078  |    -142.36   |    -172.024  |    -196.04   |    -215.384  |    -230.907  |    -243.321  |    -253.204  |    -261.022  |    -267.149  |    -271.888  |    -275.481  |    -278.126  |    -279.983  |    -281.184  |    -281.838  |    -282.032  |    -281.842  |    -281.328  |    -280.543  |    -279.529  |    -278.324  |    -276.957  |
        |            295 |       0.2503 |      24.5473 |      75.4215 |     100.257  |      91.9666 |      62.7734 |      23.8569 |     -17.8245 |     -58.516  |     -96.4046 |    -130.76   |    -161.428  |    -188.542  |    -212.369  |    -233.225  |    -251.432  |    -267.293  |    -281.087  |    -293.065  |    -303.447  |    -312.428  |    -320.178  |    -326.843  |    -332.555  |    -337.424  |    -341.55   |    -345.019  |    -347.905  |    -350.274  |
        |            300 |       0.0029 |       4.2228 |      33.7597 |      77.876  |     110.55   |     122.462  |     115.65   |      95.4196 |      66.8238 |      33.7198 |      -1.2183 |     -36.2415 |     -70.2608 |    -102.637  |    -133.028  |    -161.285  |    -187.382  |    -211.367  |    -233.335  |    -253.401  |    -271.695  |    -288.346  |    -303.484  |    -317.231  |    -329.704  |    -341.009  |    -351.248  |    -360.511  |    -368.882  |
        |            305 |       0      |       0.3942 |       9.0883 |      36.6033 |      74.8804 |     109.918  |     133.53   |     143.585  |     141.278  |     129.043  |     109.464  |      84.815  |      56.9356 |      27.2384 |      -3.2269 |     -33.7016 |     -63.6512 |     -92.7108 |    -120.641  |    -147.295  |    -172.592  |    -196.5    |    -219.022  |    -240.182  |    -260.021  |    -278.591  |    -295.949  |    -312.156  |    -327.274  |
        |            310 |       0      |       0.0217 |       1.6302 |      12.0158 |      35.8272 |      68.6326 |     102.557  |     131.402  |     151.889  |     163.059  |     165.361  |     159.928  |     148.129  |     131.32   |     110.725  |      87.3944 |      62.1953 |      35.8306 |       8.8575 |     -18.2888 |     -45.2735 |     -71.8432 |     -97.8101 |    -123.038  |    -147.432  |    -170.929  |    -193.49   |    -215.098  |    -235.748  |
        |            315 |       0      |       0.0007 |       0.2059 |       2.9369 |      13.1148 |      33.1451 |      60.8597 |      91.7987 |     121.629  |     147.237  |     166.857  |     179.806  |     186.134  |     186.327  |     181.093  |     171.224  |     157.506  |     140.674  |     121.389  |     100.227  |      77.6848 |      54.18   |      30.0621 |       5.6197 |     -18.9113 |     -43.3395 |     -67.5117 |     -91.3065 |    -114.629  |
        |            320 |       0      |       0      |       0.019  |       0.5552 |       3.8311 |      13.0025 |      29.6083 |      52.7321 |      79.896  |     108.234  |     135.246  |     159.097  |     178.637  |     193.289  |     202.906  |     207.638  |     207.815  |     203.875  |     196.297  |     185.57   |     172.164  |     156.52   |     139.037  |     120.076  |      99.9523 |      78.9449 |      57.2942 |      35.2076 |      12.8623 |
        |            325 |       0      |       0      |       0.0013 |       0.0833 |       0.9176 |       4.2658 |      12.1976 |      25.8444 |      44.9524 |      68.2083 |      93.8062 |     119.917  |     144.962  |     167.73   |     187.384  |     203.423  |     215.61   |     223.915  |     228.451  |     229.43   |     227.124  |     221.835  |     213.88   |     203.573  |     191.214  |     177.09   |     161.463  |     144.577  |     126.65   |
        |            330 |       0      |       0      |       0.0001 |       0.0101 |       0.1838 |       1.194  |       4.3453 |      11.0542 |      22.2079 |      37.8873 |      57.4604 |      79.8488 |     103.8    |     128.091  |     151.651  |     173.619  |     193.349  |     210.403  |     224.513  |     235.558  |     243.524  |     248.483  |     250.566  |     249.944  |     246.811  |     241.376  |     233.854  |     224.455  |     213.384  |
        |            335 |       0      |       0      |       0      |       0.001  |       0.0312 |       0.2894 |       1.359  |       4.1913 |       9.7937 |      18.8797 |      31.6845 |      47.9684 |      67.1329 |      88.3706 |     110.801  |     133.568  |     155.906  |     177.172  |     196.854  |     214.569  |     230.049  |     243.13   |     253.728  |     261.83   |     267.472  |     270.734  |     271.721  |     270.56   |     267.39   |
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
