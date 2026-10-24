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

        # Option Statistics
        self._prices = self._daily_historical["Adj Close"]

        # The volatility on a date is that of the year of daily returns up to it, so a past
        # pricing date uses only what was known on that date.
        self._volatility = self._daily_historical["Return"].rolling(
            TRAILING_DAYS, min_periods=MINIMUM_VOLATILITY_DAYS
        ).std() * np.sqrt(TRAILING_DAYS)

        self._risk_free_rate = risk_free_rate["Adj Close"]
        self._annual_historical = annual_historical

        # Calculate Dividend Yield, relevant for Black Scholes formula
        dividend_yield = valuation_model.get_dividend_yield(
            dividends=self._annual_historical.loc[:, "Dividends"].T,
            stock_price=self._annual_historical.loc[:, "Adj Close"].T,
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

        # The dividend yield on a date: the dividends of the year up to it relative to the
        # price. It follows from the dividend adjustment of the prices, Adj Close / Close, which
        # grows by the yield of each dividend paid and is not affected by splits (unlike the
        # dividends per share themselves). The first year, without a full year before it,
        # takes the first full year's yield.
        self._daily_dividend_yield = pd.DataFrame()
        if "Close" in self._daily_historical:
            dates = to_period_index(self._daily_historical.index).to_timestamp()
            adjustment = (
                self._daily_historical["Adj Close"] / self._daily_historical["Close"]
            ).set_axis(dates)
            year_earlier = adjustment.reindex(
                dates - pd.Timedelta(days=365), method="ffill"
            )
            daily_dividend_yield = pd.DataFrame(
                adjustment.to_numpy() / year_earlier.to_numpy() - 1,
                index=self._daily_historical.index,
                columns=adjustment.columns,
            )
            self._daily_dividend_yield = daily_dividend_yield.clip(lower=0).bfill()

    def _dividend_yield_on(self, ticker: str, date) -> float:
        """
        The dividend yield of a ticker on a pricing date: the dividends of the year up to that
        date over its price, or the latest annual yield when there are no daily dividends.

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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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

        |       |   2026-10-09 |
        |------:|-------------:|
        | 185   |       4.2155 |
        | 245   |       4.2461 |
        | 270   |       1.8906 |
        | 275   |       2.1283 |
        | 280   |       1.302  |
        | 285   |       1.9159 |
        | 290   |       1.8173 |
        | 295   |       1.2774 |
        | 310   |       1.1031 |
        | 312.5 |       0.957  |
        | 315   |       0.9168 |
        | 325   |       0.4748 |
        | 327.5 |       0.4575 |
        | 330   |       0.4352 |
        | 332.5 |       0.375  |
        | 335   |       0.3711 |
        | 337.5 |       0.3512 |
        | 340   |       0.3385 |
        | 342.5 |       0.3497 |
        | 345   |       0.3604 |
        | 347.5 |       0.373  |
        | 350   |       0.3914 |
        | 352.5 |       0.3895 |
        | 355   |       0.441  |
        | 357.5 |       0.4675 |
        | 360   |       0.5151 |
        | 362.5 |       0.5215 |
        | 365   |       0.5647 |
        | 367.5 |       0.6074 |
        | 370   |       0.6495 |
        | 372.5 |       0.691  |
        | 375   |       0.732  |
        | 380   |       0.8126 |
        | 385   |       0.9535 |
        | 390   |       0.9683 |
        | 395   |       1.3212 |
        | 400   |       1.2428 |
        | 405   |       1.322  |
        | 410   |       1.2618 |
        | 415   |       1.3318 |
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
                pricing date (the dividends of the year before it relative to the price).
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

        |   Strike Price |   2026-10-09 |   2026-10-12 |   2026-10-14 |
        |---------------:|-------------:|-------------:|-------------:|
        |          270   |       1.9963 |     nan      |     nan      |
        |          275   |       1.8884 |     nan      |     nan      |
        |          280   |       1.7787 |     nan      |     nan      |
        |          285   |       1.6668 |     nan      |     nan      |
        |          290   |       1.5527 |       0.692  |     nan      |
        |          295   |       1.4364 |     nan      |     nan      |
        |          305   |     nan      |       0.5509 |     nan      |
        |          310   |       1.0741 |       0.4993 |     nan      |
        |          312.5 |       1.0121 |       0.4724 |     nan      |
        |          315   |       0.9497 |       0.4448 |     nan      |
        |          317.5 |     nan      |       0.4166 |     nan      |
        |          320   |     nan      |       0.3876 |     nan      |
        |          322.5 |     nan      |       0.358  |       0.288  |
        |          325   |       0.7002 |     nan      |       0.279  |
        |          327.5 |       0.6389 |       0.2979 |       0.2701 |
        |          330   |       0.5788 |       0.2682 |       0.2615 |
        |          332.5 |       0.5206 |       0.2399 |       0.2533 |
        |          335   |       0.465  |       0.2143 |       0.2456 |
        |          337.5 |       0.4134 |       0.1934 |       0.2386 |
        |          340   |       0.3676 |       0.1792 |       0.2324 |
        |          342.5 |       0.3298 |       0.1736 |       0.2275 |
        |          345   |       0.3031 |       0.177  |       0.2239 |
        |          347.5 |       0.2903 |       0.1881 |       0.2221 |
        |          350   |       0.2929 |       0.2048 |       0.2225 |
        |          352.5 |       0.3102 |       0.225  |       0.2254 |
        |          355   |       0.3394 |       0.2471 |       0.2311 |
        |          357.5 |       0.3773 |       0.2699 |       0.2399 |
        |          360   |       0.4212 |       0.2928 |       0.2519 |
        |          362.5 |       0.4689 |       0.3155 |       0.2671 |
        |          365   |       0.5189 |       0.3376 |       0.2853 |
        |          367.5 |       0.5703 |       0.3592 |       0.3063 |
        |          370   |       0.6224 |       0.3801 |       0.3296 |
        |          372.5 |       0.6746 |       0.4003 |     nan      |
        |          375   |       0.7267 |       0.4198 |     nan      |
        |          380   |       0.8296 |       0.457  |     nan      |
        |          385   |       0.9298 |       0.4919 |     nan      |
        |          390   |       1.0268 |     nan      |     nan      |
        |          395   |       1.1204 |     nan      |     nan      |
        |          400   |       1.2105 |     nan      |     nan      |
        |          405   |       1.2971 |     nan      |     nan      |
        |          410   |       1.3805 |     nan      |     nan      |
        |          415   |       1.4607 |       0.6642 |     nan      |
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
                pricing date (the dividends of the year before it relative to the price).
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
        |        170.05  |      0 |
        |        171.742 |      0 |
        |        173.434 |      0 |
        |        175.126 |      0 |
        |        176.818 |      0 |
        |        178.511 |      0 |
        |        180.203 |      0 |
        |        181.895 |      0 |
        |        183.587 |      0 |
        |        185.279 |      0 |
        |        186.971 |      0 |
        |        188.663 |      0 |
        |        190.355 |      0 |
        |        192.047 |      0 |
        |        193.739 |      0 |
        |        195.431 |      0 |
        |        197.123 |      0 |
        |        198.815 |      0 |
        |        200.507 |      0 |
        |        202.199 |      0 |
        |        203.891 |      0 |
        |        205.583 |      0 |
        |        207.275 |      0 |
        |        208.967 |      0 |
        |        210.659 |      0 |
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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        | UUUUUUUUUU |      48.9262 |      59.949  |      72.1073 |      85.2847 |      99.4195 |     114.526  |     130.669  |     147.917  |     166.344  |     186.028  |     207.052  |
        | DUUUUUUUUU |     nan      |      37.7929 |      47.7184 |      58.9104 |      71.1946 |      84.4291 |      98.5757 |     113.695  |     129.852  |     147.115  |     165.559  |
        | DDUUUUUUUU |     nan      |     nan      |      27.7161 |      36.398  |      46.5394 |      57.9305 |      70.3195 |      83.5648 |      97.7233 |     112.856  |     129.027  |
        | DDDUUUUUUU |     nan      |     nan      |     nan      |      18.8578 |      26.0839 |      35.0008 |      45.4416 |      57.0368 |      69.4357 |      82.6918 |      96.8621 |
        | DDDDUUUUUU |     nan      |     nan      |     nan      |     nan      |      11.4501 |      16.966  |      24.3617 |      33.6805 |      44.5302 |      56.1344 |      68.5432 |
        | DDDDDUUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |       5.7713 |       9.3652 |      14.809  |      22.6024 |      32.7521 |      43.61   |
        | DDDDDDUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       2.0572 |       3.7446 |       6.7741 |      12.1654 |      21.6578 |
        | DDDDDDDUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0.3079 |       0.6045 |       1.1869 |       2.3302 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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

        |   Strike Price |   (Period('2026-01-29', 'D'), 'Delta') |   (Period('2026-01-29', 'D'), 'Dual Delta') |   (Period('2026-01-29', 'D'), 'Vega') |
        |---------------:|---------------------------------------:|--------------------------------------------:|--------------------------------------:|
        |            290 |                                 0.2558 |                                     -0.2265 |                                0.2457 |
        |            295 |                                 0.1996 |                                     -0.1745 |                                0.2137 |
        |            300 |                                 0.1523 |                                     -0.1314 |                                0.18   |
        |            305 |                                 0.1137 |                                     -0.0968 |                                0.1472 |
        |            310 |                                 0.0831 |                                     -0.0698 |                                0.1169 |
        |            315 |                                 0.0595 |                                     -0.0493 |                                0.0904 |
        |            320 |                                 0.0417 |                                     -0.0341 |                                0.0682 |
        |            325 |                                 0.0287 |                                     -0.0231 |                                0.0501 |
        |            330 |                                 0.0194 |                                     -0.0154 |                                0.036  |
        |            335 |                                 0.0128 |                                     -0.0101 |                                0.0253 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |            0 |       0.0027 |       0.0117 |       0.0253 |       0.0407 |       0.0565 |       0.0719 |       0.0866 |       0.1004 |       0.1133 |       0.1254 |       0.1367 |       0.1472 |       0.1571 |       0.1664 |       0.1751 |       0.1833 |       0.191  |       0.1984 |       0.2053 |       0.2119 |       0.2182 |       0.2242 |       0.2299 |       0.2353 |       0.2405 |       0.2455 |       0.2503 |       0.2549 |
        |            295 |            0 |       0.0002 |       0.0022 |       0.007  |       0.0142 |       0.023  |       0.0327 |       0.0429 |       0.0531 |       0.0633 |       0.0732 |       0.0828 |       0.0921 |       0.101  |       0.1096 |       0.1178 |       0.1256 |       0.1332 |       0.1404 |       0.1473 |       0.1539 |       0.1603 |       0.1664 |       0.1723 |       0.178  |       0.1835 |       0.1887 |       0.1938 |       0.1987 |
        |            300 |            0 |       0      |       0.0003 |       0.0016 |       0.0042 |       0.0082 |       0.0133 |       0.0193 |       0.0259 |       0.0328 |       0.04   |       0.0472 |       0.0545 |       0.0617 |       0.0688 |       0.0758 |       0.0826 |       0.0893 |       0.0958 |       0.1021 |       0.1083 |       0.1142 |       0.12   |       0.1257 |       0.1311 |       0.1364 |       0.1416 |       0.1466 |       0.1514 |
        |            305 |            0 |       0      |       0      |       0.0003 |       0.0011 |       0.0026 |       0.0049 |       0.0079 |       0.0116 |       0.0158 |       0.0204 |       0.0254 |       0.0305 |       0.0359 |       0.0413 |       0.0467 |       0.0522 |       0.0577 |       0.0631 |       0.0685 |       0.0738 |       0.079  |       0.0841 |       0.0891 |       0.0941 |       0.0989 |       0.1037 |       0.1084 |       0.1129 |
        |            310 |            0 |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0016 |       0.003  |       0.0048 |       0.0071 |       0.0098 |       0.0129 |       0.0162 |       0.0198 |       0.0236 |       0.0276 |       0.0317 |       0.0359 |       0.0401 |       0.0444 |       0.0487 |       0.053  |       0.0573 |       0.0616 |       0.0658 |       0.07   |       0.0742 |       0.0783 |       0.0824 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |       0.001  |       0.0018 |       0.003  |       0.0044 |       0.0062 |       0.0082 |       0.0105 |       0.013  |       0.0157 |       0.0185 |       0.0215 |       0.0247 |       0.0279 |       0.0312 |       0.0346 |       0.038  |       0.0414 |       0.0449 |       0.0484 |       0.0519 |       0.0554 |       0.0589 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0007 |       0.0012 |       0.0019 |       0.0028 |       0.0039 |       0.0053 |       0.0068 |       0.0085 |       0.0104 |       0.0125 |       0.0147 |       0.017  |       0.0194 |       0.0219 |       0.0245 |       0.0272 |       0.0299 |       0.0327 |       0.0355 |       0.0384 |       0.0412 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0008 |       0.0012 |       0.0018 |       0.0025 |       0.0034 |       0.0045 |       0.0057 |       0.007  |       0.0085 |       0.01   |       0.0117 |       0.0135 |       0.0154 |       0.0174 |       0.0195 |       0.0216 |       0.0238 |       0.026  |       0.0283 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0003 |       0.0005 |       0.0008 |       0.0012 |       0.0017 |       0.0023 |       0.003  |       0.0038 |       0.0047 |       0.0058 |       0.0069 |       0.0082 |       0.0095 |       0.0109 |       0.0124 |       0.014  |       0.0156 |       0.0173 |       0.0191 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0008 |       0.0011 |       0.0015 |       0.002  |       0.0026 |       0.0032 |       0.004  |       0.0048 |       0.0057 |       0.0067 |       0.0077 |       0.0088 |       0.01   |       0.0113 |       0.0126 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |           -0 |      -0.0025 |      -0.0109 |      -0.0233 |      -0.0375 |      -0.0519 |      -0.0659 |      -0.0792 |      -0.0916 |      -0.1032 |      -0.114  |      -0.1241 |      -0.1334 |      -0.1421 |      -0.1503 |      -0.1579 |      -0.165  |      -0.1718 |      -0.1781 |      -0.1841 |      -0.1897 |      -0.195  |      -0.2001 |      -0.2049 |      -0.2095 |      -0.2139 |      -0.218  |      -0.222  |      -0.2258 |
        |            295 |           -0 |      -0.0002 |      -0.002  |      -0.0063 |      -0.0128 |      -0.0208 |      -0.0295 |      -0.0386 |      -0.0478 |      -0.0568 |      -0.0656 |      -0.0741 |      -0.0823 |      -0.0901 |      -0.0976 |      -0.1047 |      -0.1116 |      -0.1181 |      -0.1243 |      -0.1303 |      -0.1359 |      -0.1414 |      -0.1466 |      -0.1516 |      -0.1564 |      -0.161  |      -0.1654 |      -0.1696 |      -0.1737 |
        |            300 |           -0 |      -0      |      -0.0003 |      -0.0014 |      -0.0037 |      -0.0073 |      -0.0118 |      -0.0171 |      -0.0229 |      -0.029  |      -0.0353 |      -0.0417 |      -0.048  |      -0.0543 |      -0.0604 |      -0.0665 |      -0.0724 |      -0.0781 |      -0.0837 |      -0.0891 |      -0.0943 |      -0.0994 |      -0.1043 |      -0.109  |      -0.1136 |      -0.1181 |      -0.1224 |      -0.1266 |      -0.1306 |
        |            305 |           -0 |      -0      |      -0      |      -0.0003 |      -0.0009 |      -0.0023 |      -0.0043 |      -0.0069 |      -0.0101 |      -0.0138 |      -0.0178 |      -0.0221 |      -0.0265 |      -0.0311 |      -0.0357 |      -0.0404 |      -0.0451 |      -0.0497 |      -0.0543 |      -0.0589 |      -0.0634 |      -0.0677 |      -0.0721 |      -0.0763 |      -0.0804 |      -0.0845 |      -0.0884 |      -0.0923 |      -0.0961 |
        |            310 |           -0 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0014 |      -0.0026 |      -0.0041 |      -0.0061 |      -0.0084 |      -0.011  |      -0.0139 |      -0.017  |      -0.0202 |      -0.0235 |      -0.027  |      -0.0305 |      -0.0341 |      -0.0377 |      -0.0413 |      -0.0448 |      -0.0484 |      -0.052  |      -0.0555 |      -0.059  |      -0.0624 |      -0.0658 |      -0.0692 |
        |            315 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0009 |      -0.0016 |      -0.0025 |      -0.0037 |      -0.0052 |      -0.0069 |      -0.0088 |      -0.0109 |      -0.0132 |      -0.0156 |      -0.0181 |      -0.0207 |      -0.0233 |      -0.0261 |      -0.0288 |      -0.0317 |      -0.0345 |      -0.0374 |      -0.0402 |      -0.0431 |      -0.0459 |      -0.0488 |
        |            320 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0005 |      -0.001  |      -0.0016 |      -0.0023 |      -0.0033 |      -0.0044 |      -0.0057 |      -0.0071 |      -0.0086 |      -0.0103 |      -0.0121 |      -0.014  |      -0.016  |      -0.018  |      -0.0202 |      -0.0223 |      -0.0245 |      -0.0268 |      -0.0291 |      -0.0314 |      -0.0337 |
        |            325 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0006 |      -0.001  |      -0.0015 |      -0.0021 |      -0.0028 |      -0.0037 |      -0.0046 |      -0.0057 |      -0.0069 |      -0.0082 |      -0.0095 |      -0.011  |      -0.0125 |      -0.0141 |      -0.0158 |      -0.0175 |      -0.0192 |      -0.021  |      -0.0228 |
        |            330 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0006 |      -0.0009 |      -0.0013 |      -0.0018 |      -0.0024 |      -0.0031 |      -0.0038 |      -0.0046 |      -0.0055 |      -0.0065 |      -0.0076 |      -0.0087 |      -0.0099 |      -0.0111 |      -0.0124 |      -0.0138 |      -0.0152 |
        |            335 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0004 |      -0.0006 |      -0.0009 |      -0.0012 |      -0.0016 |      -0.002  |      -0.0026 |      -0.0031 |      -0.0038 |      -0.0045 |      -0.0053 |      -0.0061 |      -0.007  |      -0.0079 |      -0.0089 |      -0.0099 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |            0 |       0.0017 |       0.0075 |       0.0167 |       0.0277 |       0.0395 |       0.0515 |       0.0633 |       0.0749 |       0.0862 |       0.0971 |       0.1076 |       0.1178 |       0.1276 |       0.1372 |       0.1464 |       0.1553 |       0.1639 |       0.1723 |       0.1805 |       0.1884 |       0.1961 |       0.2037 |       0.211  |       0.2182 |       0.2252 |       0.232  |       0.2387 |       0.2453 |
        |            295 |            0 |       0.0002 |       0.0017 |       0.0055 |       0.0114 |       0.0189 |       0.0274 |       0.0366 |       0.0461 |       0.0557 |       0.0654 |       0.075  |       0.0845 |       0.0939 |       0.1031 |       0.1121 |       0.1209 |       0.1295 |       0.1379 |       0.1462 |       0.1543 |       0.1622 |       0.1699 |       0.1775 |       0.1849 |       0.1921 |       0.1992 |       0.2062 |       0.2131 |
        |            300 |            0 |       0      |       0.0003 |       0.0014 |       0.0039 |       0.0078 |       0.0128 |       0.0188 |       0.0256 |       0.0329 |       0.0405 |       0.0484 |       0.0565 |       0.0647 |       0.0728 |       0.081  |       0.0891 |       0.0972 |       0.1052 |       0.1131 |       0.1209 |       0.1286 |       0.1362 |       0.1436 |       0.151  |       0.1582 |       0.1654 |       0.1724 |       0.1793 |
        |            305 |            0 |       0      |       0      |       0.0003 |       0.0011 |       0.0028 |       0.0053 |       0.0087 |       0.0129 |       0.0178 |       0.0232 |       0.0291 |       0.0353 |       0.0418 |       0.0486 |       0.0555 |       0.0625 |       0.0695 |       0.0766 |       0.0838 |       0.0909 |       0.098  |       0.1051 |       0.1121 |       0.1191 |       0.126  |       0.1329 |       0.1397 |       0.1464 |
        |            310 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0009 |       0.0019 |       0.0036 |       0.0059 |       0.0089 |       0.0123 |       0.0163 |       0.0207 |       0.0255 |       0.0307 |       0.0361 |       0.0417 |       0.0475 |       0.0534 |       0.0595 |       0.0657 |       0.0719 |       0.0782 |       0.0845 |       0.0908 |       0.0972 |       0.1035 |       0.1099 |       0.1162 |
        |            315 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0006 |       0.0014 |       0.0025 |       0.0041 |       0.0061 |       0.0086 |       0.0114 |       0.0147 |       0.0184 |       0.0223 |       0.0266 |       0.031  |       0.0358 |       0.0407 |       0.0457 |       0.0509 |       0.0562 |       0.0617 |       0.0672 |       0.0727 |       0.0783 |       0.084  |       0.0897 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0005 |       0.001  |       0.0017 |       0.0028 |       0.0042 |       0.006  |       0.0081 |       0.0105 |       0.0132 |       0.0162 |       0.0195 |       0.023  |       0.0267 |       0.0307 |       0.0348 |       0.0391 |       0.0436 |       0.0482 |       0.0529 |       0.0577 |       0.0625 |       0.0675 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0003 |       0.0007 |       0.0012 |       0.002  |       0.0029 |       0.0042 |       0.0057 |       0.0074 |       0.0095 |       0.0117 |       0.0142 |       0.017  |       0.0199 |       0.0231 |       0.0264 |       0.0299 |       0.0336 |       0.0374 |       0.0413 |       0.0454 |       0.0495 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0005 |       0.0009 |       0.0014 |       0.0021 |       0.003  |       0.004  |       0.0053 |       0.0068 |       0.0085 |       0.0104 |       0.0125 |       0.0148 |       0.0173 |       0.02   |       0.0228 |       0.0258 |       0.0289 |       0.0321 |       0.0355 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0006 |       0.001  |       0.0015 |       0.0021 |       0.0029 |       0.0038 |       0.0049 |       0.0062 |       0.0076 |       0.0093 |       0.011  |       0.013  |       0.0151 |       0.0173 |       0.0197 |       0.0223 |       0.0249 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |      -0.0004 |      -0.0135 |      -0.041  |      -0.0685 |      -0.091  |      -0.1083 |      -0.1211 |      -0.1306 |      -0.1376 |      -0.1427 |      -0.1464 |      -0.149  |      -0.1508 |      -0.1519 |      -0.1526 |      -0.1529 |      -0.1529 |      -0.1526 |      -0.1522 |      -0.1516 |      -0.151  |      -0.1502 |      -0.1494 |      -0.1485 |      -0.1476 |      -0.1466 |      -0.1457 |      -0.1447 |      -0.1437 |
        |            295 |      -0      |      -0.0014 |      -0.0092 |      -0.0225 |      -0.0374 |      -0.0517 |      -0.0644 |      -0.0753 |      -0.0844 |      -0.092  |      -0.0983 |      -0.1035 |      -0.1078 |      -0.1114 |      -0.1143 |      -0.1166 |      -0.1186 |      -0.1201 |      -0.1214 |      -0.1223 |      -0.1231 |      -0.1237 |      -0.1241 |      -0.1243 |      -0.1245 |      -0.1246 |      -0.1245 |      -0.1244 |      -0.1242 |
        |            300 |      -0      |      -0.0001 |      -0.0015 |      -0.0059 |      -0.0128 |      -0.0212 |      -0.0301 |      -0.0387 |      -0.0468 |      -0.0542 |      -0.0608 |      -0.0667 |      -0.0719 |      -0.0765 |      -0.0805 |      -0.0841 |      -0.0872 |      -0.0899 |      -0.0923 |      -0.0943 |      -0.0961 |      -0.0977 |      -0.0991 |      -0.1003 |      -0.1013 |      -0.1022 |      -0.103  |      -0.1036 |      -0.1042 |
        |            305 |      -0      |      -0      |      -0.0002 |      -0.0012 |      -0.0037 |      -0.0075 |      -0.0124 |      -0.0179 |      -0.0236 |      -0.0292 |      -0.0347 |      -0.04   |      -0.0449 |      -0.0494 |      -0.0536 |      -0.0574 |      -0.0609 |      -0.0641 |      -0.067  |      -0.0697 |      -0.0721 |      -0.0743 |      -0.0762 |      -0.078  |      -0.0797 |      -0.0811 |      -0.0825 |      -0.0837 |      -0.0848 |
        |            310 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0009 |      -0.0023 |      -0.0045 |      -0.0074 |      -0.0108 |      -0.0145 |      -0.0184 |      -0.0224 |      -0.0263 |      -0.0301 |      -0.0338 |      -0.0373 |      -0.0406 |      -0.0437 |      -0.0466 |      -0.0494 |      -0.052  |      -0.0544 |      -0.0566 |      -0.0587 |      -0.0606 |      -0.0624 |      -0.0641 |      -0.0656 |      -0.0671 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0015 |      -0.0028 |      -0.0046 |      -0.0067 |      -0.0091 |      -0.0117 |      -0.0145 |      -0.0173 |      -0.0202 |      -0.023  |      -0.0258 |      -0.0285 |      -0.0312 |      -0.0337 |      -0.0361 |      -0.0384 |      -0.0406 |      -0.0427 |      -0.0447 |      -0.0466 |      -0.0484 |      -0.0501 |      -0.0517 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0004 |      -0.001  |      -0.0018 |      -0.0029 |      -0.0042 |      -0.0058 |      -0.0075 |      -0.0095 |      -0.0115 |      -0.0136 |      -0.0157 |      -0.0179 |      -0.02   |      -0.0221 |      -0.0242 |      -0.0263 |      -0.0282 |      -0.0302 |      -0.032  |      -0.0338 |      -0.0355 |      -0.0372 |      -0.0388 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0006 |      -0.0011 |      -0.0018 |      -0.0027 |      -0.0037 |      -0.0049 |      -0.0062 |      -0.0077 |      -0.0092 |      -0.0108 |      -0.0124 |      -0.014  |      -0.0157 |      -0.0174 |      -0.019  |      -0.0207 |      -0.0223 |      -0.0239 |      -0.0254 |      -0.027  |      -0.0284 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0007 |      -0.0012 |      -0.0017 |      -0.0024 |      -0.0032 |      -0.0041 |      -0.0052 |      -0.0062 |      -0.0074 |      -0.0086 |      -0.0099 |      -0.0111 |      -0.0125 |      -0.0138 |      -0.0151 |      -0.0164 |      -0.0178 |      -0.0191 |      -0.0204 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0003 |      -0.0005 |      -0.0008 |      -0.0011 |      -0.0016 |      -0.0022 |      -0.0028 |      -0.0035 |      -0.0043 |      -0.0051 |      -0.006  |      -0.0069 |      -0.0079 |      -0.0089 |      -0.01   |      -0.011  |      -0.0121 |      -0.0132 |      -0.0143 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |            0 |       0.0039 |       0.0259 |       0.0742 |       0.149  |       0.2476 |       0.3667 |       0.5034 |       0.6552 |       0.8202 |       0.9965 |       1.1827 |       1.3778 |       1.5808 |       1.7908 |       2.0071 |       2.2291 |       2.4563 |       2.6883 |       2.9247 |       3.1651 |       3.4092 |       3.6568 |       3.9075 |       4.1613 |       4.4178 |       4.677  |       4.9386 |       5.2026 |
        |            295 |            0 |       0.0003 |       0.0049 |       0.0205 |       0.0519 |       0.1008 |       0.1671 |       0.2498 |       0.3477 |       0.4593 |       0.5835 |       0.7189 |       0.8646 |       1.0197 |       1.1832 |       1.3545 |       1.533  |       1.7179 |       1.909  |       2.1056 |       2.3074 |       2.514  |       2.7251 |       2.9404 |       3.1596 |       3.3825 |       3.6089 |       3.8386 |       4.0713 |
        |            300 |            0 |       0      |       0.0007 |       0.0046 |       0.0154 |       0.036  |       0.0682 |       0.1127 |       0.1696 |       0.2387 |       0.3193 |       0.411  |       0.5129 |       0.6246 |       0.7452 |       0.8743 |       1.0112 |       1.1554 |       1.3065 |       1.464  |       1.6276 |       1.7968 |       1.9712 |       2.1507 |       2.3349 |       2.5235 |       2.7164 |       2.9132 |       3.1137 |
        |            305 |            0 |       0      |       0.0001 |       0.0009 |       0.0039 |       0.0113 |       0.025  |       0.0463 |       0.0762 |       0.1152 |       0.1636 |       0.2213 |       0.288  |       0.3637 |       0.4479 |       0.5402 |       0.6403 |       0.7479 |       0.8626 |       0.9839 |       1.1117 |       1.2455 |       1.385  |       1.5301 |       1.6803 |       1.8355 |       1.9954 |       2.1598 |       2.3285 |
        |            310 |            0 |       0      |       0      |       0.0001 |       0.0009 |       0.0031 |       0.0082 |       0.0174 |       0.0316 |       0.0518 |       0.0786 |       0.1124 |       0.1533 |       0.2016 |       0.2572 |       0.3199 |       0.3897 |       0.4664 |       0.5498 |       0.6397 |       0.7358 |       0.838  |       0.9459 |       1.0595 |       1.1784 |       1.3025 |       1.4315 |       1.5653 |       1.7037 |
        |            315 |            0 |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0025 |       0.006  |       0.0121 |       0.0217 |       0.0355 |       0.0539 |       0.0775 |       0.1065 |       0.1413 |       0.1818 |       0.2282 |       0.2805 |       0.3387 |       0.4027 |       0.4724 |       0.5477 |       0.6285 |       0.7146 |       0.8059 |       0.9023 |       1.0036 |       1.1096 |       1.2203 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0019 |       0.0043 |       0.0085 |       0.0151 |       0.0245 |       0.0373 |       0.0538 |       0.0744 |       0.0993 |       0.1288 |       0.1629 |       0.2019 |       0.2458 |       0.2945 |       0.3481 |       0.4066 |       0.4699 |       0.538  |       0.6107 |       0.6881 |       0.77   |       0.8563 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |       0.0014 |       0.0031 |       0.0061 |       0.0106 |       0.0171 |       0.026  |       0.0376 |       0.0522 |       0.0701 |       0.0915 |       0.1166 |       0.1455 |       0.1784 |       0.2153 |       0.2564 |       0.3015 |       0.3508 |       0.4042 |       0.4617 |       0.5233 |       0.589  |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0011 |       0.0023 |       0.0043 |       0.0075 |       0.012  |       0.0182 |       0.0264 |       0.0368 |       0.0497 |       0.0653 |       0.0837 |       0.1052 |       0.1298 |       0.1577 |       0.1889 |       0.2236 |       0.2617 |       0.3034 |       0.3487 |       0.3974 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0008 |       0.0017 |       0.0031 |       0.0053 |       0.0085 |       0.0129 |       0.0187 |       0.0262 |       0.0355 |       0.0468 |       0.0604 |       0.0763 |       0.0947 |       0.1157 |       0.1394 |       0.166  |       0.1954 |       0.2279 |       0.2633 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |           -0 |      -0.004  |      -0.0261 |      -0.0751 |      -0.1513 |      -0.2519 |      -0.374  |      -0.5145 |      -0.6711 |      -0.8416 |      -1.0244 |      -1.2181 |      -1.4215 |      -1.6336 |      -1.8536 |      -2.0808 |      -2.3145 |      -2.5543 |      -2.7997 |      -3.0503 |      -3.3056 |      -3.5655 |      -3.8296 |      -4.0977 |      -4.3695 |      -4.6449 |      -4.9236 |      -5.2055 |      -5.4904 |
        |            295 |           -0 |      -0.0003 |      -0.0049 |      -0.0207 |      -0.0526 |      -0.1024 |      -0.1701 |      -0.2547 |      -0.3552 |      -0.4701 |      -0.5981 |      -0.7382 |      -0.8893 |      -1.0505 |      -1.2209 |      -1.3998 |      -1.5865 |      -1.7805 |      -1.9813 |      -2.1884 |      -2.4014 |      -2.6199 |      -2.8436 |      -3.0723 |      -3.3055 |      -3.5431 |      -3.7849 |      -4.0306 |      -4.2801 |
        |            300 |           -0 |      -0      |      -0.0007 |      -0.0047 |      -0.0156 |      -0.0365 |      -0.0692 |      -0.1146 |      -0.1729 |      -0.2437 |      -0.3266 |      -0.421  |      -0.5263 |      -0.6418 |      -0.7669 |      -0.901  |      -1.0436 |      -1.1941 |      -1.3521 |      -1.5171 |      -1.6888 |      -1.8667 |      -2.0506 |      -2.24   |      -2.4349 |      -2.6347 |      -2.8394 |      -3.0487 |      -3.2624 |
        |            305 |           -0 |      -0      |      -0.0001 |      -0.0009 |      -0.004  |      -0.0115 |      -0.0253 |      -0.047  |      -0.0775 |      -0.1174 |      -0.167  |      -0.2262 |      -0.2949 |      -0.3729 |      -0.4598 |      -0.5554 |      -0.6592 |      -0.771  |      -0.8904 |      -1.0169 |      -1.1504 |      -1.2905 |      -1.4368 |      -1.5892 |      -1.7473 |      -1.9109 |      -2.0798 |      -2.2537 |      -2.4324 |
        |            310 |           -0 |      -0      |      -0      |      -0.0001 |      -0.0009 |      -0.0032 |      -0.0083 |      -0.0176 |      -0.0321 |      -0.0527 |      -0.0801 |      -0.1147 |      -0.1567 |      -0.2063 |      -0.2635 |      -0.3282 |      -0.4004 |      -0.4798 |      -0.5663 |      -0.6597 |      -0.7597 |      -0.8662 |      -0.9789 |      -1.0977 |      -1.2223 |      -1.3525 |      -1.4882 |      -1.629  |      -1.775  |
        |            315 |           -0 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0008 |      -0.0025 |      -0.006  |      -0.0123 |      -0.0221 |      -0.0361 |      -0.055  |      -0.0791 |      -0.1089 |      -0.1445 |      -0.1862 |      -0.234  |      -0.288  |      -0.3482 |      -0.4145 |      -0.4868 |      -0.565  |      -0.649  |      -0.7388 |      -0.8341 |      -0.9349 |      -1.0409 |      -1.1521 |      -1.2684 |
        |            320 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0007 |      -0.0019 |      -0.0044 |      -0.0087 |      -0.0153 |      -0.0249 |      -0.038  |      -0.0549 |      -0.0759 |      -0.1015 |      -0.1318 |      -0.167  |      -0.2072 |      -0.2525 |      -0.3029 |      -0.3584 |      -0.4191 |      -0.4849 |      -0.5557 |      -0.6315 |      -0.7122 |      -0.7978 |      -0.8881 |
        |            325 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0015 |      -0.0032 |      -0.0062 |      -0.0107 |      -0.0174 |      -0.0264 |      -0.0383 |      -0.0533 |      -0.0716 |      -0.0936 |      -0.1195 |      -0.1493 |      -0.1832 |      -0.2214 |      -0.2638 |      -0.3106 |      -0.3617 |      -0.4172 |      -0.477  |      -0.5412 |      -0.6097 |
        |            330 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0011 |      -0.0023 |      -0.0044 |      -0.0076 |      -0.0122 |      -0.0186 |      -0.027  |      -0.0376 |      -0.0508 |      -0.0668 |      -0.0858 |      -0.1078 |      -0.1332 |      -0.162  |      -0.1943 |      -0.2302 |      -0.2697 |      -0.313  |      -0.36   |      -0.4107 |
        |            335 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0008 |      -0.0017 |      -0.0032 |      -0.0054 |      -0.0087 |      -0.0132 |      -0.0191 |      -0.0267 |      -0.0363 |      -0.0479 |      -0.0618 |      -0.0782 |      -0.0971 |      -0.1188 |      -0.1433 |      -0.1708 |      -0.2013 |      -0.2349 |      -0.2716 |
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
            uses the dividend yield on the pricing date (the dividends of the year before it relative to the price).
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
        |            290 |      259.73  |      140.338 |      99.6198 |      78.796  |      66.0284 |      57.3396 |      51.0108 |      46.1754 |      42.3476 |      39.2335 |      36.6443 |      34.4532 |      32.5717 |      30.936  |      29.499  |      28.225  |      27.0866 |      26.0622 |      25.1347 |      24.2903 |      23.5178 |      22.8079 |      22.1529 |      21.5464 |      20.9827 |      20.4574 |      19.9664 |      19.5062 |      19.0739 |
        |            295 |      314.881 |      166.673 |     116.605  |      91.2131 |      75.7582 |      65.3085 |      57.7409 |      51.9889 |      47.4568 |      43.7853 |      40.7447 |      38.1809 |      35.9867 |      34.0851 |      32.4193 |      30.9465 |      29.6338 |      28.4554 |      27.3909 |      26.4239 |      25.541  |      24.7312 |      23.9854 |      23.296  |      22.6565 |      22.0614 |      21.506  |      20.9863 |      20.4988 |
        |            300 |      370.212 |      193.319 |     133.866  |     103.862  |      85.6843 |      73.4452 |      64.6161 |      57.9292 |      52.6778 |      48.4368 |      44.9344 |      41.9892 |      39.4749 |      37.3009 |      35.4008 |      33.7243 |      32.2329 |      30.8967 |      29.6918 |      28.5992 |      27.6031 |      26.691  |      25.8523 |      25.078  |      24.3607 |      23.6942 |      23.0729 |      22.4922 |      21.9481 |
        |            305 |      425.343 |      220.047 |     151.25   |     116.635  |      95.7249 |      81.6855 |      71.5846 |      63.9536 |      57.9749 |      53.1572 |      49.1871 |      45.8551 |      43.0159 |      40.5655 |      38.4273 |      36.5438 |      34.871  |      33.3744 |      32.0267 |      30.8062 |      29.6952 |      28.679  |      27.7456 |      26.885  |      26.0886 |      25.3493 |      24.661  |      24.0183 |      23.4166 |
        |            310 |      480.059 |      246.715 |     168.655  |     129.454  |     105.819  |      89.9805 |      78.6061 |      70.0283 |      63.3193 |      57.9219 |      53.481  |      49.7594 |      46.5928 |      43.8636 |      41.4853 |      39.3929 |      37.5367 |      35.878  |      34.3862 |      33.0365 |      31.8091 |      30.6876 |      29.6586 |      28.7106 |      27.8342 |      27.0213 |      26.2651 |      25.5596 |      24.8997 |
        |            315 |      534.236 |      273.229 |     186.009  |     142.263  |     115.922  |      98.2932 |      85.6496 |      76.1268 |      68.6881 |      62.7108 |      57.7985 |      53.6867 |      50.1918 |      47.1828 |      44.5633 |      42.2611 |      40.2207 |      38.3991 |      36.7621 |      35.2825 |      33.938  |      32.7106 |      31.5852 |      30.5493 |      29.5923 |      28.7053 |      27.8807 |      27.1119 |      26.3933 |
        |            320 |      587.804 |      299.529 |     203.266  |     155.024  |     126.001  |     106.596  |      92.6914 |      82.2287 |      74.0633 |      67.5081 |      62.1256 |      57.6241 |      53.8013 |      50.5126 |      47.652  |      45.1398 |      42.915  |      40.9302 |      39.1479 |      37.538  |      36.0762 |      34.7425 |      33.5204 |      32.3962 |      31.3583 |      30.3969 |      29.5036 |      28.6713 |      27.8938 |
        |            325 |      640.721 |      325.576 |     220.39   |     167.707  |     136.032  |     114.868  |      99.7134 |      88.318  |      79.4308 |      72.3011 |      66.451  |      61.5616 |      57.4121 |      53.8447 |      50.7436 |      48.0219 |      45.6131 |      43.4653 |      41.5378 |      39.7977 |      38.2186 |      36.7786 |      35.4599 |      34.2474 |      33.1285 |      32.0926 |      31.1306 |      30.2347 |      29.3981 |
        |            330 |      692.967 |      351.347 |     237.361  |     180.293  |     145.998  |     123.094  |     106.702  |      94.3825 |      84.7798 |      77.0801 |      70.7656 |      65.4908 |      61.0167 |      57.1721 |      53.8317 |      50.9015 |      48.3093 |      45.9993 |      43.927  |      42.0572 |      40.361  |      38.8151 |      37.3999 |      36.0992 |      34.8995 |      33.7893 |      32.7586 |      31.7991 |      30.9035 |
        |            335 |      744.535 |      376.825 |     254.162  |     192.768  |     155.885  |     131.262  |     113.647  |     100.413  |      90.1015 |      81.8372 |      75.0623 |      69.4054 |      64.609  |      60.4892 |      56.9112 |      53.7737 |      50.9994 |      48.5279 |      46.3117 |      44.3127 |      42.5001 |      40.8486 |      39.3373 |      37.9489 |      36.6686 |      35.4843 |      34.3852 |      33.3623 |      32.4079 |
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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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

        |   Strike Price |   (Period('2026-01-29', 'D'), 'Gamma') |   (Period('2026-01-29', 'D'), 'Dual Gamma') |   (Period('2026-01-29', 'D'), 'Vanna') |
        |---------------:|---------------------------------------:|--------------------------------------------:|---------------------------------------:|
        |            290 |                                 0.0129 |                                      0.0113 |                                 0.7399 |
        |            295 |                                 0.0113 |                                      0.0095 |                                 0.8038 |
        |            300 |                                 0.0095 |                                      0.0077 |                                 0.8101 |
        |            305 |                                 0.0078 |                                      0.0061 |                                 0.7692 |
        |            310 |                                 0.0062 |                                      0.0047 |                                 0.6948 |
        |            315 |                                 0.0048 |                                      0.0035 |                                 0.6009 |
        |            320 |                                 0.0036 |                                      0.0026 |                                 0.5001 |
        |            325 |                                 0.0026 |                                      0.0018 |                                 0.402  |
        |            330 |                                 0.0019 |                                      0.0013 |                                 0.313  |
        |            335 |                                 0.0013 |                                      0.0009 |                                 0.2367 |
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
        |            290 |            0 |       0.0013 |       0.0038 |       0.0064 |       0.0085 |       0.0101 |       0.0113 |       0.0121 |       0.0128 |       0.0132 |       0.0135 |       0.0137 |       0.0139 |       0.014  |       0.014  |       0.014  |       0.014  |       0.014  |       0.0139 |       0.0138 |       0.0137 |       0.0137 |       0.0136 |       0.0135 |       0.0134 |       0.0133 |       0.0132 |       0.0131 |       0.013  |
        |            295 |            0 |       0.0001 |       0.0009 |       0.0021 |       0.0035 |       0.0048 |       0.006  |       0.007  |       0.0078 |       0.0085 |       0.0091 |       0.0096 |       0.01   |       0.0103 |       0.0105 |       0.0107 |       0.0109 |       0.011  |       0.0111 |       0.0112 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |       0.0113 |
        |            300 |            0 |       0      |       0.0001 |       0.0006 |       0.0012 |       0.002  |       0.0028 |       0.0036 |       0.0044 |       0.005  |       0.0056 |       0.0062 |       0.0067 |       0.0071 |       0.0074 |       0.0078 |       0.008  |       0.0083 |       0.0085 |       0.0087 |       0.0088 |       0.009  |       0.0091 |       0.0092 |       0.0093 |       0.0093 |       0.0094 |       0.0094 |       0.0095 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0007 |       0.0012 |       0.0017 |       0.0022 |       0.0027 |       0.0032 |       0.0037 |       0.0042 |       0.0046 |       0.005  |       0.0053 |       0.0056 |       0.0059 |       0.0062 |       0.0064 |       0.0066 |       0.0068 |       0.007  |       0.0072 |       0.0073 |       0.0074 |       0.0075 |       0.0076 |       0.0077 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0007 |       0.001  |       0.0014 |       0.0017 |       0.0021 |       0.0024 |       0.0028 |       0.0031 |       0.0035 |       0.0038 |       0.004  |       0.0043 |       0.0046 |       0.0048 |       0.005  |       0.0052 |       0.0054 |       0.0056 |       0.0057 |       0.0059 |       0.006  |       0.0061 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0003 |       0.0004 |       0.0006 |       0.0008 |       0.0011 |       0.0013 |       0.0016 |       0.0019 |       0.0021 |       0.0024 |       0.0026 |       0.0029 |       0.0031 |       0.0033 |       0.0035 |       0.0037 |       0.0039 |       0.0041 |       0.0043 |       0.0044 |       0.0046 |       0.0047 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0007 |       0.0009 |       0.0011 |       0.0013 |       0.0015 |       0.0017 |       0.0019 |       0.002  |       0.0022 |       0.0024 |       0.0026 |       0.0028 |       0.003  |       0.0031 |       0.0033 |       0.0034 |       0.0036 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0003 |       0.0005 |       0.0006 |       0.0007 |       0.0009 |       0.001  |       0.0011 |       0.0013 |       0.0015 |       0.0016 |       0.0018 |       0.0019 |       0.0021 |       0.0022 |       0.0023 |       0.0025 |       0.0026 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0012 |       0.0013 |       0.0014 |       0.0015 |       0.0016 |       0.0018 |       0.0019 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |
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
        |            290 |            0 |       0.0011 |       0.0034 |       0.0056 |       0.0074 |       0.0088 |       0.0098 |       0.0106 |       0.0111 |       0.0115 |       0.0118 |       0.012  |       0.0121 |       0.0122 |       0.0122 |       0.0122 |       0.0122 |       0.0122 |       0.0121 |       0.0121 |       0.012  |       0.0119 |       0.0119 |       0.0118 |       0.0117 |       0.0116 |       0.0115 |       0.0114 |       0.0113 |
        |            295 |            0 |       0.0001 |       0.0007 |       0.0018 |       0.003  |       0.0041 |       0.0051 |       0.0059 |       0.0066 |       0.0072 |       0.0077 |       0.0081 |       0.0084 |       0.0087 |       0.0089 |       0.0091 |       0.0092 |       0.0093 |       0.0094 |       0.0095 |       0.0095 |       0.0095 |       0.0096 |       0.0096 |       0.0096 |       0.0096 |       0.0095 |       0.0095 |       0.0095 |
        |            300 |            0 |       0      |       0.0001 |       0.0005 |       0.001  |       0.0016 |       0.0023 |       0.0029 |       0.0036 |       0.0041 |       0.0046 |       0.0051 |       0.0054 |       0.0058 |       0.0061 |       0.0063 |       0.0066 |       0.0068 |       0.0069 |       0.0071 |       0.0072 |       0.0073 |       0.0074 |       0.0075 |       0.0076 |       0.0076 |       0.0077 |       0.0077 |       0.0077 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0006 |       0.0009 |       0.0013 |       0.0017 |       0.0022 |       0.0026 |       0.0029 |       0.0033 |       0.0036 |       0.0039 |       0.0042 |       0.0044 |       0.0047 |       0.0049 |       0.0051 |       0.0052 |       0.0054 |       0.0055 |       0.0057 |       0.0058 |       0.0059 |       0.006  |       0.006  |       0.0061 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0008 |       0.001  |       0.0013 |       0.0016 |       0.0019 |       0.0021 |       0.0024 |       0.0026 |       0.0029 |       0.0031 |       0.0033 |       0.0035 |       0.0037 |       0.0038 |       0.004  |       0.0041 |       0.0043 |       0.0044 |       0.0045 |       0.0046 |       0.0047 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0006 |       0.0008 |       0.001  |       0.0012 |       0.0014 |       0.0016 |       0.0018 |       0.002  |       0.0021 |       0.0023 |       0.0025 |       0.0026 |       0.0028 |       0.0029 |       0.003  |       0.0032 |       0.0033 |       0.0034 |       0.0035 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0008 |       0.0009 |       0.001  |       0.0012 |       0.0013 |       0.0015 |       0.0016 |       0.0017 |       0.0019 |       0.002  |       0.0021 |       0.0022 |       0.0023 |       0.0025 |       0.0026 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |       0.0015 |       0.0016 |       0.0017 |       0.0018 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0007 |       0.0008 |       0.0009 |
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
        |            290 |        0.002 |       0.0715 |       0.2168 |       0.3621 |       0.4803 |       0.5703 |       0.6372 |       0.6864 |       0.7223 |       0.7482 |       0.7667 |       0.7796 |       0.7882 |       0.7935 |       0.7963 |       0.7971 |       0.7964 |       0.7945 |       0.7918 |       0.7883 |       0.7842 |       0.7797 |       0.7749 |       0.7699 |       0.7647 |       0.7594 |       0.754  |       0.7485 |       0.743  |
        |            295 |        0     |       0.0096 |       0.0614 |       0.1493 |       0.248  |       0.3422 |       0.4255 |       0.4967 |       0.5564 |       0.6059 |       0.6467 |       0.6804 |       0.7079 |       0.7305 |       0.7488 |       0.7637 |       0.7757 |       0.7853 |       0.7928 |       0.7986 |       0.803  |       0.8061 |       0.8082 |       0.8095 |       0.81   |       0.8098 |       0.8091 |       0.808  |       0.8064 |
        |            300 |        0     |       0.0008 |       0.0123 |       0.0468 |       0.102  |       0.1686 |       0.2385 |       0.3067 |       0.3703 |       0.4283 |       0.4802 |       0.5263 |       0.5669 |       0.6026 |       0.6339 |       0.6612 |       0.685  |       0.7058 |       0.7239 |       0.7396 |       0.7533 |       0.7651 |       0.7753 |       0.7841 |       0.7916 |       0.798  |       0.8034 |       0.808  |       0.8118 |
        |            305 |        0     |       0      |       0.0018 |       0.0115 |       0.0342 |       0.0697 |       0.1146 |       0.1648 |       0.2171 |       0.2693 |       0.3197 |       0.3674 |       0.412  |       0.4534 |       0.4913 |       0.5261 |       0.5578 |       0.5866 |       0.6127 |       0.6365 |       0.6579 |       0.6774 |       0.695  |       0.7108 |       0.7252 |       0.7381 |       0.7498 |       0.7603 |       0.7698 |
        |            310 |        0     |       0      |       0.0002 |       0.0022 |       0.0095 |       0.0246 |       0.0478 |       0.0781 |       0.1137 |       0.1525 |       0.1932 |       0.2343 |       0.275  |       0.3146 |       0.3527 |       0.389  |       0.4233 |       0.4556 |       0.4859 |       0.5142 |       0.5407 |       0.5652 |       0.5881 |       0.6093 |       0.629  |       0.6472 |       0.6641 |       0.6798 |       0.6942 |
        |            315 |        0     |       0      |       0      |       0.0004 |       0.0022 |       0.0075 |       0.0176 |       0.033  |       0.0536 |       0.0786 |       0.1069 |       0.1376 |       0.1699 |       0.203  |       0.2363 |       0.2693 |       0.3017 |       0.3332 |       0.3637 |       0.393  |       0.421  |       0.4477 |       0.4731 |       0.4972 |       0.52   |       0.5416 |       0.562  |       0.5813 |       0.5995 |
        |            320 |        0     |       0      |       0      |       0      |       0.0004 |       0.002  |       0.0057 |       0.0126 |       0.023  |       0.0371 |       0.0546 |       0.075  |       0.0979 |       0.1226 |       0.1487 |       0.1757 |       0.2032 |       0.2308 |       0.2584 |       0.2856 |       0.3123 |       0.3383 |       0.3637 |       0.3883 |       0.412  |       0.4348 |       0.4568 |       0.4778 |       0.498  |
        |            325 |        0     |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0017 |       0.0043 |       0.009  |       0.0161 |       0.0258 |       0.0381 |       0.0528 |       0.0697 |       0.0883 |       0.1085 |       0.1299 |       0.1521 |       0.175  |       0.1982 |       0.2216 |       0.245  |       0.2683 |       0.2913 |       0.3139 |       0.3361 |       0.3578 |       0.379  |       0.3995 |
        |            330 |        0     |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0014 |       0.0033 |       0.0065 |       0.0114 |       0.0182 |       0.0268 |       0.0374 |       0.0497 |       0.0637 |       0.0791 |       0.0957 |       0.1134 |       0.1319 |       0.151  |       0.1706 |       0.1905 |       0.2107 |       0.2309 |       0.251  |       0.2711 |       0.2909 |       0.3105 |
        |            335 |        0     |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0011 |       0.0024 |       0.0047 |       0.0081 |       0.0129 |       0.019  |       0.0266 |       0.0357 |       0.0461 |       0.0577 |       0.0705 |       0.0844 |       0.0991 |       0.1145 |       0.1307 |       0.1473 |       0.1643 |       0.1816 |       0.1991 |       0.2167 |       0.2343 |
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
        |            290 |      -0.1169 |      -2.1261 |      -4.3136 |      -5.4185 |      -5.7674 |      -5.7237 |      -5.4974 |      -5.1966 |      -4.8748 |      -4.5581 |      -4.2584 |      -3.9804 |      -3.7252 |      -3.4923 |      -3.2801 |      -3.087  |      -2.911  |      -2.7505 |      -2.6038 |      -2.4694 |      -2.3462 |      -2.2329 |      -2.1284 |      -2.032  |      -1.9428 |      -1.86   |      -1.7831 |      -1.7115 |      -1.6447 |
        |            295 |      -0.0016 |      -0.2838 |      -1.2195 |      -2.2283 |      -2.969  |      -3.4217 |      -3.6561 |      -3.7429 |      -3.7353 |      -3.6694 |      -3.5691 |      -3.4496 |      -3.3208 |      -3.1891 |      -3.0583 |      -2.9308 |      -2.808  |      -2.6908 |      -2.5794 |      -2.4739 |      -2.3743 |      -2.2803 |      -2.1917 |      -2.1082 |      -2.0295 |      -1.9554 |      -1.8855 |      -1.8195 |      -1.7571 |
        |            300 |      -0      |      -0.0229 |      -0.2432 |      -0.6978 |      -1.2186 |      -1.6817 |      -2.0435 |      -2.3036 |      -2.4775 |      -2.5838 |      -2.6388 |      -2.6562 |      -2.6463 |      -2.6169 |      -2.5741 |      -2.522  |      -2.4639 |      -2.4022 |      -2.3385 |      -2.2741 |      -2.2099 |      -2.1466 |      -2.0845 |      -2.0241 |      -1.9654 |      -1.9086 |      -1.8539 |      -1.8011 |      -1.7503 |
        |            305 |      -0      |      -0.0012 |      -0.0352 |      -0.1707 |      -0.4081 |      -0.6943 |      -0.9796 |      -1.2351 |      -1.4491 |      -1.6201 |      -1.7514 |      -1.8484 |      -1.9166 |      -1.9613 |      -1.9872 |      -1.9981 |      -1.9971 |      -1.9869 |      -1.9695 |      -1.9466 |      -1.9196 |      -1.8895 |      -1.8573 |      -1.8235 |      -1.7888 |      -1.7535 |      -1.718  |      -1.6825 |      -1.6473 |
        |            310 |      -0      |      -0      |      -0.0038 |      -0.0333 |      -0.1134 |      -0.2445 |      -0.4086 |      -0.5847 |      -0.7571 |      -0.9158 |      -1.0559 |      -1.1757 |      -1.2757 |      -1.3572 |      -1.4221 |      -1.4726 |      -1.5105 |      -1.5377 |      -1.5559 |      -1.5665 |      -1.5708 |      -1.5699 |      -1.5646 |      -1.5557 |      -1.5439 |      -1.5297 |      -1.5137 |      -1.4961 |      -1.4774 |
        |            315 |      -0      |      -0      |      -0.0003 |      -0.0052 |      -0.0265 |      -0.0744 |      -0.1499 |      -0.247  |      -0.3567 |      -0.471  |      -0.5833 |      -0.6895 |      -0.7868 |      -0.874  |      -0.9507 |      -1.0171 |      -1.0739 |      -1.1216 |      -1.1612 |      -1.1934 |      -1.2192 |      -1.2392 |      -1.2542 |      -1.2648 |      -1.2716 |      -1.2751 |      -1.2758 |      -1.274  |      -1.2701 |
        |            320 |      -0      |      -0      |      -0      |      -0.0007 |      -0.0053 |      -0.0197 |      -0.0488 |      -0.0938 |      -0.1528 |      -0.222  |      -0.2974 |      -0.3752 |      -0.4524 |      -0.5269 |      -0.5972 |      -0.6622 |      -0.7216 |      -0.7752 |      -0.8229 |      -0.8651 |      -0.902  |      -0.934  |      -0.9615 |      -0.9848 |      -1.0043 |      -1.0204 |      -1.0334 |      -1.0437 |      -1.0515 |
        |            325 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0009 |      -0.0046 |      -0.0142 |      -0.0323 |      -0.0599 |      -0.0965 |      -0.1407 |      -0.1905 |      -0.2439 |      -0.299  |      -0.3542 |      -0.4083 |      -0.4605 |      -0.5099 |      -0.5563 |      -0.5993 |      -0.6388 |      -0.6749 |      -0.7076 |      -0.7371 |      -0.7634 |      -0.7868 |      -0.8074 |      -0.8255 |      -0.8411 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0009 |      -0.0037 |      -0.0101 |      -0.0216 |      -0.0389 |      -0.0621 |      -0.0907 |      -0.1238 |      -0.1603 |      -0.1992 |      -0.2394 |      -0.28   |      -0.3204 |      -0.3599 |      -0.398  |      -0.4345 |      -0.4691 |      -0.5016 |      -0.532  |      -0.5602 |      -0.5863 |      -0.6102 |      -0.6321 |      -0.6521 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0009 |      -0.0029 |      -0.0072 |      -0.0146 |      -0.0257 |      -0.0406 |      -0.0594 |      -0.0815 |      -0.1066 |      -0.1339 |      -0.1629 |      -0.1929 |      -0.2235 |      -0.2542 |      -0.2847 |      -0.3144 |      -0.3434 |      -0.3712 |      -0.3979 |      -0.4233 |      -0.4473 |      -0.4699 |      -0.4911 |
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
        |            290 |       0.1106 |       3.9945 |      12.0759 |      20.0912 |      26.5537 |      31.4133 |      34.9663 |      37.5235 |      39.3371 |      40.5971 |      41.4434 |      41.9789 |      42.2794 |      42.401  |      42.3861 |      42.2665 |      42.0665 |      41.8047 |      41.4956 |      41.1505 |      40.7782 |      40.3856 |      39.9782 |      39.5604 |      39.1357 |      38.7069 |      38.2761 |      37.8451 |      37.4153 |
        |            295 |       0.0019 |       0.6705 |       4.2981 |      10.4165 |      17.257  |      23.7391 |      29.4357 |      34.2577 |      38.2579 |      41.538  |      44.2065 |      46.3638 |      48.0967 |      49.4781 |      50.5683 |      51.4171 |      52.065  |      52.5457 |      52.8867 |      53.1106 |      53.2363 |      53.2795 |      53.2531 |      53.168  |      53.0335 |      52.8572 |      52.6456 |      52.4043 |      52.1379 |
        |            300 |       0      |       0.0651 |       1.0306 |       3.9253 |       8.5308 |      14.0649 |      19.8507 |      25.4611 |      30.6704 |      35.3821 |      39.574  |      43.2638 |      46.4878 |      49.2902 |      51.7163 |      53.8092 |      55.6089 |      57.1509 |      58.4671 |      59.5853 |      60.5299 |      61.3222 |      61.9808 |      62.5217 |      62.9591 |      63.3053 |      63.571  |      63.7656 |      63.8974 |
        |            305 |       0      |       0.0039 |       0.1741 |       1.1199 |       3.3339 |       6.7804 |      11.1184 |      15.96   |      20.9861 |      25.9698 |      30.7646 |      35.2845 |      39.4853 |      43.3504 |      46.881  |      50.0885 |      52.9907 |      55.6081 |      57.9626 |      60.0756 |      61.9682 |      63.6599 |      65.1689 |      66.5122 |      67.7051 |      68.7616 |      69.6943 |      70.5148 |      71.2334 |
        |            310 |       0      |       0.0001 |       0.0214 |       0.2487 |       1.057  |       2.7253 |       5.2945 |       8.6312 |      12.5303 |      16.7846 |      21.2163 |      25.6861 |      30.0916 |      34.3614 |      38.4486 |      42.3243 |      45.9733 |      49.3898 |      52.5747 |      55.5336 |      58.2749 |      60.8086 |      63.1459 |      65.2986 |      67.2782 |      69.0961 |      70.7634 |      72.2905 |      73.6875 |
        |            315 |       0      |       0      |       0.002  |       0.0439 |       0.2771 |       0.9296 |       2.1797 |       4.0921 |       6.6303 |       9.6974 |      13.1721 |      16.9329 |      20.8705 |      24.8927 |      28.9257 |      32.912  |      36.8088 |      40.5848 |      44.2187 |      47.6962 |      51.0093 |      54.1542 |      57.1303 |      59.9396 |      62.586  |      65.0744 |      67.4106 |      69.601  |      71.6522 |
        |            320 |       0      |       0      |       0.0001 |       0.0063 |       0.061  |       0.2728 |       0.7859 |       1.7217 |       3.1456 |       5.0649 |       7.4429 |      10.2165 |      13.311  |      16.6504 |      20.1633 |      23.7862 |      27.4647 |      31.1537 |      34.8162 |      38.4232 |      41.9521 |      45.3855 |      48.7111 |      51.9199 |      55.0062 |      57.9667 |      60.8    |      63.5062 |      66.0866 |
        |            325 |       0      |       0      |       0      |       0.0007 |       0.0114 |       0.0697 |       0.2507 |       0.6491 |       1.3505 |       2.4127 |       3.8604 |       5.6884 |       7.8699 |      10.3641 |      13.1235 |      16.0981 |      19.2396 |      22.5032 |      25.8485 |      29.2401 |      32.6477 |      36.0456 |      39.4126 |      42.7311 |      45.9873 |      49.1699 |      52.2705 |      55.2827 |      58.2018 |
        |            330 |       0      |       0      |       0      |       0.0001 |       0.0018 |       0.0156 |       0.0714 |       0.2211 |       0.5287 |       1.0559 |       1.8507 |       2.9423 |       4.3412 |       6.0412 |       8.0242 |      10.2637 |      12.7281 |      15.3842 |      18.1983 |      21.1381 |      24.1736 |      27.2773 |      30.4246 |      33.5938 |      36.766  |      39.925  |      43.0569 |      46.1502 |      49.195  |
        |            335 |       0      |       0      |       0      |       0      |       0.0003 |       0.0031 |       0.0183 |       0.0685 |       0.1899 |       0.4271 |       0.8248 |       1.4218 |       2.2463 |       3.3148 |       4.6326 |       6.1951 |       7.9904 |      10.001  |      12.2059 |      14.5823 |      17.1064 |      19.7549 |      22.5052 |      25.3361 |      28.2279 |      31.1627 |      34.1243 |      37.0982 |      40.0718 |
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
        |            290 |       0.0015 |       0.1052 |       0.477  |       1.0574 |       1.7458 |       2.4768 |       3.2145 |       3.9399 |       4.6437 |       5.3216 |       5.9721 |       6.5951 |       7.1913 |       7.7619 |       8.3083 |       8.8316 |       9.3334 |       9.8148 |      10.2771 |      10.7214 |      11.1487 |      11.56   |      11.9561 |      12.338  |      12.7062 |      13.0616 |      13.4047 |      13.7362 |      14.0565 |
        |            295 |       0      |       0.0141 |       0.1354 |       0.4374 |       0.9054 |       1.4938 |       2.16   |       2.8715 |       3.6058 |       4.3478 |       5.0873 |       5.8177 |       6.5348 |       7.236  |       7.9198 |       8.5853 |       9.2322 |       9.8606 |      10.4708 |      11.0631 |      11.638  |      12.196  |      12.7377 |      13.2638 |      13.7746 |      14.2709 |      14.7532 |      15.222  |      15.6777 |
        |            300 |       0      |       0.0011 |       0.0271 |       0.1375 |       0.3734 |       0.7385 |       1.2155 |       1.781  |       2.4125 |       3.0911 |       3.8014 |       4.5318 |       5.2731 |       6.0186 |       6.763  |       7.5027 |       8.2349 |       8.9573 |       9.6687 |      10.3679 |      11.0544 |      11.7275 |      12.3871 |      13.0331 |      13.6655 |      14.2843 |      14.8898 |      15.4822 |      16.0616 |
        |            305 |       0      |       0.0001 |       0.0039 |       0.0337 |       0.1255 |       0.3061 |       0.5855 |       0.9601 |       1.4198 |       1.9514 |       2.542  |       3.1794 |       3.853  |       4.5539 |       5.2747 |       6.0092 |       6.7523 |       7.4999 |       8.2488 |       8.9963 |       9.7403 |      10.4789 |      11.2109 |      11.9352 |      12.651  |      13.3576 |      14.0545 |      14.7414 |      15.418  |
        |            310 |       0      |       0      |       0.0004 |       0.0066 |       0.035  |       0.1081 |       0.245  |       0.4564 |       0.7451 |       1.1087 |       1.5411 |       2.0347 |       2.5815 |       3.1736 |       3.8035 |       4.4647 |       5.1511 |       5.8576 |       6.5796 |       7.3134 |       8.0556 |       8.8034 |       9.5543 |      10.3064 |      11.0578 |      11.8072 |      12.5532 |      13.295  |      14.0315 |
        |            315 |       0      |       0      |       0      |       0.001  |       0.0082 |       0.033  |       0.0901 |       0.1933 |       0.3523 |       0.5724 |       0.855  |       1.1988 |       1.6002 |       2.0548 |       2.5576 |       3.1032 |       3.6864 |       4.3025 |       4.9468 |       5.6151 |       6.3036 |       7.0089 |       7.7281 |       8.4582 |       9.1971 |       9.9425 |      10.6926 |      11.4457 |      12.2004 |
        |            320 |       0      |       0      |       0      |       0.0001 |       0.0016 |       0.0088 |       0.0294 |       0.0736 |       0.1513 |       0.2706 |       0.4374 |       0.6548 |       0.9239 |       1.2443 |       1.6141 |       2.0305 |       2.4904 |       2.9903 |       3.5267 |       4.0958 |       4.6944 |       5.3191 |       5.9668 |       6.6347 |       7.3201 |       8.0207 |       8.734  |       9.4583 |      10.1915 |
        |            325 |       0      |       0      |       0      |       0      |       0.0003 |       0.002  |       0.0086 |       0.0254 |       0.0594 |       0.1179 |       0.2075 |       0.3335 |       0.4997 |       0.7085 |       0.961  |       1.2571 |       1.596  |       1.976  |       2.3953 |       2.8516 |       3.3423 |       3.865  |       4.4171 |       4.996  |       5.5995 |       6.225  |       6.8705 |       7.5338 |       8.213  |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0.0004 |       0.0023 |       0.008  |       0.0215 |       0.0476 |       0.0918 |       0.1591 |       0.2543 |       0.381  |       0.5421 |       0.7394 |       0.9741 |       1.2464 |       1.5559 |       1.902  |       2.2834 |       2.6987 |       3.1462 |       3.6242 |       4.1308 |       4.6641 |       5.2224 |       5.8036 |       6.4061 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0023 |       0.0072 |       0.0179 |       0.038  |       0.0714 |       0.1222 |       0.1942 |       0.2908 |       0.4147 |       0.5682 |       0.7529 |       0.9697 |       1.2192 |       1.5015 |       1.8161 |       2.1626 |       2.54   |       2.9472 |       3.3831 |       3.8463 |       4.3355 |       4.8493 |
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
        |            290 |      -0.0002 |      -0.0037 |      -0.0078 |      -0.0103 |      -0.0115 |      -0.0119 |      -0.0119 |      -0.0117 |      -0.0114 |      -0.0111 |      -0.0107 |      -0.0104 |      -0.01   |      -0.0097 |      -0.0094 |      -0.0091 |      -0.0088 |      -0.0085 |      -0.0083 |      -0.008  |      -0.0078 |      -0.0076 |      -0.0074 |      -0.0073 |      -0.0071 |      -0.0069 |      -0.0068 |      -0.0066 |      -0.0065 |
        |            295 |      -0      |      -0.0006 |      -0.0026 |      -0.0049 |      -0.0068 |      -0.0081 |      -0.0089 |      -0.0094 |      -0.0096 |      -0.0097 |      -0.0097 |      -0.0096 |      -0.0094 |      -0.0093 |      -0.0091 |      -0.0089 |      -0.0087 |      -0.0085 |      -0.0083 |      -0.0082 |      -0.008  |      -0.0078 |      -0.0076 |      -0.0075 |      -0.0073 |      -0.0072 |      -0.007  |      -0.0069 |      -0.0068 |
        |            300 |      -0      |      -0.0001 |      -0.0006 |      -0.0018 |      -0.0032 |      -0.0045 |      -0.0056 |      -0.0064 |      -0.007  |      -0.0075 |      -0.0078 |      -0.008  |      -0.0081 |      -0.0082 |      -0.0082 |      -0.0082 |      -0.0081 |      -0.008  |      -0.0079 |      -0.0078 |      -0.0077 |      -0.0076 |      -0.0075 |      -0.0074 |      -0.0073 |      -0.0072 |      -0.0071 |      -0.007  |      -0.0069 |
        |            305 |      -0      |      -0      |      -0.0001 |      -0.0005 |      -0.0012 |      -0.0021 |      -0.003  |      -0.0038 |      -0.0046 |      -0.0052 |      -0.0057 |      -0.0061 |      -0.0064 |      -0.0066 |      -0.0068 |      -0.0069 |      -0.007  |      -0.0071 |      -0.0071 |      -0.0071 |      -0.0071 |      -0.0071 |      -0.0071 |      -0.007  |      -0.007  |      -0.0069 |      -0.0068 |      -0.0068 |      -0.0067 |
        |            310 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0008 |      -0.0014 |      -0.002  |      -0.0026 |      -0.0032 |      -0.0037 |      -0.0042 |      -0.0046 |      -0.005  |      -0.0053 |      -0.0055 |      -0.0057 |      -0.0059 |      -0.006  |      -0.0061 |      -0.0062 |      -0.0063 |      -0.0063 |      -0.0063 |      -0.0063 |      -0.0063 |      -0.0063 |      -0.0063 |      -0.0063 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0006 |      -0.0009 |      -0.0013 |      -0.0018 |      -0.0022 |      -0.0027 |      -0.0031 |      -0.0035 |      -0.0038 |      -0.0041 |      -0.0044 |      -0.0046 |      -0.0048 |      -0.005  |      -0.0051 |      -0.0053 |      -0.0054 |      -0.0055 |      -0.0055 |      -0.0056 |      -0.0056 |      -0.0057 |      -0.0057 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0006 |      -0.0009 |      -0.0012 |      -0.0016 |      -0.0019 |      -0.0022 |      -0.0026 |      -0.0029 |      -0.0031 |      -0.0034 |      -0.0036 |      -0.0039 |      -0.0041 |      -0.0042 |      -0.0044 |      -0.0045 |      -0.0046 |      -0.0047 |      -0.0048 |      -0.0049 |      -0.005  |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0003 |      -0.0004 |      -0.0006 |      -0.0009 |      -0.0011 |      -0.0014 |      -0.0016 |      -0.0019 |      -0.0021 |      -0.0024 |      -0.0026 |      -0.0028 |      -0.0031 |      -0.0032 |      -0.0034 |      -0.0036 |      -0.0037 |      -0.0039 |      -0.004  |      -0.0041 |      -0.0042 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0004 |      -0.0006 |      -0.0008 |      -0.001  |      -0.0012 |      -0.0014 |      -0.0016 |      -0.0018 |      -0.002  |      -0.0022 |      -0.0024 |      -0.0026 |      -0.0027 |      -0.0029 |      -0.0031 |      -0.0032 |      -0.0033 |      -0.0034 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0002 |      -0.0003 |      -0.0004 |      -0.0006 |      -0.0007 |      -0.0009 |      -0.001  |      -0.0012 |      -0.0014 |      -0.0015 |      -0.0017 |      -0.0019 |      -0.002  |      -0.0022 |      -0.0023 |      -0.0025 |      -0.0026 |      -0.0027 |
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
        |            290 |            0 |       0.0011 |       0.0034 |       0.0056 |       0.0074 |       0.0088 |       0.0098 |       0.0106 |       0.0111 |       0.0115 |       0.0118 |       0.012  |       0.0121 |       0.0122 |       0.0122 |       0.0122 |       0.0122 |       0.0122 |       0.0121 |       0.0121 |       0.012  |       0.0119 |       0.0119 |       0.0118 |       0.0117 |       0.0116 |       0.0115 |       0.0114 |       0.0113 |
        |            295 |            0 |       0.0001 |       0.0007 |       0.0018 |       0.003  |       0.0041 |       0.0051 |       0.0059 |       0.0066 |       0.0072 |       0.0077 |       0.0081 |       0.0084 |       0.0087 |       0.0089 |       0.0091 |       0.0092 |       0.0093 |       0.0094 |       0.0095 |       0.0095 |       0.0095 |       0.0096 |       0.0096 |       0.0096 |       0.0096 |       0.0095 |       0.0095 |       0.0095 |
        |            300 |            0 |       0      |       0.0001 |       0.0005 |       0.001  |       0.0016 |       0.0023 |       0.0029 |       0.0036 |       0.0041 |       0.0046 |       0.0051 |       0.0054 |       0.0058 |       0.0061 |       0.0063 |       0.0066 |       0.0068 |       0.0069 |       0.0071 |       0.0072 |       0.0073 |       0.0074 |       0.0075 |       0.0076 |       0.0076 |       0.0077 |       0.0077 |       0.0077 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0006 |       0.0009 |       0.0013 |       0.0017 |       0.0022 |       0.0026 |       0.0029 |       0.0033 |       0.0036 |       0.0039 |       0.0042 |       0.0044 |       0.0047 |       0.0049 |       0.0051 |       0.0052 |       0.0054 |       0.0055 |       0.0057 |       0.0058 |       0.0059 |       0.006  |       0.006  |       0.0061 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0008 |       0.001  |       0.0013 |       0.0016 |       0.0019 |       0.0021 |       0.0024 |       0.0026 |       0.0029 |       0.0031 |       0.0033 |       0.0035 |       0.0037 |       0.0038 |       0.004  |       0.0041 |       0.0043 |       0.0044 |       0.0045 |       0.0046 |       0.0047 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0006 |       0.0008 |       0.001  |       0.0012 |       0.0014 |       0.0016 |       0.0018 |       0.002  |       0.0021 |       0.0023 |       0.0025 |       0.0026 |       0.0028 |       0.0029 |       0.003  |       0.0032 |       0.0033 |       0.0034 |       0.0035 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0008 |       0.0009 |       0.001  |       0.0012 |       0.0013 |       0.0015 |       0.0016 |       0.0017 |       0.0019 |       0.002  |       0.0021 |       0.0022 |       0.0023 |       0.0025 |       0.0026 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |       0.0014 |       0.0015 |       0.0016 |       0.0017 |       0.0018 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |       0.001  |       0.0011 |       0.0012 |       0.0013 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0004 |       0.0004 |       0.0005 |       0.0005 |       0.0006 |       0.0007 |       0.0007 |       0.0008 |       0.0009 |
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

        By default the stock price, risk free rate, dividend yield and volatility of the most recent date are used;
        changing the start date prices the option as of that date instead. The volatility is the annualized standard
        deviation of the daily returns over the year before that date and the dividend yield the dividends of that
        year relative to the price, so an earlier start date only uses what was known on that date.

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

        |   Strike Price |   (Period('2026-01-29', 'D'), 'Speed') |   (Period('2026-01-29', 'D'), 'Zomma') |   (Period('2026-01-29', 'D'), 'Color') |
        |---------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|
        |            215 |                                -0.0001 |                                 0.0048 |                                -0.018  |
        |            220 |                                -0.0001 |                                 0.0044 |                                -0.0166 |
        |            225 |                                -0.0001 |                                 0.0038 |                                -0.0139 |
        |            230 |                                -0.0001 |                                 0.0028 |                                -0.0099 |
        |            235 |                                -0.0001 |                                 0.0015 |                                -0.0047 |
        |            240 |                                -0.0001 |                                -0.0001 |                                 0.0015 |
        |            245 |                                -0.0001 |                                -0.0018 |                                 0.0084 |
        |            250 |                                -0.0001 |                                -0.0037 |                                 0.0157 |
        |            255 |                                -0.0001 |                                -0.0056 |                                 0.023  |
        |            260 |                                -0.0001 |                                -0.0074 |                                 0.0301 |
        |            265 |                                -0.0001 |                                -0.009  |                                 0.0364 |
        |            270 |                                -0.0001 |                                -0.0104 |                                 0.0418 |
        |            275 |                                -0.0001 |                                -0.0115 |                                 0.046  |
        |            280 |                                -0.0001 |                                -0.0123 |                                 0.0488 |
        |            285 |                                -0      |                                -0.0127 |                                 0.0501 |
        |            290 |                                -0      |                                -0.0127 |                                 0.05   |
        |            295 |                                -0      |                                -0.0123 |                                 0.0485 |
        |            300 |                                 0      |                                -0.0117 |                                 0.0456 |
        |            305 |                                 0      |                                -0.0107 |                                 0.0417 |
        |            310 |                                 0      |                                -0.0095 |                                 0.0368 |
        |            315 |                                 0      |                                -0.0081 |                                 0.0313 |
        |            320 |                                 0.0001 |                                -0.0066 |                                 0.0252 |
        |            325 |                                 0.0001 |                                -0.0051 |                                 0.019  |
        |            330 |                                 0.0001 |                                -0.0035 |                                 0.0127 |
        |            335 |                                 0.0001 |                                -0.0019 |                                 0.0065 |
        |            340 |                                 0.0001 |                                -0.0005 |                                 0.0007 |
        |            345 |                                 0.0001 |                                 0.0009 |                                -0.0047 |
        |            350 |                                 0.0001 |                                 0.0021 |                                -0.0095 |
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
        |            290 |            0 |       0.0005 |       0.0011 |       0.0013 |       0.0014 |       0.0014 |       0.0013 |       0.0012 |       0.0011 |       0.001  |       0.001  |       0.0009 |       0.0008 |       0.0008 |       0.0007 |       0.0007 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            295 |            0 |       0.0001 |       0.0003 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |       0.0009 |       0.0009 |       0.0008 |       0.0008 |       0.0008 |       0.0007 |       0.0007 |       0.0007 |       0.0006 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |
        |            300 |            0 |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0006 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |
        |            305 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0005 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |
        |            310 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0004 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |       0.0003 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |       0.0002 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0002 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |       0.0001 |
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
        |            290 |       0.0016 |       0.0267 |       0.0498 |       0.0572 |       0.0551 |       0.0491 |       0.0418 |       0.0344 |       0.0276 |       0.0215 |       0.016  |       0.0112 |       0.007  |       0.0033 |       0.0001 |      -0.0028 |      -0.0053 |      -0.0075 |      -0.0094 |      -0.0111 |      -0.0127 |      -0.014  |      -0.0152 |      -0.0163 |      -0.0173 |      -0.0181 |      -0.0189 |      -0.0196 |      -0.0202 |
        |            295 |       0      |       0.0047 |       0.0193 |       0.0334 |       0.0421 |       0.0457 |       0.0459 |       0.044  |       0.0409 |       0.0373 |       0.0335 |       0.0296 |       0.026  |       0.0225 |       0.0192 |       0.0161 |       0.0133 |       0.0107 |       0.0083 |       0.0061 |       0.0041 |       0.0023 |       0.0006 |      -0.001  |      -0.0025 |      -0.0038 |      -0.005  |      -0.0061 |      -0.0072 |
        |            300 |       0      |       0.0005 |       0.0048 |       0.0133 |       0.0224 |       0.0298 |       0.0348 |       0.0376 |       0.0388 |       0.0387 |       0.0377 |       0.0362 |       0.0342 |       0.0321 |       0.0299 |       0.0276 |       0.0253 |       0.0231 |       0.021  |       0.0189 |       0.0169 |       0.0151 |       0.0133 |       0.0116 |       0.01   |       0.0085 |       0.0071 |       0.0058 |       0.0045 |
        |            305 |       0      |       0      |       0.0008 |       0.0039 |       0.0091 |       0.0151 |       0.0208 |       0.0254 |       0.0289 |       0.0314 |       0.0329 |       0.0336 |       0.0337 |       0.0333 |       0.0326 |       0.0316 |       0.0304 |       0.0291 |       0.0277 |       0.0262 |       0.0248 |       0.0233 |       0.0218 |       0.0204 |       0.019  |       0.0176 |       0.0163 |       0.015  |       0.0138 |
        |            310 |       0      |       0      |       0.0001 |       0.0009 |       0.003  |       0.0063 |       0.0103 |       0.0144 |       0.0182 |       0.0215 |       0.0243 |       0.0264 |       0.0279 |       0.029  |       0.0296 |       0.0299 |       0.0298 |       0.0296 |       0.0291 |       0.0285 |       0.0277 |       0.0269 |       0.026  |       0.025  |       0.0241 |       0.023  |       0.022  |       0.021  |       0.02   |
        |            315 |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0022 |       0.0043 |       0.007  |       0.01   |       0.0129 |       0.0157 |       0.0182 |       0.0204 |       0.0223 |       0.0238 |       0.0249 |       0.0258 |       0.0264 |       0.0268 |       0.0269 |       0.0269 |       0.0268 |       0.0265 |       0.0261 |       0.0257 |       0.0251 |       0.0245 |       0.0239 |       0.0232 |
        |            320 |       0      |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0016 |       0.003  |       0.0048 |       0.0069 |       0.0092 |       0.0114 |       0.0135 |       0.0155 |       0.0173 |       0.0189 |       0.0202 |       0.0214 |       0.0224 |       0.0231 |       0.0237 |       0.0241 |       0.0244 |       0.0246 |       0.0246 |       0.0245 |       0.0244 |       0.0242 |       0.0239 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |       0.0012 |       0.0021 |       0.0034 |       0.0049 |       0.0065 |       0.0082 |       0.0099 |       0.0116 |       0.0132 |       0.0147 |       0.0161 |       0.0173 |       0.0184 |       0.0193 |       0.0201 |       0.0208 |       0.0214 |       0.0218 |       0.0222 |       0.0224 |       0.0226 |       0.0227 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0008 |       0.0015 |       0.0024 |       0.0034 |       0.0046 |       0.0059 |       0.0073 |       0.0086 |       0.01   |       0.0113 |       0.0126 |       0.0137 |       0.0148 |       0.0158 |       0.0167 |       0.0175 |       0.0182 |       0.0188 |       0.0194 |       0.0198 |       0.0202 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0006 |       0.0011 |       0.0017 |       0.0024 |       0.0033 |       0.0043 |       0.0053 |       0.0064 |       0.0075 |       0.0086 |       0.0097 |       0.0108 |       0.0118 |       0.0127 |       0.0136 |       0.0144 |       0.0152 |       0.0159 |       0.0165 |       0.0171 |
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
        |            290 |      -0.0941 |      -0.7948 |      -0.9931 |      -0.859  |      -0.6666 |      -0.4981 |      -0.3665 |      -0.2674 |      -0.1933 |      -0.1379 |      -0.0962 |      -0.0647 |      -0.0406 |      -0.0221 |      -0.0079 |       0.0031 |       0.0117 |       0.0183 |       0.0234 |       0.0274 |       0.0305 |       0.0328 |       0.0345 |       0.0358 |       0.0367 |       0.0373 |       0.0377 |       0.0378 |       0.0378 |
        |            295 |      -0.0017 |      -0.1403 |      -0.3831 |      -0.4994 |      -0.5052 |      -0.4592 |      -0.3969 |      -0.3345 |      -0.2782 |      -0.2296 |      -0.1886 |      -0.1545 |      -0.1261 |      -0.1025 |      -0.0829 |      -0.0665 |      -0.0529 |      -0.0415 |      -0.0319 |      -0.0238 |      -0.017  |      -0.0113 |      -0.0064 |      -0.0022 |       0.0013 |       0.0043 |       0.0068 |       0.009  |       0.0108 |
        |            300 |      -0      |      -0.014  |      -0.0956 |      -0.1989 |      -0.2685 |      -0.2979 |      -0.299  |      -0.2839 |      -0.2609 |      -0.2351 |      -0.2092 |      -0.1847 |      -0.1622 |      -0.142  |      -0.124  |      -0.108  |      -0.094  |      -0.0817 |      -0.0708 |      -0.0613 |      -0.0529 |      -0.0455 |      -0.039  |      -0.0333 |      -0.0282 |      -0.0238 |      -0.0198 |      -0.0163 |      -0.0132 |
        |            305 |      -0      |      -0.0008 |      -0.0165 |      -0.0586 |      -0.1092 |      -0.151  |      -0.1779 |      -0.191  |      -0.1939 |      -0.1897 |      -0.1811 |      -0.1701 |      -0.158  |      -0.1455 |      -0.1333 |      -0.1216 |      -0.1106 |      -0.1003 |      -0.0909 |      -0.0822 |      -0.0743 |      -0.0671 |      -0.0605 |      -0.0545 |      -0.0491 |      -0.0441 |      -0.0396 |      -0.0356 |      -0.0319 |
        |            310 |      -0      |      -0      |      -0.0021 |      -0.0133 |      -0.0355 |      -0.0626 |      -0.0879 |      -0.1078 |      -0.1216 |      -0.1296 |      -0.133  |      -0.1329 |      -0.1302 |      -0.1258 |      -0.1203 |      -0.114  |      -0.1075 |      -0.1009 |      -0.0944 |      -0.088  |      -0.0819 |      -0.0761 |      -0.0706 |      -0.0654 |      -0.0606 |      -0.056  |      -0.0518 |      -0.0479 |      -0.0442 |
        |            315 |      -0      |      -0      |      -0.0002 |      -0.0024 |      -0.0095 |      -0.0218 |      -0.037  |      -0.0526 |      -0.0664 |      -0.0777 |      -0.086  |      -0.0917 |      -0.0949 |      -0.0963 |      -0.096  |      -0.0946 |      -0.0924 |      -0.0895 |      -0.0861 |      -0.0825 |      -0.0788 |      -0.075  |      -0.0712 |      -0.0674 |      -0.0637 |      -0.0602 |      -0.0568 |      -0.0535 |      -0.0504 |
        |            320 |      -0      |      -0      |      -0      |      -0.0003 |      -0.0021 |      -0.0065 |      -0.0136 |      -0.0226 |      -0.0322 |      -0.0416 |      -0.05   |      -0.057  |      -0.0626 |      -0.0668 |      -0.0697 |      -0.0714 |      -0.0722 |      -0.0722 |      -0.0716 |      -0.0705 |      -0.0689 |      -0.0671 |      -0.0651 |      -0.0629 |      -0.0606 |      -0.0583 |      -0.0559 |      -0.0536 |      -0.0513 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0.0004 |      -0.0017 |      -0.0044 |      -0.0086 |      -0.0141 |      -0.0202 |      -0.0264 |      -0.0325 |      -0.0379 |      -0.0427 |      -0.0467 |      -0.0499 |      -0.0523 |      -0.0541 |      -0.0552 |      -0.0558 |      -0.056  |      -0.0558 |      -0.0553 |      -0.0545 |      -0.0535 |      -0.0523 |      -0.0511 |      -0.0497 |      -0.0483 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0013 |      -0.003  |      -0.0056 |      -0.0089 |      -0.0129 |      -0.0171 |      -0.0213 |      -0.0254 |      -0.0291 |      -0.0325 |      -0.0355 |      -0.0379 |      -0.04   |      -0.0416 |      -0.0428 |      -0.0437 |      -0.0442 |      -0.0445 |      -0.0445 |      -0.0443 |      -0.0439 |      -0.0434 |      -0.0428 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0009 |      -0.002  |      -0.0037 |      -0.0058 |      -0.0084 |      -0.0112 |      -0.0141 |      -0.0171 |      -0.02   |      -0.0227 |      -0.0252 |      -0.0274 |      -0.0293 |      -0.031  |      -0.0324 |      -0.0336 |      -0.0345 |      -0.0352 |      -0.0356 |      -0.0359 |      -0.0361 |      -0.0361 |
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
        |            290 |       4.333  |      59.4712 |      82.0031 |      55.0813 |       8.2795 |     -41.1001 |     -86.2254 |    -125.122  |    -157.756  |    -184.773  |    -206.985  |    -225.172  |    -240.019  |    -252.101  |    -261.897  |    -269.798  |    -276.125  |    -281.138  |    -285.053  |    -288.047  |    -290.266  |    -291.832  |    -292.844  |    -293.387  |    -293.533  |    -293.34   |    -292.858  |    -292.131  |    -291.195  |
        |            295 |       0.1305 |      19.2791 |      68.8654 |     100.582  |     100.716  |      78.0883 |      43.2729 |       3.6086 |     -36.5881 |     -75.0146 |    -110.572  |    -142.844  |    -171.785  |    -197.541  |    -220.35   |    -240.48   |    -258.203  |    -273.781  |    -287.451  |    -299.43   |    -309.912  |    -319.068  |    -327.048  |    -333.988  |    -340.003  |    -345.197  |    -349.66   |    -353.473  |    -356.706  |
        |            300 |       0.0013 |       2.9589 |      27.9645 |      70.5814 |     106.543  |     124.16   |     123.314  |     108.216  |      83.553  |      53.1876 |      19.9486 |     -14.221  |     -48.0467 |     -80.7293 |    -111.796  |    -140.996  |    -168.221  |    -193.459  |    -216.755  |    -238.191  |    -257.869  |    -275.9    |    -292.399  |    -307.477  |    -321.243  |    -333.801  |    -345.247  |    -355.671  |    -365.155  |
        |            305 |       0      |       0.2498 |       6.9678 |      30.9594 |      67.3902 |     103.46   |     130.3    |     144.674  |     146.902  |     138.886  |     122.942  |     101.254  |      75.6642 |      47.6403 |      18.3071 |     -11.4963 |     -41.1591 |     -70.2468 |     -98.46   |    -125.6    |    -151.543  |    -176.222  |    -199.607  |    -221.7    |    -242.521  |    -262.106  |    -280.499  |    -297.75   |    -313.912  |
        |            310 |       0      |       0.0125 |       1.1673 |       9.5967 |      30.6046 |      61.4209 |      95.0555 |     125.284  |     148.372  |     162.837  |     168.689  |     166.753  |     158.203  |     144.286  |     126.179  |     104.923  |      81.401  |      56.3454 |      30.3484 |       3.8826 |     -22.6802 |     -49.0519 |     -75.0126 |    -100.397  |    -125.086  |    -148.992  |    -172.059  |    -194.25   |    -215.546  |
        |            315 |       0      |       0.0004 |       0.1384 |       2.2283 |      10.712  |      28.4653 |      54.2284 |      84.1758 |     114.17   |     140.987  |     162.613  |     178.088  |     187.218  |     190.294  |     187.883  |     180.678  |     169.406  |     154.768  |     137.41   |     117.91   |      96.771  |      74.4262 |      51.2422 |      27.5266 |       3.5347 |     -20.5232 |     -44.4757 |     -68.1843 |     -91.5383 |
        |            320 |       0      |       0      |       0.012  |       0.4016 |       3.0045 |      10.768  |      25.5093 |      46.8292 |      72.6907 |     100.465  |     127.701  |     152.488  |     173.539  |     190.123  |     201.946  |     209.029  |     211.601  |     210.021  |     204.71   |     196.117  |     184.686  |     170.842  |     154.978  |     137.453  |     118.588  |      98.6673 |      77.9409 |      56.6258 |      34.9098 |
        |            325 |       0      |       0      |       0.0008 |       0.0576 |       0.6928 |       3.4168 |      10.1954 |      22.3132 |      39.8171 |      61.6926 |      86.3458 |     112.053  |     137.254  |     160.694  |     181.461  |     198.96   |     212.866  |     223.063  |     229.591  |     232.601  |     232.314  |     228.995  |     222.931  |     214.415  |     203.733  |     191.163  |     176.962  |     161.372  |     144.613  |
        |            330 |       0      |       0      |       0      |       0.0067 |       0.1338 |       0.9268 |       3.5314 |       9.3008 |      19.2021 |      33.4916 |      51.7358 |      73.024  |      96.2174 |     120.151  |     143.766  |     166.178  |     186.7    |     204.837  |     220.267  |     232.809  |     242.4    |     249.065  |     252.893  |     254.022  |     252.62   |     248.875  |     242.981  |     235.138  |     225.541  |
        |            335 |       0      |       0      |       0      |       0.0006 |       0.022  |       0.2181 |       1.0755 |       3.4424 |       8.2812 |      16.343  |      27.9646 |      43.0355 |      61.0818 |      81.3964 |     103.167  |     125.576  |     147.867  |     169.389  |     189.605  |     208.1    |     224.572  |     238.816  |     250.714  |     260.216  |     267.33   |     272.105  |     274.625  |     274.995  |     273.339  |
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
