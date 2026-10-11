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
from financetoolkit.risk import risk_model
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

        yearly_volatility = risk_model.get_volatility(
            self._daily_historical["Return"], "yearly"
        )
        year_labels = to_period_index(self._daily_historical.index).asfreq("Y")
        self._volatility = yearly_volatility.reindex(year_labels)
        self._volatility.index = self._daily_historical.index

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

    def get_option_chains(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        show_expiration_dates: bool = False,
        rounding: int | None = None,
    ):
        """
        Get the Option Chains which gives information about the currently available
        options as reported by Yahoo Finance. This returns the Contract Symbol, Strike
        Currency, Last Price, Absolute Change, Percent Change, Volume, Open Interest,
        Bid Pirce, Ask Price, Expiration, Last Trade Date, Implied Volatility and
        whether the option is In The Money.

        The data comes from Yahoo Finance. When Yahoo Finance has no options for a ticker,
        Cboe's delayed quotes (15 minutes) are used, which cover every option listed on a US
        equity or index, such as "^SPX" with expiries several years out. If neither has
        data, it is advised to use the theoretical calculations as provided by the
        Black Scholes Model as well as the Greeks to get a better understanding of the
        option prices over time.

        Also known as: calls, puts, strike prices, expiry dates, option data.

        Args:
            expiration_date (str | None, optional): The expiration date to use. Defaults to None which means it will
            put_option (bool, optional): Whether to show the put options instead of the call options.
                Defaults to False.
            show_expiration_dates (bool, optional): Whether to return the available expiration dates instead of
                the option chains. Defaults to False.
            use the first available expiration date.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: the option chains containing the tickers and strike prices as
            the index and the time to expiration as the columns.

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

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

        The formulas are as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Option Price = S * e^(—q * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            175 |      55.8399 |      55.8599 |      55.8798 |      55.8998 |      55.9197 |      55.9396 |      55.9596 |      55.9795 |      55.9994 |
        |            180 |      50.8405 |      50.861  |      50.8815 |      50.902  |      50.9225 |      50.943  |      50.9635 |      50.984  |      51.0045 |
        |            185 |      45.8411 |      45.8622 |      45.8832 |      45.9043 |      45.9254 |      45.9465 |      45.9675 |      45.9886 |      46.0097 |
        |            190 |      40.8417 |      40.8633 |      40.885  |      40.9066 |      40.9282 |      40.9499 |      40.9716 |      40.9933 |      41.0152 |
        |            195 |      35.8422 |      35.8644 |      35.8867 |      35.9089 |      35.9311 |      35.9534 |      35.976  |      35.999  |      36.0226 |
        |            200 |      30.8428 |      30.8656 |      30.8884 |      30.9112 |      30.9343 |      30.9581 |      30.9831 |      31.0098 |      31.0386 |
        |            205 |      25.8434 |      25.8667 |      25.8902 |      25.9144 |      25.9407 |      25.9702 |      26.0039 |      26.0422 |      26.0853 |
        |            210 |      20.8439 |      20.868  |      20.8941 |      20.926  |      20.9667 |      21.017  |      21.0765 |      21.1442 |      21.2191 |
        |            215 |      15.8445 |      15.8734 |      15.92   |      15.9892 |      16.0782 |      16.1824 |      16.2979 |      16.4216 |      16.5511 |
        |            220 |      10.8498 |      10.9347 |      11.079  |      11.2556 |      11.4481 |      11.6476 |      11.8495 |      12.0512 |      12.2513 |
        |            225 |       5.9884 |       6.3445 |       6.708  |       7.0532 |       7.3783 |       7.6851 |       7.9758 |       8.2527 |       8.5173 |
        |            230 |       2.1132 |       2.8035 |       3.3391 |       3.7938 |       4.1963 |       4.5617 |       4.8989 |       5.2137 |       5.5102 |
        |            235 |       0.3563 |       0.8541 |       1.2982 |       1.6971 |       2.0612 |       2.3983 |       2.7135 |       3.0107 |       3.2927 |
        |            240 |       0.0233 |       0.1671 |       0.38   |       0.6172 |       0.8611 |       1.1044 |       1.344  |       1.5787 |       1.8078 |
        |            245 |       0.0005 |       0.0202 |       0.082  |       0.1801 |       0.3031 |       0.4424 |       0.5922 |       0.7486 |       0.9091 |
        |            250 |       0      |       0.0015 |       0.013  |       0.0419 |       0.0896 |       0.1537 |       0.2316 |       0.3205 |       0.418  |
        |            255 |       0      |       0.0001 |       0.0015 |       0.0078 |       0.0222 |       0.0463 |       0.0804 |       0.1238 |       0.1758 |
        |            260 |       0      |       0      |       0.0001 |       0.0012 |       0.0046 |       0.0121 |       0.0248 |       0.0433 |       0.0677 |
        |            265 |       0      |       0      |       0      |       0.0001 |       0.0008 |       0.0028 |       0.0068 |       0.0137 |       0.0239 |
        |            270 |       0      |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0017 |       0.0039 |       0.0078 |
        |            275 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.001  |       0.0023 |
        |            280 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0006 |
        |            285 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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

        Also known as: IV, option-implied volatility.

        Args:
            expiration_date (str | None, optional): The expiration date to use for the calculation. Defaults to None
            which means it will use the most recent expiration date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
            show_expiration_dates (bool, optional): Whether to show the expiration dates. Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.

        Returns:
            pd.Series | list[str]: Implied Volatility values containing the tickers as the index and the expiration
            dates as the columns. If show_expiration_dates is True, it will return a list of expiration dates.

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
                else self._dividend_yield[ticker].iloc[-1]
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
                calculation. Defaults to None which means it will use the dividend
                yield as obtained through annual historical data.
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
                    else self._dividend_yield[ticker].iloc[-1]
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
                calculation. Defaults to None which means it will use the dividend
                yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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
            american_option (bool, optional): Whether to value an American option, which can be exercised at every
                node, instead of a European option. Defaults to False.
            it will calculate the call option price.
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
        | UUUUUUUUUU |      49.8119 |      62.0866 |      75.8572 |      90.9918 |     107.405  |     125.099  |     144.156  |     164.677  |     186.771  |     210.556  |     236.159  |
        | DUUUUUUUUU |     nan      |      37.5692 |      48.4029 |      60.8806 |      74.8219 |      90.0481 |     106.484  |     124.188  |     143.255  |     163.787  |     185.894  |
        | DDUUUUUUUU |     nan      |     nan      |      26.7114 |      35.9396 |      47.0111 |      59.7425 |      73.8464 |      89.1104 |     105.555  |     123.269  |     142.346  |
        | DDDUUUUUUU |     nan      |     nan      |     nan      |      17.4193 |      24.8233 |      34.2749 |      45.6962 |      58.7199 |      72.8924 |      88.164  |     104.617  |
        | DDDDUUUUUU |     nan      |     nan      |     nan      |     nan      |       9.9312 |      15.2854 |      22.7845 |      32.6478 |      44.5944 |      57.7503 |      71.9295 |
        | DDDDDUUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |       4.4941 |       7.6818 |      12.8055 |      20.6027 |      31.4008 |      43.61   |
        | DDDDDDUUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       1.2456 |       2.464  |       4.8743 |       9.6424 |      19.0748 |
        | DDDDDDDUUU |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |       0      |       0      |
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
                    else self._dividend_yield[ticker].iloc[-1]
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
        on the stock price, volatility, risk free rate, dividend yield and time to expiration. The stock price is then
        simulated based on the up and down movements.

        By default the most recent risk free rate and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

        The formulas are as follows:

        - up movement (u) = e^(σ * sqrt(t))
        - down movement (d) = 1 / u
        - stock price at each node = S * u^j * d^(n — j)

        Where S is the stock price, r is the risk free rate, σ is the volatility, t is the time to
        expiration, j is the number of up movements, n is the number of time steps.

        The resulting output is a DataFrame containing the tickers and movements as the index and the
        time to expiration as the columns. The movements index contains the number of up movements and the number of
        down movements. The output is the binomial tree displayed in a table. E.g. when using 10 time steps, the
        table from each company will contain the actual binomial tree's stock prices as also depicted
        in the image found here: https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method

        **Hint:** consider plotting the resulting DataFrame for each company to visualize the binomial tree.
        For example for below's example use `stock_price_simulation.loc['AMZN'].T.plot(legend=False)`

        Also known as: Monte Carlo simulation, GBM, stock price path.

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            time_to_expiration (int): The number of year to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None which
            means it will use the current risk free rate.
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
        | UUUU       |       209.69 |      249.064 |      295.832 |      351.382 |      417.362 |
        | UUUD       |       209.69 |      249.064 |      295.832 |      351.382 |      295.832 |
        | UUDU       |       209.69 |      249.064 |      295.832 |      249.064 |      295.832 |
        | UUDD       |       209.69 |      249.064 |      295.832 |      249.064 |      209.69  |
        | UDUU       |       209.69 |      249.064 |      209.69  |      249.064 |      295.832 |
        | UDUD       |       209.69 |      249.064 |      209.69  |      249.064 |      209.69  |
        | UDDU       |       209.69 |      249.064 |      209.69  |      176.54  |      209.69  |
        | UDDD       |       209.69 |      249.064 |      209.69  |      176.54  |      148.631 |
        | DUUU       |       209.69 |      176.54  |      209.69  |      249.064 |      295.832 |
        | DUUD       |       209.69 |      176.54  |      209.69  |      249.064 |      209.69  |
        | DUDU       |       209.69 |      176.54  |      209.69  |      176.54  |      209.69  |
        | DUDD       |       209.69 |      176.54  |      209.69  |      176.54  |      148.631 |
        | DDUU       |       209.69 |      176.54  |      148.631 |      176.54  |      209.69  |
        | DDUD       |       209.69 |      176.54  |      148.631 |      176.54  |      148.631 |
        | DDDU       |       209.69 |      176.54  |      148.631 |      125.134 |      148.631 |
        | DDDD       |       209.69 |      176.54  |      148.631 |      125.134 |      105.352 |
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
        the standard error roughly 95% of the time. A fixed ``seed`` is used by default
        for reproducibility of documentation examples; set it explicitly (or leave it as
        None) to control this behavior.

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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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
                else self._dividend_yield[ticker].iloc[-1]
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
        - Partial Derivative: measures the rate of change in the option price with respect to the strike price.

        **Third Order Greeks:**

        - Speed: measures the rate of change in Gamma with respect to changes in the underlying price.
        - Zomma: measures the rate of change of Gamma with respect to changes in volatility.
        - Color: also referred to as gamma decay or DgammaDtime measures the rate of change of gamma over
        the passage of time.
        - Ultima: measures the sensitivity of the option vomma with respect to change in volatility.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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
            means it will use the dividend yield as obtained through annual historical data.
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

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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
            means it will use the dividend yield as obtained through annual historical data.
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
                means it will use the current dividend yield.
            means it will use the current risk free rate.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0027 |       0.0119 |       0.0256 |       0.0412 |       0.0571 |       0.0726 |       0.0873 |       0.1011 |
        |            295 |            0 |       0.0002 |       0.0023 |       0.0071 |       0.0144 |       0.0233 |       0.0331 |       0.0434 |       0.0537 |
        |            300 |            0 |       0      |       0.0003 |       0.0016 |       0.0043 |       0.0084 |       0.0135 |       0.0196 |       0.0262 |
        |            305 |            0 |       0      |       0      |       0.0003 |       0.0011 |       0.0026 |       0.005  |       0.0081 |       0.0118 |
        |            310 |            0 |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0016 |       0.003  |       0.0049 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |       0.001  |       0.0019 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |       0.0007 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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
        the actual probability of an option finishing in the money which is the first derivative
        of option price with respect to strike.

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

        Also known as: cash delta, binary option delta.

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
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |           -0 |      -0.0025 |      -0.011  |      -0.0237 |      -0.0379 |      -0.0524 |      -0.0665 |      -0.0798 |      -0.0923 |
        |            295 |           -0 |      -0.0002 |      -0.0021 |      -0.0065 |      -0.0131 |      -0.0211 |      -0.0299 |      -0.0391 |      -0.0483 |
        |            300 |           -0 |      -0      |      -0.0003 |      -0.0014 |      -0.0038 |      -0.0074 |      -0.0121 |      -0.0174 |      -0.0232 |
        |            305 |           -0 |      -0      |      -0      |      -0.0003 |      -0.001  |      -0.0023 |      -0.0044 |      -0.0071 |      -0.0103 |
        |            310 |           -0 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0014 |      -0.0026 |      -0.0042 |
        |            315 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0004 |      -0.0009 |      -0.0016 |
        |            320 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0006 |
        |            325 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |
        |            330 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |
        |            335 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0017 |       0.0076 |       0.0169 |       0.028  |       0.0398 |       0.0518 |       0.0637 |       0.0753 |
        |            295 |            0 |       0.0002 |       0.0017 |       0.0056 |       0.0116 |       0.0191 |       0.0277 |       0.0369 |       0.0465 |
        |            300 |            0 |       0      |       0.0003 |       0.0015 |       0.004  |       0.0079 |       0.013  |       0.0191 |       0.0259 |
        |            305 |            0 |       0      |       0      |       0.0003 |       0.0012 |       0.0028 |       0.0054 |       0.0089 |       0.0131 |
        |            310 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0009 |       0.002  |       0.0037 |       0.0061 |
        |            315 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0007 |       0.0014 |       0.0026 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0005 |       0.001  |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0004 |      -0.0138 |      -0.0417 |      -0.0695 |      -0.0922 |      -0.1094 |      -0.1223 |      -0.1318 |      -0.1388 |
        |            295 |      -0      |      -0.0015 |      -0.0095 |      -0.023  |      -0.0381 |      -0.0525 |      -0.0653 |      -0.0762 |      -0.0854 |
        |            300 |      -0      |      -0.0001 |      -0.0016 |      -0.0061 |      -0.0131 |      -0.0217 |      -0.0306 |      -0.0393 |      -0.0475 |
        |            305 |      -0      |      -0      |      -0.0002 |      -0.0013 |      -0.0038 |      -0.0077 |      -0.0127 |      -0.0182 |      -0.024  |
        |            310 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0009 |      -0.0024 |      -0.0047 |      -0.0076 |      -0.0111 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0007 |      -0.0015 |      -0.0029 |      -0.0047 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0005 |      -0.001  |      -0.0018 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0007 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.004  |       0.0263 |       0.0752 |       0.1507 |       0.25   |       0.3698 |       0.5073 |       0.6599 |
        |            295 |            0 |       0.0004 |       0.005  |       0.0209 |       0.0528 |       0.1022 |       0.1692 |       0.2526 |       0.3512 |
        |            300 |            0 |       0      |       0.0007 |       0.0047 |       0.0158 |       0.0367 |       0.0693 |       0.1144 |       0.1719 |
        |            305 |            0 |       0      |       0.0001 |       0.0009 |       0.004  |       0.0116 |       0.0255 |       0.0472 |       0.0776 |
        |            310 |            0 |       0      |       0      |       0.0001 |       0.0009 |       0.0032 |       0.0085 |       0.0178 |       0.0323 |
        |            315 |            0 |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0025 |       0.0061 |       0.0125 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.002  |       0.0045 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0006 |       0.0015 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |
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
                else self._dividend_yield[ticker].iloc[-1]
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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |           -0 |      -0.0041 |      -0.0266 |      -0.0761 |      -0.153  |      -0.2544 |      -0.3773 |      -0.5186 |      -0.6759 |
        |            295 |           -0 |      -0.0004 |      -0.005  |      -0.0211 |      -0.0535 |      -0.1038 |      -0.1722 |      -0.2576 |      -0.3588 |
        |            300 |           -0 |      -0      |      -0.0007 |      -0.0048 |      -0.0159 |      -0.0372 |      -0.0704 |      -0.1164 |      -0.1753 |
        |            305 |           -0 |      -0      |      -0.0001 |      -0.0009 |      -0.0041 |      -0.0117 |      -0.0259 |      -0.0479 |      -0.0789 |
        |            310 |           -0 |      -0      |      -0      |      -0.0001 |      -0.0009 |      -0.0033 |      -0.0086 |      -0.018  |      -0.0328 |
        |            315 |           -0 |      -0      |      -0      |      -0      |      -0.0002 |      -0.0008 |      -0.0026 |      -0.0062 |      -0.0127 |
        |            320 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0007 |      -0.002  |      -0.0045 |
        |            325 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0006 |      -0.0015 |
        |            330 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0005 |
        |            335 |           -0 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |
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
                else self._dividend_yield[ticker].iloc[-1]
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
        is a mathematical model used to estimate the price of European—style options. The lambda is
        the rate of change of the option price with respect to the underlying price.

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
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the dividend yield as obtained through annual historical data.
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      258.394 |      139.654 |      99.1523 |      78.4369 |      65.7343 |      57.0891 |      50.7916 |      45.9798 |      42.1704 |
        |            295 |      313.218 |      165.83  |     116.034  |      90.7777 |      75.4041 |      65.0088 |      57.4801 |      51.7573 |      47.2479 |
        |            300 |      368.224 |      192.315 |     133.19   |     103.35   |      85.2695 |      73.0955 |      64.3129 |      57.6609 |      52.4367 |
        |            305 |      423.035 |      218.886 |     150.47   |     116.045  |      95.2489 |      81.2854 |      71.2387 |      63.6482 |      57.7011 |
        |            310 |      477.435 |      245.396 |     167.771  |     128.787  |     105.282  |      89.53   |      78.2174 |      69.6858 |      63.0127 |
        |            315 |      531.3   |      271.755 |     185.023  |     141.52   |     115.324  |      97.7925 |      85.2182 |      75.7472 |      68.3487 |
        |            320 |      584.56  |      297.902 |     202.178  |     154.205  |     125.343  |     106.045  |      92.2175 |      81.8122 |      73.6913 |
        |            325 |      637.175 |      323.8   |     219.202  |     166.813  |     135.315  |     114.268  |      99.1974 |      87.8648 |      79.0265 |
        |            330 |      689.124 |      349.421 |     236.074  |     179.325  |     145.222  |     122.445  |     106.144  |      93.893  |      84.3433 |
        |            335 |      740.398 |      374.753 |     252.778  |     191.727  |     155.051  |     130.565  |     113.048  |      99.8874 |      89.6332 |
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
                else self._dividend_yield[ticker].iloc[-1]
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
        - Partial Derivative: measures the rate of change in the option price with respect to the strike price.

        For a deeper explanation, please have a look at: https://en.wikipedia.org/wiki/Greeks_(finance) and the
        references to the literature as found on this page.

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0013 |       0.0039 |       0.0065 |       0.0086 |       0.0101 |       0.0113 |       0.0122 |       0.0128 |
        |            295 |            0 |       0.0001 |       0.0009 |       0.0021 |       0.0035 |       0.0049 |       0.0061 |       0.0071 |       0.0079 |
        |            300 |            0 |       0      |       0.0001 |       0.0006 |       0.0012 |       0.002  |       0.0028 |       0.0036 |       0.0044 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0004 |       0.0007 |       0.0012 |       0.0017 |       0.0022 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0004 |       0.0007 |       0.001  |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0003 |       0.0004 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        Also known as: cash gamma, binary option gamma.

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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0011 |       0.0034 |       0.0057 |       0.0075 |       0.0089 |       0.0099 |       0.0106 |       0.0112 |
        |            295 |            0 |       0.0001 |       0.0007 |       0.0018 |       0.003  |       0.0041 |       0.0051 |       0.006  |       0.0067 |
        |            300 |            0 |       0      |       0.0001 |       0.0005 |       0.001  |       0.0016 |       0.0023 |       0.003  |       0.0036 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0006 |       0.0009 |       0.0013 |       0.0018 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0008 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |        0.002 |       0.0727 |       0.2189 |       0.3641 |       0.482  |       0.5714 |       0.6378 |       0.6864 |       0.7219 |
        |            295 |        0     |       0.0098 |       0.0625 |       0.1511 |       0.2501 |       0.3443 |       0.4275 |       0.4984 |       0.5577 |
        |            300 |        0     |       0.0008 |       0.0126 |       0.0477 |       0.1035 |       0.1705 |       0.2407 |       0.3089 |       0.3725 |
        |            305 |        0     |       0      |       0.0019 |       0.0118 |       0.035  |       0.0709 |       0.1162 |       0.1667 |       0.2193 |
        |            310 |        0     |       0      |       0.0002 |       0.0023 |       0.0098 |       0.0252 |       0.0488 |       0.0795 |       0.1153 |
        |            315 |        0     |       0      |       0      |       0.0004 |       0.0023 |       0.0077 |       0.018  |       0.0338 |       0.0547 |
        |            320 |        0     |       0      |       0      |       0      |       0.0005 |       0.0021 |       0.0059 |       0.0129 |       0.0236 |
        |            325 |        0     |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0017 |       0.0045 |       0.0093 |
        |            330 |        0     |       0      |       0      |       0      |       0      |       0.0001 |       0.0005 |       0.0014 |       0.0034 |
        |            335 |        0     |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0004 |       0.0011 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.1219 |      -2.1684 |      -4.3673 |      -5.466  |      -5.8053 |      -5.7531 |      -5.52   |      -5.214  |      -4.8883 |
        |            295 |      -0.0018 |      -0.2932 |      -1.2452 |      -2.2622 |      -3.0038 |      -3.4539 |      -3.6845 |      -3.7674 |      -3.7563 |
        |            300 |      -0      |      -0.0241 |      -0.2509 |      -0.7139 |      -1.2405 |      -1.7063 |      -2.0685 |      -2.3276 |      -2.5    |
        |            305 |      -0      |      -0.0012 |      -0.0368 |      -0.1762 |      -0.4184 |      -0.7087 |      -0.9966 |      -1.2536 |      -1.4681 |
        |            310 |      -0      |      -0      |      -0.004  |      -0.0347 |      -0.1172 |      -0.2513 |      -0.4181 |      -0.5965 |      -0.7705 |
        |            315 |      -0      |      -0      |      -0.0003 |      -0.0055 |      -0.0276 |      -0.077  |      -0.1544 |      -0.2533 |      -0.3648 |
        |            320 |      -0      |      -0      |      -0      |      -0.0007 |      -0.0056 |      -0.0206 |      -0.0506 |      -0.0968 |      -0.1571 |
        |            325 |      -0      |      -0      |      -0      |      -0.0001 |      -0.001  |      -0.0048 |      -0.0149 |      -0.0335 |      -0.0619 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.001  |      -0.0039 |      -0.0106 |      -0.0225 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0002 |      -0.0009 |      -0.0031 |      -0.0075 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.1147 |       4.0503 |      12.1539 |      20.1454 |      26.5649 |      31.3784 |      34.8886 |      37.4084 |      39.19   |
        |            295 |       0.0021 |       0.6886 |       4.3631 |      10.5122 |      17.3543 |      23.8166 |      29.4816 |      34.2668 |      38.2294 |
        |            300 |       0      |       0.0679 |       1.0569 |       3.9921 |       8.6324 |      14.1844 |      19.971  |      25.5686 |      30.756  |
        |            305 |       0      |       0.0041 |       0.1806 |       1.1491 |       3.3979 |       6.8791 |      11.2436 |      16.1003 |      21.1301 |
        |            310 |       0      |       0.0002 |       0.0225 |       0.2578 |       1.086  |       2.7837 |       5.3854 |       8.7516 |      12.6739 |
        |            315 |       0      |       0      |       0.0021 |       0.046  |       0.2872 |       0.9566 |       2.2315 |       4.1727 |       6.7401 |
        |            320 |       0      |       0      |       0.0001 |       0.0067 |       0.0638 |       0.283  |       0.8102 |       1.7664 |       3.2153 |
        |            325 |       0      |       0      |       0      |       0.0008 |       0.0121 |       0.0729 |       0.2604 |       0.6704 |       1.3886 |
        |            330 |       0      |       0      |       0      |       0.0001 |       0.002  |       0.0165 |       0.0748 |       0.23   |       0.5471 |
        |            335 |       0      |       0      |       0      |       0      |       0.0003 |       0.0033 |       0.0193 |       0.0718 |       0.1979 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0015 |       0.107  |       0.4814 |       1.0633 |       1.7516 |       2.4813 |       3.2167 |       3.9394 |       4.6401 |
        |            295 |       0      |       0.0145 |       0.1379 |       0.4427 |       0.9131 |       1.5031 |       2.1696 |       2.8806 |       3.6137 |
        |            300 |       0      |       0.0012 |       0.0279 |       0.1403 |       0.3789 |       0.7469 |       1.2264 |       1.7937 |       2.4263 |
        |            305 |       0      |       0.0001 |       0.0041 |       0.0347 |       0.1283 |       0.3115 |       0.5938 |       0.9714 |       1.4337 |
        |            310 |       0      |       0      |       0.0004 |       0.0068 |       0.036  |       0.1108 |       0.25   |       0.4641 |       0.7559 |
        |            315 |       0      |       0      |       0      |       0.0011 |       0.0085 |       0.034  |       0.0925 |       0.1977 |       0.3592 |
        |            320 |       0      |       0      |       0      |       0.0001 |       0.0017 |       0.0091 |       0.0304 |       0.0758 |       0.1551 |
        |            325 |       0      |       0      |       0      |       0      |       0.0003 |       0.0021 |       0.0089 |       0.0263 |       0.0613 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0.0004 |       0.0024 |       0.0083 |       0.0223 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0006 |       0.0024 |       0.0075 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0002 |      -0.0037 |      -0.0079 |      -0.0104 |      -0.0116 |      -0.012  |      -0.012  |      -0.0118 |      -0.0115 |
        |            295 |      -0      |      -0.0006 |      -0.0027 |      -0.005  |      -0.0069 |      -0.0081 |      -0.0089 |      -0.0094 |      -0.0096 |
        |            300 |      -0      |      -0.0001 |      -0.0006 |      -0.0018 |      -0.0032 |      -0.0045 |      -0.0056 |      -0.0065 |      -0.0071 |
        |            305 |      -0      |      -0      |      -0.0001 |      -0.0005 |      -0.0012 |      -0.0021 |      -0.003  |      -0.0039 |      -0.0046 |
        |            310 |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0008 |      -0.0014 |      -0.002  |      -0.0027 |
        |            315 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.0006 |      -0.0009 |      -0.0014 |
        |            320 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0002 |      -0.0004 |      -0.0006 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0001 |      -0.0003 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |      -0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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
        the rate of change of the option price with respect to the strike price.

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

        Also known as: numerical derivative, option sensitivity.

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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0011 |       0.0034 |       0.0057 |       0.0075 |       0.0089 |       0.0099 |       0.0106 |       0.0112 |
        |            295 |            0 |       0.0001 |       0.0007 |       0.0018 |       0.003  |       0.0041 |       0.0051 |       0.006  |       0.0067 |
        |            300 |            0 |       0      |       0.0001 |       0.0005 |       0.001  |       0.0016 |       0.0023 |       0.003  |       0.0036 |
        |            305 |            0 |       0      |       0      |       0.0001 |       0.0003 |       0.0006 |       0.0009 |       0.0013 |       0.0018 |
        |            310 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0005 |       0.0008 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0003 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        By default the most recent risk free rate, dividend yield and stock price is used, you can alter this by changing
        the start date. The volatility is calculated based on the daily returns of the stock price and the selected
        period (this can be altered by defining this accordingly when defining the Toolkit class, start_date and end_date).

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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |            0 |       0.0005 |       0.0011 |       0.0013 |       0.0014 |       0.0014 |       0.0013 |       0.0012 |       0.0011 |
        |            295 |            0 |       0.0001 |       0.0003 |       0.0006 |       0.0007 |       0.0008 |       0.0009 |       0.0009 |       0.0009 |
        |            300 |            0 |       0      |       0.0001 |       0.0002 |       0.0003 |       0.0004 |       0.0005 |       0.0006 |       0.0006 |
        |            305 |            0 |       0      |       0      |       0      |       0.0001 |       0.0002 |       0.0002 |       0.0003 |       0.0004 |
        |            310 |            0 |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |       0.0001 |       0.0002 |
        |            315 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0001 |
        |            320 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            325 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            330 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
        |            335 |            0 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0      |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       0.0016 |       0.027  |       0.0499 |       0.057  |       0.0548 |       0.0487 |       0.0413 |       0.034  |       0.0272 |
        |            295 |       0      |       0.0048 |       0.0195 |       0.0336 |       0.0421 |       0.0456 |       0.0457 |       0.0437 |       0.0406 |
        |            300 |       0      |       0.0005 |       0.0049 |       0.0135 |       0.0226 |       0.0299 |       0.0348 |       0.0376 |       0.0387 |
        |            305 |       0      |       0      |       0.0009 |       0.004  |       0.0093 |       0.0153 |       0.0209 |       0.0255 |       0.029  |
        |            310 |       0      |       0      |       0.0001 |       0.0009 |       0.003  |       0.0064 |       0.0104 |       0.0145 |       0.0183 |
        |            315 |       0      |       0      |       0      |       0.0002 |       0.0008 |       0.0022 |       0.0044 |       0.0071 |       0.0101 |
        |            320 |       0      |       0      |       0      |       0      |       0.0002 |       0.0007 |       0.0016 |       0.0031 |       0.0049 |
        |            325 |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0005 |       0.0012 |       0.0022 |
        |            330 |       0      |       0      |       0      |       0      |       0      |       0      |       0.0002 |       0.0004 |       0.0009 |
        |            335 |       0      |       0      |       0      |       0      |       0      |       0      |       0      |       0.0001 |       0.0003 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |      -0.0976 |      -0.8053 |      -0.9984 |      -0.8599 |      -0.6654 |      -0.496  |      -0.3642 |      -0.2651 |      -0.1912 |
        |            295 |      -0.0018 |      -0.144  |      -0.3886 |      -0.5035 |      -0.5074 |      -0.46   |      -0.3968 |      -0.3338 |      -0.2772 |
        |            300 |      -0      |      -0.0146 |      -0.098  |      -0.2022 |      -0.2714 |      -0.3002 |      -0.3005 |      -0.2847 |      -0.2613 |
        |            305 |      -0      |      -0.0009 |      -0.0171 |      -0.0601 |      -0.1113 |      -0.1531 |      -0.1797 |      -0.1926 |      -0.195  |
        |            310 |      -0      |      -0      |      -0.0022 |      -0.0137 |      -0.0365 |      -0.0639 |      -0.0893 |      -0.1093 |      -0.1229 |
        |            315 |      -0      |      -0      |      -0.0002 |      -0.0025 |      -0.0098 |      -0.0224 |      -0.0379 |      -0.0536 |      -0.0675 |
        |            320 |      -0      |      -0      |      -0      |      -0.0004 |      -0.0022 |      -0.0067 |      -0.014  |      -0.0231 |      -0.0329 |
        |            325 |      -0      |      -0      |      -0      |      -0      |      -0.0004 |      -0.0018 |      -0.0046 |      -0.0089 |      -0.0145 |
        |            330 |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0004 |      -0.0013 |      -0.0031 |      -0.0058 |
        |            335 |      -0      |      -0      |      -0      |      -0      |      -0      |      -0.0001 |      -0.0003 |      -0.001  |      -0.0021 |
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
                else self._dividend_yield[ticker].iloc[-1]
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

        |   Strike Price |   2026-01-01 |   2026-01-02 |   2026-01-03 |   2026-01-04 |   2026-01-05 |   2026-01-06 |   2026-01-07 |   2026-01-08 |   2026-01-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            290 |       4.4479 |      59.5728 |      81.1739 |      53.6643 |       6.762  |     -42.4233 |     -87.2163 |    -125.733  |    -157.986  |
        |            295 |       0.1375 |      19.5947 |      69.0758 |     100.071  |      99.471  |      76.366  |      41.3441 |       1.6829 |     -38.3689 |
        |            300 |       0.0014 |       3.0559 |      28.3789 |      70.9546 |     106.42   |     123.366  |     121.898  |     106.326  |      81.3579 |
        |            305 |       0      |       0.2627 |       7.1602 |      31.4362 |      67.9124 |     103.693  |     130.022  |     143.805  |     145.467  |
        |            310 |       0      |       0.0135 |       1.2162 |       9.8489 |      31.1181 |      62.0504 |      95.5641 |     125.461  |     148.082  |
        |            315 |       0      |       0.0004 |       0.1464 |       2.3133 |      10.9951 |      28.9922 |      54.9184 |      84.8689 |     114.693  |
        |            320 |       0      |       0      |       0.0129 |       0.4222 |       3.1152 |      11.062  |      26.0298 |      47.5385 |      73.4872 |
        |            325 |       0      |       0      |       0.0009 |       0.0614 |       0.7262 |       3.5422 |      10.4864 |      22.8123 |      40.5143 |
        |            330 |       0      |       0      |       0      |       0.0073 |       0.1419 |       0.9702 |       3.6628 |       9.5798 |      19.6698 |
        |            335 |       0      |       0      |       0      |       0.0007 |       0.0236 |       0.2306 |       1.1254 |       3.5734 |       8.5425 |
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
                else self._dividend_yield[ticker].iloc[-1]
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
