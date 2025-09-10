"""Options Module"""

__docformat__ = "google"

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from financetoolkit.options import (
    binomial_trees_model,
    black_scholes_model,
    greeks_model,
    helpers,
    options_model,
)
from financetoolkit.ratios import valuation_model

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,too-many-locals,cell-var-from-loop
# pylint: disable=line-too-long,too-many-public-methods
# ruff: noqa: E501

try:
    from tqdm import tqdm

    ENABLE_TQDM = True
except ImportError:
    ENABLE_TQDM = False


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

        toolkit = Toolkit(["TSLA", "MU"], api_key="FINANCIAL_MODELING_PREP_KEY")

        all_greeks = toolkit.options.collect_all_greeks(start_date='2024-01-03')

        all_greeks.loc['TSLA', '2024-01-04']
        ```

        Which returns:

        |   Strike Price |   Delta |   Dual Delta |   Vega |   Theta |    Rho |   Epsilon |   Lambda |   Gamma |   Dual Gamma |   Vanna |    Charm |   Vomma |    Vera |      Veta |     PD |   Speed |   Zomma |   Color |   Ultima |
        |---------------:|--------:|-------------:|-------:|--------:|-------:|----------:|---------:|--------:|-------------:|--------:|---------:|--------:|--------:|----------:|-------:|--------:|--------:|--------:|---------:|
        |            205 |  1      |      -0.9999 | 0      | -0.022  | 0.0056 |   -0.6533 |   0.0712 |  0      |       0      | -0      |   0.0005 |  0.0003 | -0      |    0.1236 | 0      | -0      |  0      |  0.0004 |   0.0001 |
        |            210 |  1      |      -0.9999 | 0      | -0.0226 | 0.0058 |   -0.6533 |   0.0837 |  0      |       0      | -0.0002 |   0.0221 |  0.0119 | -0.0001 |    4.6313 | 0      | -0      |  0.0001 |  0.0132 |   0.0034 |
        |            215 |  0.9998 |      -0.9997 | 0.0001 | -0.0254 | 0.0059 |   -0.6532 |   0.1016 |  0.0001 |       0.0001 | -0.0044 |   0.4426 |  0.1942 | -0.0029 |   77.6496 | 0.0001 | -0.0001 |  0.0021 |  0.209  |   0.0336 |
        |            220 |  0.9973 |      -0.9969 | 0.001  | -0.0526 | 0.006  |   -0.6515 |   0.1287 |  0.0012 |       0.0014 | -0.0414 |   4.1955 |  1.4351 | -0.0273 |  600.92   | 0.0014 | -0.0005 |  0.0144 |  1.4569 |   0.1196 |
        |            225 |  0.9777 |      -0.976  | 0.0066 | -0.2079 | 0.006  |   -0.6387 |   0.1723 |  0.0076 |       0.0086 | -0.1884 |  19.0888 |  4.7244 | -0.1249 | 2187.89   | 0.0086 | -0.0022 |  0.0407 |  4.1228 |   0.0829 |
        |            230 |  0.8953 |      -0.8898 | 0.0226 | -0.6528 | 0.0056 |   -0.5849 |   0.2419 |  0.0261 |       0.028  | -0.3993 |  40.3564 |  6.2557 | -0.267  | 3816.31   | 0.028  | -0.0048 |  0.0253 |  2.5239 |  -0.1641 |
        |            235 |  0.6978 |      -0.6874 | 0.0435 | -1.2304 | 0.0044 |   -0.4558 |   0.3442 |  0.0502 |       0.0516 | -0.306  |  30.653  |  1.9785 | -0.2119 | 3623.7    | 0.0516 | -0.0039 | -0.0672 | -6.8719 |  -0.0977 |
        |            240 |  0.4192 |      -0.4078 | 0.0488 | -1.3691 | 0.0027 |   -0.2739 |   0.4789 |  0.0562 |       0.0555 |  0.1634 | -17.1438 |  0.4159 |  0.0934 | 3407.79   | 0.0555 |  0.0014 | -0.096  | -9.7512 |  -0.0222 |
        |            245 |  0.1812 |      -0.1736 | 0.0329 | -0.9207 | 0.0012 |   -0.1184 |   0.6396 |  0.0379 |       0.0359 |  0.4445 | -45.5549 |  5.0536 |  0.2814 | 4080.87   | 0.0359 |  0.0048 | -0.0098 | -0.9474 |  -0.1945 |
        |            250 |  0.0544 |      -0.0513 | 0.0138 | -0.3848 | 0.0004 |   -0.0355 |   0.8183 |  0.0159 |       0.0144 |  0.3232 | -33.01   |  6.468  |  0.2073 | 3328.37   | 0.0144 |  0.0036 |  0.0461 |  4.7176 |  -0.0443 |
        |            255 |  0.0112 |      -0.0104 | 0.0037 | -0.1028 | 0.0001 |   -0.0073 |   1.0084 |  0.0042 |       0.0037 |  0.1223 | -12.477  |  3.4845 |  0.0789 | 1542.52   | 0.0037 |  0.0014 |  0.0325 |  3.3216 |   0.1424 |
        |            260 |  0.0016 |      -0.0015 | 0.0006 | -0.018  | 0      |   -0.001  |   1.205  |  0.0007 |       0.0006 |  0.0276 |  -2.8148 |  1.0161 |  0.0179 |  421.028  | 0.0006 |  0.0003 |  0.0104 |  1.0578 |   0.1054 |
        |            265 |  0.0002 |      -0.0001 | 0.0001 | -0.0021 | 0      |   -0.0001 |   1.4049 |  0.0001 |       0.0001 |  0.004  |  -0.4041 |  0.1783 |  0.0026 |   71.3544 | 0.0001 |  0      |  0.0019 |  0.1933 |   0.0322 |
        |            270 |  0      |      -0      | 0      | -0.0002 | 0      |   -0      |   1.6059 |  0      |       0      |  0.0004 |  -0.0385 |  0.02   |  0.0002 |    7.8471 | 0      |  0      |  0.0002 |  0.0222 |   0.0054 |
        |            275 |  0      |      -0      | 0      | -0      | 0      |   -0      |   1.8068 |  0      |       0      |  0      |  -0.0025 |  0.0015 |  0      |    0.5804 | 0      |  0      |  0      |  0.0017 |   0.0006 |
        """
        self._tickers = tickers
        self._daily_historical = daily_historical
        self._quarterly = quarterly
        self._rounding: int | None = rounding

        # Option Statistics
        self._prices = self._daily_historical["Adj Close"]
        self._volatility = self._daily_historical["Volatility"] * np.sqrt(252)
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
                # If empty, it doesn't matter that the value is 0
                # given that this implies the company doesn't pay dividends
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

        The data comes from Yahoo Finance and is not always available. If the data is not
        available, it is advised to use the theoretical calculations as provided by the
        Black Scholes Model as well as the Greeks to get a better understanding of the
        option prices over time.

        Args:
            expiration_date (str | None, optional): The expiration date to use. Defaults to None which means it will
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

        |   Strike Price | Contract Symbol     | Currency   |   Last Price |   Change |   Percent Change |   Volume |   Open Interest |   Bid |   Ask | Expiration   | Last Trade Date   |   Implied Volatility | In The Money   |
        |---------------:|:--------------------|:-----------|-------------:|---------:|-----------------:|---------:|----------------:|------:|------:|:-------------|:------------------|---------------------:|:---------------|
        |          155   | AAPL240112C00155000 | USD        |        29.75 |        0 |                0 |        9 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          157.5 | AAPL240112C00157500 | USD        |        24.15 |        0 |                0 |        1 |               0 |     0 |     0 | 2024-01-12   | 2024-01-05        |                    0 | True           |
        |          160   | AAPL240112C00160000 | USD        |        25.75 |        0 |                0 |       11 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          162.5 | AAPL240112C00162500 | USD        |        18.5  |        0 |                0 |        7 |               0 |     0 |     0 | 2024-01-12   | 2024-01-05        |                    0 | True           |
        |          165   | AAPL240112C00165000 | USD        |        20.47 |        0 |                0 |       11 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          167.5 | AAPL240112C00167500 | USD        |        17.22 |        0 |                0 |        1 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          170   | AAPL240112C00170000 | USD        |        15.65 |        0 |                0 |      176 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          172.5 | AAPL240112C00172500 | USD        |        13.19 |        0 |                0 |       34 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          175   | AAPL240112C00175000 | USD        |        10.64 |        0 |                0 |      258 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          177.5 | AAPL240112C00177500 | USD        |         8.3  |        0 |                0 |      489 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          180   | AAPL240112C00180000 | USD        |         5.65 |        0 |                0 |     6152 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          182.5 | AAPL240112C00182500 | USD        |         3.25 |        0 |                0 |    14721 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
        |          185   | AAPL240112C00185000 | USD        |         1.18 |        0 |                0 |   102803 |               0 |     0 |     0 | 2024-01-12   | 2024-01-11        |                    0 | True           |
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

        option_chains["Change"] = option_chains["Change"].round(
            rounding if rounding else self._rounding
        )
        option_chains["Percent Change"] = option_chains["Percent Change"].round(
            rounding if rounding else self._rounding
        )
        option_chains["Implied Volatility"] = option_chains["Implied Volatility"].round(
            rounding if rounding else self._rounding
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

        Returns:
            pd.DataFrame: Black Scholes values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        black_scholes = toolkit.options.get_black_scholes_model()

        black_scholes.loc['AMZN']
        ```

        Which returns:

        |   Strike Price |   2024-01-12 |   2024-01-13 |   2024-01-14 |   2024-01-15 |   2024-01-16 |   2024-01-17 |   2024-01-18 |   2024-01-19 |   2024-01-20 |   2024-01-21 |   2024-01-22 |   2024-01-23 |   2024-01-24 |   2024-01-25 |   2024-01-26 |   2024-01-27 |   2024-01-28 |   2024-01-29 |   2024-01-30 |   2024-01-31 |   2024-02-01 |   2024-02-02 |   2024-02-03 |   2024-02-04 |   2024-02-05 |   2024-02-06 |   2024-02-07 |   2024-02-08 |   2024-02-09 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        |            115 |      40.1925 |      40.2051 |      40.2176 |      40.2301 |      40.2427 |      40.2552 |      40.2677 |      40.2803 |      40.2928 |      40.3053 |      40.3179 |      40.3304 |      40.3429 |      40.3554 |      40.3679 |      40.3805 |      40.393  |      40.4055 |      40.4181 |      40.4306 |      40.4432 |      40.4558 |      40.4684 |      40.4811 |      40.4938 |      40.5065 |      40.5193 |      40.5322 |      40.5451 |
        |            120 |      35.1931 |      35.2062 |      35.2192 |      35.2323 |      35.2454 |      35.2585 |      35.2716 |      35.2846 |      35.2977 |      35.3108 |      35.3239 |      35.3369 |      35.35   |      35.3631 |      35.3762 |      35.3894 |      35.4026 |      35.4158 |      35.4292 |      35.4426 |      35.4561 |      35.4697 |      35.4834 |      35.4973 |      35.5113 |      35.5255 |      35.5399 |      35.5544 |      35.5691 |
        |            125 |      30.1936 |      30.2073 |      30.2209 |      30.2345 |      30.2481 |      30.2618 |      30.2754 |      30.289  |      30.3026 |      30.3163 |      30.33   |      30.3438 |      30.3576 |      30.3716 |      30.3858 |      30.4001 |      30.4147 |      30.4296 |      30.4447 |      30.4602 |      30.476  |      30.4921 |      30.5087 |      30.5256 |      30.5429 |      30.5606 |      30.5787 |      30.5972 |      30.6161 |
        |            130 |      25.1942 |      25.2083 |      25.2225 |      25.2367 |      25.2509 |      25.265  |      25.2793 |      25.2936 |      25.3081 |      25.3229 |      25.3381 |      25.3537 |      25.3699 |      25.3866 |      25.4041 |      25.4223 |      25.4412 |      25.4609 |      25.4814 |      25.5026 |      25.5246 |      25.5474 |      25.571  |      25.5952 |      25.6202 |      25.6459 |      25.6723 |      25.6993 |      25.7269 |
        |            135 |      20.1947 |      20.2094 |      20.2242 |      20.2389 |      20.2538 |      20.2691 |      20.2851 |      20.3022 |      20.3206 |      20.3405 |      20.3619 |      20.385  |      20.4098 |      20.4363 |      20.4643 |      20.4938 |      20.5248 |      20.5572 |      20.5908 |      20.6257 |      20.6617 |      20.6987 |      20.7367 |      20.7756 |      20.8153 |      20.8558 |      20.897  |      20.9388 |      20.9813 |
        |            140 |      15.1953 |      15.2105 |      15.2261 |      15.2432 |      15.2631 |      15.2869 |      15.3149 |      15.3471 |      15.3834 |      15.4233 |      15.4664 |      15.5125 |      15.5611 |      15.6119 |      15.6645 |      15.7189 |      15.7747 |      15.8317 |      15.8898 |      15.9488 |      16.0085 |      16.0689 |      16.1299 |      16.1913 |      16.2531 |      16.3152 |      16.3776 |      16.4402 |      16.5029 |
        |            145 |      10.1958 |      10.2147 |      10.2456 |      10.2916 |      10.3506 |      10.4194 |      10.4956 |      10.5769 |      10.662  |      10.7497 |      10.8392 |      10.9299 |      11.0213 |      11.113  |      11.2049 |      11.2967 |      11.3882 |      11.4794 |      11.5701 |      11.6603 |      11.75   |      11.839  |      11.9275 |      12.0153 |      12.1024 |      12.1889 |      12.2747 |      12.3598 |      12.4443 |
        |            150 |       5.2213 |       5.3527 |       5.52   |       5.6952 |       5.8693 |       6.0395 |       6.2047 |       6.3647 |       6.5198 |       6.6702 |       6.8162 |       6.9581 |       7.0962 |       7.2308 |       7.3621 |       7.4903 |       7.6157 |       7.7384 |       7.8586 |       7.9765 |       8.0921 |       8.2056 |       8.3172 |       8.4268 |       8.5347 |       8.6409 |       8.7455 |       8.8485 |       8.9501 |
        |            155 |       1.1757 |       1.6286 |       1.9783 |       2.2744 |       2.5363 |       2.7739 |       2.9931 |       3.1976 |       3.3902 |       3.5728 |       3.7469 |       3.9136 |       4.0737 |       4.2282 |       4.3775 |       4.5221 |       4.6626 |       4.7991 |       4.9322 |       5.062  |       5.1888 |       5.3128 |       5.4341 |       5.5531 |       5.6697 |       5.7842 |       5.8966 |       6.0071 |       6.1158 |
        |            160 |       0.0437 |       0.2013 |       0.3903 |       0.5823 |       0.77   |       0.9513 |       1.1259 |       1.2942 |       1.4565 |       1.6133 |       1.7651 |       1.9124 |       2.0554 |       2.1946 |       2.3302 |       2.4624 |       2.5916 |       2.7179 |       2.8416 |       2.9627 |       3.0814 |       3.198  |       3.3124 |       3.4249 |       3.5355 |       3.6443 |       3.7514 |       3.8569 |       3.9608 |
        |            165 |       0.0001 |       0.0081 |       0.0378 |       0.0889 |       0.1563 |       0.235  |       0.3213 |       0.413  |       0.5081 |       0.6055 |       0.7043 |       0.804  |       0.9039 |       1.0039 |       1.1036 |       1.2029 |       1.3017 |       1.3999 |       1.4974 |       1.5941 |       1.69   |       1.7852 |       1.8795 |       1.973  |       2.0657 |       2.1576 |       2.2487 |       2.339  |       2.4285 |
        |            170 |       0      |       0.0001 |       0.0017 |       0.0079 |       0.0208 |       0.0412 |       0.0689 |       0.103  |       0.143  |       0.1879 |       0.237  |       0.2897 |       0.3454 |       0.4037 |       0.4641 |       0.5263 |       0.59   |       0.6549 |       0.721  |       0.7878 |       0.8555 |       0.9237 |       0.9923 |       1.0614 |       1.1307 |       1.2003 |       1.27   |       1.3398 |       1.4096 |
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

            for strike_price in strike_prices:
                black_scholes[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    black_scholes[ticker][strike_price][time_to_expiration] = (
                        black_scholes_model.get_black_scholes(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            time_to_expiration=time_to_expiration,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        black_scholes_df = helpers.create_greek_dataframe(
            greek_dictionary=black_scholes,
            start_date=start_date,
        )

        black_scholes_df = black_scholes_df.round(
            rounding if rounding else self._rounding
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

        |       |   2024-02-09 |
        |------:|-------------:|
        | 162.5 |       0.2828 |
        | 165   |       1.0238 |
        | 167.5 |       0.7867 |
        | 170   |       0.6984 |
        | 172.5 |       0.6796 |
        | 175   |       0.4611 |
        | 177.5 |       0.4423 |
        | 180   |       0.4154 |
        | 182.5 |       0.3541 |
        | 185   |       0.3506 |
        | 187.5 |       0.3331 |
        | 190   |       0.3329 |
        | 192.5 |       0.3411 |
        | 195   |       0.361  |
        | 197.5 |       0.3833 |
        | 200   |       0.4033 |
        | 202.5 |       0.4477 |
        | 205   |       0.4452 |
        | 207.5 |       0.518  |
        """
        option_chains = self.get_option_chains(
            expiration_date=expiration_date if expiration_date is not None else None,
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
        today = datetime.today()
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
                # The expiration date is used to calculate the days to expiration
                # which serves as input for the time to expiration parameter in the Black Scholes Model.
                days_to_expiration = (pd.to_datetime(option_chains.name) - today).days

                def objective_function(sigma: float):
                    return (
                        black_scholes_model.get_black_scholes(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            risk_free_rate=risk_free_rate,
                            time_to_expiration=days_to_expiration / 365,
                            volatility=sigma,
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                        - row["Last Price"]
                    ) ** 2

                # The minimize function is used to find the implied volatility value that minimizes
                # the objective function. This means that the difference between the output of the
                # Black Scholes Model and the market option price is minimized.
                implied_volatility_value = minimize(
                    objective_function, volatility.loc[ticker]
                ).x[0]

                # Values that are equal to the current volatility refer to not being able to resolve
                # and thus are not added to the implied volatility dictionary.
                if round(implied_volatility_value, 4) != round(
                    volatility.loc[ticker], 4
                ):
                    implied_volatility[ticker][strike_price] = implied_volatility_value

        implied_volatility_df = pd.DataFrame(implied_volatility).unstack().dropna()

        implied_volatility_df = implied_volatility_df.round(
            rounding if rounding else self._rounding
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

        Args:
            start_date (str | None, optional): The start date which determines the stock price. Defaults to None
            which means it will use the most recent date.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
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

        Returns:
            pd.DataFrame: Binomial Trees values containing the tickers, strike prices and movements as the index and the
            timesteps as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        binomial_trees_model = toolkit.options.get_binomial_trees_model(
            start_date='2024-02-02'
        )

        binomial_trees_model.loc['AAPL', 140]
        ```

        Which returns:

        | Movement   |   2024-02-02 |   2024-03-09 |   2024-04-15 |   2024-05-21 |   2024-06-27 |   2024-08-02 |   2024-09-08 |   2024-10-14 |   2024-11-20 |   2024-12-26 |   2025-02-01 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUUUUUUUU |      53.9518 |      69.0761 |      86.6159 |     106.476  |     128.56   |     152.858  |     179.496  |     208.695  |     240.695  |     275.76   |     314.18   |
        | UUUUUUUUUD |     nan      |      39.3262 |      52.173  |      67.5522 |      85.3691 |     105.438  |     127.617  |     151.936  |     178.598  |     207.823  |     239.852  |
        | UUUUUUUUDD |     nan      |     nan      |      26.8446 |      37.2767 |      50.3585 |      66.0853 |      84.2272 |     104.466  |     126.663  |     151.003  |     177.689  |
        | UUUUUUUDDD |     nan      |     nan      |     nan      |      16.6633 |      24.5421 |      35.0986 |      48.5539 |      64.764  |      83.2271 |     103.482  |     125.698  |
        | UUUUUUDDDD |     nan      |     nan      |     nan      |     nan      |       8.9417 |      14.216  |      21.9711 |      32.795  |      46.8998 |      63.7382 |      82.2161 |
        | UUUUUDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |       3.7526 |       6.596  |      11.3548 |      18.9981 |      30.4984 |      45.85   |
        | UUUUDDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0.9457 |       1.9009 |       3.8207 |       7.6794 |      15.4353 |
        | UUUDDDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |       0      |       0      |
        | UUDDDDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |       0      |
        | UDDDDDDDDD |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |       0      |
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

        binomial_trees: dict[str, dict[float, dict[float, float]]] = {}
        binomial_trees_statistics: dict[str, dict[float, dict[str, float]]] = {
            "Up Movement": {},
            "Down Movement": {},
            "Risk Neutral Probability": {},
        }

        iterator = (
            tqdm(strike_prices_per_ticker.items(), desc="Calculating Binomial Trees")
            if ENABLE_TQDM
            else strike_prices_per_ticker.items()
        )
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in iterator:
            binomial_trees[ticker] = {}

            for strike_price in strike_prices:
                binomial_trees[ticker][strike_price] = {}
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

        binomial_trees_df = binomial_trees_df.round(
            rounding if rounding else self._rounding
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

        Returns:
            pd.DataFrame: Simulated Stock Price values containing the tickers and movements as the index and the
            timesteps as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "ASML"], api_key=API_KEY)

        stock_price_simulation = toolkit.options.get_stock_price_simulation(
            start_date='2020-06-22', timesteps=4
        )

        stock_price_simulation.loc['AMZN']
        ```

        Which returns:

        | Movement   |   2020-06-22 |   2020-09-21 |   2020-12-21 |   2021-03-22 |   2021-06-22 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUU       |       135.69 |      160.047 |     188.776  |     222.663  |     262.632  |
        | UUUD       |       135.69 |      160.047 |     188.776  |     222.663  |     188.776  |
        | UUDU       |       135.69 |      160.047 |     188.776  |     160.047  |     188.776  |
        | UUDD       |       135.69 |      160.047 |     188.776  |     160.047  |     135.69   |
        | UDUU       |       135.69 |      160.047 |     135.69   |     160.047  |     188.776  |
        | UDUD       |       135.69 |      160.047 |     135.69   |     160.047  |     135.69   |
        | UDDU       |       135.69 |      160.047 |     135.69   |     115.04   |     135.69   |
        | UDDD       |       135.69 |      160.047 |     135.69   |     115.04   |      97.5323 |
        | DUUU       |       135.69 |      115.04  |     135.69   |     160.047  |     188.776  |
        | DUUD       |       135.69 |      115.04  |     135.69   |     160.047  |     135.69   |
        | DUDU       |       135.69 |      115.04  |     135.69   |     115.04   |     135.69   |
        | DUDD       |       135.69 |      115.04  |     135.69   |     115.04   |      97.5323 |
        | DDUU       |       135.69 |      115.04  |      97.5323 |     115.04   |     135.69   |
        | DDUD       |       135.69 |      115.04  |      97.5323 |     115.04   |      97.5323 |
        | DDDU       |       135.69 |      115.04  |      97.5323 |      82.6891 |      97.5323 |
        | DDDD       |       135.69 |      115.04  |      97.5323 |      82.6891 |      70.1049 |
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

        stock_price_simulation: dict[str, dict[float, dict[float, float]]] = {}
        stock_price_statistics: dict[str, dict[float, dict[str, float]]] = {
            "Up Movement": {},
            "Down Movement": {},
        }

        iterator = (
            tqdm(self._tickers, desc="Simulating Stock Prices")
            if ENABLE_TQDM
            else self._tickers
        )

        for ticker in iterator:
            stock_price_simulation[ticker] = {}

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

        stock_price_simulation_df = stock_price_simulation_df.round(
            rounding if rounding else self._rounding
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
    ):
        """
        Calculate all Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        **First Order Greeks:**

        - Delta: measures the rate of change of the theoretical option value with respect to changes in the underlying
        asset's price.
        - Dual Delta: the actual probability of an option finishing in the money which is the first derivative
        of option price with respect to strike.
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

        Returns:
            pd.DataFrame: the greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["TSLA", "MU"], api_key="FINANCIAL_MODELING_PREP_KEY")

        all_greeks = toolkit.options.collect_all_greeks(start_date='2024-01-03')

        all_greeks.loc['TSLA', '2024-01-04']
        ```

        Which returns:

        |   Strike Price |   Delta |   Dual Delta |   Vega |   Theta |    Rho |   Epsilon |   Lambda |   Gamma |   Dual Gamma |   Vanna |    Charm |   Vomma |    Vera |      Veta |     PD |   Speed |   Zomma |   Color |   Ultima |
        |---------------:|--------:|-------------:|-------:|--------:|-------:|----------:|---------:|--------:|-------------:|--------:|---------:|--------:|--------:|----------:|-------:|--------:|--------:|--------:|---------:|
        |            215 |  0.9998 |      -0.9997 | 0.0001 | -0.0254 | 0.0059 |   -0.6532 |   0.1016 |  0.0001 |       0.0001 | -0.0044 |   0.4426 |  0.1942 | -0.0029 |   77.6496 | 0.0001 | -0.0001 |  0.0021 |  0.209  |   0.0336 |
        |            220 |  0.9973 |      -0.9969 | 0.001  | -0.0526 | 0.006  |   -0.6515 |   0.1287 |  0.0012 |       0.0014 | -0.0414 |   4.1955 |  1.4351 | -0.0273 |  600.92   | 0.0014 | -0.0005 |  0.0144 |  1.4569 |   0.1196 |
        |            225 |  0.9777 |      -0.976  | 0.0066 | -0.2079 | 0.006  |   -0.6387 |   0.1723 |  0.0076 |       0.0086 | -0.1884 |  19.0888 |  4.7244 | -0.1249 | 2187.89   | 0.0086 | -0.0022 |  0.0407 |  4.1228 |   0.0829 |
        |            230 |  0.8953 |      -0.8898 | 0.0226 | -0.6528 | 0.0056 |   -0.5849 |   0.2419 |  0.0261 |       0.028  | -0.3993 |  40.3564 |  6.2557 | -0.267  | 3816.31   | 0.028  | -0.0048 |  0.0253 |  2.5239 |  -0.1641 |
        |            235 |  0.6978 |      -0.6874 | 0.0435 | -1.2304 | 0.0044 |   -0.4558 |   0.3442 |  0.0502 |       0.0516 | -0.306  |  30.653  |  1.9785 | -0.2119 | 3623.7    | 0.0516 | -0.0039 | -0.0672 | -6.8719 |  -0.0977 |
        |            240 |  0.4192 |      -0.4078 | 0.0488 | -1.3691 | 0.0027 |   -0.2739 |   0.4789 |  0.0562 |       0.0555 |  0.1634 | -17.1438 |  0.4159 |  0.0934 | 3407.79   | 0.0555 |  0.0014 | -0.096  | -9.7512 |  -0.0222 |
        |            245 |  0.1812 |      -0.1736 | 0.0329 | -0.9207 | 0.0012 |   -0.1184 |   0.6396 |  0.0379 |       0.0359 |  0.4445 | -45.5549 |  5.0536 |  0.2814 | 4080.87   | 0.0359 |  0.0048 | -0.0098 | -0.9474 |  -0.1945 |
        |            250 |  0.0544 |      -0.0513 | 0.0138 | -0.3848 | 0.0004 |   -0.0355 |   0.8183 |  0.0159 |       0.0144 |  0.3232 | -33.01   |  6.468  |  0.2073 | 3328.37   | 0.0144 |  0.0036 |  0.0461 |  4.7176 |  -0.0443 |
        |            255 |  0.0112 |      -0.0104 | 0.0037 | -0.1028 | 0.0001 |   -0.0073 |   1.0084 |  0.0042 |       0.0037 |  0.1223 | -12.477  |  3.4845 |  0.0789 | 1542.52   | 0.0037 |  0.0014 |  0.0325 |  3.3216 |   0.1424 |
        |            260 |  0.0016 |      -0.0015 | 0.0006 | -0.018  | 0      |   -0.001  |   1.205  |  0.0007 |       0.0006 |  0.0276 |  -2.8148 |  1.0161 |  0.0179 |  421.028  | 0.0006 |  0.0003 |  0.0104 |  1.0578 |   0.1054 |
        |            265 |  0.0002 |      -0.0001 | 0.0001 | -0.0021 | 0      |   -0.0001 |   1.4049 |  0.0001 |       0.0001 |  0.004  |  -0.4041 |  0.1783 |  0.0026 |   71.3544 | 0.0001 |  0      |  0.0019 |  0.1933 |   0.0322 |
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
    ):
        """
        Calculate the first order Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        - Delta: measures the rate of change of the theoretical option value with respect to changes in the underlying
        asset's price.
        - Dual Delta: the actual probability of an option finishing in the money which is the first derivative
        of option price with respect to strike.
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

        Returns:
            pd.DataFrame: the first order greek values containing the tickers and strike prices as the index
            and the time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.collect_first_order_greeks()
        ```
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
        - Call Option Delta = N(d1)
        - Put Option Delta = N(d1) — 1

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Delta can be interpreted as follows:

        - For call options, Delta is positive, indicating that the option price tends to move in the same direction as the
        underlying asset's price.
        - For put options, Delta is negative, indicating that the option price tends to move in the opposite direction to the
        underlying asset's price.

        Note that the delta of a call option is always between 0 and 1, while the delta of a put option
        is always between —1 and 0.

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
            put_option (bool, optional): Whether to calculate the put option delta. Defaults to False which means
            it will calculate the call option delta.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: the delta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_delta()
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

            for strike_price in strike_prices:
                delta[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    delta[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_delta(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        delta_df = helpers.create_greek_dataframe(
            greek_dictionary=delta,
            start_date=start_date,
        )

        delta_df = delta_df.round(rounding if rounding else self._rounding)

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
        - Call Dual Delta = e^(—r * t) * N(d2)
        - Put Dual Delta = e^(—r * t) * N(—d2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Dual Delta can be interpreted as the probability of an option finishing in the money. For example, if the
        Dual Delta is 0.5, then the probability of the option finishing in the money is 50%.

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

        Returns:
            pd.DataFrame: the dual delta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_dual_delta()
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

            for strike_price in strike_prices:
                dual_delta[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    dual_delta[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_dual_delta(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        dual_delta_df = helpers.create_greek_dataframe(
            greek_dictionary=dual_delta,
            start_date=start_date,
        )

        dual_delta_df = dual_delta_df.round(rounding if rounding else self._rounding)

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
        - Vega = S * e^(—q * t) * N'(d1) * sqrt(t)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Vega can be interpreted as follows:

        - If Vega is positive, it indicates that the option value will increase as the volatility increases,
        and vice versa.
        - If Vega is negative, it implies that the option value will decrease as the volatility increases,
        and vice versa.

        Note that the vega of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the vega values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_vega()
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

            for strike_price in strike_prices:
                vega[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    vega[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_vega(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        vega_df = helpers.create_greek_dataframe(
            greek_dictionary=vega,
            start_date=start_date,
        )

        vega_df = vega_df.round(rounding if rounding else self._rounding)

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
        - Call Theta = e^(—q * t) * (stock_price * N'(d1) * σ) / (2 * sqrt(t)) — r * K * e^(—r * t) * N(d2)
        + q * S * e^(—q * t) * N(d1)
        - Put Theta = e^(—q * t) * (stock_price * N'(d1) * σ) / (2 * sqrt(t)) + r * K * e^(—r * t) * N(d2)
        — q * S * e^(—q * t) * N(d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Theta can be interpreted as follows:

        - If Theta is positive, it indicates that the option value will increase as the time to expiration increases,
        and vice versa.
        - If Theta is negative, it implies that the option value will decrease as the time to expiration increases,
        and vice versa.

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

        Returns:
            pd.DataFrame: the theta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_theta()
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

            for strike_price in strike_prices:
                theta[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    theta[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_theta(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        theta_df = helpers.create_greek_dataframe(
            greek_dictionary=theta,
            start_date=start_date,
        )

        theta_df = theta_df.round(rounding if rounding else self._rounding)

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

        Rho is typically expressed as the amount of money, per share of the underlying, that the value of the option
        will gain or lose as the risk—free interest rate rises or falls by 1.0% per annum (100 basis points).

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

        Returns:
            pd.DataFrame: the rho values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_rho()
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

            for strike_price in strike_prices:
                rho[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    rho[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_rho(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        rho_df = helpers.create_greek_dataframe(
            greek_dictionary=rho,
            start_date=start_date,
        )

        rho_df = rho_df.round(rounding if rounding else self._rounding)

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
        - Call Epsilon = —S * t * e^(—q * t) * N'(d1)
        - Put Epislon = S * t * e^(—q * t) * N'(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Epsilon can be interpreted as follows:

        - If Epislon is positive, it indicates that the option value will increase as the dividend yield increases,
        and vice versa.
        - If Epislon is negative, it implies that the option value will decrease as the dividend yield increases,
        and vice versa.

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

        Returns:
            pd.DataFrame: the epsilon values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_epsilon()
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

            for strike_price in strike_prices:
                epsilon[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    epsilon[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_epsilon(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        epsilon_df = helpers.create_greek_dataframe(
            greek_dictionary=epsilon,
            start_date=start_date,
        )

        epsilon_df = epsilon_df.round(rounding if rounding else self._rounding)

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
        - Delta = N(d1)
        - Call Option = N'(d1) / (S * σ * sqrt(t))
        - Put Option = N'(d1) / (S * σ * sqrt(t))
        - Lambda = Delta * (Stock Price / Call Option or Put Option)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Lambda can be interpreted as follows:

        - If Lambda is positive, it indicates that the option value will increase as the underlying price increases,
        and vice versa.
        - If Lambda is negative, it implies that the option value will decrease as the underlying price increases,
        and vice versa.

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

        Returns:
            pd.DataFrame: the lambda values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_lambda()
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

            for strike_price in strike_prices:
                lambda_greek[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    lambda_greek[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_lambda(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        lambda_df = helpers.create_greek_dataframe(
            greek_dictionary=lambda_greek,
            start_date=start_date,
        )

        lambda_df = lambda_df.round(rounding if rounding else self._rounding)

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
    ):
        """
        Calculate the second order Greeks of an option based on the Black Scholes Model. This will return the following Greeks
        per Strike Price and Expiration Date:

        - Gamma: measures the rate of change in the delta with respect to changes in the underlying price. Gamma is
        the second derivative of the value function with respect to the underlying price.
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

        Returns:
            pd.DataFrame: the second order greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.collect_second_order_greeks()
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
        - Gamma = N'(d1) / (S * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Gamma can be interpreted as follows:

        - If Gamma is high, it indicates that the option's Delta is highly sensitive to changes in the underlying
        asset's price. The option's Delta will change more significantly with small movements in the stock price.
        - If Gamma is low, it suggests that the option's Delta is relatively insensitive to changes in the
        underlying asset's price. The option's Delta changes more gradually with movements in the stock price.

        Note that the gamma of a call option and put option are equal to each other.

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
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        Returns:
            pd.DataFrame: the gamma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_gamma()
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

            for strike_price in strike_prices:
                gamma[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    gamma[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_gamma(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        gamma_df = helpers.create_greek_dataframe(
            greek_dictionary=gamma,
            start_date=start_date,
        )

        gamma_df = gamma_df.round(rounding if rounding else self._rounding)

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
        - d2 = d1 — σ * sqrt(t)
        - Dual Gamma = e^(—r * t) * N'(d2) / (S * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        Note that the dual gamma of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the dual gamma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_dual_gamma()
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

            for strike_price in strike_prices:
                dual_gamma[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    dual_gamma[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_dual_gamma(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        dual_gamma_df = helpers.create_greek_dataframe(
            greek_dictionary=dual_gamma,
            start_date=start_date,
        )

        dual_gamma_df = dual_gamma_df.round(rounding if rounding else self._rounding)

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

        Returns:
            pd.DataFrame: the vanna values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_vanna()
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

            for strike_price in strike_prices:
                vanna[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    vanna[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_vanna(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        vanna_df = helpers.create_greek_dataframe(
            greek_dictionary=vanna,
            start_date=start_date,
        )

        vanna_df = vanna_df.round(rounding if rounding else self._rounding)

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
        - Call Charm = q * e^(—q * t) * N'(d1) — e^(—q * t) * N(d1) * (2 * (r — q) * t — d2 * σ * sqrt(t)) / (2 * t * σ * sqrt(t))
        - Put Charm = —q * e^(—q * t) * N'(—d1) — e^(—q * t) * N(d1) * (2 * (r — q) * t — d2 * σ * sqrt(t)) / (2 * t * σ * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Charm can be interpreted as follows:

        - If Charm is positive, it suggests that the option's Delta is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in the underlying asset's price as time passes.
        - If Charm is negative, it indicates that the option's Delta is becoming more negative over time. The
        option is losing sensitivity to changes in the underlying asset's price as time passes.

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

        Returns:
            pd.DataFrame: the charm values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_charm()
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

            for strike_price in strike_prices:
                charm[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    charm[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_charm(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                            put_option=put_option,
                        )
                    )

        charm_df = helpers.create_greek_dataframe(
            greek_dictionary=charm,
            start_date=start_date,
        )

        charm_df = charm_df.round(rounding if rounding else self._rounding)

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

        Returns:
            pd.DataFrame: the vomma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_vomma()
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

            for strike_price in strike_prices:
                vomma[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    vomma[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_vomma(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        vomma_df = helpers.create_greek_dataframe(
            greek_dictionary=vomma,
            start_date=start_date,
        )

        vomma_df = vomma_df.round(rounding if rounding else self._rounding)

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
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Vera can be interpreted as follows:

        - If Vera is positive, it indicates that the option's Rho is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in the risk free rate as time passes.
        - If Vera is negative, it suggests that the option's Rho is becoming more negative over time. The
        option is losing sensitivity to changes in the risk free rate as time passes.

        Note that the vera of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the vera values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_vera()
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

            for strike_price in strike_prices:
                vera[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    vera[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_vera(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        vera_df = helpers.create_greek_dataframe(
            greek_dictionary=vera,
            start_date=start_date,
        )

        vera_df = vera_df.round(rounding if rounding else self._rounding)

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
        - Veta = —S * e^(—q * t) * N'(d1) * sqrt(t) * (q + ((r — q) * d1) / (σ * sqrt(t)) — (1 + d1 * d2) / (2 * t)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        It is common practice to divide the mathematical result of veta by 100 times the number of days per year to
        reduce the value to the percentage change in vega per one day. This is also done here.

        The Veta can be interpreted as follows:

        - If Veta is positive, it indicates that the option's Vega is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in implied volatility as time passes.
        - If Veta is negative, it suggests that the option's Vega is becoming more negative over time. The
        option is losing sensitivity to changes in implied volatility as time passes.

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

        Returns:
            pd.DataFrame: the veta values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_veta()
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

            for strike_price in strike_prices:
                veta[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    veta[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_veta(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        veta_df = helpers.create_greek_dataframe(
            greek_dictionary=veta,
            start_date=start_date,
        )

        veta_df = veta_df.round(rounding if rounding else self._rounding)

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
    ):
        """
        Calculate the partial derivative of an option based on the Black Scholes Model. The Black Scholes Model
        is a mathematical model used to estimate the price of European—style options. The partial derivative is
        the rate of change of the option price with respect to the strike price.

        The partial derivative is used in the Breeden-Litzenberger theorem is used for risk-neutral valuation and
        was developed by Fischer Black and Robert Litzenberger in 1978. The theorem states that the price of any
        derivative security can be calculated by finding the expected value of the derivative under a risk-neutral
        measure. The theorem is based on the Black-Scholes model and the assumption that the underlying asset
        follows a lognormal distribution. See the paper: https://www.jstor.org/stable/2352653

        The formula is as follows:

        - Partial Derivative (PD) = e^(—r * t) * (1 / K) * (1 sqrt(2 * pi * volatility ** 2 * t)) *
        e^(—(1 / (2 * volatility ** 2 * t)) * (ln(S / K) — ((r — q) — (0.5 * volatility ** 2)) * t) ** 2

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

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

        Returns:
            pd.DataFrame: the partial derivative values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_partial_derivative()
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

            for strike_price in strike_prices:
                partial_derivative[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    partial_derivative[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_second_order_partial_derivative(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        partial_derivative_df = helpers.create_greek_dataframe(
            greek_dictionary=partial_derivative,
            start_date=start_date,
        )

        partial_derivative_df = partial_derivative_df.round(
            rounding if rounding else self._rounding
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

        Returns:
            pd.DataFrame: the third order greeks values containing the tickers and strike prices as the index and the
            time to expiration and greeks as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MU", "AMZN"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.collect_third_order_greeks()
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
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Speed can be interpreted as follows:

        - If Speed is positive, it indicates that the option's Gamma is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in the underlying price as time passes.
        - If Speed is negative, it suggests that the option's Gamma is becoming more negative over time. The
        option is losing sensitivity to changes in the underlying price as time passes.

        Note that the speed of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the speed values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_speed()
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

            for strike_price in strike_prices:
                speed[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    speed[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_speed(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        speed_df = helpers.create_greek_dataframe(
            greek_dictionary=speed,
            start_date=start_date,
        )

        speed_df = speed_df.round(rounding if rounding else self._rounding)

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
        - Zomma = e^(—q * t) * (N'(d1) / (d1 * d2 — 1)) / (S * σ **2 * sqrt(t))

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Zomma can be interpreted as follows:

        - If Zomma is positive, it indicates that the option's Gamma is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in volatility as time passes.
        - If Zomma is negative, it suggests that the option's Gamma is becoming more negative over time. The
        option is losing sensitivity to changes in volatility as time passes.

        Note that the zomma of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the zomma values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_zomma()
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

            for strike_price in strike_prices:
                zomma[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    zomma[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_zomma(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        zomma_df = helpers.create_greek_dataframe(
            greek_dictionary=zomma,
            start_date=start_date,
        )

        zomma_df = zomma_df.round(rounding if rounding else self._rounding)

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
        - Color = —e^(—q * t) * (N'(d1) / (2 * S * t * σ * sqrt(t))) * (2 * q * t + 1 + ((2 * (r — q) * t — d2 * σ * sqrt(t)) / (σ * sqrt(t))) * d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Color can be interpreted as follows:

        - If Color is positive, it indicates that the option's Gamma is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in time to expiration as time passes.
        - If Color is negative, it suggests that the option's Gamma is becoming more negative over time. The
        option is losing sensitivity to changes in time to expiration as time passes.

        Note that the color of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the color values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_color()
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

            for strike_price in strike_prices:
                color[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    color[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_color(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        color_df = helpers.create_greek_dataframe(
            greek_dictionary=color,
            start_date=start_date,
        )

        color_df = color_df.round(rounding if rounding else self._rounding)

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
        - Ultima = (—vega / volatility ** 2) * (d1 * d2 * (1 — d1 * d2) + d1 ** 2 + d2 ** 2)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration, N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        The Ultima can be interpreted as follows:

        - If Ultima is positive, it indicates that the option's vomma is becoming more positive over time. In
        other words, the option is gaining sensitivity to changes in volatility as time passes.
        - If Ultima is negative, it suggests that the option's vomma is becoming more negative over time. The
        option is losing sensitivity to changes in volatility as time passes.

        Note that the ultima of a call option and put option are equal to each other.

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

        Returns:
            pd.DataFrame: the ultima values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "ASML"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_ultima()
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

            for strike_price in strike_prices:
                ultima[ticker][strike_price] = {}

                for time_to_expiration in time_to_expiration_list:
                    ultima[ticker][strike_price][time_to_expiration] = (
                        greeks_model.get_ultima(
                            stock_price=stock_price.loc[ticker],
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            risk_free_rate=risk_free_rate,
                            volatility=volatility.loc[ticker],
                            dividend_yield=dividend_yield_value[ticker],
                        )
                    )

        ultima_df = helpers.create_greek_dataframe(
            greek_dictionary=ultima,
            start_date=start_date,
        )

        ultima_df = ultima_df.round(rounding if rounding else self._rounding)

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
