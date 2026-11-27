"""Option Greeks Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.options import helpers
from financetoolkit.options.greeks import greeks_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_standardization,
)

# ruff: noqa: E501

logger = logger_model.get_logger()


@filter_shown_columns
@track_sources
class GreeksOptions:
    """
    The first, second and third order Greeks of the Black-Scholes model.
    """

    @handle_errors
    @logger_model.log_each_message_once()
    def collect_all_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate all Greeks of an option based on the Black Scholes Model: how the option value responds to the
        price of the underlying asset, the strike price, the volatility, time, the risk free rate and the dividend
        yield.

        This will return the following Greeks per Strike Price and Expiration Date:

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            180 |  1      |      -0.9999 | 0      | -0.0191 | 0.4931 |   -0.6533 |   4.0782 |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
        |            185 |  1      |      -0.9999 | 0      | -0.0196 | 0.5068 |   -0.6533 |   4.4595 |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
        |            190 |  1      |      -0.9999 | 0      | -0.0201 | 0.5205 |   -0.6533 |   4.9195 |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
        |            195 |  1      |      -0.9999 | 0      | -0.0207 | 0.5342 |   -0.6533 |   5.4853 |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
        |            200 |  1      |      -0.9999 | 0      | -0.0212 | 0.5479 |   -0.6533 |   6.1981 |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0.0001 |
        |            205 |  1      |      -0.9999 | 0      | -0.0217 | 0.5616 |   -0.6533 |   7.1239 |  0      |       0      |  0      |   0.0002 |  0.0001 |  0      |  0      | 0      |  0      |  0      | -0.0002 |   0.0071 |
        |            210 |  1      |      -0.9999 | 0      | -0.0223 | 0.5753 |   -0.6533 |   8.3748 |  0      |       0      | -0.0001 |   0.0124 |  0.0071 | -0.0001 |  0      | 0      |  0      |  0.0001 | -0.0079 |   0.2259 |
        |            215 |  0.9999 |      -0.9998 | 0.0001 | -0.0243 | 0.5889 |   -0.6532 |  10.1573 |  0.0001 |       0.0001 | -0.0031 |   0.3046 |  0.1422 | -0.002  | -0.0004 | 0.0001 |  0      |  0.0016 | -0.1539 |   2.7473 |
        |            220 |  0.9979 |      -0.9976 | 0.0008 | -0.0454 | 0.6013 |   -0.6519 |  12.8781 |  0.001  |       0.0011 | -0.0345 |   3.3878 |  1.2328 | -0.0227 | -0.0037 | 0.0011 | -0.0004 |  0.0129 | -1.2629 |  11.7206 |
        |            225 |  0.9808 |      -0.9794 | 0.0058 | -0.1807 | 0.6037 |   -0.6408 |  17.2971 |  0.0069 |       0.0078 | -0.1766 |  17.3413 |  4.5652 | -0.1169 | -0.0152 | 0.0078 | -0.0022 |  0.0414 | -4.0651 |  10.4164 |
        |            230 |  0.9022 |      -0.8971 | 0.0216 | -0.6039 | 0.5653 |   -0.5894 |  24.4771 |  0.0256 |       0.0275 | -0.4047 |  39.6564 |  6.5366 | -0.2703 | -0.0283 | 0.0275 | -0.005  |  0.0303 | -2.9376 | -16.4877 |
        |            235 |  0.7031 |      -0.6932 | 0.0432 | -1.1837 | 0.4463 |   -0.4593 |  35.2284 |  0.0513 |       0.0528 | -0.3236 |  31.4261 |  2.1544 | -0.2232 | -0.0273 | 0.0528 | -0.0043 | -0.0694 |  6.8815 | -10.9032 |
        |            240 |  0.4163 |      -0.4053 | 0.0487 | -1.3253 | 0.2665 |   -0.272  |  49.5403 |  0.0579 |       0.0571 |  0.173  | -17.591  |  0.4563 |  0.0997 | -0.0256 | 0.0571 |  0.0016 | -0.1017 | 10.0103 |  -2.5052 |
        |            245 |  0.1734 |      -0.1663 | 0.032  | -0.8683 | 0.1116 |   -0.1133 |  66.7064 |  0.038  |       0.036  |  0.4598 | -45.6743 |  5.398  |  0.2916 | -0.0307 | 0.036  |  0.0051 | -0.0062 |  0.5661 | -20.8824 |
        |            250 |  0.0489 |      -0.0461 | 0.0127 | -0.343  | 0.0316 |   -0.032  |  85.8355 |  0.015  |       0.0137 |  0.316  | -31.2885 |  6.5283 |  0.203  | -0.024  | 0.0137 |  0.0036 |  0.0497 | -4.9376 |  -2.5834 |
        |            255 |  0.0092 |      -0.0086 | 0.0031 | -0.0841 | 0.006  |   -0.006  | 106.18   |  0.0037 |       0.0032 |  0.1098 | -10.8605 |  3.2292 |  0.0709 | -0.0103 | 0.0032 |  0.0013 |  0.0315 | -3.1214 |  15.6375 |
        |            260 |  0.0012 |      -0.0011 | 0.0005 | -0.0132 | 0.0008 |   -0.0008 | 127.21   |  0.0006 |       0.0005 |  0.0222 |  -2.1897 |  0.8415 |  0.0143 | -0.0025 | 0.0005 |  0.0003 |  0.0089 | -0.8827 |   9.8817 |
        |            265 |  0.0001 |      -0.0001 | 0      | -0.0014 | 0.0001 |   -0.0001 | 148.577  |  0.0001 |       0      |  0.0028 |  -0.2738 |  0.1286 |  0.0018 | -0.0004 | 0      |  0      |  0.0014 | -0.1401 |   2.5989 |
        |            270 |  0      |       0      | 0      | -0.0001 | 0      |    0      | 170.056  |  0      |       0      |  0.0002 |  -0.0222 |  0.0123 |  0.0001 |  0      | 0      |  0      |  0.0001 | -0.0137 |   0.37   |
        |            275 |  0      |       0      | 0      |  0      | 0      |    0      | 191.505  |  0      |       0      |  0      |  -0.0012 |  0.0008 |  0      |  0      | 0      |  0      |  0      | -0.0009 |   0.0317 |
        |            280 |  0      |       0      | 0      |  0      | 0      |    0      | 212.831  |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0.0017 |
        |            285 |  0      |       0      | 0      |  0      | 0      |    0      | 233.975  |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0.0001 |
        |            290 |  0      |       0      | 0      |  0      | 0      |    0      | 254.901  |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
        |            295 |  0      |       0      | 0      |  0      | 0      |    0      | 275.584  |  0      |       0      |  0      |   0      |  0      |  0      |  0      | 0      |  0      |  0      |  0      |   0      |
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

        if first_order_greeks.empty:
            # The first order greeks have logged why none could be calculated; the others share the reason.
            return first_order_greeks

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

    @handle_errors
    @logger_model.log_each_message_once()
    def collect_first_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the first order Greeks of an option based on the Black Scholes Model: how the option value
        responds to the price of the underlying asset, the strike price, the volatility, time, the risk free rate
        and the dividend yield.

        This will return the following Greeks per Strike Price and Expiration Date:

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            290 |                                 0.0001 |                                     -0.0001 |                                     0 |                                -0.0007 |                               0.0001 |                                  -0.0001 |                                 250.96  |                                 0.0038 |                                     -0.0035 |                                0.0023 |                                -0.0185 |                               0.0056 |                                  -0.0056 |                                 136.168 |                                 0.0149 |                                     -0.0138 |                                0.0093 |                                -0.0506 |                               0.033  |                                  -0.0333 |                                 96.9339 |                                 0.0304 |                                     -0.0282 |                                0.0196 |                                -0.0803 |                               0.0895 |                                  -0.0907 |                                 76.8328 |                                 0.0475 |                                     -0.0438 |                                0.0315 |                                -0.1034 |                               0.174  |                                  -0.1768 |                                 64.4894 |                                 0.0644 |                                     -0.0593 |                                0.0439 |                                -0.1204 |                               0.2829 |                                  -0.288  |                                 56.0783 |                                 0.0807 |                                     -0.0741 |                                0.0563 |                                -0.1327 |                               0.4123 |                                  -0.4208 |                                 49.9447 |                                 0.096  |                                     -0.088  |                                0.0685 |                                -0.1415 |                               0.5593 |                                  -0.5719 |                                 45.2537 |                                 0.1102 |                                     -0.1008 |                                0.0804 |                                -0.1478 |                               0.7211 |                                  -0.7389 |                                 41.5368 |                                 0.1235 |                                     -0.1127 |                                0.0918 |                                -0.1522 |                               0.8957 |                                  -0.9195 |                                 38.5103 |                                 0.1357 |                                     -0.1237 |                                0.1029 |                                -0.1552 |                               1.0813 |                                  -1.1122 |                                 35.9922 |                                 0.1472 |                                     -0.1339 |                                0.1135 |                                -0.1573 |                               1.2766 |                                  -1.3154 |                                 33.8598 |                                 0.1578 |                                     -0.1433 |                                0.1237 |                                -0.1585 |                               1.4804 |                                  -1.5281 |                                 32.0276 |                                 0.1677 |                                     -0.1521 |                                0.1336 |                                -0.1592 |                               1.6917 |                                  -1.7492 |                                 30.4338 |                                 0.177  |                                     -0.1602 |                                0.1431 |                                -0.1594 |                               1.9098 |                                  -1.978  |                                 29.0328 |                                 0.1858 |                                     -0.1679 |                                0.1523 |                                -0.1593 |                               2.134  |                                  -2.2136 |                                 27.7901 |                                 0.1939 |                                     -0.175  |                                0.1612 |                                -0.1589 |                               2.3636 |                                  -2.4557 |                                 26.6792 |                                 0.2017 |                                     -0.1817 |                                0.1698 |                                -0.1583 |                               2.5982 |                                  -2.7035 |                                 25.679  |                                 0.2089 |                                     -0.188  |                                0.1782 |                                -0.1576 |                               2.8374 |                                  -2.9568 |                                 24.7732 |                                 0.2158 |                                     -0.1939 |                                0.1863 |                                -0.1568 |                               3.0808 |                                  -3.215  |                                 23.9482 |                                 0.2224 |                                     -0.1995 |                                0.1942 |                                -0.1558 |                               3.3279 |                                  -3.4779 |                                 23.1931 |                                 0.2286 |                                     -0.2047 |                                0.2019 |                                -0.1548 |                               3.5787 |                                  -3.7451 |                                 22.499  |                                 0.2345 |                                     -0.2097 |                                0.2094 |                                -0.1538 |                               3.8327 |                                  -4.0164 |                                 21.8583 |                                 0.2401 |                                     -0.2145 |                                0.2167 |                                -0.1527 |                               4.0897 |                                  -4.2916 |                                 21.2648 |                                 0.2454 |                                     -0.219  |                                0.2239 |                                -0.1516 |                               4.3497 |                                  -4.5703 |                                 20.7132 |                                 0.2506 |                                     -0.2233 |                                0.2308 |                                -0.1505 |                               4.6122 |                                  -4.8524 |                                 20.1989 |                                 0.2555 |                                     -0.2274 |                                0.2376 |                                -0.1494 |                               4.8773 |                                  -5.1378 |                                 19.7181 |                                 0.2602 |                                     -0.2313 |                                0.2443 |                                -0.1483 |                               5.1447 |                                  -5.4263 |                                 19.2673 |                                 0.2647 |                                     -0.235  |                                0.2508 |                                -0.1471 |                               5.4142 |                                  -5.7177 |                                 18.8438 |
        |            295 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 305.85  |                                 0.0004 |                                     -0.0003 |                                0.0003 |                                -0.0021 |                               0.0005 |                                  -0.0005 |                                 162.339 |                                 0.0029 |                                     -0.0027 |                                0.0022 |                                -0.0121 |                               0.0065 |                                  -0.0066 |                                113.802  |                                 0.0087 |                                     -0.0079 |                                0.0067 |                                -0.0275 |                               0.0256 |                                  -0.0259 |                                 89.1597 |                                 0.017  |                                     -0.0154 |                                0.0134 |                                -0.0439 |                               0.0624 |                                  -0.0633 |                                 74.1468 |                                 0.0268 |                                     -0.0243 |                                0.0216 |                                -0.0591 |                               0.118  |                                  -0.1199 |                                 63.9872 |                                 0.0374 |                                     -0.0339 |                                0.0307 |                                -0.0722 |                               0.1918 |                                  -0.1952 |                                 56.624  |                                 0.0484 |                                     -0.0437 |                                0.0404 |                                -0.0832 |                               0.2825 |                                  -0.2882 |                                 51.0232 |                                 0.0592 |                                     -0.0534 |                                0.0504 |                                -0.0923 |                               0.3886 |                                  -0.3972 |                                 46.6075 |                                 0.0699 |                                     -0.0629 |                                0.0604 |                                -0.0997 |                               0.5086 |                                  -0.5207 |                                 43.0281 |                                 0.0803 |                                     -0.0721 |                                0.0703 |                                -0.1058 |                               0.6411 |                                  -0.6576 |                                 40.0622 |                                 0.0902 |                                     -0.0809 |                                0.0802 |                                -0.1108 |                               0.7849 |                                  -0.8064 |                                 37.5601 |                                 0.0998 |                                     -0.0893 |                                0.0899 |                                -0.1148 |                               0.9388 |                                  -0.9661 |                                 35.4176 |                                 0.1089 |                                     -0.0974 |                                0.0994 |                                -0.118  |                               1.1019 |                                  -1.1357 |                                 33.56   |                                 0.1177 |                                     -0.105  |                                0.1087 |                                -0.1206 |                               1.2733 |                                  -1.3145 |                                 31.9321 |                                 0.126  |                                     -0.1123 |                                0.1178 |                                -0.1227 |                               1.4524 |                                  -1.5016 |                                 30.4922 |                                 0.134  |                                     -0.1192 |                                0.1267 |                                -0.1244 |                               1.6384 |                                  -1.6964 |                                 29.2084 |                                 0.1416 |                                     -0.1258 |                                0.1354 |                                -0.1257 |                               1.8307 |                                  -1.8984 |                                 28.0555 |                                 0.1489 |                                     -0.1321 |                                0.1439 |                                -0.1267 |                               2.029  |                                  -2.107  |                                 27.0137 |                                 0.1559 |                                     -0.1381 |                                0.1522 |                                -0.1275 |                               2.2327 |                                  -2.3218 |                                 26.067  |                                 0.1625 |                                     -0.1438 |                                0.1603 |                                -0.1281 |                               2.4414 |                                  -2.5423 |                                 25.2024 |                                 0.1689 |                                     -0.1493 |                                0.1682 |                                -0.1284 |                               2.6548 |                                  -2.7682 |                                 24.4092 |                                 0.1751 |                                     -0.1545 |                                0.176  |                                -0.1287 |                               2.8726 |                                  -2.9992 |                                 23.6785 |                                 0.181  |                                     -0.1595 |                                0.1835 |                                -0.1288 |                               3.0944 |                                  -3.235  |                                 23.0028 |                                 0.1866 |                                     -0.1643 |                                0.1909 |                                -0.1287 |                               3.32   |                                  -3.4753 |                                 22.3759 |                                 0.1921 |                                     -0.1689 |                                0.1982 |                                -0.1286 |                               3.5492 |                                  -3.7199 |                                 21.7924 |                                 0.1973 |                                     -0.1733 |                                0.2053 |                                -0.1285 |                               3.7817 |                                  -3.9685 |                                 21.2477 |                                 0.2024 |                                     -0.1775 |                                0.2123 |                                -0.1282 |                               4.0173 |                                  -4.2209 |                                 20.7378 |                                 0.2073 |                                     -0.1816 |                                0.2191 |                                -0.1279 |                               4.256  |                                  -4.477  |                                 20.2595 |
        |            300 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 361.004 |                                 0      |                                      0      |                                0      |                                -0.0002 |                               0      |                                   0      |                                 188.866 |                                 0.0004 |                                     -0.0004 |                                0.0004 |                                -0.0021 |                               0.001  |                                  -0.001  |                                130.973  |                                 0.002  |                                     -0.0018 |                                0.0018 |                                -0.0075 |                               0.006  |                                  -0.0061 |                                101.738  |                                 0.0052 |                                     -0.0046 |                                0.0048 |                                -0.0155 |                               0.0191 |                                  -0.0193 |                                 84.0149 |                                 0.0098 |                                     -0.0088 |                                0.0091 |                                -0.0249 |                               0.0432 |                                  -0.0438 |                                 72.0749 |                                 0.0156 |                                     -0.0139 |                                0.0147 |                                -0.0345 |                               0.0799 |                                  -0.0811 |                                 63.457  |                                 0.0222 |                                     -0.0197 |                                0.0213 |                                -0.0437 |                               0.1297 |                                  -0.132  |                                 56.9267 |                                 0.0293 |                                     -0.026  |                                0.0285 |                                -0.0521 |                               0.1926 |                                  -0.1964 |                                 51.7959 |                                 0.0368 |                                     -0.0326 |                                0.0362 |                                -0.0597 |                               0.268  |                                  -0.2738 |                                 47.6504 |                                 0.0444 |                                     -0.0393 |                                0.0442 |                                -0.0664 |                               0.3554 |                                  -0.3636 |                                 44.2256 |                                 0.052  |                                     -0.046  |                                0.0525 |                                -0.0723 |                               0.4539 |                                  -0.4652 |                                 41.3445 |                                 0.0597 |                                     -0.0527 |                                0.0609 |                                -0.0775 |                               0.5629 |                                  -0.5778 |                                 38.8839 |                                 0.0672 |                                     -0.0592 |                                0.0693 |                                -0.082  |                               0.6816 |                                  -0.7007 |                                 36.7558 |                                 0.0746 |                                     -0.0656 |                                0.0777 |                                -0.086  |                               0.8093 |                                  -0.8332 |                                 34.895  |                                 0.0818 |                                     -0.0719 |                                0.086  |                                -0.0894 |                               0.9455 |                                  -0.9748 |                                 33.2528 |                                 0.0888 |                                     -0.078  |                                0.0944 |                                -0.0924 |                               1.0894 |                                  -1.1248 |                                 31.7915 |                                 0.0957 |                                     -0.0839 |                                0.1026 |                                -0.0949 |                               1.2406 |                                  -1.2827 |                                 30.4819 |                                 0.1023 |                                     -0.0896 |                                0.1107 |                                -0.0972 |                               1.3986 |                                  -1.448  |                                 29.3006 |                                 0.1088 |                                     -0.0951 |                                0.1187 |                                -0.0991 |                               1.563  |                                  -1.6204 |                                 28.2291 |                                 0.115  |                                     -0.1004 |                                0.1266 |                                -0.1008 |                               1.7332 |                                  -1.7992 |                                 27.2521 |                                 0.1211 |                                     -0.1056 |                                0.1344 |                                -0.1022 |                               1.909  |                                  -1.9843 |                                 26.3573 |                                 0.127  |                                     -0.1106 |                                0.142  |                                -0.1035 |                               2.0901 |                                  -2.1753 |                                 25.5342 |                                 0.1327 |                                     -0.1154 |                                0.1496 |                                -0.1045 |                               2.276  |                                  -2.3717 |                                 24.7742 |                                 0.1382 |                                     -0.12   |                                0.157  |                                -0.1054 |                               2.4665 |                                  -2.5734 |                                 24.07   |                                 0.1436 |                                     -0.1245 |                                0.1643 |                                -0.1062 |                               2.6614 |                                  -2.7801 |                                 23.4155 |                                 0.1488 |                                     -0.1289 |                                0.1714 |                                -0.1069 |                               2.8604 |                                  -2.9916 |                                 22.8053 |                                 0.1538 |                                     -0.1331 |                                0.1785 |                                -0.1074 |                               3.0633 |                                  -3.2075 |                                 22.2349 |                                 0.1587 |                                     -0.1372 |                                0.1854 |                                -0.1078 |                               3.2698 |                                  -3.4278 |                                 21.7004 |
        |            305 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 416.007 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 215.505 |                                 0.0001 |                                      0      |                                0.0001 |                                -0.0003 |                               0.0001 |                                  -0.0001 |                                148.289  |                                 0.0004 |                                     -0.0003 |                                0.0004 |                                -0.0016 |                               0.0012 |                                  -0.0012 |                                114.455  |                                 0.0014 |                                     -0.0012 |                                0.0014 |                                -0.0046 |                               0.005  |                                  -0.005  |                                 94.0084 |                                 0.0032 |                                     -0.0028 |                                0.0033 |                                -0.0091 |                               0.0139 |                                  -0.0141 |                                 80.2749 |                                 0.0058 |                                     -0.0051 |                                0.0062 |                                -0.0146 |                               0.0299 |                                  -0.0303 |                                 70.3903 |                                 0.0092 |                                     -0.0081 |                                0.01   |                                -0.0206 |                               0.0542 |                                  -0.0551 |                                 62.92   |                                 0.0133 |                                     -0.0117 |                                0.0146 |                                -0.0267 |                               0.0879 |                                  -0.0895 |                                 57.0652 |                                 0.018  |                                     -0.0157 |                                0.0199 |                                -0.0327 |                               0.1313 |                                  -0.1338 |                                 52.3459 |                                 0.023  |                                     -0.0201 |                                0.0257 |                                -0.0385 |                               0.1844 |                                  -0.1883 |                                 48.4555 |                                 0.0283 |                                     -0.0247 |                                0.0319 |                                -0.0439 |                               0.2473 |                                  -0.2529 |                                 45.1895 |                                 0.0338 |                                     -0.0294 |                                0.0385 |                                -0.0489 |                               0.3195 |                                  -0.3273 |                                 42.4058 |                                 0.0394 |                                     -0.0343 |                                0.0453 |                                -0.0536 |                               0.4009 |                                  -0.4112 |                                 40.0026 |                                 0.0451 |                                     -0.0392 |                                0.0524 |                                -0.0578 |                               0.4909 |                                  -0.5042 |                                 37.9051 |                                 0.0509 |                                     -0.0441 |                                0.0595 |                                -0.0617 |                               0.5893 |                                  -0.6061 |                                 36.0571 |                                 0.0566 |                                     -0.049  |                                0.0667 |                                -0.0652 |                               0.6955 |                                  -0.7163 |                                 34.4153 |                                 0.0622 |                                     -0.0538 |                                0.074  |                                -0.0683 |                               0.8092 |                                  -0.8345 |                                 32.9461 |                                 0.0679 |                                     -0.0586 |                                0.0813 |                                -0.0712 |                               0.93   |                                  -0.9604 |                                 31.6229 |                                 0.0734 |                                     -0.0633 |                                0.0886 |                                -0.0738 |                               1.0576 |                                  -1.0935 |                                 30.4243 |                                 0.0789 |                                     -0.0679 |                                0.0959 |                                -0.0761 |                               1.1915 |                                  -1.2336 |                                 29.3329 |                                 0.0842 |                                     -0.0724 |                                0.1031 |                                -0.0783 |                               1.3315 |                                  -1.3802 |                                 28.3346 |                                 0.0895 |                                     -0.0769 |                                0.1103 |                                -0.0802 |                               1.4772 |                                  -1.5331 |                                 27.4174 |                                 0.0947 |                                     -0.0812 |                                0.1175 |                                -0.0819 |                               1.6283 |                                  -1.692  |                                 26.5716 |                                 0.0997 |                                     -0.0854 |                                0.1246 |                                -0.0834 |                               1.7846 |                                  -1.8566 |                                 25.7888 |                                 0.1047 |                                     -0.0896 |                                0.1316 |                                -0.0848 |                               1.9458 |                                  -2.0267 |                                 25.0619 |                                 0.1095 |                                     -0.0936 |                                0.1386 |                                -0.0861 |                               2.1117 |                                  -2.202  |                                 24.3851 |                                 0.1142 |                                     -0.0975 |                                0.1454 |                                -0.0872 |                               2.282  |                                  -2.3823 |                                 23.753  |                                 0.1189 |                                     -0.1014 |                                0.1522 |                                -0.0883 |                               2.4565 |                                  -2.5674 |                                 23.1612 |
        |            310 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 470.625 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 242.104 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                165.639  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                127.229  |                                 0.0003 |                                     -0.0003 |                                0.0004 |                                -0.0012 |                               0.0011 |                                  -0.0011 |                                104.064  |                                 0.0009 |                                     -0.0008 |                                0.0011 |                                -0.0029 |                               0.004  |                                  -0.004  |                                 88.5364 |                                 0.002  |                                     -0.0017 |                                0.0023 |                                -0.0055 |                               0.01   |                                  -0.0102 |                                 77.3824 |                                 0.0035 |                                     -0.003  |                                0.0043 |                                -0.0087 |                               0.0207 |                                  -0.021  |                                 68.9684 |                                 0.0056 |                                     -0.0048 |                                0.0068 |                                -0.0125 |                               0.037  |                                  -0.0377 |                                 62.3859 |                                 0.0082 |                                     -0.0071 |                                0.0101 |                                -0.0165 |                               0.0599 |                                  -0.0609 |                                 57.089  |                                 0.0112 |                                     -0.0096 |                                0.0139 |                                -0.0207 |                               0.0898 |                                  -0.0915 |                                 52.7297 |                                 0.0145 |                                     -0.0125 |                                0.0181 |                                -0.0249 |                               0.1271 |                                  -0.1297 |                                 49.0758 |                                 0.0182 |                                     -0.0156 |                                0.0229 |                                -0.029  |                               0.172  |                                  -0.1758 |                                 45.9661 |                                 0.022  |                                     -0.0189 |                                0.028  |                                -0.033  |                               0.2245 |                                  -0.2298 |                                 43.2852 |                                 0.0261 |                                     -0.0223 |                                0.0334 |                                -0.0368 |                               0.2845 |                                  -0.2916 |                                 40.9486 |                                 0.0303 |                                     -0.0259 |                                0.0391 |                                -0.0404 |                               0.352  |                                  -0.3613 |                                 38.8926 |                                 0.0346 |                                     -0.0296 |                                0.0449 |                                -0.0438 |                               0.4267 |                                  -0.4386 |                                 37.0683 |                                 0.039  |                                     -0.0333 |                                0.051  |                                -0.047  |                               0.5085 |                                  -0.5233 |                                 35.4378 |                                 0.0435 |                                     -0.037  |                                0.0572 |                                -0.05   |                               0.5971 |                                  -0.6152 |                                 33.971  |                                 0.0479 |                                     -0.0408 |                                0.0635 |                                -0.0527 |                               0.6923 |                                  -0.7142 |                                 32.6438 |                                 0.0524 |                                     -0.0445 |                                0.0698 |                                -0.0553 |                               0.7938 |                                  -0.8199 |                                 31.4367 |                                 0.0569 |                                     -0.0482 |                                0.0762 |                                -0.0577 |                               0.9014 |                                  -0.9321 |                                 30.3336 |                                 0.0613 |                                     -0.052  |                                0.0827 |                                -0.0599 |                               1.0149 |                                  -1.0507 |                                 29.3212 |                                 0.0657 |                                     -0.0556 |                                0.0892 |                                -0.062  |                               1.1339 |                                  -1.1753 |                                 28.3884 |                                 0.0701 |                                     -0.0593 |                                0.0957 |                                -0.0639 |                               1.2583 |                                  -1.3058 |                                 27.526  |                                 0.0745 |                                     -0.0629 |                                0.1021 |                                -0.0657 |                               1.3879 |                                  -1.4419 |                                 26.726  |                                 0.0787 |                                     -0.0664 |                                0.1086 |                                -0.0673 |                               1.5225 |                                  -1.5834 |                                 25.9816 |                                 0.083  |                                     -0.0699 |                                0.115  |                                -0.0688 |                               1.6618 |                                  -1.7302 |                                 25.287  |                                 0.0871 |                                     -0.0733 |                                0.1215 |                                -0.0702 |                               1.8057 |                                  -1.8821 |                                 24.6373 |
        |            315 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 524.724 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 268.562 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                182.949  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                140.001  |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0002 |                                  -0.0002 |                                114.135  |                                 0.0002 |                                     -0.0002 |                                0.0003 |                                -0.0008 |                               0.001  |                                  -0.001  |                                 96.8211 |                                 0.0006 |                                     -0.0005 |                                0.0008 |                                -0.0018 |                               0.0031 |                                  -0.0031 |                                 84.4009 |                                 0.0012 |                                     -0.001  |                                0.0016 |                                -0.0034 |                               0.0072 |                                  -0.0073 |                                 75.0445 |                                 0.0022 |                                     -0.0019 |                                0.0029 |                                -0.0053 |                               0.0145 |                                  -0.0147 |                                 67.7343 |                                 0.0035 |                                     -0.003  |                                0.0047 |                                -0.0077 |                               0.0255 |                                  -0.0259 |                                 61.8593 |                                 0.0051 |                                     -0.0043 |                                0.0069 |                                -0.0104 |                               0.0411 |                                  -0.0418 |                                 57.0302 |                                 0.007  |                                     -0.006  |                                0.0097 |                                -0.0132 |                               0.0617 |                                  -0.0629 |                                 52.9872 |                                 0.0093 |                                     -0.0078 |                                0.0128 |                                -0.0162 |                               0.0879 |                                  -0.0897 |                                 49.5503 |                                 0.0117 |                                     -0.0099 |                                0.0163 |                                -0.0192 |                               0.1198 |                                  -0.1225 |                                 46.5907 |                                 0.0144 |                                     -0.0122 |                                0.0202 |                                -0.0222 |                               0.1578 |                                  -0.1614 |                                 44.0139 |                                 0.0173 |                                     -0.0146 |                                0.0244 |                                -0.0252 |                               0.2018 |                                  -0.2067 |                                 41.7487 |                                 0.0204 |                                     -0.0172 |                                0.0289 |                                -0.0281 |                               0.252  |                                  -0.2585 |                                 39.741  |                                 0.0236 |                                     -0.0198 |                                0.0336 |                                -0.0309 |                               0.3082 |                                  -0.3166 |                                 37.9482 |                                 0.0269 |                                     -0.0226 |                                0.0386 |                                -0.0336 |                               0.3706 |                                  -0.3811 |                                 36.3369 |                                 0.0303 |                                     -0.0254 |                                0.0437 |                                -0.0362 |                               0.4389 |                                  -0.4518 |                                 34.8802 |                                 0.0338 |                                     -0.0283 |                                0.049  |                                -0.0387 |                               0.513  |                                  -0.5288 |                                 33.5565 |                                 0.0373 |                                     -0.0312 |                                0.0544 |                                -0.0411 |                               0.5929 |                                  -0.6118 |                                 32.3479 |                                 0.0409 |                                     -0.0342 |                                0.0599 |                                -0.0433 |                               0.6783 |                                  -0.7007 |                                 31.2395 |                                 0.0445 |                                     -0.0371 |                                0.0655 |                                -0.0454 |                               0.7692 |                                  -0.7955 |                                 30.2192 |                                 0.0481 |                                     -0.0401 |                                0.0712 |                                -0.0474 |                               0.8653 |                                  -0.8959 |                                 29.2765 |                                 0.0517 |                                     -0.0431 |                                0.0769 |                                -0.0493 |                               0.9665 |                                  -1.0018 |                                 28.4026 |                                 0.0553 |                                     -0.046  |                                0.0827 |                                -0.0511 |                               1.0727 |                                  -1.1131 |                                 27.5901 |                                 0.059  |                                     -0.049  |                                0.0885 |                                -0.0528 |                               1.1837 |                                  -1.2296 |                                 26.8326 |                                 0.0626 |                                     -0.0519 |                                0.0943 |                                -0.0544 |                               1.2994 |                                  -1.3511 |                                 26.1244 |
        |            320 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 578.226 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 294.817 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                200.169  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                152.73   |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                124.187  |                                 0.0001 |                                      0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                105.1    |                                 0.0002 |                                     -0.0001 |                                0.0002 |                                -0.0006 |                               0.0008 |                                  -0.0009 |                                 91.4214 |                                 0.0004 |                                     -0.0003 |                                0.0006 |                                -0.0012 |                               0.0023 |                                  -0.0024 |                                 81.1271 |                                 0.0008 |                                     -0.0007 |                                0.0012 |                                -0.0021 |                               0.0052 |                                  -0.0053 |                                 73.0919 |                                 0.0014 |                                     -0.0012 |                                0.002  |                                -0.0033 |                               0.0102 |                                  -0.0103 |                                 66.6404 |                                 0.0022 |                                     -0.0018 |                                0.0033 |                                -0.0049 |                               0.0177 |                                  -0.018  |                                 61.3423 |                                 0.0032 |                                     -0.0027 |                                0.0048 |                                -0.0066 |                               0.0284 |                                  -0.0289 |                                 56.9107 |                                 0.0045 |                                     -0.0037 |                                0.0067 |                                -0.0085 |                               0.0427 |                                  -0.0435 |                                 53.1468 |                                 0.006  |                                     -0.005  |                                0.009  |                                -0.0106 |                               0.0611 |                                  -0.0623 |                                 49.9084 |                                 0.0077 |                                     -0.0064 |                                0.0116 |                                -0.0128 |                               0.0838 |                                  -0.0856 |                                 47.0912 |                                 0.0095 |                                     -0.0079 |                                0.0146 |                                -0.015  |                               0.1112 |                                  -0.1137 |                                 44.6167 |                                 0.0116 |                                     -0.0096 |                                0.0178 |                                -0.0173 |                               0.1433 |                                  -0.1468 |                                 42.4251 |                                 0.0138 |                                     -0.0114 |                                0.0213 |                                -0.0195 |                               0.1804 |                                  -0.185  |                                 40.4697 |                                 0.0161 |                                     -0.0134 |                                0.025  |                                -0.0218 |                               0.2225 |                                  -0.2284 |                                 38.7135 |                                 0.0186 |                                     -0.0154 |                                0.029  |                                -0.024  |                               0.2697 |                                  -0.2771 |                                 37.1271 |                                 0.0212 |                                     -0.0175 |                                0.0331 |                                -0.0261 |                               0.3219 |                                  -0.3312 |                                 35.6864 |                                 0.0238 |                                     -0.0197 |                                0.0375 |                                -0.0283 |                               0.3792 |                                  -0.3905 |                                 34.3718 |                                 0.0266 |                                     -0.0219 |                                0.042  |                                -0.0303 |                               0.4414 |                                  -0.4552 |                                 33.1672 |                                 0.0294 |                                     -0.0242 |                                0.0466 |                                -0.0323 |                               0.5087 |                                  -0.525  |                                 32.0589 |                                 0.0322 |                                     -0.0265 |                                0.0514 |                                -0.0342 |                               0.5807 |                                  -0.6001 |                                 31.0356 |                                 0.0351 |                                     -0.0288 |                                0.0562 |                                -0.036  |                               0.6576 |                                  -0.6802 |                                 30.0876 |                                 0.0381 |                                     -0.0312 |                                0.0612 |                                -0.0378 |                               0.7392 |                                  -0.7654 |                                 29.2067 |                                 0.041  |                                     -0.0336 |                                0.0662 |                                -0.0394 |                               0.8253 |                                  -0.8555 |                                 28.3859 |                                 0.044  |                                     -0.036  |                                0.0713 |                                -0.041  |                               0.916  |                                  -0.9504 |                                 27.619  |
        |            325 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 631.087 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 320.826 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                217.263  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                165.387  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                134.196  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                113.352  |                                 0      |                                      0      |                                0.0001 |                                -0.0002 |                               0.0002 |                                  -0.0002 |                                 98.425  |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0004 |                               0.0007 |                                  -0.0007 |                                 87.1996 |                                 0.0003 |                                     -0.0002 |                                0.0004 |                                -0.0008 |                               0.0018 |                                  -0.0018 |                                 78.4441 |                                 0.0005 |                                     -0.0004 |                                0.0008 |                                -0.0014 |                               0.0038 |                                  -0.0038 |                                 71.4193 |                                 0.0009 |                                     -0.0007 |                                0.0014 |                                -0.0021 |                               0.0072 |                                  -0.0073 |                                 65.6545 |                                 0.0014 |                                     -0.0012 |                                0.0023 |                                -0.0031 |                               0.0124 |                                  -0.0126 |                                 60.8359 |                                 0.0021 |                                     -0.0017 |                                0.0034 |                                -0.0043 |                               0.0198 |                                  -0.0201 |                                 56.7461 |                                 0.0029 |                                     -0.0024 |                                0.0047 |                                -0.0056 |                               0.0298 |                                  -0.0304 |                                 53.2297 |                                 0.0039 |                                     -0.0032 |                                0.0064 |                                -0.007  |                               0.0427 |                                  -0.0436 |                                 50.1726 |                                 0.0051 |                                     -0.0041 |                                0.0083 |                                -0.0086 |                               0.0589 |                                  -0.0602 |                                 47.4892 |                                 0.0064 |                                     -0.0052 |                                0.0105 |                                -0.0102 |                               0.0786 |                                  -0.0804 |                                 45.1141 |                                 0.0078 |                                     -0.0064 |                                0.0129 |                                -0.0119 |                               0.1021 |                                  -0.1045 |                                 42.9961 |                                 0.0094 |                                     -0.0077 |                                0.0156 |                                -0.0136 |                               0.1294 |                                  -0.1327 |                                 41.0952 |                                 0.0111 |                                     -0.009  |                                0.0185 |                                -0.0153 |                               0.1608 |                                  -0.165  |                                 39.379  |                                 0.0129 |                                     -0.0105 |                                0.0217 |                                -0.0171 |                               0.1963 |                                  -0.2016 |                                 37.8213 |                                 0.0148 |                                     -0.012  |                                0.025  |                                -0.0188 |                               0.236  |                                  -0.2427 |                                 36.4008 |                                 0.0168 |                                     -0.0137 |                                0.0285 |                                -0.0205 |                               0.28   |                                  -0.2882 |                                 35.0998 |                                 0.0189 |                                     -0.0154 |                                0.0322 |                                -0.0223 |                               0.3282 |                                  -0.3382 |                                 33.9035 |                                 0.0211 |                                     -0.0171 |                                0.036  |                                -0.0239 |                               0.3807 |                                  -0.3927 |                                 32.7995 |                                 0.0233 |                                     -0.0189 |                                0.04   |                                -0.0256 |                               0.4375 |                                  -0.4517 |                                 31.7773 |                                 0.0256 |                                     -0.0207 |                                0.0441 |                                -0.0272 |                               0.4985 |                                  -0.5152 |                                 30.8279 |                                 0.028  |                                     -0.0226 |                                0.0483 |                                -0.0287 |                               0.5637 |                                  -0.5832 |                                 29.9436 |                                 0.0303 |                                     -0.0245 |                                0.0526 |                                -0.0302 |                               0.633  |                                  -0.6555 |                                 29.1178 |
        |            330 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 683.285 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 346.564 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                234.207  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                177.951  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                144.141  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                121.56   |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                105.397  |                                 0      |                                      0      |                                0.0001 |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 93.2495 |                                 0.0001 |                                     -0.0001 |                                0.0001 |                                -0.0003 |                               0.0006 |                                  -0.0006 |                                 83.7795 |                                 0.0002 |                                     -0.0001 |                                0.0003 |                                -0.0005 |                               0.0013 |                                  -0.0014 |                                 76.1857 |                                 0.0003 |                                     -0.0003 |                                0.0006 |                                -0.0009 |                               0.0028 |                                  -0.0028 |                                 69.9574 |                                 0.0006 |                                     -0.0005 |                                0.001  |                                -0.0014 |                               0.0051 |                                  -0.0052 |                                 64.7543 |                                 0.0009 |                                     -0.0007 |                                0.0016 |                                -0.002  |                               0.0087 |                                  -0.0089 |                                 60.3405 |                                 0.0014 |                                     -0.0011 |                                0.0024 |                                -0.0028 |                               0.0139 |                                  -0.0142 |                                 56.5474 |                                 0.0019 |                                     -0.0015 |                                0.0034 |                                -0.0037 |                               0.0209 |                                  -0.0213 |                                 53.2516 |                                 0.0026 |                                     -0.0021 |                                0.0045 |                                -0.0047 |                               0.0301 |                                  -0.0307 |                                 50.3602 |                                 0.0034 |                                     -0.0027 |                                0.0059 |                                -0.0058 |                               0.0417 |                                  -0.0426 |                                 47.8021 |                                 0.0043 |                                     -0.0034 |                                0.0076 |                                -0.0069 |                               0.0559 |                                  -0.0572 |                                 45.5223 |                                 0.0053 |                                     -0.0042 |                                0.0094 |                                -0.0082 |                               0.073  |                                  -0.0747 |                                 43.477  |                                 0.0064 |                                     -0.0052 |                                0.0115 |                                -0.0095 |                               0.0931 |                                  -0.0954 |                                 41.6313 |                                 0.0076 |                                     -0.0061 |                                0.0137 |                                -0.0108 |                               0.1165 |                                  -0.1195 |                                 39.957  |                                 0.009  |                                     -0.0072 |                                0.0162 |                                -0.0122 |                               0.1431 |                                  -0.147  |                                 38.4308 |                                 0.0104 |                                     -0.0083 |                                0.0188 |                                -0.0135 |                               0.1732 |                                  -0.178  |                                 37.0335 |                                 0.0119 |                                     -0.0095 |                                0.0216 |                                -0.0149 |                               0.2068 |                                  -0.2128 |                                 35.7493 |                                 0.0135 |                                     -0.0108 |                                0.0246 |                                -0.0163 |                               0.244  |                                  -0.2513 |                                 34.5647 |                                 0.0152 |                                     -0.0121 |                                0.0277 |                                -0.0177 |                               0.2848 |                                  -0.2936 |                                 33.4684 |                                 0.0169 |                                     -0.0135 |                                0.031  |                                -0.0191 |                               0.3292 |                                  -0.3397 |                                 32.4505 |                                 0.0187 |                                     -0.0149 |                                0.0344 |                                -0.0204 |                               0.3774 |                                  -0.3897 |                                 31.5029 |                                 0.0205 |                                     -0.0164 |                                0.0379 |                                -0.0218 |                               0.4291 |                                  -0.4436 |                                 30.6183 |
        |            335 |                                 0      |                                      0      |                                     0 |                                 0      |                               0      |                                   0      |                                 734.808 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 372.013 |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                250.985  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                190.406  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                154.011  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                129.713  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                112.328  |                                 0      |                                      0      |                                0      |                                 0      |                               0      |                                   0      |                                 99.2668 |                                 0      |                                      0      |                                0      |                                -0.0001 |                               0.0002 |                                  -0.0002 |                                 89.0894 |                                 0.0001 |                                      0      |                                0.0001 |                                -0.0002 |                               0.0004 |                                  -0.0004 |                                 80.9316 |                                 0.0001 |                                     -0.0001 |                                0.0002 |                                -0.0003 |                               0.001  |                                  -0.001  |                                 74.2438 |                                 0.0002 |                                     -0.0002 |                                0.0004 |                                -0.0006 |                               0.002  |                                  -0.0021 |                                 68.6591 |                                 0.0004 |                                     -0.0003 |                                0.0007 |                                -0.0009 |                               0.0037 |                                  -0.0038 |                                 63.9236 |                                 0.0006 |                                     -0.0005 |                                0.0011 |                                -0.0013 |                               0.0062 |                                  -0.0063 |                                 59.8559 |                                 0.0009 |                                     -0.0007 |                                0.0017 |                                -0.0018 |                               0.0099 |                                  -0.0101 |                                 56.3229 |                                 0.0013 |                                     -0.001  |                                0.0024 |                                -0.0024 |                               0.0148 |                                  -0.0151 |                                 53.2246 |                                 0.0017 |                                     -0.0014 |                                0.0032 |                                -0.0031 |                               0.0214 |                                  -0.0218 |                                 50.4847 |                                 0.0023 |                                     -0.0018 |                                0.0043 |                                -0.0039 |                               0.0297 |                                  -0.0303 |                                 48.0438 |                                 0.0029 |                                     -0.0023 |                                0.0055 |                                -0.0048 |                               0.04   |                                  -0.0409 |                                 45.8549 |                                 0.0036 |                                     -0.0029 |                                0.0069 |                                -0.0057 |                               0.0525 |                                  -0.0537 |                                 43.8804 |                                 0.0044 |                                     -0.0035 |                                0.0084 |                                -0.0066 |                               0.0673 |                                  -0.0689 |                                 42.0898 |                                 0.0053 |                                     -0.0042 |                                0.0102 |                                -0.0076 |                               0.0846 |                                  -0.0868 |                                 40.4583 |                                 0.0063 |                                     -0.005  |                                0.0121 |                                -0.0087 |                               0.1046 |                                  -0.1074 |                                 38.9653 |                                 0.0073 |                                     -0.0058 |                                0.0141 |                                -0.0097 |                               0.1274 |                                  -0.1309 |                                 37.5935 |                                 0.0084 |                                     -0.0067 |                                0.0164 |                                -0.0108 |                               0.153  |                                  -0.1573 |                                 36.3286 |                                 0.0096 |                                     -0.0076 |                                0.0187 |                                -0.0119 |                               0.1816 |                                  -0.1869 |                                 35.1582 |                                 0.0109 |                                     -0.0086 |                                0.0213 |                                -0.0131 |                               0.2131 |                                  -0.2196 |                                 34.0721 |                                 0.0123 |                                     -0.0096 |                                0.0239 |                                -0.0142 |                               0.2478 |                                  -0.2555 |                                 33.0613 |                                 0.0136 |                                     -0.0107 |                                0.0267 |                                -0.0153 |                               0.2856 |                                  -0.2948 |                                 32.118  |
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

        if all(greek.empty for greek in greeks.values()):
            # Each greek has logged why it is empty, which is the same reason for all of them.
            logger.warning("None of the greeks could be calculated.")
            return pd.DataFrame()

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    @handle_errors
    def get_delta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the delta of an option based on the Black Scholes Model: how much the option value changes for a
        1.00 change in the price of the underlying asset.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        delta is the rate of change of the option price with respect to the price of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            200 |       0.9374 |       0.8735 |       0.8369 |
        |            205 |       0.9003 |       0.8331 |       0.7989 |
        |            210 |       0.8508 |       0.7865 |       0.7571 |
        |            215 |       0.7886 |       0.7344 |       0.7121 |
        |            220 |       0.7154 |       0.6778 |       0.6647 |
        |            225 |       0.6338 |       0.6183 |       0.6156 |
        |            230 |       0.5476 |       0.5572 |       0.5658 |
        |            235 |       0.461  |       0.496  |       0.516  |
        |            240 |       0.378  |       0.4362 |       0.467  |
        |            245 |       0.3018 |       0.379  |       0.4195 |
        |            250 |       0.2346 |       0.3254 |       0.374  |
        |            255 |       0.1778 |       0.2762 |       0.3311 |
        |            260 |       0.1313 |       0.2318 |       0.2911 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_dual_delta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the dual delta of an option based on the Black Scholes Model: how much the option value changes
        for a 1.00 change in the strike price.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        dual delta is the first derivative of the option price with respect to the strike price; up to the discount
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            280 |      -0.9717 |      -0.91   |      -0.8603 |
        |            285 |      -0.9533 |      -0.8798 |      -0.8278 |
        |            290 |      -0.9258 |      -0.8433 |      -0.791  |
        |            295 |      -0.8873 |      -0.8005 |      -0.7504 |
        |            300 |      -0.8368 |      -0.7518 |      -0.7065 |
        |            305 |      -0.7742 |      -0.6982 |      -0.6598 |
        |            310 |      -0.7008 |      -0.6406 |      -0.6113 |
        |            315 |      -0.6194 |      -0.5805 |      -0.5616 |
        |            320 |      -0.5336 |      -0.5193 |      -0.5116 |
        |            325 |      -0.4474 |      -0.4585 |      -0.4621 |
        |            330 |      -0.3649 |      -0.3995 |      -0.4139 |
        |            335 |      -0.2892 |      -0.3435 |      -0.3676 |
        |            340 |      -0.2228 |      -0.2915 |      -0.3237 |
        |            345 |      -0.1667 |      -0.2441 |      -0.2827 |
        |            350 |      -0.1213 |      -0.2018 |      -0.2449 |
        |            355 |      -0.0858 |      -0.1647 |      -0.2105 |
        |            360 |      -0.0591 |      -0.1327 |      -0.1795 |
        |            365 |      -0.0396 |      -0.1057 |      -0.1518 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_vega(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the vega of an option based on the Black Scholes Model: how much the option value changes for a
        1.00 change in the volatility of the underlying asset.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        vega is the rate of change of the option price with respect to the volatility of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_theta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the theta of an option based on the Black Scholes Model: how much the option value changes as time
        passes, its time decay.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        theta is the rate of change of the option price with respect to the passage of time.

        The theta calculation is the theoretical value of the theta. The actual theta can differ from this
        value due to several factors such as the volatility of the underlying asset, the time to expiration,
        the risk free rate and more.

        The formula is as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Theta = [—e^(—q * t) * (S * N'(d1) * σ) / (2 * sqrt(t)) — r * K * e^(—r * t) * N(d2) +
          q * S * e^(—q * t) * N(d1)] / 365
        - Put Theta = [—e^(—q * t) * (S * N'(d1) * σ) / (2 * sqrt(t)) + r * K * e^(—r * t) * N(—d2) —
          q * S * e^(—q * t) * N(—d1)] / 365

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            290 |      -0.0622 |      -0.1333 |      -0.18   |
        |            300 |      -0.1273 |      -0.2073 |      -0.2285 |
        |            310 |      -0.2475 |      -0.2972 |      -0.2762 |
        |            320 |      -0.4168 |      -0.3897 |      -0.3185 |
        |            330 |      -0.5935 |      -0.4679 |      -0.3511 |
        |            340 |      -0.7151 |      -0.5166 |      -0.3711 |
        |            350 |      -0.7359 |      -0.5271 |      -0.3771 |
        |            360 |      -0.6538 |      -0.5    |      -0.3696 |
        |            370 |      -0.5072 |      -0.4432 |      -0.3502 |
        |            380 |      -0.347  |      -0.3691 |      -0.3217 |
        |            390 |      -0.2115 |      -0.2901 |      -0.2871 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_rho(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the rho of an option based on the Black Scholes Model: how much the option value changes for a
        1.00 change in the risk free rate.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        rho is the rate of change of the option price with respect to the risk free interest rate.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            120 |       9.8242 |      28.3968 |      52.5187 |
        |            125 |      10.1774 |      28.2215 |      50.9318 |
        |            130 |      10.2553 |      26.8656 |      47.8678 |
        |            135 |       9.5388 |      24.1422 |      43.4471 |
        |            140 |       7.6094 |      20.2487 |      38.0035 |
        |            145 |       4.885  |      15.739  |      32.0077 |
        |            150 |       2.4331 |      11.3016 |      25.96   |
        |            155 |       0.9288 |       7.4961 |      20.293  |
        |            160 |       0.2726 |       4.6014 |      15.3096 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_epsilon(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the epsilon of an option based on the Black Scholes Model: how much the option value changes for a
        1.00 change in the dividend yield.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        epsilon is the rate of change of the option price with respect to the dividend yield.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             60 |      -5.7266 |     -16.5449 |     -30.9929 |
        |             62 |      -5.6895 |     -15.8742 |     -29.2736 |
        |             64 |      -5.5271 |     -14.7661 |     -27.0636 |
        |             66 |      -5.0612 |     -13.1819 |     -24.4224 |
        |             68 |      -4.153  |     -11.1972 |     -21.4687 |
        |             70 |      -2.9116 |      -8.9943 |     -18.3604 |
        |             72 |      -1.6884 |      -6.8063 |     -15.2669 |
        |             74 |      -0.7967 |      -4.8441 |     -12.3419 |
        |             76 |      -0.3046 |      -3.2422 |      -9.7037 |
        |             78 |      -0.0947 |      -2.043  |      -7.425  |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_lambda(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the lambda of an option based on the Black Scholes Model: the percentage change in the option
        value per percentage change in the price of the underlying asset, the leverage the option gives.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        lambda, or elasticity, is the percentage change in the option price for a 1% change in the price of the
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            315 |       9.0902 |       8.0791 |       7.3278 |
        |            320 |      10.1194 |       8.7387 |       7.8119 |
        |            325 |      11.2886 |       9.4486 |       8.322  |
        |            330 |      12.601  |      10.2075 |       8.857  |
        |            335 |      14.0561 |      11.0131 |       9.4157 |
        |            340 |      15.6492 |      11.8631 |       9.9965 |
        |            345 |      17.3729 |      12.7543 |      10.5978 |
        |            350 |      19.2169 |      13.6837 |      11.2182 |
        |            355 |      21.1699 |      14.6477 |      11.8559 |
        |            360 |      23.2196 |      15.6431 |      12.5094 |
        |            365 |      25.3536 |      16.6664 |      13.177  |
        |            370 |      27.5599 |      17.7143 |      13.8571 |
        |            375 |      29.8274 |      18.7837 |      14.5484 |
        |            380 |      32.1457 |      19.8717 |      15.2494 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    @logger_model.log_each_message_once()
    def collect_second_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the second order Greeks of an option based on the Black Scholes Model: how the first order Greeks
        themselves respond to the price of the underlying asset, the strike price, the volatility and time.

        This will return the following Greeks per Strike Price and Expiration Date:

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            290 |                                 0.0001 |                                      0.0001 |                                 0.0035 |                                -0.2099 |                                 0.1908 |                                0.0026 |                               -0.0003 |                              0.0001 |                                 0.0017 |                                      0.0015 |                                 0.0936 |                                -2.7872 |                                 5.0326 |                                0.1382 |                               -0.0047 |                              0.0015 |                                 0.0047 |                                      0.0041 |                                 0.2558 |                                -5.0935 |                                13.6993 |                                0.564  |                               -0.009  |                              0.0041 |                                 0.0075 |                                      0.0066 |                                 0.4054 |                                -6.0728 |                                21.626  |                                1.1864 |                               -0.0113 |                              0.0066 |                                 0.0096 |                                      0.0084 |                                 0.5212 |                                -6.265  |                                27.6937 |                                1.8979 |                               -0.0123 |                              0.0084 |                                 0.0111 |                                      0.0098 |                                 0.6061 |                                -6.0897 |                                32.0779 |                                2.6364 |                               -0.0125 |                              0.0098 |                                 0.0123 |                                      0.0108 |                                 0.6671 |                                -5.763  |                                35.1702 |                                3.3701 |                               -0.0123 |                              0.0108 |                                 0.0131 |                                      0.0115 |                                 0.7106 |                                -5.3877 |                                37.3152 |                                4.0839 |                               -0.012  |                              0.0115 |                                 0.0136 |                                      0.012  |                                 0.7413 |                                -5.0109 |                                38.7726 |                                4.7708 |                               -0.0116 |                              0.012  |                                 0.014  |                                      0.0123 |                                 0.7626 |                                -4.6535 |                                39.7296 |                                5.4282 |                               -0.0112 |                              0.0123 |                                 0.0142 |                                      0.0125 |                                 0.777  |                                -4.3235 |                                40.3201 |                                6.0559 |                               -0.0108 |                              0.0125 |                                 0.0144 |                                      0.0127 |                                 0.7863 |                                -4.0227 |                                40.6404 |                                6.6547 |                               -0.0104 |                              0.0127 |                                 0.0145 |                                      0.0127 |                                 0.7918 |                                -3.7503 |                                40.7599 |                                7.2258 |                               -0.01   |                              0.0127 |                                 0.0145 |                                      0.0128 |                                 0.7944 |                                -3.5042 |                                40.7294 |                                7.7709 |                               -0.0097 |                              0.0128 |                                 0.0145 |                                      0.0128 |                                 0.7948 |                                -3.282  |                                40.5865 |                                8.2914 |                               -0.0094 |                              0.0128 |                                 0.0145 |                                      0.0127 |                                 0.7935 |                                -3.0811 |                                40.3592 |                                8.7891 |                               -0.0091 |                              0.0127 |                                 0.0144 |                                      0.0127 |                                 0.791  |                                -2.8992 |                                40.0686 |                                9.2652 |                               -0.0088 |                              0.0127 |                                 0.0144 |                                      0.0126 |                                 0.7876 |                                -2.7341 |                                39.7306 |                                9.7213 |                               -0.0085 |                              0.0126 |                                 0.0143 |                                      0.0126 |                                 0.7834 |                                -2.5839 |                                39.3575 |                               10.1586 |                               -0.0082 |                              0.0126 |                                 0.0142 |                                      0.0125 |                                 0.7787 |                                -2.447  |                                38.9587 |                               10.5782 |                               -0.008  |                              0.0125 |                                 0.0141 |                                      0.0124 |                                 0.7735 |                                -2.3217 |                                38.5415 |                               10.9812 |                               -0.0078 |                              0.0124 |                                 0.014  |                                      0.0123 |                                 0.7681 |                                -2.2069 |                                38.1117 |                               11.3686 |                               -0.0076 |                              0.0123 |                                 0.0139 |                                      0.0122 |                                 0.7624 |                                -2.1014 |                                37.6735 |                               11.7412 |                               -0.0074 |                              0.0122 |                                 0.0138 |                                      0.0121 |                                 0.7566 |                                -2.0042 |                                37.2305 |                               12.1    |                               -0.0072 |                              0.0121 |                                 0.0136 |                                      0.012  |                                 0.7507 |                                -1.9144 |                                36.7854 |                               12.4456 |                               -0.007  |                              0.012  |                                 0.0135 |                                      0.0119 |                                 0.7448 |                                -1.8314 |                                36.3403 |                               12.7787 |                               -0.0069 |                              0.0119 |                                 0.0134 |                                      0.0118 |                                 0.7388 |                                -1.7543 |                                35.8969 |                               13.1    |                               -0.0067 |                              0.0118 |                                 0.0133 |                                      0.0117 |                                 0.7329 |                                -1.6827 |                                35.4564 |                               13.41   |                               -0.0066 |                              0.0117 |                                 0.0132 |                                      0.0116 |                                 0.727  |                                -1.6161 |                                35.0199 |                               13.7093 |                               -0.0065 |                              0.0116 |
        |            295 |                                 0      |                                      0      |                                 0.0001 |                                -0.0035 |                                 0.004  |                                0      |                                0      |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0137 |                                -0.4073 |                                 0.9328 |                                0.0203 |                               -0.0008 |                              0.0002 |                                 0.0011 |                                      0.001  |                                 0.0772 |                                -1.5337 |                                 5.2396 |                                0.1707 |                               -0.0032 |                              0.001  |                                 0.0026 |                                      0.0022 |                                 0.1756 |                                -2.6237 |                                11.8853 |                                0.5159 |                               -0.0057 |                              0.0022 |                                 0.0041 |                                      0.0035 |                                 0.2805 |                                -3.3605 |                                18.9235 |                                1.0262 |                               -0.0076 |                              0.0035 |                                 0.0055 |                                      0.0047 |                                 0.3769 |                                -3.7722 |                                25.3505 |                                1.6488 |                               -0.0088 |                              0.0047 |                                 0.0067 |                                      0.0057 |                                 0.46   |                                -3.9557 |                                30.8431 |                                2.3392 |                               -0.0095 |                              0.0057 |                                 0.0077 |                                      0.0065 |                                 0.5294 |                                -3.9932 |                                35.3875 |                                3.0657 |                               -0.0098 |                              0.0065 |                                 0.0085 |                                      0.0072 |                                 0.5865 |                                -3.9419 |                                39.083  |                                3.8072 |                               -0.01   |                              0.0072 |                                 0.0092 |                                      0.0078 |                                 0.6331 |                                -3.8388 |                                42.0574 |                                4.5498 |                               -0.01   |                              0.0078 |                                 0.0097 |                                      0.0083 |                                 0.6709 |                                -3.7074 |                                44.4333 |                                5.2849 |                               -0.0099 |                              0.0083 |                                 0.0102 |                                      0.0086 |                                 0.7016 |                                -3.5622 |                                46.3178 |                                6.0068 |                               -0.0098 |                              0.0086 |                                 0.0105 |                                      0.0089 |                                 0.7263 |                                -3.4122 |                                47.8004 |                                6.7123 |                               -0.0096 |                              0.0089 |                                 0.0108 |                                      0.0092 |                                 0.7462 |                                -3.2629 |                                48.9544 |                                7.3994 |                               -0.0094 |                              0.0092 |                                 0.011  |                                      0.0094 |                                 0.7621 |                                -3.1176 |                                49.8392 |                                8.0671 |                               -0.0092 |                              0.0094 |                                 0.0112 |                                      0.0095 |                                 0.7747 |                                -2.9781 |                                50.5031 |                                8.7152 |                               -0.009  |                              0.0095 |                                 0.0114 |                                      0.0096 |                                 0.7845 |                                -2.8454 |                                50.9852 |                                9.3436 |                               -0.0088 |                              0.0096 |                                 0.0115 |                                      0.0097 |                                 0.7922 |                                -2.7198 |                                51.3171 |                                9.9526 |                               -0.0086 |                              0.0097 |                                 0.0115 |                                      0.0098 |                                 0.7979 |                                -2.6014 |                                51.5248 |                               10.5427 |                               -0.0084 |                              0.0098 |                                 0.0116 |                                      0.0098 |                                 0.8021 |                                -2.4901 |                                51.6294 |                               11.1145 |                               -0.0082 |                              0.0098 |                                 0.0116 |                                      0.0099 |                                 0.805  |                                -2.3855 |                                51.6482 |                               11.6686 |                               -0.008  |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8068 |                                -2.2874 |                                51.5956 |                               12.2057 |                               -0.0078 |                              0.0099 |                                 0.0117 |                                      0.0099 |                                 0.8076 |                                -2.1953 |                                51.4835 |                               12.7263 |                               -0.0077 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8077 |                                -2.1089 |                                51.3216 |                               13.2312 |                               -0.0075 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.8071 |                                -2.0277 |                                51.1182 |                               13.721  |                               -0.0073 |                              0.0099 |                                 0.0116 |                                      0.0099 |                                 0.806  |                                -1.9514 |                                50.8802 |                               14.1963 |                               -0.0072 |                              0.0099 |                                 0.0116 |                                      0.0098 |                                 0.8044 |                                -1.8797 |                                50.6134 |                               14.6577 |                               -0.007  |                              0.0098 |                                 0.0115 |                                      0.0098 |                                 0.8024 |                                -1.8122 |                                50.3227 |                               15.1056 |                               -0.0069 |                              0.0098 |                                 0.0115 |                                      0.0098 |                                 0.8001 |                                -1.7486 |                                50.0121 |                               15.5408 |                               -0.0068 |                              0.0098 |
        |            300 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0.0012 |                                -0.0359 |                                 0.0993 |                                0.0018 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0002 |                                 0.0164 |                                -0.3246 |                                 1.3409 |                                0.0362 |                               -0.0008 |                              0.0002 |                                 0.0007 |                                      0.0006 |                                 0.0577 |                                -0.8602 |                                 4.7163 |                                0.1698 |                               -0.0022 |                              0.0006 |                                 0.0014 |                                      0.0012 |                                 0.1198 |                                -1.4324 |                                 9.7716 |                                0.4397 |                               -0.0037 |                              0.0012 |                                 0.0023 |                                      0.0019 |                                 0.1918 |                                -1.915  |                                15.6051 |                                0.8422 |                               -0.005  |                              0.0019 |                                 0.0032 |                                      0.0026 |                                 0.2653 |                                -2.2748 |                                21.5282 |                                1.355  |                               -0.0061 |                              0.0026 |                                 0.004  |                                      0.0033 |                                 0.3354 |                                -2.5212 |                                27.1441 |                                1.9517 |                               -0.0069 |                              0.0033 |                                 0.0048 |                                      0.004  |                                 0.3996 |                                -2.6761 |                                32.2648 |                                2.6087 |                               -0.0075 |                              0.004  |                                 0.0055 |                                      0.0045 |                                 0.4573 |                                -2.7616 |                                36.8261 |                                3.307  |                               -0.0079 |                              0.0045 |                                 0.0061 |                                      0.005  |                                 0.5083 |                                -2.7963 |                                40.8303 |                                4.0315 |                               -0.0082 |                              0.005  |                                 0.0067 |                                      0.0055 |                                 0.5531 |                                -2.7946 |                                44.3123 |                                4.771  |                               -0.0083 |                              0.0055 |                                 0.0071 |                                      0.0059 |                                 0.5922 |                                -2.7674 |                                47.3205 |                                5.5172 |                               -0.0084 |                              0.0059 |                                 0.0075 |                                      0.0062 |                                 0.6262 |                                -2.7226 |                                49.9069 |                                6.2637 |                               -0.0084 |                              0.0062 |                                 0.0079 |                                      0.0065 |                                 0.6557 |                                -2.6661 |                                52.122  |                                7.0061 |                               -0.0084 |                              0.0065 |                                 0.0082 |                                      0.0067 |                                 0.6812 |                                -2.602  |                                54.012  |                                7.7409 |                               -0.0083 |                              0.0067 |                                 0.0085 |                                      0.0069 |                                 0.7033 |                                -2.5334 |                                55.6186 |                                8.4658 |                               -0.0083 |                              0.0069 |                                 0.0087 |                                      0.0071 |                                 0.7224 |                                -2.4624 |                                56.9785 |                                9.1791 |                               -0.0082 |                              0.0071 |                                 0.0089 |                                      0.0073 |                                 0.7388 |                                -2.3906 |                                58.1238 |                                9.8797 |                               -0.0081 |                              0.0073 |                                 0.009  |                                      0.0074 |                                 0.753  |                                -2.3191 |                                59.0822 |                               10.5667 |                               -0.008  |                              0.0074 |                                 0.0092 |                                      0.0075 |                                 0.7652 |                                -2.2487 |                                59.8779 |                               11.2398 |                               -0.0078 |                              0.0075 |                                 0.0093 |                                      0.0076 |                                 0.7756 |                                -2.1799 |                                60.5317 |                               11.8986 |                               -0.0077 |                              0.0076 |                                 0.0094 |                                      0.0077 |                                 0.7844 |                                -2.113  |                                61.0616 |                               12.5431 |                               -0.0076 |                              0.0077 |                                 0.0095 |                                      0.0078 |                                 0.7919 |                                -2.0483 |                                61.4832 |                               13.1733 |                               -0.0075 |                              0.0078 |                                 0.0096 |                                      0.0079 |                                 0.7983 |                                -1.9859 |                                61.81   |                               13.7894 |                               -0.0074 |                              0.0079 |                                 0.0096 |                                      0.0079 |                                 0.8036 |                                -1.9259 |                                62.0536 |                               14.3915 |                               -0.0072 |                              0.0079 |                                 0.0097 |                                      0.0079 |                                 0.8079 |                                -1.8682 |                                62.2244 |                               14.9799 |                               -0.0071 |                              0.0079 |                                 0.0097 |                                      0.008  |                                 0.8115 |                                -1.8129 |                                62.331  |                               15.5548 |                               -0.007  |                              0.008  |                                 0.0097 |                                      0.008  |                                 0.8144 |                                -1.7599 |                                62.3813 |                               16.1166 |                               -0.0069 |                              0.008  |
        |            305 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.002  |                                 0.0064 |                                0.0001 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0498 |                                 0.2407 |                                0.0056 |                               -0.0001 |                              0      |                                 0.0002 |                                      0.0001 |                                 0.0148 |                                -0.2199 |                                 1.4114 |                                0.0435 |                               -0.0006 |                              0.0001 |                                 0.0004 |                                      0.0003 |                                 0.0416 |                                -0.4972 |                                 3.974  |                                0.1532 |                               -0.0014 |                              0.0003 |                                 0.0008 |                                      0.0007 |                                 0.0818 |                                -0.8151 |                                 7.7871 |                                0.36   |                               -0.0024 |                              0.0007 |                                 0.0014 |                                      0.0011 |                                 0.1309 |                                -1.1199 |                                12.4342 |                                0.6704 |                               -0.0034 |                              0.0011 |                                 0.0019 |                                      0.0015 |                                 0.1846 |                                -1.3843 |                                17.4963 |                                1.0777 |                               -0.0042 |                              0.0015 |                                 0.0025 |                                      0.002  |                                 0.2395 |                                -1.5993 |                                22.6518 |                                1.5691 |                               -0.005  |                              0.002  |                                 0.003  |                                      0.0024 |                                 0.2933 |                                -1.7661 |                                27.6846 |                                2.1301 |                               -0.0056 |                              0.0024 |                                 0.0036 |                                      0.0028 |                                 0.3447 |                                -1.8901 |                                32.4637 |                                2.7466 |                               -0.006  |                              0.0028 |                                 0.0041 |                                      0.0032 |                                 0.3928 |                                -1.978  |                                36.9182 |                                3.4062 |                               -0.0064 |                              0.0032 |                                 0.0045 |                                      0.0036 |                                 0.4374 |                                -2.0366 |                                41.0173 |                                4.0983 |                               -0.0067 |                              0.0036 |                                 0.0049 |                                      0.0039 |                                 0.4783 |                                -2.0716 |                                44.7551 |                                4.814  |                               -0.0069 |                              0.0039 |                                 0.0053 |                                      0.0042 |                                 0.5157 |                                -2.0879 |                                48.1412 |                                5.5461 |                               -0.0071 |                              0.0042 |                                 0.0057 |                                      0.0045 |                                 0.5496 |                                -2.0897 |                                51.1938 |                                6.2887 |                               -0.0072 |                              0.0045 |                                 0.006  |                                      0.0048 |                                 0.5803 |                                -2.0803 |                                53.9356 |                                7.0371 |                               -0.0073 |                              0.0048 |                                 0.0063 |                                      0.005  |                                 0.6081 |                                -2.0622 |                                56.3907 |                                7.7874 |                               -0.0073 |                              0.005  |                                 0.0065 |                                      0.0052 |                                 0.6331 |                                -2.0376 |                                58.5838 |                                8.5366 |                               -0.0073 |                              0.0052 |                                 0.0067 |                                      0.0054 |                                 0.6557 |                                -2.0082 |                                60.5382 |                                9.2824 |                               -0.0073 |                              0.0054 |                                 0.007  |                                      0.0055 |                                 0.6761 |                                -1.9752 |                                62.2762 |                               10.0227 |                               -0.0073 |                              0.0055 |                                 0.0071 |                                      0.0057 |                                 0.6944 |                                -1.9397 |                                63.8185 |                               10.7562 |                               -0.0072 |                              0.0057 |                                 0.0073 |                                      0.0058 |                                 0.7108 |                                -1.9025 |                                65.1838 |                               11.4816 |                               -0.0072 |                              0.0058 |                                 0.0075 |                                      0.0059 |                                 0.7256 |                                -1.8642 |                                66.3893 |                               12.198  |                               -0.0071 |                              0.0059 |                                 0.0076 |                                      0.006  |                                 0.7389 |                                -1.8254 |                                67.4506 |                               12.9047 |                               -0.0071 |                              0.006  |                                 0.0077 |                                      0.0061 |                                 0.7508 |                                -1.7864 |                                68.3816 |                               13.6013 |                               -0.007  |                              0.0061 |                                 0.0078 |                                      0.0062 |                                 0.7614 |                                -1.7476 |                                69.1951 |                               14.2873 |                               -0.0069 |                              0.0062 |                                 0.0079 |                                      0.0063 |                                 0.771  |                                -1.7091 |                                69.9024 |                               14.9626 |                               -0.0068 |                              0.0063 |                                 0.008  |                                      0.0064 |                                 0.7795 |                                -1.6711 |                                70.5137 |                               15.6269 |                               -0.0068 |                              0.0064 |
        |            310 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0003 |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0.0003 |                                -0.0057 |                                 0.0313 |                                0.0006 |                                0      |                              0      |                                 0      |                                      0      |                                 0.003  |                                -0.0447 |                                 0.328  |                                0.0089 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.012  |                                -0.143  |                                 1.3078 |                                0.0442 |                               -0.0005 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0297 |                                -0.2955 |                                 3.232  |                                0.1309 |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0561 |                                -0.4791 |                                 6.0915 |                                0.2878 |                               -0.0016 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.0895 |                                -0.6703 |                                 9.7071 |                                0.5241 |                               -0.0023 |                              0.0006 |                                 0.0012 |                                      0.0009 |                                 0.1279 |                                -0.8527 |                                13.8449 |                                0.8406 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0012 |                                 0.1693 |                                -1.017  |                                18.2846 |                                1.2331 |                               -0.0035 |                              0.0012 |                                 0.0019 |                                      0.0015 |                                 0.2119 |                                -1.1592 |                                22.8457 |                                1.6943 |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0018 |                                 0.2545 |                                -1.2784 |                                27.3924 |                                2.2155 |                               -0.0045 |                              0.0018 |                                 0.0027 |                                      0.0021 |                                 0.2963 |                                -1.3759 |                                31.8286 |                                2.7879 |                               -0.0049 |                              0.0021 |                                 0.003  |                                      0.0023 |                                 0.3367 |                                -1.4537 |                                36.0903 |                                3.4033 |                               -0.0053 |                              0.0023 |                                 0.0034 |                                      0.0026 |                                 0.3751 |                                -1.5141 |                                40.1375 |                                4.054  |                               -0.0056 |                              0.0026 |                                 0.0037 |                                      0.0029 |                                 0.4116 |                                -1.5596 |                                43.948  |                                4.7333 |                               -0.0058 |                              0.0029 |                                 0.004  |                                      0.0031 |                                 0.4458 |                                -1.5923 |                                47.5122 |                                5.4353 |                               -0.006  |                              0.0031 |                                 0.0043 |                                      0.0033 |                                 0.4778 |                                -1.6144 |                                50.8291 |                                6.1549 |                               -0.0061 |                              0.0033 |                                 0.0046 |                                      0.0035 |                                 0.5077 |                                -1.6275 |                                53.9036 |                                6.8876 |                               -0.0062 |                              0.0035 |                                 0.0048 |                                      0.0037 |                                 0.5355 |                                -1.6332 |                                56.7445 |                                7.6298 |                               -0.0063 |                              0.0037 |                                 0.0051 |                                      0.0039 |                                 0.5613 |                                -1.6328 |                                59.3628 |                                8.3783 |                               -0.0064 |                              0.0039 |                                 0.0053 |                                      0.0041 |                                 0.5853 |                                -1.6274 |                                61.7707 |                                9.1305 |                               -0.0064 |                              0.0041 |                                 0.0055 |                                      0.0042 |                                 0.6074 |                                -1.6179 |                                63.9811 |                                9.884  |                               -0.0065 |                              0.0042 |                                 0.0057 |                                      0.0044 |                                 0.6279 |                                -1.6051 |                                66.007  |                               10.6369 |                               -0.0065 |                              0.0044 |                                 0.0058 |                                      0.0045 |                                 0.6468 |                                -1.5896 |                                67.8609 |                               11.3878 |                               -0.0065 |                              0.0045 |                                 0.006  |                                      0.0046 |                                 0.6642 |                                -1.572  |                                69.5551 |                               12.1351 |                               -0.0065 |                              0.0046 |                                 0.0061 |                                      0.0047 |                                 0.6803 |                                -1.5528 |                                71.1012 |                               12.878  |                               -0.0065 |                              0.0047 |                                 0.0063 |                                      0.0048 |                                 0.6952 |                                -1.5323 |                                72.5101 |                               13.6153 |                               -0.0064 |                              0.0048 |                                 0.0064 |                                      0.0049 |                                 0.7089 |                                -1.5108 |                                73.7921 |                               14.3464 |                               -0.0064 |                              0.0049 |
        |            315 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0005 |                                 0.003  |                                0.0001 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0073 |                                 0.0605 |                                0.0015 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0346 |                                 0.3552 |                                0.0107 |                               -0.0001 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0093 |                                -0.0924 |                                 1.1365 |                                0.0411 |                               -0.0003 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0211 |                                -0.1801 |                                 2.5758 |                                0.1085 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0387 |                                -0.2893 |                                 4.7138 |                                0.2269 |                               -0.0011 |                              0.0002 |                                 0.0005 |                                      0.0004 |                                 0.0616 |                                -0.4097 |                                 7.4875 |                                0.4054 |                               -0.0015 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.0888 |                                -0.5324 |                                10.7784 |                                0.6482 |                               -0.002  |                              0.0005 |                                 0.001  |                                      0.0007 |                                 0.1192 |                                -0.6509 |                                14.4513 |                                0.9557 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1518 |                                -0.7611 |                                18.3769 |                                1.3254 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.0011 |                                 0.1858 |                                -0.8606 |                                22.4432 |                                1.7531 |                               -0.0033 |                              0.0011 |                                 0.0018 |                                      0.0013 |                                 0.2202 |                                -0.9486 |                                26.5585 |                                2.2334 |                               -0.0037 |                              0.0013 |                                 0.0021 |                                      0.0015 |                                 0.2546 |                                -1.0249 |                                30.6512 |                                2.761  |                               -0.0041 |                              0.0015 |                                 0.0023 |                                      0.0017 |                                 0.2884 |                                -1.0901 |                                34.6673 |                                3.33   |                               -0.0044 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3214 |                                -1.1449 |                                38.5676 |                                3.9351 |                               -0.0046 |                              0.0019 |                                 0.0028 |                                      0.0021 |                                 0.3533 |                                -1.1902 |                                42.3247 |                                4.5711 |                               -0.0048 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.384  |                                -1.2271 |                                45.9206 |                                5.2335 |                               -0.005  |                              0.0023 |                                 0.0033 |                                      0.0025 |                                 0.4134 |                                -1.2565 |                                49.3445 |                                5.9181 |                               -0.0052 |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4413 |                                -1.2794 |                                52.5912 |                                6.621  |                               -0.0053 |                              0.0026 |                                 0.0038 |                                      0.0028 |                                 0.4679 |                                -1.2964 |                                55.6592 |                                7.3389 |                               -0.0055 |                              0.0028 |                                 0.004  |                                      0.003  |                                 0.493  |                                -1.3085 |                                58.5505 |                                8.0688 |                               -0.0056 |                              0.003  |                                 0.0042 |                                      0.0031 |                                 0.5168 |                                -1.3162 |                                61.2688 |                                8.808  |                               -0.0056 |                              0.0031 |                                 0.0043 |                                      0.0032 |                                 0.5393 |                                -1.3202 |                                63.8195 |                                9.5543 |                               -0.0057 |                              0.0032 |                                 0.0045 |                                      0.0034 |                                 0.5604 |                                -1.321  |                                66.2088 |                               10.3056 |                               -0.0058 |                              0.0034 |                                 0.0047 |                                      0.0035 |                                 0.5804 |                                -1.319  |                                68.4439 |                               11.0601 |                               -0.0058 |                              0.0035 |                                 0.0048 |                                      0.0036 |                                 0.5991 |                                -1.3147 |                                70.5319 |                               11.8164 |                               -0.0058 |                              0.0036 |                                 0.005  |                                      0.0037 |                                 0.6168 |                                -1.3085 |                                72.4803 |                               12.5729 |                               -0.0058 |                              0.0037 |
        |            320 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0.0002 |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.001  |                                 0.009  |                                0.0002 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0071 |                                 0.0809 |                                0.0022 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0025 |                                -0.0252 |                                 0.3434 |                                0.0112 |                               -0.0001 |                              0      |                                 0.0001 |                                      0      |                                 0.007  |                                -0.0601 |                                 0.9527 |                                0.0363 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.015  |                                -0.1122 |                                 2.029  |                                0.0883 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0269 |                                -0.1788 |                                 3.6264 |                                0.1774 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0426 |                                -0.2553 |                                 5.7373 |                                0.3118 |                               -0.001  |                              0.0002 |                                 0.0005 |                                      0.0003 |                                 0.0618 |                                -0.3371 |                                 8.3104 |                                0.4966 |                               -0.0014 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.084  |                                -0.4202 |                                11.2712 |                                0.7346 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0006 |                                 0.1085 |                                -0.5016 |                                14.5367 |                                1.0262 |                               -0.0021 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1347 |                                -0.5792 |                                18.0258 |                                1.37   |                               -0.0024 |                              0.0007 |                                 0.0012 |                                      0.0009 |                                 0.1621 |                                -0.6515 |                                21.6646 |                                1.7637 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.1903 |                                -0.7178 |                                25.3887 |                                2.2041 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0012 |                                 0.2188 |                                -0.7777 |                                29.1444 |                                2.6876 |                               -0.0034 |                              0.0012 |                                 0.0018 |                                      0.0013 |                                 0.2473 |                                -0.8311 |                                32.8876 |                                3.2104 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.2755 |                                -0.8782 |                                36.5834 |                                3.7686 |                               -0.0039 |                              0.0014 |                                 0.0022 |                                      0.0016 |                                 0.3032 |                                -0.9194 |                                40.2047 |                                4.3585 |                               -0.0041 |                              0.0016 |                                 0.0024 |                                      0.0017 |                                 0.3304 |                                -0.9551 |                                43.7309 |                                4.9765 |                               -0.0043 |                              0.0017 |                                 0.0026 |                                      0.0019 |                                 0.3567 |                                -0.9856 |                                47.1469 |                                5.6194 |                               -0.0044 |                              0.0019 |                                 0.0028 |                                      0.002  |                                 0.3822 |                                -1.0115 |                                50.4421 |                                6.2838 |                               -0.0046 |                              0.002  |                                 0.003  |                                      0.0021 |                                 0.4069 |                                -1.033  |                                53.6096 |                                6.967  |                               -0.0047 |                              0.0021 |                                 0.0031 |                                      0.0023 |                                 0.4306 |                                -1.0508 |                                56.6452 |                                7.6663 |                               -0.0048 |                              0.0023 |                                 0.0033 |                                      0.0024 |                                 0.4534 |                                -1.0651 |                                59.547  |                                8.3792 |                               -0.0049 |                              0.0024 |                                 0.0035 |                                      0.0025 |                                 0.4752 |                                -1.0763 |                                62.3151 |                                9.1037 |                               -0.005  |                              0.0025 |                                 0.0036 |                                      0.0026 |                                 0.4961 |                                -1.0847 |                                64.9508 |                                9.8377 |                               -0.0051 |                              0.0026 |                                 0.0037 |                                      0.0027 |                                 0.516  |                                -1.0907 |                                67.4563 |                               10.5794 |                               -0.0051 |                              0.0027 |
        |            325 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0001 |                                 0.0011 |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0156 |                                0.0004 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.006  |                                 0.0902 |                                0.0027 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0021 |                                -0.0179 |                                 0.3115 |                                0.0108 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0053 |                                -0.0394 |                                 0.7819 |                                0.0311 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0107 |                                -0.0713 |                                 1.588  |                                0.071  |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0188 |                                -0.1128 |                                 2.7829 |                                0.1381 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0297 |                                -0.1619 |                                 4.383  |                                0.2392 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0433 |                                -0.2164 |                                 6.3741 |                                0.3795 |                               -0.001  |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0593 |                                -0.274  |                                 8.721  |                                0.5623 |                               -0.0012 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.0775 |                                -0.3327 |                                11.376  |                                0.7898 |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0005 |                                 0.0974 |                                -0.3909 |                                14.286  |                                1.0624 |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0006 |                                 0.1188 |                                -0.4474 |                                17.3977 |                                1.3797 |                               -0.0021 |                              0.0006 |                                 0.0009 |                                      0.0007 |                                 0.1413 |                                -0.5014 |                                20.6603 |                                1.7404 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.1646 |                                -0.5521 |                                24.0279 |                                2.1427 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.1884 |                                -0.5992 |                                27.4597 |                                2.5842 |                               -0.0028 |                              0.0009 |                                 0.0014 |                                      0.001  |                                 0.2124 |                                -0.6426 |                                30.9207 |                                3.0623 |                               -0.003  |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2365 |                                -0.6822 |                                34.3811 |                                3.5744 |                               -0.0032 |                              0.0011 |                                 0.0017 |                                      0.0012 |                                 0.2605 |                                -0.7181 |                                37.8164 |                                4.1179 |                               -0.0034 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.2843 |                                -0.7504 |                                41.2064 |                                4.6899 |                               -0.0036 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.3077 |                                -0.7791 |                                44.5349 |                                5.2878 |                               -0.0038 |                              0.0014 |                                 0.0022 |                                      0.0015 |                                 0.3306 |                                -0.8047 |                                47.789  |                                5.9093 |                               -0.0039 |                              0.0015 |                                 0.0023 |                                      0.0016 |                                 0.353  |                                -0.8271 |                                50.959  |                                6.5518 |                               -0.004  |                              0.0016 |                                 0.0025 |                                      0.0017 |                                 0.3749 |                                -0.8467 |                                54.0375 |                                7.2132 |                               -0.0042 |                              0.0017 |                                 0.0026 |                                      0.0018 |                                 0.3962 |                                -0.8637 |                                57.0192 |                                7.8912 |                               -0.0043 |                              0.0018 |                                 0.0028 |                                      0.0019 |                                 0.4168 |                                -0.8783 |                                59.9005 |                                8.5841 |                               -0.0044 |                              0.0019 |
        |            330 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0.0001 |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0026 |                                0.0001 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0013 |                                 0.0208 |                                0.0006 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0006 |                                -0.0048 |                                 0.0908 |                                0.0029 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0017 |                                -0.0126 |                                 0.272  |                                0.01   |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0039 |                                -0.0262 |                                 0.6336 |                                0.0261 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0077 |                                -0.0462 |                                 1.2392 |                                0.0567 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0133 |                                -0.0725 |                                 2.1352 |                                0.1074 |                               -0.0003 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0209 |                                -0.1044 |                                 3.3466 |                                0.1836 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0305 |                                -0.1408 |                                 4.8784 |                                0.2898 |                               -0.0007 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0421 |                                -0.1805 |                                 6.7188 |                                0.4298 |                               -0.0009 |                              0.0002 |                                 0.0003 |                                      0.0002 |                                 0.0555 |                                -0.2223 |                                 8.8444 |                                0.6061 |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0705 |                                -0.2651 |                                11.2241 |                                0.8202 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0004 |                                 0.0869 |                                -0.308  |                                13.8229 |                                1.073  |                               -0.0015 |                              0.0004 |                                 0.0006 |                                      0.0004 |                                 0.1046 |                                -0.3502 |                                16.6046 |                                1.3645 |                               -0.0017 |                              0.0004 |                                 0.0008 |                                      0.0005 |                                 0.1232 |                                -0.3912 |                                19.5338 |                                1.694  |                               -0.0019 |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.1425 |                                -0.4305 |                                22.5768 |                                2.0606 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1625 |                                -0.4679 |                                25.7029 |                                2.4626 |                               -0.0024 |                              0.0007 |                                 0.0011 |                                      0.0008 |                                 0.1828 |                                -0.5031 |                                28.8844 |                                2.8986 |                               -0.0025 |                              0.0008 |                                 0.0012 |                                      0.0008 |                                 0.2034 |                                -0.536  |                                32.0968 |                                3.3667 |                               -0.0027 |                              0.0008 |                                 0.0014 |                                      0.0009 |                                 0.2242 |                                -0.5665 |                                35.3188 |                                3.8649 |                               -0.0029 |                              0.0009 |                                 0.0015 |                                      0.001  |                                 0.2449 |                                -0.5947 |                                38.532  |                                4.3913 |                               -0.0031 |                              0.001  |                                 0.0016 |                                      0.0011 |                                 0.2655 |                                -0.6206 |                                41.7208 |                                4.9438 |                               -0.0032 |                              0.0011 |                                 0.0017 |                                      0.0012 |                                 0.2859 |                                -0.6443 |                                44.8721 |                                5.5206 |                               -0.0033 |                              0.0012 |                                 0.0019 |                                      0.0013 |                                 0.3061 |                                -0.6658 |                                47.975  |                                6.1196 |                               -0.0035 |                              0.0013 |                                 0.002  |                                      0.0014 |                                 0.326  |                                -0.6852 |                                51.0208 |                                6.7391 |                               -0.0036 |                              0.0014 |
        |            335 |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0      |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                 0      |                                 0.0004 |                                0      |                                0      |                              0      |                                 0      |                                      0      |                                 0      |                                -0.0002 |                                 0.0042 |                                0.0001 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0001 |                                -0.0012 |                                 0.0238 |                                0.0007 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0005 |                                -0.0037 |                                 0.086  |                                0.0029 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0013 |                                -0.0089 |                                 0.2319 |                                0.0089 |                                0      |                              0      |                                 0      |                                      0      |                                 0.0029 |                                -0.0176 |                                 0.5098 |                                0.0216 |                               -0.0001 |                              0      |                                 0      |                                      0      |                                 0.0056 |                                -0.0304 |                                 0.9665 |                                0.0451 |                               -0.0002 |                              0      |                                 0.0001 |                                      0      |                                 0.0095 |                                -0.0474 |                                 1.6406 |                                0.0835 |                               -0.0002 |                              0      |                                 0.0001 |                                      0.0001 |                                 0.0148 |                                -0.0684 |                                 2.5584 |                                0.1411 |                               -0.0004 |                              0.0001 |                                 0.0001 |                                      0.0001 |                                 0.0217 |                                -0.0929 |                                 3.7335 |                                0.2217 |                               -0.0005 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.03   |                                -0.1202 |                                 5.1674 |                                0.3287 |                               -0.0006 |                              0.0001 |                                 0.0002 |                                      0.0001 |                                 0.0399 |                                -0.1498 |                                 6.8519 |                                0.4648 |                               -0.0008 |                              0.0001 |                                 0.0003 |                                      0.0002 |                                 0.0511 |                                -0.1808 |                                 8.7717 |                                0.6321 |                               -0.0009 |                              0.0002 |                                 0.0004 |                                      0.0002 |                                 0.0636 |                                -0.2128 |                                10.9061 |                                0.8319 |                               -0.0011 |                              0.0002 |                                 0.0004 |                                      0.0003 |                                 0.0773 |                                -0.2451 |                                13.2316 |                                1.0652 |                               -0.0013 |                              0.0003 |                                 0.0005 |                                      0.0003 |                                 0.092  |                                -0.2773 |                                15.7231 |                                1.3321 |                               -0.0015 |                              0.0003 |                                 0.0006 |                                      0.0004 |                                 0.1075 |                                -0.309  |                                18.3554 |                                1.6326 |                               -0.0016 |                              0.0004 |                                 0.0007 |                                      0.0005 |                                 0.1237 |                                -0.3398 |                                21.104  |                                1.966  |                               -0.0018 |                              0.0005 |                                 0.0008 |                                      0.0005 |                                 0.1405 |                                -0.3696 |                                23.9454 |                                2.3317 |                               -0.002  |                              0.0005 |                                 0.0009 |                                      0.0006 |                                 0.1578 |                                -0.3982 |                                26.858  |                                2.7284 |                               -0.0022 |                              0.0006 |                                 0.001  |                                      0.0007 |                                 0.1755 |                                -0.4254 |                                29.8219 |                                3.1551 |                               -0.0023 |                              0.0007 |                                 0.0011 |                                      0.0007 |                                 0.1933 |                                -0.4511 |                                32.8191 |                                3.6104 |                               -0.0025 |                              0.0007 |                                 0.0012 |                                      0.0008 |                                 0.2113 |                                -0.4753 |                                35.8336 |                                4.0928 |                               -0.0026 |                              0.0008 |                                 0.0013 |                                      0.0009 |                                 0.2294 |                                -0.498  |                                38.8513 |                                4.6009 |                               -0.0027 |                              0.0009 |                                 0.0014 |                                      0.0009 |                                 0.2475 |                                -0.5192 |                                41.8597 |                                5.1332 |                               -0.0029 |                              0.0009 |
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

        if all(greek.empty for greek in greeks.values()):
            # Each greek has logged why it is empty, which is the same reason for all of them.
            logger.warning("None of the greeks could be calculated.")
            return pd.DataFrame()

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    @handle_errors
    def get_gamma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the gamma of an option based on the Black Scholes Model: how much the delta changes for a 1.00
        change in the price of the underlying asset.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        gamma is the rate of change of the delta with respect to the price of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
                means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             22 |       0.0012 |       0.0165 |       0.0502 |
        |             23 |       0.0395 |       0.0929 |       0.1195 |
        |             24 |       0.2623 |       0.24   |       0.1926 |
        |             25 |       0.4397 |       0.3115 |       0.22   |
        |             26 |       0.2178 |       0.2197 |       0.1852 |
        |             27 |       0.0366 |       0.0902 |       0.119  |
        |             28 |       0.0023 |       0.0229 |       0.06   |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_dual_gamma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the dual gamma of an option based on the Black Scholes Model: how much the dual delta changes for
        a 1.00 change in the strike price.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        dual gamma is the second derivative of the option price with respect to the strike price, i.e. the rate of
        change of the dual delta as the strike price changes. It is the strike-space counterpart of gamma and
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             21 |       0      |       0.0003 |       0.0081 |
        |             22 |       0.0003 |       0.009  |       0.0446 |
        |             23 |       0.0255 |       0.0866 |       0.1339 |
        |             24 |       0.2943 |       0.2845 |       0.2348 |
        |             25 |       0.5059 |       0.3613 |       0.2564 |
        |             26 |       0.1614 |       0.198  |       0.1841 |
        |             27 |       0.0115 |       0.0514 |       0.0911 |
        |             28 |       0.0002 |       0.0069 |       0.0324 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_vanna(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the vanna of an option based on the Black Scholes Model: how much the delta changes for a 1.00
        change in the volatility, or equally the vega for a 1.00 change in the price of the underlying asset.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        vanna is the rate of change of the vega with respect to the price of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             96 |      -0.3178 |      -0.595  |      -0.6673 |
        |             98 |      -0.4447 |      -0.6644 |      -0.6908 |
        |            100 |      -0.5731 |      -0.706  |      -0.6877 |
        |            102 |      -0.6795 |      -0.7114 |      -0.6556 |
        |            104 |      -0.7374 |      -0.675  |      -0.5939 |
        |            106 |      -0.7247 |      -0.5955 |      -0.504  |
        |            108 |      -0.6296 |      -0.4753 |      -0.3894 |
        |            110 |      -0.4551 |      -0.3213 |      -0.2548 |
        |            112 |      -0.2189 |      -0.1429 |      -0.1063 |
        |            114 |       0.0498 |       0.048  |       0.0496 |
        |            116 |       0.3173 |       0.2394 |       0.2062 |
        |            118 |       0.5519 |       0.4197 |       0.3572 |
        |            120 |       0.7301 |       0.5791 |       0.497  |
        |            122 |       0.8397 |       0.7105 |       0.621  |
        |            124 |       0.8799 |       0.8093 |       0.726  |
        |            126 |       0.8593 |       0.874  |       0.8097 |
        |            128 |       0.7923 |       0.9056 |       0.8714 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_charm(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        put_option: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the charm of an option based on the Black Scholes Model: how much the delta changes as time
        passes, the delta decay.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        charm is the rate of change of the delta with respect to the time to expiration.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
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
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            290 |       0.0264 |       0.0405 |       0.1977 |
        |            295 |       0.0268 |       0.076  |       0.332  |
        |            300 |       0.0307 |       0.1726 |       0.5239 |
        |            305 |       0.057  |       0.3914 |       0.7663 |
        |            310 |       0.1843 |       0.8006 |       1.0315 |
        |            315 |       0.6286 |       1.4229 |       1.2704 |
        |            320 |       1.7336 |       2.1641 |       1.4208 |
        |            325 |       3.6153 |       2.7782 |       1.4233 |
        |            330 |       5.5281 |       2.9352 |       1.2406 |
        |            335 |       5.8384 |       2.3897 |       0.8727 |
        |            340 |       3.3616 |       1.1521 |       0.3608 |
        |            345 |      -1.0896 |      -0.4685 |      -0.2214 |
        |            350 |      -5.0641 |      -1.9984 |      -0.7863 |
        |            355 |      -6.6128 |      -3.0258 |      -1.2543 |
        |            360 |      -5.729  |      -3.3722 |      -1.5714 |
        |            365 |      -3.7544 |      -3.1181 |      -1.7176 |
        |            370 |      -1.9575 |      -2.5022 |      -1.7048 |
        |            375 |      -0.8345 |      -1.7834 |      -1.5678 |
        |            380 |      -0.2961 |      -1.1451 |      -1.352  |
        |            385 |      -0.0886 |      -0.6688 |      -1.1021 |
        |            390 |      -0.0226 |      -0.358  |      -0.8539 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_vomma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the vomma of an option based on the Black Scholes Model: how much the vega changes for a 1.00
        change in the volatility, the convexity of the option value in the volatility.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        vomma is the rate of change of the vega with respect to the volatility of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            280 |      51.2097 |      52.1616 |      47.8182 |
        |            285 |      47.8381 |      45.2178 |      40.3535 |
        |            290 |      42.7853 |      37.8675 |      32.9635 |
        |            295 |      36.4289 |      30.4352 |      25.8688 |
        |            300 |      29.2674 |      23.2408 |      19.2724 |
        |            305 |      21.8715 |      16.5838 |      13.3537 |
        |            310 |      14.8278 |      10.7301 |       8.2647 |
        |            315 |       8.6846 |       5.9013 |       4.1267 |
        |            320 |       3.9039 |       2.2678 |       1.0291 |
        |            325 |       0.8251 |      -0.0552 |      -0.9709 |
        |            330 |      -0.3565 |      -1.0091 |      -1.8476 |
        |            335 |       0.4045 |      -0.5891 |      -1.6048 |
        |            340 |       3.0113 |       1.1589 |      -0.2734 |
        |            345 |       7.245  |       4.1456 |       2.0917 |
        |            350 |      12.7927 |       8.2446 |       5.4153 |
        |            355 |      19.2798 |      13.3011 |       9.6054 |
        |            360 |      26.3039 |      19.1402 |      14.5571 |
        |            365 |      33.4654 |      25.575  |      20.1562 |
        |            370 |      40.3941 |      32.4143 |      26.2832 |
        |            375 |      46.7696 |      39.4695 |      32.8159 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_vera(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the vera of an option based on the Black Scholes Model: how much the rho changes for a 1.00 change
        in the volatility.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        vera is the rate of change of the rho with respect to volatility.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            920 |     -45.0096 |    -157.215  |    -295.291  |
        |            940 |     -50.1022 |    -152.342  |    -280.248  |
        |            960 |     -52.6538 |    -143.623  |    -261.571  |
        |            980 |     -52.0109 |    -131.27   |    -239.609  |
        |           1000 |     -47.8614 |    -115.641  |    -214.763  |
        |           1020 |     -40.299  |     -97.212  |    -187.471  |
        |           1040 |     -29.8143 |     -76.5443 |    -158.19   |
        |           1060 |     -17.2215 |     -54.2545 |    -127.386  |
        |           1080 |      -3.5354 |     -30.9812 |     -95.5234 |
        |           1100 |      10.1711 |      -7.3547 |     -63.0518 |
        |           1120 |      22.9037 |      16.0281 |     -30.3999 |
        |           1140 |      33.8555 |      38.6251 |       2.0327 |
        |           1160 |      42.4735 |      59.9646 |      33.8803 |
        |           1180 |      48.481  |      79.6557 |      64.8152 |
        |           1200 |      51.8615 |      97.3928 |      94.55   |
        |           1220 |      52.8126 |     112.957  |     122.838  |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_veta(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the veta of an option based on the Black Scholes Model: how much the vega changes as time passes.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        veta is the rate of change of the vega with respect to the time to expiration.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_partial_derivative(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the partial derivative of an option based on the Black Scholes Model: how fast the dual delta
        changes with the strike price, the second derivative of the option value with respect to the strike price.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        partial derivative is the second derivative of the option price with respect to the strike price, the rate
        at which the dual delta changes with the strike.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             49 |       0.0017 |       0.014  |       0.0333 |
        |             50 |       0.008  |       0.0296 |       0.0478 |
        |             51 |       0.0269 |       0.0536 |       0.0635 |
        |             52 |       0.0665 |       0.0833 |       0.0783 |
        |             53 |       0.1241 |       0.1125 |       0.09   |
        |             54 |       0.1768 |       0.1328 |       0.0967 |
        |             55 |       0.1955 |       0.1382 |       0.0975 |
        |             56 |       0.17   |       0.1275 |       0.0927 |
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
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    @logger_model.log_each_message_once()
    def collect_third_order_greeks(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the third order Greeks of an option based on the Black Scholes Model: how the second order Greeks
        themselves respond to the price of the underlying asset, the volatility and time.

        This will return the following Greeks per Strike Price and Expiration Date:

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            215 |                                 0      |                                 0      |                                 0      |                                  0      |                                 0      |                                 0      |                                 0      |                                  0.0002 |                                 0      |                                 0      |                                -0.0001 |                                  0.0462 |                                 0      |                                 0      |                                -0.0012 |                                  0.6517 |                                 0      |                                 0.0002 |                                -0.0046 |                                  2.8961 |                                 0      |                                 0.0005 |                                -0.01   |                                  7.3099 |                                 0      |                                 0.001  |                                -0.0163 |                                 13.4344 |                                 0      |                                 0.0016 |                                -0.0224 |                                 20.3143 |                                 0      |                                 0.0022 |                                -0.0277 |                                 27.0161 |                                 0      |                                 0.0028 |                                -0.0318 |                                 32.8626 |                                 0      |                                 0.0034 |                                -0.0346 |                                 37.4632 |                                 0      |                                 0.0039 |                                -0.0364 |                                 40.6573 |                                 0      |                                 0.0043 |                                -0.0373 |                                 42.4422 |                                 0      |                                 0.0047 |                                -0.0374 |                                 42.9098 |                                -0.0001 |                                 0.0049 |                                -0.037  |                                 42.202  |                                -0.0001 |                                 0.0052 |                                -0.0362 |                                 40.4798 |                                -0.0001 |                                 0.0053 |                                -0.0351 |                                 37.9056 |                                -0.0001 |                                 0.0054 |                                -0.0338 |                                 34.6322 |                                -0.0001 |                                 0.0055 |                                -0.0323 |                                 30.7979 |                                -0.0001 |                                 0.0055 |                                -0.0308 |                                 26.5244 |                                -0.0001 |                                 0.0055 |                                -0.0292 |                                 21.9169 |                                -0.0001 |                                 0.0055 |                                -0.0277 |                                 17.0648 |                                -0.0001 |                                 0.0054 |                                -0.0262 |                                 12.0433 |                                -0.0001 |                                 0.0053 |                                -0.0247 |                                  6.9149 |                                -0.0001 |                                 0.0052 |                                -0.0232 |                                  1.7314 |                                -0.0001 |                                 0.0051 |                                -0.0219 |                                 -3.465  |                                -0.0001 |                                 0.005  |                                -0.0205 |                                 -8.6399 |                                -0.0001 |                                 0.0049 |                                -0.0193 |                                -13.7657 |                                -0.0001 |                                 0.0048 |                                -0.0181 |                                -18.8205 |
        |            220 |                                 0      |                                 0      |                                 0      |                                  0      |                                 0      |                                 0      |                                 0      |                                  0.0023 |                                 0      |                                 0      |                                -0.0007 |                                  0.2279 |                                 0      |                                 0.0002 |                                -0.0045 |                                  1.9556 |                                 0      |                                 0.0005 |                                -0.0124 |                                  6.4192 |                                 0      |                                 0.0012 |                                -0.0222 |                                 13.1727 |                                 0      |                                 0.0019 |                                -0.0315 |                                 20.7745 |                                 0      |                                 0.0028 |                                -0.0389 |                                 27.857  |                                 0      |                                 0.0035 |                                -0.0442 |                                 33.5461 |                                 0      |                                 0.0042 |                                -0.0473 |                                 37.4484 |                                -0.0001 |                                 0.0048 |                                -0.0488 |                                 39.5045 |                                -0.0001 |                                 0.0052 |                                -0.0489 |                                 39.8469 |                                -0.0001 |                                 0.0056 |                                -0.0481 |                                 38.6988 |                                -0.0001 |                                 0.0058 |                                -0.0465 |                                 36.3117 |                                -0.0001 |                                 0.006  |                                -0.0446 |                                 32.9304 |                                -0.0001 |                                 0.0061 |                                -0.0424 |                                 28.7768 |                                -0.0001 |                                 0.0061 |                                -0.04   |                                 24.0431 |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                 18.8907 |                                -0.0001 |                                 0.006  |                                -0.0352 |                                 13.453  |                                -0.0001 |                                 0.0059 |                                -0.0329 |                                  7.838  |                                -0.0001 |                                 0.0058 |                                -0.0307 |                                  2.1323 |                                -0.0001 |                                 0.0057 |                                -0.0285 |                                 -3.5957 |                                -0.0001 |                                 0.0055 |                                -0.0265 |                                 -9.2924 |                                -0.0001 |                                 0.0053 |                                -0.0246 |                                -14.9166 |                                -0.0001 |                                 0.0052 |                                -0.0228 |                                -20.4368 |                                -0.0001 |                                 0.005  |                                -0.0211 |                                -25.8298 |                                -0.0001 |                                 0.0048 |                                -0.0195 |                                -31.0786 |                                -0.0001 |                                 0.0046 |                                -0.018  |                                -36.1715 |                                -0.0001 |                                 0.0044 |                                -0.0167 |                                -41.1007 |
        |            225 |                                 0      |                                 0      |                                 0      |                                  0      |                                 0      |                                 0      |                                -0.0001 |                                  0.0214 |                                 0      |                                 0.0001 |                                -0.0032 |                                  0.8894 |                                 0      |                                 0.0005 |                                -0.014  |                                  4.8702 |                                 0      |                                 0.0013 |                                -0.0293 |                                 12.1181 |                                 0      |                                 0.0023 |                                -0.0439 |                                 20.5173 |                                 0      |                                 0.0034 |                                -0.0548 |                                 27.9778 |                                 0      |                                 0.0044 |                                -0.0615 |                                 33.339  |                                -0.0001 |                                 0.0051 |                                -0.0646 |                                 36.2508 |                                -0.0001 |                                 0.0058 |                                -0.065  |                                 36.839  |                                -0.0001 |                                 0.0062 |                                -0.0636 |                                 35.4428 |                                -0.0001 |                                 0.0065 |                                -0.061  |                                 32.4608 |                                -0.0001 |                                 0.0067 |                                -0.0577 |                                 28.2735 |                                -0.0001 |                                 0.0067 |                                -0.054  |                                 23.212  |                                -0.0001 |                                 0.0067 |                                -0.0502 |                                 17.5493 |                                -0.0001 |                                 0.0066 |                                -0.0464 |                                 11.5032 |                                -0.0001 |                                 0.0065 |                                -0.0427 |                                  5.2435 |                                -0.0001 |                                 0.0063 |                                -0.0392 |                                 -1.1    |                                -0.0001 |                                 0.0061 |                                -0.0359 |                                 -7.43   |                                -0.0001 |                                 0.0059 |                                -0.0328 |                                -13.6744 |                                -0.0001 |                                 0.0057 |                                -0.03   |                                -19.7812 |                                -0.0001 |                                 0.0055 |                                -0.0273 |                                -25.7141 |                                -0.0001 |                                 0.0052 |                                -0.0249 |                                -31.4482 |                                -0.0001 |                                 0.005  |                                -0.0227 |                                -36.9681 |                                -0.0001 |                                 0.0047 |                                -0.0206 |                                -42.2649 |                                -0.0001 |                                 0.0045 |                                -0.0187 |                                -47.3352 |                                -0.0001 |                                 0.0043 |                                -0.017  |                                -52.179  |                                -0.0001 |                                 0.004  |                                -0.0155 |                                -56.7994 |                                -0.0001 |                                 0.0038 |                                -0.014  |                                -61.2015 |
        |            230 |                                 0      |                                 0      |                                 0      |                                  0      |                                 0      |                                 0      |                                -0.0009 |                                  0.1466 |                                 0      |                                 0.0003 |                                -0.0126 |                                  2.7617 |                                 0      |                                 0.0013 |                                -0.0367 |                                 10.0594 |                                 0      |                                 0.0027 |                                -0.0608 |                                 19.3768 |                                -0.0001 |                                 0.0041 |                                -0.0774 |                                 27.319  |                                -0.0001 |                                 0.0053 |                                -0.086  |                                 32.2265 |                                -0.0001 |                                 0.0063 |                                -0.0883 |                                 33.8626 |                                -0.0001 |                                 0.0069 |                                -0.0865 |                                 32.6687 |                                -0.0001 |                                 0.0073 |                                -0.0821 |                                 29.2873 |                                -0.0001 |                                 0.0075 |                                -0.0765 |                                 24.3399 |                                -0.0001 |                                 0.0075 |                                -0.0703 |                                 18.3477 |                                -0.0001 |                                 0.0074 |                                -0.064  |                                 11.7171 |                                -0.0001 |                                 0.0073 |                                -0.0579 |                                  4.7523 |                                -0.0001 |                                 0.007  |                                -0.0522 |                                 -2.326  |                                -0.0001 |                                 0.0068 |                                -0.0469 |                                 -9.362  |                                -0.0001 |                                 0.0064 |                                -0.042  |                                -16.2482 |                                -0.0001 |                                 0.0061 |                                -0.0376 |                                -22.9134 |                                -0.0001 |                                 0.0058 |                                -0.0336 |                                -29.3124 |                                -0.0001 |                                 0.0055 |                                -0.03   |                                -35.4191 |                                -0.0001 |                                 0.0051 |                                -0.0267 |                                -41.2206 |                                -0.0001 |                                 0.0048 |                                -0.0238 |                                -46.7137 |                                -0.0001 |                                 0.0045 |                                -0.0212 |                                -51.9017 |                                -0.0001 |                                 0.0042 |                                -0.0188 |                                -56.7925 |                                -0.0001 |                                 0.0039 |                                -0.0167 |                                -61.3966 |                                -0.0001 |                                 0.0036 |                                -0.0148 |                                -65.7265 |                                -0.0001 |                                 0.0033 |                                -0.013  |                                -69.7957 |                                -0.0001 |                                 0.0031 |                                -0.0115 |                                -73.618  |                                -0.0001 |                                 0.0028 |                                -0.0101 |                                -77.2074 |
        |            235 |                                 0      |                                 0      |                                 0      |                                  0.0003 |                                 0      |                                 0.0001 |                                -0.006  |                                  0.7369 |                                 0      |                                 0.0011 |                                -0.0402 |                                  6.8209 |                                 0      |                                 0.0029 |                                -0.0824 |                                 17.1105 |                                -0.0001 |                                 0.0049 |                                -0.1103 |                                 25.8464 |                                -0.0001 |                                 0.0064 |                                -0.1216 |                                 30.285  |                                -0.0001 |                                 0.0075 |                                -0.1215 |                                 30.3906 |                                -0.0001 |                                 0.0082 |                                -0.115  |                                 27.1327 |                                -0.0001 |                                 0.0084 |                                -0.1054 |                                 21.593  |                                -0.0001 |                                 0.0084 |                                -0.0947 |                                 14.6651 |                                -0.0001 |                                 0.0083 |                                -0.0841 |                                  7.0056 |                                -0.0001 |                                 0.008  |                                -0.0741 |                                 -0.9318 |                                -0.0001 |                                 0.0076 |                                -0.0649 |                                 -8.8472 |                                -0.0001 |                                 0.0071 |                                -0.0567 |                                -16.55   |                                -0.0001 |                                 0.0067 |                                -0.0494 |                                -23.9252 |                                -0.0001 |                                 0.0062 |                                -0.043  |                                -30.9089 |                                -0.0001 |                                 0.0058 |                                -0.0373 |                                -37.4714 |                                -0.0001 |                                 0.0053 |                                -0.0324 |                                -43.6049 |                                -0.0001 |                                 0.0049 |                                -0.028  |                                -49.3159 |                                -0.0001 |                                 0.0045 |                                -0.0242 |                                -54.6196 |                                -0.0001 |                                 0.0041 |                                -0.0208 |                                -59.5361 |                                -0.0001 |                                 0.0037 |                                -0.0178 |                                -64.0882 |                                -0.0001 |                                 0.0033 |                                -0.0152 |                                -68.2999 |                                -0.0001 |                                 0.003  |                                -0.013  |                                -72.1949 |                                -0.0001 |                                 0.0027 |                                -0.0109 |                                -75.7964 |                                -0.0001 |                                 0.0023 |                                -0.0091 |                                -79.1265 |                                -0.0001 |                                 0.0021 |                                -0.0076 |                                -82.2061 |                                -0.0001 |                                 0.0018 |                                -0.0062 |                                -85.0546 |                                -0.0001 |                                 0.0015 |                                -0.0049 |                                -87.6902 |
        |            240 |                                 0      |                                 0      |                                -0.0002 |                                  0.0079 |                                 0      |                                 0.0005 |                                -0.0292 |                                  2.7239 |                                 0      |                                 0.0028 |                                -0.1053 |                                 13.2964 |                                -0.0001 |                                 0.0056 |                                -0.1584 |                                 23.4961 |                                -0.0001 |                                 0.0077 |                                -0.1751 |                                 27.6952 |                                -0.0002 |                                 0.009  |                                -0.1694 |                                 26.08   |                                -0.0002 |                                 0.0095 |                                -0.1535 |                                 20.5215 |                                -0.0002 |                                 0.0096 |                                -0.1344 |                                 12.7461 |                                -0.0002 |                                 0.0093 |                                -0.1155 |                                  3.965  |                                -0.0002 |                                 0.0088 |                                -0.0982 |                                 -5.0605 |                                -0.0002 |                                 0.0082 |                                -0.083  |                                -13.8837 |                                -0.0002 |                                 0.0075 |                                -0.0698 |                                -22.2608 |                                -0.0002 |                                 0.0069 |                                -0.0586 |                                -30.0726 |                                -0.0002 |                                 0.0062 |                                -0.049  |                                -37.275  |                                -0.0002 |                                 0.0056 |                                -0.0409 |                                -43.8674 |                                -0.0002 |                                 0.005  |                                -0.034  |                                -49.8736 |                                -0.0002 |                                 0.0044 |                                -0.0281 |                                -55.3296 |                                -0.0002 |                                 0.0039 |                                -0.0232 |                                -60.2772 |                                -0.0001 |                                 0.0034 |                                -0.0189 |                                -64.7595 |                                -0.0001 |                                 0.0029 |                                -0.0153 |                                -68.8186 |                                -0.0001 |                                 0.0025 |                                -0.0122 |                                -72.4942 |                                -0.0001 |                                 0.0021 |                                -0.0095 |                                -75.8232 |                                -0.0001 |                                 0.0017 |                                -0.0072 |                                -78.8393 |                                -0.0001 |                                 0.0014 |                                -0.0053 |                                -81.5732 |                                -0.0001 |                                 0.0011 |                                -0.0036 |                                -84.0525 |                                -0.0001 |                                 0.0007 |                                -0.0021 |                                -86.3021 |                                -0.0001 |                                 0.0005 |                                -0.0008 |                                -88.3441 |                                -0.0001 |                                 0.0002 |                                 0.0003 |                                -90.1987 |                                -0.0001 |                                 0      |                                 0.0012 |                                -91.8837 |
        |            245 |                                 0      |                                 0      |                                -0.0028 |                                  0.107  |                                 0      |                                 0.0019 |                                -0.1076 |                                  7.3596 |                                -0.0001 |                                 0.006  |                                -0.2274 |                                 19.9799 |                                -0.0002 |                                 0.0092 |                                -0.2596 |                                 24.7341 |                                -0.0002 |                                 0.0107 |                                -0.2416 |                                 21.3396 |                                -0.0002 |                                 0.011  |                                -0.2069 |                                 13.2918 |                                -0.0002 |                                 0.0106 |                                -0.1707 |                                  3.3131 |                                -0.0002 |                                 0.0099 |                                -0.1383 |                                 -7.0298 |                                -0.0002 |                                 0.009  |                                -0.111  |                                -16.9456 |                                -0.0002 |                                 0.008  |                                -0.0886 |                                -26.0822 |                                -0.0002 |                                 0.007  |                                -0.0705 |                                -34.3207 |                                -0.0002 |                                 0.0061 |                                -0.0558 |                                -41.6595 |                                -0.0002 |                                 0.0052 |                                -0.0438 |                                -48.1522 |                                -0.0002 |                                 0.0044 |                                -0.0342 |                                -53.875  |                                -0.0002 |                                 0.0037 |                                -0.0263 |                                -58.9102 |                                -0.0002 |                                 0.003  |                                -0.0199 |                                -63.3373 |                                -0.0002 |                                 0.0024 |                                -0.0146 |                                -67.23   |                                -0.0002 |                                 0.0019 |                                -0.0102 |                                -70.654  |                                -0.0002 |                                 0.0014 |                                -0.0067 |                                -73.6678 |                                -0.0002 |                                 0.0009 |                                -0.0037 |                                -76.3224 |                                -0.0001 |                                 0.0005 |                                -0.0012 |                                -78.6621 |                                -0.0001 |                                 0.0001 |                                 0.0008 |                                -80.7256 |                                -0.0001 |                                -0.0002 |                                 0.0025 |                                -82.5462 |                                -0.0001 |                                -0.0006 |                                 0.0039 |                                -84.1532 |                                -0.0001 |                                -0.0008 |                                 0.0051 |                                -85.5716 |                                -0.0001 |                                -0.0011 |                                 0.0061 |                                -86.8235 |                                -0.0001 |                                -0.0014 |                                 0.0069 |                                -87.928  |                                -0.0001 |                                -0.0016 |                                 0.0076 |                                -88.9017 |                                -0.0001 |                                -0.0018 |                                 0.0081 |                                -89.7592 |
        |            250 |                                 0      |                                 0.0003 |                                -0.0306 |                                  0.8572 |                                -0.0001 |                                 0.0054 |                                -0.3043 |                                 14.2006 |                                -0.0002 |                                 0.0106 |                                -0.4024 |                                 21.678  |                                -0.0003 |                                 0.0127 |                                -0.3589 |                                 16.771  |                                -0.0003 |                                 0.0126 |                                -0.2843 |                                  6.0745 |                                -0.0003 |                                 0.0115 |                                -0.2159 |                                 -5.9652 |                                -0.0003 |                                 0.0101 |                                -0.1611 |                                -17.4028 |                                -0.0003 |                                 0.0086 |                                -0.1192 |                                -27.5629 |                                -0.0003 |                                 0.0071 |                                -0.0875 |                                -36.3189 |                                -0.0003 |                                 0.0058 |                                -0.0635 |                                -43.7584 |                                -0.0003 |                                 0.0046 |                                -0.0454 |                                -50.0391 |                                -0.0002 |                                 0.0036 |                                -0.0315 |                                -55.3289 |                                -0.0002 |                                 0.0026 |                                -0.0208 |                                -59.7821 |                                -0.0002 |                                 0.0018 |                                -0.0126 |                                -63.5334 |                                -0.0002 |                                 0.0011 |                                -0.0062 |                                -66.6964 |                                -0.0002 |                                 0.0004 |                                -0.0012 |                                -69.366  |                                -0.0002 |                                -0.0001 |                                 0.0027 |                                -71.6207 |                                -0.0002 |                                -0.0006 |                                 0.0058 |                                -73.5258 |                                -0.0002 |                                -0.0011 |                                 0.0082 |                                -75.1352 |                                -0.0002 |                                -0.0015 |                                 0.01   |                                -76.4938 |                                -0.0002 |                                -0.0018 |                                 0.0115 |                                -77.639  |                                -0.0001 |                                -0.0022 |                                 0.0126 |                                -78.6019 |                                -0.0001 |                                -0.0024 |                                 0.0135 |                                -79.4087 |                                -0.0001 |                                -0.0027 |                                 0.0142 |                                -80.0814 |                                -0.0001 |                                -0.0029 |                                 0.0146 |                                -80.6388 |                                -0.0001 |                                -0.0031 |                                 0.015  |                                -81.0966 |                                -0.0001 |                                -0.0033 |                                 0.0152 |                                -81.4682 |                                -0.0001 |                                -0.0035 |                                 0.0154 |                                -81.7651 |                                -0.0001 |                                -0.0036 |                                 0.0154 |                                -81.9973 |
        |            255 |                                 0      |                                 0.0019 |                                -0.2109 |                                  4.0061 |                                -0.0003 |                                 0.0115 |                                -0.655  |                                 18.248  |                                -0.0004 |                                 0.0152 |                                -0.5732 |                                 13.2126 |                                -0.0004 |                                 0.0144 |                                -0.4056 |                                 -0.3183 |                                -0.0004 |                                 0.0121 |                                -0.2723 |                                -14.2004 |                                -0.0004 |                                 0.0097 |                                -0.1794 |                                -26.0917 |                                -0.0004 |                                 0.0074 |                                -0.1163 |                                -35.7063 |                                -0.0003 |                                 0.0054 |                                -0.0733 |                                -43.3254 |                                -0.0003 |                                 0.0037 |                                -0.0437 |                                -49.3264 |                                -0.0003 |                                 0.0023 |                                -0.023  |                                -54.0499 |                                -0.0003 |                                 0.0011 |                                -0.0085 |                                -57.7723 |                                -0.0003 |                                 0.0001 |                                 0.0018 |                                -60.7097 |                                -0.0002 |                                -0.0008 |                                 0.0092 |                                -63.029  |                                -0.0002 |                                -0.0015 |                                 0.0144 |                                -64.8587 |                                -0.0002 |                                -0.0021 |                                 0.0182 |                                -66.2982 |                                -0.0002 |                                -0.0026 |                                 0.0208 |                                -67.425  |                                -0.0002 |                                -0.0031 |                                 0.0226 |                                -68.2995 |                                -0.0002 |                                -0.0035 |                                 0.0238 |                                -68.9696 |                                -0.0002 |                                -0.0038 |                                 0.0246 |                                -69.473  |                                -0.0002 |                                -0.0041 |                                 0.025  |                                -69.8402 |                                -0.0002 |                                -0.0044 |                                 0.0252 |                                -70.0954 |                                -0.0001 |                                -0.0046 |                                 0.0252 |                                -70.2585 |                                -0.0001 |                                -0.0048 |                                 0.0251 |                                -70.3456 |                                -0.0001 |                                -0.005  |                                 0.0248 |                                -70.3698 |                                -0.0001 |                                -0.0051 |                                 0.0245 |                                -70.342  |                                -0.0001 |                                -0.0052 |                                 0.0241 |                                -70.2711 |                                -0.0001 |                                -0.0053 |                                 0.0237 |                                -70.1646 |                                -0.0001 |                                -0.0054 |                                 0.0233 |                                -70.0287 |                                -0.0001 |                                -0.0055 |                                 0.0228 |                                -69.8686 |
        |            260 |                                -0.0002 |                                 0.008  |                                -0.9052 |                                 10.3989 |                                -0.0006 |                                 0.0185 |                                -1.0466 |                                 11.7618 |                                -0.0006 |                                 0.0166 |                                -0.6246 |                                 -4.8618 |                                -0.0006 |                                 0.0123 |                                -0.3431 |                                -20.4075 |                                -0.0005 |                                 0.0082 |                                -0.1818 |                                -31.9627 |                                -0.0005 |                                 0.005  |                                -0.0893 |                                -40.1559 |                                -0.0004 |                                 0.0024 |                                -0.035  |                                -45.9201 |                                -0.0004 |                                 0.0004 |                                -0.0023 |                                -49.9803 |                                -0.0003 |                                -0.0011 |                                 0.0176 |                                -52.8433 |                                -0.0003 |                                -0.0023 |                                 0.0298 |                                -54.8557 |                                -0.0003 |                                -0.0033 |                                 0.0371 |                                -56.2561 |                                -0.0003 |                                -0.0041 |                                 0.0414 |                                -57.2105 |                                -0.0002 |                                -0.0047 |                                 0.0437 |                                -57.8362 |                                -0.0002 |                                -0.0052 |                                 0.0447 |                                -58.2172 |                                -0.0002 |                                -0.0056 |                                 0.0448 |                                -58.4149 |                                -0.0002 |                                -0.006  |                                 0.0444 |                                -58.4742 |                                -0.0002 |                                -0.0062 |                                 0.0437 |                                -58.4288 |                                -0.0002 |                                -0.0065 |                                 0.0427 |                                -58.3042 |                                -0.0002 |                                -0.0067 |                                 0.0415 |                                -58.1195 |                                -0.0002 |                                -0.0068 |                                 0.0403 |                                -57.8895 |                                -0.0001 |                                -0.0069 |                                 0.0391 |                                -57.6258 |                                -0.0001 |                                -0.007  |                                 0.0378 |                                -57.3371 |                                -0.0001 |                                -0.0071 |                                 0.0366 |                                -57.0304 |                                -0.0001 |                                -0.0072 |                                 0.0354 |                                -56.7112 |                                -0.0001 |                                -0.0072 |                                 0.0342 |                                -56.3837 |                                -0.0001 |                                -0.0073 |                                 0.033  |                                -56.0514 |                                -0.0001 |                                -0.0073 |                                 0.0319 |                                -55.7168 |                                -0.0001 |                                -0.0073 |                                 0.0309 |                                -55.3822 |                                -0.0001 |                                -0.0073 |                                 0.0298 |                                -55.0491 |
        |            265 |                                -0.0008 |                                 0.0208 |                                -2.3611 |                                 12.3941 |                                -0.001  |                                 0.0202 |                                -1.1439 |                                 -6.1306 |                                -0.0009 |                                 0.0116 |                                -0.4336 |                                -23.5887 |                                -0.0007 |                                 0.0051 |                                -0.1389 |                                -33.9313 |                                -0.0006 |                                 0.0007 |                                -0.0104 |                                -39.8485 |                                -0.0005 |                                -0.0023 |                                 0.0481 |                                -43.2327 |                                -0.0004 |                                -0.0043 |                                 0.0747 |                                -45.1384 |                                -0.0004 |                                -0.0058 |                                 0.086  |                                -46.1536 |                                -0.0003 |                                -0.0068 |                                 0.0896 |                                -46.6172 |                                -0.0003 |                                -0.0076 |                                 0.0891 |                                -46.732  |                                -0.0003 |                                -0.0081 |                                 0.0867 |                                -46.6233 |                                -0.0002 |                                -0.0085 |                                 0.0832 |                                -46.3707 |                                -0.0002 |                                -0.0088 |                                 0.0794 |                                -46.026  |                                -0.0002 |                                -0.009  |                                 0.0754 |                                -45.6238 |                                -0.0002 |                                -0.0092 |                                 0.0715 |                                -45.187  |                                -0.0002 |                                -0.0093 |                                 0.0677 |                                -44.7316 |                                -0.0002 |                                -0.0093 |                                 0.0642 |                                -44.2683 |                                -0.0002 |                                -0.0094 |                                 0.0608 |                                -43.8046 |                                -0.0001 |                                -0.0094 |                                 0.0577 |                                -43.3456 |                                -0.0001 |                                -0.0094 |                                 0.0548 |                                -42.8947 |                                -0.0001 |                                -0.0094 |                                 0.0521 |                                -42.4543 |                                -0.0001 |                                -0.0093 |                                 0.0496 |                                -42.026  |                                -0.0001 |                                -0.0093 |                                 0.0473 |                                -41.6105 |                                -0.0001 |                                -0.0093 |                                 0.0451 |                                -41.2085 |                                -0.0001 |                                -0.0092 |                                 0.0431 |                                -40.8201 |                                -0.0001 |                                -0.0092 |                                 0.0412 |                                -40.4454 |                                -0.0001 |                                -0.0091 |                                 0.0394 |                                -40.0842 |                                -0.0001 |                                -0.0091 |                                 0.0378 |                                -39.7362 |                                -0.0001 |                                -0.009  |                                 0.0362 |                                -39.4012 |
        |            270 |                                -0.0018 |                                 0.0296 |                                -3.3565 |                                 -1.1373 |                                -0.0014 |                                 0.0102 |                                -0.568  |                                -22.8451 |                                -0.001  |                                -0.0007 |                                 0.0379 |                                -31.1609 |                                -0.0007 |                                -0.0062 |                                 0.1856 |                                -34.1545 |                                -0.0006 |                                -0.0092 |                                 0.2156 |                                -35.0519 |                                -0.0005 |                                -0.0108 |                                 0.2108 |                                -35.0499 |                                -0.0004 |                                -0.0118 |                                 0.1959 |                                -34.6438 |                                -0.0003 |                                -0.0123 |                                 0.1789 |                                -34.058  |                                -0.0003 |                                -0.0126 |                                 0.1625 |                                -33.4007 |                                -0.0003 |                                -0.0127 |                                 0.1476 |                                -32.7262 |                                -0.0002 |                                -0.0127 |                                 0.1344 |                                -32.062  |                                -0.0002 |                                -0.0127 |                                 0.1228 |                                -31.4219 |                                -0.0002 |                                -0.0126 |                                 0.1126 |                                -30.8125 |                                -0.0002 |                                -0.0125 |                                 0.1036 |                                -30.2362 |                                -0.0002 |                                -0.0124 |                                 0.0957 |                                -29.6932 |                                -0.0002 |                                -0.0122 |                                 0.0887 |                                -29.1828 |                                -0.0001 |                                -0.0121 |                                 0.0825 |                                -28.7034 |                                -0.0001 |                                -0.0119 |                                 0.077  |                                -28.2532 |                                -0.0001 |                                -0.0118 |                                 0.072  |                                -27.8305 |                                -0.0001 |                                -0.0116 |                                 0.0675 |                                -27.4333 |                                -0.0001 |                                -0.0115 |                                 0.0635 |                                -27.0599 |                                -0.0001 |                                -0.0113 |                                 0.0598 |                                -26.7087 |                                -0.0001 |                                -0.0112 |                                 0.0565 |                                -26.378  |                                -0.0001 |                                -0.0111 |                                 0.0535 |                                -26.0664 |                                -0.0001 |                                -0.0109 |                                 0.0507 |                                -25.7726 |                                -0.0001 |                                -0.0108 |                                 0.0482 |                                -25.4954 |                                -0.0001 |                                -0.0107 |                                 0.0458 |                                -25.2336 |                                -0.0001 |                                -0.0105 |                                 0.0437 |                                -24.9863 |                                -0.0001 |                                -0.0104 |                                 0.0417 |                                -24.7525 |
        |            275 |                                -0.0028 |                                 0.0104 |                                -1.1551 |                                -17.2898 |                                -0.0014 |                                -0.0119 |                                 0.6953 |                                -23.4002 |                                -0.0009 |                                -0.0175 |                                 0.6759 |                                -23.3625 |                                -0.0006 |                                -0.0191 |                                 0.5505 |                                -22.3833 |                                -0.0005 |                                -0.0193 |                                 0.4456 |                                -21.3102 |                                -0.0004 |                                -0.0191 |                                 0.3664 |                                -20.3193 |                                -0.0003 |                                -0.0186 |                                 0.3069 |                                -19.4415 |                                -0.0003 |                                -0.0181 |                                 0.2613 |                                -18.6715 |                                -0.0002 |                                -0.0176 |                                 0.2257 |                                -17.9959 |                                -0.0002 |                                -0.0171 |                                 0.1974 |                                -17.4008 |                                -0.0002 |                                -0.0166 |                                 0.1744 |                                -16.8741 |                                -0.0002 |                                -0.0162 |                                 0.1556 |                                -16.4056 |                                -0.0002 |                                -0.0158 |                                 0.1399 |                                -15.9866 |                                -0.0002 |                                -0.0154 |                                 0.1266 |                                -15.6103 |                                -0.0001 |                                -0.015  |                                 0.1153 |                                -15.271  |                                -0.0001 |                                -0.0146 |                                 0.1056 |                                -14.9638 |                                -0.0001 |                                -0.0143 |                                 0.0971 |                                -14.6849 |                                -0.0001 |                                -0.014  |                                 0.0898 |                                -14.4309 |                                -0.0001 |                                -0.0137 |                                 0.0833 |                                -14.1989 |                                -0.0001 |                                -0.0134 |                                 0.0776 |                                -13.9866 |                                -0.0001 |                                -0.0132 |                                 0.0724 |                                -13.792  |                                -0.0001 |                                -0.0129 |                                 0.0679 |                                -13.6133 |                                -0.0001 |                                -0.0127 |                                 0.0638 |                                -13.449  |                                -0.0001 |                                -0.0125 |                                 0.06   |                                -13.2976 |                                -0.0001 |                                -0.0123 |                                 0.0567 |                                -13.1582 |                                -0.0001 |                                -0.0121 |                                 0.0536 |                                -13.0297 |                                -0.0001 |                                -0.0119 |                                 0.0508 |                                -12.9111 |                                -0.0001 |                                -0.0117 |                                 0.0483 |                                -12.8016 |                                -0.0001 |                                -0.0115 |                                 0.0459 |                                -12.7007 |
        |            280 |                                -0.0024 |                                -0.0372 |                                 4.2755 |                                -11.6654 |                                -0.001  |                                -0.0362 |                                 2.0747 |                                 -9.6792 |                                -0.0006 |                                -0.0325 |                                 1.2421 |                                 -8.3993 |                                -0.0004 |                                -0.0295 |                                 0.8448 |                                 -7.5475 |                                -0.0003 |                                -0.0271 |                                 0.6212 |                                 -6.9398 |                                -0.0003 |                                -0.0252 |                                 0.4812 |                                 -6.4836 |                                -0.0002 |                                -0.0236 |                                 0.3868 |                                 -6.1286 |                                -0.0002 |                                -0.0223 |                                 0.3197 |                                 -5.845  |                                -0.0002 |                                -0.0212 |                                 0.2699 |                                 -5.6139 |                                -0.0002 |                                -0.0202 |                                 0.2319 |                                 -5.4229 |                                -0.0001 |                                -0.0194 |                                 0.202  |                                 -5.2632 |                                -0.0001 |                                -0.0186 |                                 0.178  |                                 -5.1285 |                                -0.0001 |                                -0.0179 |                                 0.1584 |                                 -5.0143 |                                -0.0001 |                                -0.0173 |                                 0.1422 |                                 -4.9171 |                                -0.0001 |                                -0.0168 |                                 0.1285 |                                 -4.8342 |                                -0.0001 |                                -0.0163 |                                 0.117  |                                 -4.7634 |                                -0.0001 |                                -0.0158 |                                 0.107  |                                 -4.703  |                                -0.0001 |                                -0.0154 |                                 0.0984 |                                 -4.6517 |                                -0.0001 |                                -0.015  |                                 0.0909 |                                 -4.6084 |                                -0.0001 |                                -0.0147 |                                 0.0843 |                                 -4.5721 |                                -0.0001 |                                -0.0143 |                                 0.0785 |                                 -4.5421 |                                -0.0001 |                                -0.014  |                                 0.0733 |                                 -4.5177 |                                -0.0001 |                                -0.0137 |                                 0.0686 |                                 -4.4983 |                                -0.0001 |                                -0.0135 |                                 0.0645 |                                 -4.4834 |                                -0.0001 |                                -0.0132 |                                 0.0607 |                                 -4.4728 |                                -0.0001 |                                -0.013  |                                 0.0573 |                                 -4.4658 |                                -0.0001 |                                -0.0127 |                                 0.0542 |                                 -4.4624 |                                -0.0001 |                                -0.0125 |                                 0.0514 |                                 -4.4622 |                                -0.0001 |                                -0.0123 |                                 0.0488 |                                 -4.4649 |
        |            285 |                                -0.0004 |                                -0.068  |                                 7.7675 |                                 -0.1059 |                                -0.0002 |                                -0.0482 |                                 2.7508 |                                 -0.094  |                                -0.0002 |                                -0.0394 |                                 1.4983 |                                 -0.0977 |                                -0.0001 |                                -0.0341 |                                 0.9736 |                                 -0.1072 |                                -0.0001 |                                -0.0305 |                                 0.6969 |                                 -0.1202 |                                -0.0001 |                                -0.0279 |                                 0.5303 |                                 -0.1357 |                                -0.0001 |                                -0.0258 |                                 0.4209 |                                 -0.153  |                                -0.0001 |                                -0.0241 |                                 0.3446 |                                 -0.172  |                                -0.0001 |                                -0.0228 |                                 0.2889 |                                 -0.1924 |                                -0.0001 |                                -0.0216 |                                 0.2467 |                                 -0.2141 |                                -0.0001 |                                -0.0206 |                                 0.2139 |                                 -0.237  |                                -0.0001 |                                -0.0197 |                                 0.1878 |                                 -0.2609 |                                -0.0001 |                                -0.0189 |                                 0.1666 |                                 -0.2859 |                                -0.0001 |                                -0.0182 |                                 0.1491 |                                 -0.3118 |                                -0.0001 |                                -0.0176 |                                 0.1344 |                                 -0.3386 |                                -0.0001 |                                -0.0171 |                                 0.1221 |                                 -0.3663 |                                -0.0001 |                                -0.0166 |                                 0.1115 |                                 -0.3949 |                                -0.0001 |                                -0.0161 |                                 0.1023 |                                 -0.4243 |                                -0.0001 |                                -0.0157 |                                 0.0944 |                                 -0.4544 |                                -0.0001 |                                -0.0153 |                                 0.0874 |                                 -0.4854 |                                -0.0001 |                                -0.0149 |                                 0.0813 |                                 -0.517  |                                -0.0001 |                                -0.0146 |                                 0.0758 |                                 -0.5494 |                                -0.0001 |                                -0.0142 |                                 0.0709 |                                 -0.5824 |                                -0.0001 |                                -0.0139 |                                 0.0665 |                                 -0.6162 |                                 0      |                                -0.0137 |                                 0.0626 |                                 -0.6506 |                                 0      |                                -0.0134 |                                 0.059  |                                 -0.6856 |                                 0      |                                -0.0132 |                                 0.0558 |                                 -0.7213 |                                 0      |                                -0.0129 |                                 0.0529 |                                 -0.7576 |                                 0      |                                -0.0127 |                                 0.0502 |                                 -0.7945 |
        |            290 |                                 0.0018 |                                -0.0469 |                                 5.3291 |                                 -8.8672 |                                 0.0006 |                                -0.0407 |                                 2.3116 |                                 -6.8532 |                                 0.0003 |                                -0.0354 |                                 1.3414 |                                 -5.7178 |                                 0.0002 |                                -0.0316 |                                 0.8991 |                                 -4.9793 |                                 0.0001 |                                -0.0288 |                                 0.6554 |                                 -4.452  |                                 0.0001 |                                -0.0266 |                                 0.5049 |                                 -4.0522 |                                 0      |                                -0.0249 |                                 0.4042 |                                 -3.7367 |                                 0      |                                -0.0234 |                                 0.3331 |                                 -3.4801 |                                 0      |                                -0.0222 |                                 0.2806 |                                 -3.2671 |                                 0      |                                -0.0211 |                                 0.2406 |                                 -3.0871 |                                 0      |                                -0.0202 |                                 0.2093 |                                 -2.9332 |                                 0      |                                -0.0194 |                                 0.1842 |                                 -2.8002 |                                 0      |                                -0.0187 |                                 0.1638 |                                 -2.6844 |                                 0      |                                -0.018  |                                 0.1469 |                                 -2.5829 |                                 0      |                                -0.0175 |                                 0.1327 |                                 -2.4936 |                                 0      |                                -0.0169 |                                 0.1206 |                                 -2.4146 |                                 0      |                                -0.0165 |                                 0.1103 |                                 -2.3447 |                                 0      |                                -0.016  |                                 0.1014 |                                 -2.2827 |                                 0      |                                -0.0156 |                                 0.0936 |                                 -2.2277 |                                 0      |                                -0.0152 |                                 0.0868 |                                 -2.1789 |                                 0      |                                -0.0149 |                                 0.0807 |                                 -2.1357 |                                 0      |                                -0.0145 |                                 0.0754 |                                 -2.0974 |                                 0      |                                -0.0142 |                                 0.0706 |                                 -2.0638 |                                 0      |                                -0.0139 |                                 0.0663 |                                 -2.0343 |                                 0      |                                -0.0137 |                                 0.0624 |                                 -2.0086 |                                 0      |                                -0.0134 |                                 0.0589 |                                 -1.9864 |                                 0      |                                -0.0132 |                                 0.0557 |                                 -1.9674 |                                 0      |                                -0.0129 |                                 0.0527 |                                 -1.9514 |                                 0      |                                -0.0127 |                                 0.0501 |                                 -1.9383 |
        |            295 |                                 0.0027 |                                 0.0005 |                                -0.0931 |                                -18.806  |                                 0.0012 |                                -0.019  |                                 1.0689 |                                -21.4282 |                                 0.0006 |                                -0.0226 |                                 0.8532 |                                -20.2199 |                                 0.0004 |                                -0.0231 |                                 0.6528 |                                -18.7406 |                                 0.0003 |                                -0.0226 |                                 0.512  |                                -17.4118 |                                 0.0002 |                                -0.0219 |                                 0.4132 |                                -16.2709 |                                 0.0002 |                                -0.0211 |                                 0.3417 |                                -15.2942 |                                 0.0001 |                                -0.0203 |                                 0.2883 |                                -14.4521 |                                 0.0001 |                                -0.0196 |                                 0.2473 |                                -13.7192 |                                 0.0001 |                                -0.019  |                                 0.2151 |                                -13.0753 |                                 0.0001 |                                -0.0184 |                                 0.1893 |                                -12.5048 |                                 0      |                                -0.0178 |                                 0.1682 |                                -11.9953 |                                 0      |                                -0.0173 |                                 0.1508 |                                -11.5373 |                                 0      |                                -0.0168 |                                 0.1361 |                                -11.1231 |                                 0      |                                -0.0163 |                                 0.1237 |                                -10.7465 |                                 0      |                                -0.0159 |                                 0.1131 |                                -10.4025 |                                 0      |                                -0.0155 |                                 0.1039 |                                -10.0869 |                                 0      |                                -0.0152 |                                 0.0959 |                                 -9.7963 |                                 0      |                                -0.0149 |                                 0.0888 |                                 -9.5279 |                                 0      |                                -0.0145 |                                 0.0826 |                                 -9.2793 |                                 0      |                                -0.0142 |                                 0.0771 |                                 -9.0483 |                                 0      |                                -0.014  |                                 0.0721 |                                 -8.8333 |                                 0      |                                -0.0137 |                                 0.0677 |                                 -8.6326 |                                 0      |                                -0.0134 |                                 0.0637 |                                 -8.4451 |                                 0      |                                -0.0132 |                                 0.0601 |                                 -8.2695 |                                 0      |                                -0.013  |                                 0.0568 |                                 -8.1049 |                                 0      |                                -0.0128 |                                 0.0538 |                                 -7.9503 |                                 0      |                                -0.0126 |                                 0.0511 |                                 -7.805  |                                 0      |                                -0.0124 |                                 0.0486 |                                 -7.6682 |
        |            300 |                                 0.0022 |                                 0.0289 |                                -3.3231 |                                 -7.8444 |                                 0.0013 |                                 0.0042 |                                -0.2538 |                                -26.4755 |                                 0.0008 |                                -0.0065 |                                 0.2387 |                                -31.5828 |                                 0.0006 |                                -0.0113 |                                 0.3162 |                                -32.6    |                                 0.0004 |                                -0.0136 |                                 0.306  |                                -32.2457 |                                 0.0003 |                                -0.0148 |                                 0.2769 |                                -31.4023 |                                 0.0002 |                                -0.0153 |                                 0.2463 |                                -30.3985 |                                 0.0002 |                                -0.0155 |                                 0.2186 |                                -29.3676 |                                 0.0002 |                                -0.0155 |                                 0.1947 |                                -28.3651 |                                 0.0001 |                                -0.0154 |                                 0.1742 |                                -27.4134 |                                 0.0001 |                                -0.0153 |                                 0.1568 |                                -26.5201 |                                 0.0001 |                                -0.0151 |                                 0.142  |                                -25.6858 |                                 0.0001 |                                -0.0149 |                                 0.1292 |                                -24.908  |                                 0.0001 |                                -0.0146 |                                 0.1182 |                                -24.1829 |                                 0.0001 |                                -0.0144 |                                 0.1086 |                                -23.5064 |                                 0      |                                -0.0142 |                                 0.1002 |                                -22.8744 |                                 0      |                                -0.014  |                                 0.0928 |                                -22.283  |                                 0      |                                -0.0137 |                                 0.0863 |                                -21.7287 |                                 0      |                                -0.0135 |                                 0.0805 |                                -21.2081 |                                 0      |                                -0.0133 |                                 0.0753 |                                -20.7185 |                                 0      |                                -0.0131 |                                 0.0706 |                                -20.2571 |                                 0      |                                -0.0129 |                                 0.0664 |                                -19.8217 |                                 0      |                                -0.0127 |                                 0.0626 |                                -19.4101 |                                 0      |                                -0.0125 |                                 0.0591 |                                -19.0204 |                                 0      |                                -0.0124 |                                 0.056  |                                -18.6511 |                                 0      |                                -0.0122 |                                 0.0531 |                                -18.3005 |                                 0      |                                -0.012  |                                 0.0504 |                                -17.9673 |                                 0      |                                -0.0119 |                                 0.048  |                                -17.6502 |                                 0      |                                -0.0117 |                                 0.0458 |                                -17.3482 |
        |            305 |                                 0.0012 |                                 0.0282 |                                -3.2278 |                                  9.1553 |                                 0.0012 |                                 0.0187 |                                -1.083  |                                -15.6878 |                                 0.0009 |                                 0.0074 |                                -0.2912 |                                -30.891  |                                 0.0006 |                                 0.0003 |                                -0.0165 |                                -38.2668 |                                 0.0005 |                                -0.004  |                                 0.086  |                                -41.7381 |                                 0.0004 |                                -0.0068 |                                 0.1238 |                                -43.2291 |                                 0.0003 |                                -0.0085 |                                 0.1353 |                                -43.6692 |                                 0.0003 |                                -0.0097 |                                 0.1353 |                                -43.5318 |                                 0.0002 |                                -0.0105 |                                 0.1304 |                                -43.0714 |                                 0.0002 |                                -0.011  |                                 0.1235 |                                -42.4311 |                                 0.0002 |                                -0.0114 |                                 0.116  |                                -41.6942 |                                 0.0001 |                                -0.0116 |                                 0.1085 |                                -40.9103 |                                 0.0001 |                                -0.0118 |                                 0.1015 |                                -40.1097 |                                 0.0001 |                                -0.0118 |                                 0.0949 |                                -39.3108 |                                 0.0001 |                                -0.0119 |                                 0.0888 |                                -38.5249 |                                 0.0001 |                                -0.0118 |                                 0.0832 |                                -37.7587 |                                 0.0001 |                                -0.0118 |                                 0.0782 |                                -37.0161 |                                 0.0001 |                                -0.0118 |                                 0.0735 |                                -36.299  |                                 0.0001 |                                -0.0117 |                                 0.0693 |                                -35.6085 |                                 0      |                                -0.0116 |                                 0.0654 |                                -34.9444 |                                 0      |                                -0.0115 |                                 0.0618 |                                -34.3064 |                                 0      |                                -0.0114 |                                 0.0586 |                                -33.6937 |                                 0      |                                -0.0113 |                                 0.0556 |                                -33.1055 |                                 0      |                                -0.0112 |                                 0.0528 |                                -32.5408 |                                 0      |                                -0.0111 |                                 0.0503 |                                -31.9985 |                                 0      |                                -0.011  |                                 0.0479 |                                -31.4776 |                                 0      |                                -0.0109 |                                 0.0457 |                                -30.977  |                                 0      |                                -0.0108 |                                 0.0437 |                                -30.4958 |                                 0      |                                -0.0107 |                                 0.0418 |                                -30.0329 |
        |            310 |                                 0.0005 |                                 0.0159 |                                -1.8152 |                                 14.1154 |                                 0.0008 |                                 0.0224 |                                -1.2862 |                                  2.4962 |                                 0.0008 |                                 0.0159 |                                -0.6122 |                                -18.04   |                                 0.0006 |                                 0.0093 |                                -0.2734 |                                -32.4401 |                                 0.0005 |                                 0.0044 |                                -0.1059 |                                -41.551  |                                 0.0004 |                                 0.0008 |                                -0.0198 |                                -47.2149 |                                 0.0004 |                                -0.0018 |                                 0.0257 |                                -50.6996 |                                 0.0003 |                                -0.0038 |                                 0.05   |                                -52.7898 |                                 0.0003 |                                -0.0052 |                                 0.0627 |                                -53.9701 |                                 0.0002 |                                -0.0063 |                                 0.0688 |                                -54.5454 |                                 0.0002 |                                -0.0071 |                                 0.0712 |                                -54.7122 |                                 0.0002 |                                -0.0077 |                                 0.0713 |                                -54.6002 |                                 0.0001 |                                -0.0082 |                                 0.0702 |                                -54.2971 |                                 0.0001 |                                -0.0086 |                                 0.0683 |                                -53.8626 |                                 0.0001 |                                -0.0089 |                                 0.066  |                                -53.3388 |                                 0.0001 |                                -0.0091 |                                 0.0635 |                                -52.7552 |                                 0.0001 |                                -0.0093 |                                 0.061  |                                -52.1328 |                                 0.0001 |                                -0.0094 |                                 0.0584 |                                -51.487  |                                 0.0001 |                                -0.0095 |                                 0.056  |                                -50.8286 |                                 0.0001 |                                -0.0096 |                                 0.0536 |                                -50.1658 |                                 0.0001 |                                -0.0096 |                                 0.0513 |                                -49.5044 |                                 0.0001 |                                -0.0097 |                                 0.0492 |                                -48.8485 |                                 0      |                                -0.0097 |                                 0.0471 |                                -48.2012 |                                 0      |                                -0.0097 |                                 0.0452 |                                -47.5646 |                                 0      |                                -0.0097 |                                 0.0433 |                                -46.9404 |                                 0      |                                -0.0096 |                                 0.0416 |                                -46.3295 |                                 0      |                                -0.0096 |                                 0.04   |                                -45.7325 |                                 0      |                                -0.0096 |                                 0.0384 |                                -45.1499 |                                 0      |                                -0.0096 |                                 0.037  |                                -44.5818 |
        |            315 |                                 0.0002 |                                 0.0062 |                                -0.7113 |                                  9.4638 |                                 0.0005 |                                 0.0186 |                                -1.0681 |                                 16.2431 |                                 0.0006 |                                 0.0185 |                                -0.7095 |                                  0.1321 |                                 0.0005 |                                 0.0145 |                                -0.4214 |                                -17.5555 |                                 0.0005 |                                 0.0104 |                                -0.2429 |                                -31.5765 |                                 0.0004 |                                 0.0068 |                                -0.135  |                                -41.885  |                                 0.0004 |                                 0.004  |                                -0.0688 |                                -49.3073 |                                 0.0003 |                                 0.0017 |                                -0.0274 |                                -54.617  |                                 0.0003 |                                -0.0002 |                                -0.0011 |                                -58.3983 |                                 0.0002 |                                -0.0017 |                                 0.0158 |                                -61.0701 |                                 0.0002 |                                -0.0028 |                                 0.0267 |                                -62.9293 |                                 0.0002 |                                -0.0038 |                                 0.0336 |                                -64.1874 |                                 0.0002 |                                -0.0046 |                                 0.038  |                                -64.9967 |                                 0.0002 |                                -0.0052 |                                 0.0406 |                                -65.4679 |                                 0.0001 |                                -0.0058 |                                 0.042  |                                -65.6828 |                                 0.0001 |                                -0.0062 |                                 0.0425 |                                -65.7022 |                                 0.0001 |                                -0.0066 |                                 0.0425 |                                -65.5721 |                                 0.0001 |                                -0.0069 |                                 0.0421 |                                -65.3272 |                                 0.0001 |                                -0.0071 |                                 0.0415 |                                -64.9944 |                                 0.0001 |                                -0.0073 |                                 0.0406 |                                -64.5944 |                                 0.0001 |                                -0.0075 |                                 0.0397 |                                -64.1434 |                                 0.0001 |                                -0.0077 |                                 0.0387 |                                -63.6539 |                                 0.0001 |                                -0.0078 |                                 0.0376 |                                -63.1359 |                                 0.0001 |                                -0.0079 |                                 0.0366 |                                -62.5975 |                                 0.0001 |                                -0.008  |                                 0.0355 |                                -62.0447 |                                 0.0001 |                                -0.008  |                                 0.0344 |                                -61.4826 |                                 0      |                                -0.0081 |                                 0.0334 |                                -60.9151 |                                 0      |                                -0.0081 |                                 0.0324 |                                -60.3455 |                                 0      |                                -0.0082 |                                 0.0314 |                                -59.7764 |
        |            320 |                                 0      |                                 0.0018 |                                -0.208  |                                  4.0675 |                                 0.0003 |                                 0.0124 |                                -0.7105 |                                 20.4355 |                                 0.0004 |                                 0.0168 |                                -0.6437 |                                 15.6615 |                                 0.0004 |                                 0.0161 |                                -0.4649 |                                  0.63   |                                 0.0004 |                                 0.0137 |                                -0.317  |                                -15.1564 |                                 0.0004 |                                 0.0109 |                                -0.212  |                                -28.7858 |                                 0.0003 |                                 0.0083 |                                -0.1397 |                                -39.8192 |                                 0.0003 |                                 0.006  |                                -0.09   |                                -48.5363 |                                 0.0003 |                                 0.0041 |                                -0.0555 |                                -55.3576 |                                 0.0002 |                                 0.0025 |                                -0.0312 |                                -60.674  |                                 0.0002 |                                 0.0011 |                                -0.014  |                                -64.8074 |                                 0.0002 |                                -0.0001 |                                -0.0016 |                                -68.0115 |                                 0.0002 |                                -0.0011 |                                 0.0072 |                                -70.4833 |                                 0.0002 |                                -0.0019 |                                 0.0136 |                                -72.3752 |                                 0.0002 |                                -0.0027 |                                 0.0182 |                                -73.8055 |                                 0.0001 |                                -0.0033 |                                 0.0215 |                                -74.8663 |                                 0.0001 |                                -0.0038 |                                 0.0239 |                                -75.6297 |                                 0.0001 |                                -0.0043 |                                 0.0255 |                                -76.1527 |                                 0.0001 |                                -0.0047 |                                 0.0265 |                                -76.4807 |                                 0.0001 |                                -0.005  |                                 0.0272 |                                -76.6498 |                                 0.0001 |                                -0.0053 |                                 0.0275 |                                -76.689  |                                 0.0001 |                                -0.0056 |                                 0.0277 |                                -76.6221 |                                 0.0001 |                                -0.0058 |                                 0.0276 |                                -76.4681 |                                 0.0001 |                                -0.006  |                                 0.0274 |                                -76.2428 |                                 0.0001 |                                -0.0062 |                                 0.0271 |                                -75.9591 |                                 0.0001 |                                -0.0063 |                                 0.0268 |                                -75.6275 |                                 0.0001 |                                -0.0065 |                                 0.0263 |                                -75.257  |                                 0.0001 |                                -0.0066 |                                 0.0259 |                                -74.8548 |                                 0      |                                -0.0067 |                                 0.0254 |                                -74.4271 |
        |            325 |                                 0      |                                 0.0004 |                                -0.0471 |                                  1.2505 |                                 0.0001 |                                 0.007  |                                -0.3988 |                                 17.2785 |                                 0.0003 |                                 0.013  |                                -0.4987 |                                 24.0019 |                                 0.0003 |                                 0.015  |                                -0.4314 |                                 16.3753 |                                 0.0003 |                                 0.0145 |                                -0.3349 |                                  2.9038 |                                 0.0003 |                                 0.013  |                                -0.2502 |                                -11.395  |                                 0.0003 |                                 0.0111 |                                -0.1841 |                                -24.5228 |                                 0.0003 |                                 0.0092 |                                -0.1342 |                                -35.8895 |                                 0.0003 |                                 0.0074 |                                -0.097  |                                -45.4719 |                                 0.0002 |                                 0.0058 |                                -0.0692 |                                -53.4473 |                                 0.0002 |                                 0.0044 |                                -0.0482 |                                -60.0437 |                                 0.0002 |                                 0.0031 |                                -0.0323 |                                -65.4825 |                                 0.0002 |                                 0.002  |                                -0.0202 |                                -69.9586 |                                 0.0002 |                                 0.0011 |                                -0.011  |                                -73.6368 |                                 0.0002 |                                 0.0002 |                                -0.0038 |                                -76.6536 |                                 0.0001 |                                -0.0005 |                                 0.0017 |                                -79.1213 |                                 0.0001 |                                -0.0012 |                                 0.006  |                                -81.1318 |                                 0.0001 |                                -0.0017 |                                 0.0094 |                                -82.7604 |                                 0.0001 |                                -0.0023 |                                 0.0119 |                                -84.0691 |                                 0.0001 |                                -0.0027 |                                 0.0139 |                                -85.1088 |                                 0.0001 |                                -0.0031 |                                 0.0155 |                                -85.9217 |                                 0.0001 |                                -0.0035 |                                 0.0166 |                                -86.5427 |                                 0.0001 |                                -0.0038 |                                 0.0175 |                                -87.001  |                                 0.0001 |                                -0.0041 |                                 0.0181 |                                -87.3211 |                                 0.0001 |                                -0.0043 |                                 0.0186 |                                -87.5234 |                                 0.0001 |                                -0.0045 |                                 0.0189 |                                -87.6251 |                                 0.0001 |                                -0.0047 |                                 0.0191 |                                -87.641  |                                 0.0001 |                                -0.0049 |                                 0.0191 |                                -87.5835 |                                 0.0001 |                                -0.0051 |                                 0.0191 |                                -87.463  |
        |            330 |                                 0      |                                 0.0001 |                                -0.0085 |                                  0.2902 |                                 0.0001 |                                 0.0034 |                                -0.194  |                                 11.4717 |                                 0.0002 |                                 0.009  |                                -0.3423 |                                 24.9764 |                                 0.0002 |                                 0.0124 |                                -0.3557 |                                 26.2306 |                                 0.0003 |                                 0.0135 |                                -0.3116 |                                 18.3284 |                                 0.0003 |                                 0.0133 |                                -0.2554 |                                  6.3315 |                                 0.0003 |                                 0.0123 |                                -0.2034 |                                 -6.6745 |                                 0.0003 |                                 0.011  |                                -0.1598 |                                -19.1847 |                                 0.0002 |                                 0.0096 |                                -0.1247 |                                -30.5727 |                                 0.0002 |                                 0.0082 |                                -0.0967 |                                -40.6498 |                                 0.0002 |                                 0.0069 |                                -0.0746 |                                -49.4309 |                                 0.0002 |                                 0.0057 |                                -0.0571 |                                -57.0173 |                                 0.0002 |                                 0.0047 |                                -0.0432 |                                -63.54   |                                 0.0002 |                                 0.0037 |                                -0.0321 |                                -69.1327 |                                 0.0002 |                                 0.0028 |                                -0.0232 |                                -73.9199 |                                 0.0002 |                                 0.002  |                                -0.016  |                                -78.0129 |                                 0.0001 |                                 0.0013 |                                -0.0102 |                                -81.5087 |                                 0.0001 |                                 0.0006 |                                -0.0055 |                                -84.491  |                                 0.0001 |                                 0      |                                -0.0017 |                                -87.0312 |                                 0.0001 |                                -0.0005 |                                 0.0014 |                                -89.1905 |                                 0.0001 |                                -0.001  |                                 0.0039 |                                -91.0207 |                                 0.0001 |                                -0.0014 |                                 0.006  |                                -92.5662 |                                 0.0001 |                                -0.0018 |                                 0.0077 |                                -93.8649 |                                 0.0001 |                                -0.0022 |                                 0.0091 |                                -94.9491 |                                 0.0001 |                                -0.0025 |                                 0.0102 |                                -95.8465 |                                 0.0001 |                                -0.0028 |                                 0.0111 |                                -96.5811 |                                 0.0001 |                                -0.003  |                                 0.0118 |                                -97.1732 |                                 0.0001 |                                -0.0033 |                                 0.0124 |                                -97.6407 |                                 0.0001 |                                -0.0035 |                                 0.0128 |                                -97.9989 |
        |            335 |                                 0      |                                 0      |                                -0.0012 |                                  0.0526 |                                 0      |                                 0.0015 |                                -0.0833 |                                  6.3455 |                                 0.0001 |                                 0.0056 |                                -0.2125 |                                 21.1018 |                                 0.0001 |                                 0.0093 |                                -0.2675 |                                 29.558  |                                 0.0002 |                                 0.0115 |                                -0.2647 |                                 28.5708 |                                 0.0002 |                                 0.0123 |                                -0.2369 |                                 21.1676 |                                 0.0002 |                                 0.0122 |                                -0.2023 |                                 10.5125 |                                 0.0002 |                                 0.0116 |                                -0.1687 |                                 -1.306  |                                 0.0002 |                                 0.0108 |                                -0.1391 |                                -13.1035 |                                 0.0002 |                                 0.0098 |                                -0.1138 |                                -24.272  |                                 0.0002 |                                 0.0087 |                                -0.0928 |                                -34.538  |                                 0.0002 |                                 0.0077 |                                -0.0753 |                                -43.8119 |                                 0.0002 |                                 0.0067 |                                -0.0609 |                                -52.1018 |                                 0.0002 |                                 0.0058 |                                -0.049  |                                -59.4638 |                                 0.0002 |                                 0.0049 |                                -0.0392 |                                -65.9752 |                                 0.0002 |                                 0.0041 |                                -0.0311 |                                -71.7194 |                                 0.0001 |                                 0.0034 |                                -0.0243 |                                -76.7785 |                                 0.0001 |                                 0.0027 |                                -0.0187 |                                -81.2291 |                                 0.0001 |                                 0.0021 |                                -0.014  |                                -85.1413 |                                 0.0001 |                                 0.0015 |                                -0.01   |                                -88.5775 |                                 0.0001 |                                 0.001  |                                -0.0067 |                                -91.5934 |                                 0.0001 |                                 0.0005 |                                -0.0039 |                                -94.2379 |                                 0.0001 |                                 0      |                                -0.0015 |                                -96.5542 |                                 0.0001 |                                -0.0004 |                                 0.0005 |                                -98.5799 |                                 0.0001 |                                -0.0007 |                                 0.0022 |                               -100.348  |                                 0.0001 |                                -0.0011 |                                 0.0036 |                               -101.888  |                                 0.0001 |                                -0.0014 |                                 0.0048 |                               -103.225  |                                 0.0001 |                                -0.0017 |                                 0.0058 |                               -104.382  |                                 0.0001 |                                -0.002  |                                 0.0067 |                               -105.377  |
        |            340 |                                 0      |                                 0      |                                -0.0001 |                                  0.0076 |                                 0      |                                 0.0006 |                                -0.0319 |                                  3.0177 |                                 0      |                                 0.0032 |                                -0.1209 |                                 15.3892 |                                 0.0001 |                                 0.0065 |                                -0.1863 |                                 27.7438 |                                 0.0001 |                                 0.0091 |                                -0.209  |                                 33.0366 |                                 0.0002 |                                 0.0106 |                                -0.2043 |                                 31.2554 |                                 0.0002 |                                 0.0113 |                                -0.1866 |                                 24.5975 |                                 0.0002 |                                 0.0114 |                                -0.1645 |                                 15.1676 |                                 0.0002 |                                 0.011  |                                -0.1421 |                                  4.4673 |                                 0.0002 |                                 0.0104 |                                -0.1214 |                                 -6.5493 |                                 0.0002 |                                 0.0097 |                                -0.103  |                                -17.3181 |                                 0.0002 |                                 0.0089 |                                -0.087  |                                -27.5287 |                                 0.0002 |                                 0.0081 |                                -0.0733 |                                -37.0282 |                                 0.0002 |                                 0.0073 |                                -0.0616 |                                -45.7591 |                                 0.0002 |                                 0.0065 |                                -0.0516 |                                -53.7193 |                                 0.0001 |                                 0.0058 |                                -0.0431 |                                -60.9379 |                                 0.0001 |                                 0.0051 |                                -0.0359 |                                -67.4605 |                                 0.0001 |                                 0.0044 |                                -0.0297 |                                -73.3397 |                                 0.0001 |                                 0.0038 |                                -0.0244 |                                -78.63   |                                 0.0001 |                                 0.0032 |                                -0.0199 |                                -83.3849 |                                 0.0001 |                                 0.0027 |                                -0.016  |                                -87.6549 |                                 0.0001 |                                 0.0022 |                                -0.0127 |                                -91.487  |                                 0.0001 |                                 0.0017 |                                -0.0098 |                                -94.9242 |                                 0.0001 |                                 0.0013 |                                -0.0074 |                                -98.0055 |                                 0.0001 |                                 0.0009 |                                -0.0052 |                               -100.766  |                                 0.0001 |                                 0.0005 |                                -0.0034 |                               -103.238  |                                 0.0001 |                                 0.0001 |                                -0.0018 |                               -105.449  |                                 0.0001 |                                -0.0002 |                                -0.0004 |                               -107.424  |                                 0.0001 |                                -0.0005 |                                 0.0008 |                               -109.188  |
        |            345 |                                 0      |                                 0      |                                 0      |                                  0.0009 |                                 0      |                                 0.0002 |                                -0.011  |                                  1.2588 |                                 0      |                                 0.0017 |                                -0.0636 |                                  9.9869 |                                 0.0001 |                                 0.0042 |                                -0.1213 |                                 22.977  |                                 0.0001 |                                 0.0068 |                                -0.1553 |                                 32.6056 |                                 0.0001 |                                 0.0087 |                                -0.1662 |                                 36.1614 |                                 0.0001 |                                 0.0099 |                                -0.1624 |                                 34.2431 |                                 0.0001 |                                 0.0105 |                                -0.151  |                                 28.3973 |                                 0.0002 |                                 0.0106 |                                -0.1364 |                                 20.0943 |                                 0.0002 |                                 0.0104 |                                -0.1211 |                                 10.4568 |                                 0.0002 |                                 0.0101 |                                -0.1064 |                                  0.2635 |                                 0.0002 |                                 0.0096 |                                -0.0928 |                                 -9.9754 |                                 0.0002 |                                 0.009  |                                -0.0806 |                                -19.9413 |                                 0.0001 |                                 0.0083 |                                -0.0698 |                                -29.4465 |                                 0.0001 |                                 0.0077 |                                -0.0603 |                                -38.3896 |                                 0.0001 |                                 0.0071 |                                -0.052  |                                -46.7256 |                                 0.0001 |                                 0.0064 |                                -0.0448 |                                -54.4449 |                                 0.0001 |                                 0.0058 |                                -0.0385 |                                -61.5604 |                                 0.0001 |                                 0.0052 |                                -0.033  |                                -68.0975 |                                 0.0001 |                                 0.0047 |                                -0.0282 |                                -74.0892 |                                 0.0001 |                                 0.0042 |                                -0.024  |                                -79.5715 |                                 0.0001 |                                 0.0037 |                                -0.0203 |                                -84.5816 |                                 0.0001 |                                 0.0032 |                                -0.0171 |                                -89.156  |                                 0.0001 |                                 0.0027 |                                -0.0143 |                                -93.3297 |                                 0.0001 |                                 0.0023 |                                -0.0118 |                                -97.1357 |                                 0.0001 |                                 0.0019 |                                -0.0097 |                               -100.605  |                                 0.0001 |                                 0.0016 |                                -0.0077 |                               -103.766  |                                 0.0001 |                                 0.0012 |                                -0.0061 |                               -106.644  |                                 0.0001 |                                 0.0009 |                                -0.0046 |                               -109.264  |
        |            350 |                                 0      |                                 0      |                                 0      |                                  0.0001 |                                 0      |                                 0.0001 |                                -0.0035 |                                  0.4671 |                                 0      |                                 0.0008 |                                -0.0312 |                                  5.8737 |                                 0      |                                 0.0026 |                                -0.0745 |                                 17.2698 |                                 0.0001 |                                 0.0048 |                                -0.1094 |                                 28.8862 |                                 0.0001 |                                 0.0067 |                                -0.1286 |                                 36.5375 |                                 0.0001 |                                 0.0082 |                                -0.1346 |                                 39.1848 |                                 0.0001 |                                 0.0092 |                                -0.1321 |                                 37.4404 |                                 0.0001 |                                 0.0097 |                                -0.1247 |                                 32.4049 |                                 0.0001 |                                 0.0099 |                                -0.1148 |                                 25.1434 |                                 0.0001 |                                 0.0099 |                                -0.1042 |                                 16.5173 |                                 0.0001 |                                 0.0097 |                                -0.0936 |                                  7.1678 |                                 0.0001 |                                 0.0093 |                                -0.0835 |                                 -2.4513 |                                 0.0001 |                                 0.0089 |                                -0.0741 |                                -12.0304 |                                 0.0001 |                                 0.0084 |                                -0.0656 |                                -21.3656 |                                 0.0001 |                                 0.0079 |                                -0.058  |                                -30.3287 |                                 0.0001 |                                 0.0074 |                                -0.0511 |                                -38.8445 |                                 0.0001 |                                 0.0069 |                                -0.045  |                                -46.8741 |                                 0.0001 |                                 0.0063 |                                -0.0396 |                                -54.4032 |                                 0.0001 |                                 0.0058 |                                -0.0347 |                                -61.4341 |                                 0.0001 |                                 0.0053 |                                -0.0304 |                                -67.9799 |                                 0.0001 |                                 0.0049 |                                -0.0266 |                                -74.0599 |                                 0.0001 |                                 0.0044 |                                -0.0233 |                                -79.6977 |                                 0.0001 |                                 0.004  |                                -0.0202 |                                -84.9187 |                                 0.0001 |                                 0.0036 |                                -0.0176 |                                -89.7488 |                                 0.0001 |                                 0.0032 |                                -0.0152 |                                -94.2139 |                                 0.0001 |                                 0.0028 |                                -0.013  |                                -98.3392 |                                 0.0001 |                                 0.0025 |                                -0.0111 |                               -102.149  |                                 0.0001 |                                 0.0021 |                                -0.0094 |                               -105.665  |
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

        if all(greek.empty for greek in greeks.values()):
            # Each greek has logged why it is empty, which is the same reason for all of them.
            logger.warning("None of the greeks could be calculated.")
            return pd.DataFrame()

        greeks_df = (
            pd.concat(greeks, axis=1)
            .swaplevel(axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
        )

        return greeks_df

    @handle_errors
    def get_speed(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the speed of an option based on the Black Scholes Model: how much the gamma changes for a 1.00
        change in the price of the underlying asset.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        speed is the rate of change of the gamma with respect to the price of the underlying asset.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             10 |       0      |      -0.0003 |      -0.0094 |
        |             11 |      -0.0025 |      -0.0357 |      -0.0821 |
        |             12 |      -0.3206 |      -0.3006 |      -0.181  |
        |             13 |      -0.2865 |      -0.1279 |      -0.0636 |
        |             14 |       0.5608 |       0.31   |       0.1256 |
        |             15 |       0.0475 |       0.137  |       0.1331 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_zomma(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the zomma of an option based on the Black Scholes Model: how much the gamma changes for a 1.00
        change in the volatility.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        zomma is the rate of change of the gamma with respect to volatility.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             66 |       0.0002 |       0.0169 |       0.085  |
        |             68 |       0.0062 |       0.0729 |       0.1319 |
        |             70 |       0.0671 |       0.1714 |       0.1245 |
        |             72 |       0.2478 |       0.177  |       0.0181 |
        |             74 |       0.1558 |      -0.0714 |      -0.1588 |
        |             76 |      -0.4946 |      -0.4013 |      -0.3027 |
        |             78 |      -0.5371 |      -0.4263 |      -0.3183 |
        |             80 |       0.1029 |      -0.1275 |      -0.2014 |
        |             82 |       0.2942 |       0.1513 |      -0.032  |
        |             84 |       0.1312 |       0.2137 |       0.0996 |
        |             86 |       0.0274 |       0.1414 |       0.1536 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_color(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the color of an option based on the Black Scholes Model: how much the gamma changes as time
        passes.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        color is the rate of change of the gamma with respect to time to expiration.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |            120 |       0      |      -0.0035 |      -0.0324 |
        |            125 |      -0.0026 |      -0.0528 |      -0.0911 |
        |            130 |      -0.1388 |      -0.2563 |      -0.1154 |
        |            135 |      -0.9333 |      -0.2702 |       0.0109 |
        |            140 |       0.6316 |       0.475  |       0.2228 |
        |            145 |       1.9639 |       0.756  |       0.2789 |
        |            150 |      -0.8502 |       0.0104 |       0.1184 |
        |            155 |      -0.6023 |      -0.3775 |      -0.0688 |
        |            160 |      -0.069  |      -0.2174 |      -0.1353 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

    @handle_errors
    def get_ultima(
        self,
        start_date: str | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the ultima of an option based on the Black Scholes Model: how much the vomma changes for a 1.00
        change in the volatility.

        The Black Scholes Model is a mathematical model used to estimate the price of European—style options. The
        ultima is the rate of change of the vomma with respect to volatility.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None which
            means it will use the current dividend yield.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |             54 |     -30.462  |     -65.1596 |     -73.0513 |
        |             56 |     -46.1469 |     -58.2358 |     -57.7726 |
        |             58 |     -42.7099 |     -40.8971 |     -37.7182 |
        |             60 |     -25.0057 |     -20.7534 |     -18.6318 |
        |             62 |      -6.6506 |      -5.5439 |      -5.3419 |
        |             64 |      -0.2363 |      -0.4387 |      -0.7623 |
        |             66 |      -9.4787 |      -6.7179 |      -5.544  |
        |             68 |     -28.2953 |     -21.9944 |     -18.308  |
        |             70 |     -45.8728 |     -41.5316 |     -36.2623 |
        """
        start_date = self._pricing_date(start_date)
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
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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
