"""Multi-Asset Options Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.options import helpers
from financetoolkit.options.multiasset import multi_asset_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    handle_errors,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()

# The correlation of two assets on a date is that of the trading year of daily returns up to it
# on which both traded, once at least a month of such days is available, like the volatility.
CORRELATION_DAYS = 252
MINIMUM_CORRELATION_DAYS = 21

# Both spread approximations take the second asset's forward value plus the strike as a lognormal price, which
# strikes of at most about half the price below a spread of 0 keep positive.
MAXIMUM_SPREAD_STRIKE_RANGE = 0.5


@filter_shown_columns
@track_sources
class MultiAssetOptions:
    """
    Options on two assets: exchange (Margrabe, European and American) and spread options (Bjerksund-Stensland and Kirk) on each ticker against the benchmark or another ticker.
    """

    def _two_asset_inputs(
        self, start_date: str | None, second_ticker: str | None
    ) -> tuple[pd.Period, str, dict[str, dict[str, float]]]:
        """
        The inputs of an option on each ticker against a second asset on the pricing date: the
        prices, volatilities and dividend yields of both and the correlation of their daily
        returns over the trading year up to that date, on the days both traded.

        Args:
            start_date (str | None): The pricing date, None for the last date.
            second_ticker (str | None): The second asset, None for the benchmark.

        Returns:
            tuple[pd.Period, str, dict[str, dict[str, float]]]: The pricing date, the second
            asset and the inputs per ticker. A ticker without a price, volatility or a month of
            shared returns on the date, or that is the second asset itself, is left out with a
            warning.

        Raises:
            ValueError: If the second asset is not in the historical data.
        """
        second = "Benchmark" if second_ticker is None else second_ticker
        if second not in self._prices.columns:
            raise ValueError(
                f"{second} is not in the historical data; the second asset is the benchmark "
                "(when the Toolkit has one) or one of the tickers."
            )

        date = self._pricing_date(start_date)
        prices = self._prices.loc[date]
        volatility = self._volatility.loc[date]
        returns = self._daily_historical["Return"].loc[:date]
        second_dividend_yield = self._dividend_yield_of_any(second, date)

        inputs: dict[str, dict[str, float]] = {}

        for ticker in self._tickers:
            if ticker == second:
                logger.warning(
                    "%s is the second asset itself, so it is left out of the two-asset option values.",
                    ticker,
                )
                continue

            shared = returns[[ticker, second]].dropna().tail(CORRELATION_DAYS)
            values = [
                prices.get(ticker),
                prices.get(second),
                volatility.get(ticker),
                volatility.get(second),
            ]
            if len(shared) < MINIMUM_CORRELATION_DAYS or any(
                pd.isna(value) for value in values
            ):
                logger.warning(
                    "%s has no price, volatility or month of returns shared with %s on %s, so it is left out of the "
                    "two-asset option values.",
                    ticker,
                    second,
                    date,
                )
                continue

            inputs[ticker] = {
                "first_stock_price": float(prices[ticker]),
                "second_stock_price": float(prices[second]),
                "first_volatility": float(volatility[ticker]),
                "second_volatility": float(volatility[second]),
                "correlation": float(shared[ticker].corr(shared[second])),
                "first_dividend_yield": self._dividend_yield_of_any(ticker, date),
                "second_dividend_yield": second_dividend_yield,
            }

        return date, second, inputs

    def _dividend_yield_of_any(self, ticker: str, date: pd.Period) -> float:
        """
        The continuously compounded dividend yield of a ticker or the benchmark on a date (see
        _dividend_yield_on). The benchmark has no annual dividend yield to fall back on, so a
        benchmark without dividends on the date counts as paying none.
        """
        if ticker in self._tickers:
            return self._dividend_yield_on(ticker, date)

        if (
            not self._daily_dividend_yield.empty
            and ticker in self._daily_dividend_yield
            and date in self._daily_dividend_yield.index
            and pd.notna(self._daily_dividend_yield.loc[date, ticker])
        ):
            return float(np.log1p(self._daily_dividend_yield.loc[date, ticker]))

        return 0.0

    @handle_errors
    def get_exchange_option(
        self,
        start_date: str | None = None,
        second_ticker: str | None = None,
        american_option: bool = False,
        exchange_ratio: float = 1.0,
        expiration_time_range: int = 30,
        show_input_info: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the price of an option to exchange a second asset (the benchmark by default) for each ticker, an
        outperformance option: what the right to receive the ticker for the second asset at expiration is worth,
        with the formula of Margrabe (1978).

        The option receives one share of the ticker and delivers the number of units of the second asset that are
        worth the exchange ratio times the ticker's price today, so at an exchange ratio of 1 it pays at expiration
        what the ticker gained over the second asset since today, if anything, per share of the ticker.

        Measured in units of the second asset the payoff is a call on the ratio of the two prices, so the
        risk-free rate drops out and the volatility is that of the ratio. An American option
        may be exercised at any time, which is an American call on the ticker with the second asset as strike, a
        risk-free rate of q2 and a dividend yield of q1 (Bjerksund and Stensland, 1993), priced with the QD+
        exercise boundary and its integral (Li, 2010; Andersen, Lake and Offengenden, 2016), within about 0.1% of a
        999-step Leisen-Reimer tree. Without a dividend yield on the ticker early exercise is never optimal and the
        American option is worth the European one.

        The formula is as follows:

        - S2 = Exchange Ratio * S1, the value today of the units of the second asset delivered
        - Exchange = S1 * e^(-q1 * T) * N(d1) - S2 * e^(-q2 * T) * N(d2)
        - σ = √(σ1² + σ2² - 2ρσ1σ2)
        - d1 = (ln(S1 / S2) + (q2 - q1 + σ² / 2) * T) / (σ√T), d2 = d1 - σ√T

        Where S1 is the ticker's price (the traded close on the pricing date), S2 the second asset's price, σ1
        and σ2 the annualized volatilities (the standard deviation of the daily returns over the trailing 252
        trading days of each times √252), ρ the correlation of the daily returns of the two over the trailing 252
        days on which both traded, q1 and q2 the dividend yields (ln(1 + dividend yield), continuously
        compounded), T the time to expiration in years (days / 365) and N the cumulative standard normal
        distribution.

        Options on other two-asset payoffs, and the exchange option on prices and correlations of your own, are
        priced by the model functions in financetoolkit.options.multiasset.multi_asset_model.

        See: Margrabe, W. (1978), "The Value of an Option to Exchange One Asset for Another", Journal of Finance,
        33(1), 177-186; Bjerksund, P. & Stensland, G. (1993), "American Exchange Options and a Put-Call
        Transformation: A Note", Journal of Business Finance & Accounting, 20(5), 761-764; Haug, E. G. (2007),
        The Complete Guide to Option Pricing Formulas, 2nd ed., chapter 5. The American value is checked
        against a Leisen-Reimer tree of 999 steps on the call it equals (QuantLib's AnalyticAmericanMargrabeEngine
        takes the Bjerksund-Stensland (1993) approximation, a lower bound about 1% below it).

        Also known as: Margrabe option, exchange option, outperformance option, swap option.

        Referenced against:
        - QuantLib, AnalyticEuropeanMargrabeEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_european_margrabe_engine.html

        Shares similar logic:
        - QuantLib, AnalyticAmericanMargrabeEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_american_margrabe_engine.html

        Args:
            start_date (str | None, optional): The date whose prices, volatilities and correlation the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price or a month of returns shared with the second asset is left out with a warning.
            Defaults to None, which uses the last date of the historical data.
            second_ticker (str | None, optional): The asset delivered, "Benchmark" or one of the tickers (which is
            then left out of the rows). Defaults to None, which uses the benchmark.
            american_option (bool, optional): Whether the option can be exercised before expiration. Defaults to
            False.
            exchange_ratio (float, optional): The value today of the units of the second asset delivered, as a
            share of the ticker's price: below 1 the option is in the money, above 1 out of the money. Defaults
            to 1.0, at the money.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Exchange option values per share of the ticker, with the tickers as the index and the
            expiration dates as the columns.

        Raises:
            InvalidArgumentError: If exchange_ratio is not positive.
            ValueError: If the second asset is not in the historical data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        exchange_option = toolkit.options.get_exchange_option(expiration_time_range=91)

        exchange_option[["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        | Ticker   |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |:---------|-------------:|-------------:|-------------:|
        | AAPL     |       6.7664 |       9.5415 |      11.659  |
        | MSFT     |       9.5013 |      13.351  |      16.2696 |
        """
        if exchange_ratio <= 0:
            raise InvalidArgumentError(
                f"exchange_ratio must be positive, received {exchange_ratio!r}."
            )

        start_date, second, inputs = self._two_asset_inputs(start_date, second_ticker)

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

        exchange_option: dict[str, dict[float, float]] = {}

        for ticker, values in inputs.items():
            try:
                # The units of the second asset delivered are worth exchange_ratio times the ticker today.
                pair = {
                    **values,
                    "second_stock_price": exchange_ratio * values["first_stock_price"],
                }
                exchange_option[ticker] = {
                    time_to_expiration: multi_asset_model.get_exchange_option(
                        **pair,
                        time_to_expiration=time_to_expiration,
                        american_option=american_option,
                    )
                    for time_to_expiration in time_to_expiration_list
                }
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No exchange option value for %s: %s", ticker, error)

        exchange_option_df = helpers.create_expiration_dataframe(
            value_dictionary=exchange_option, start_date=start_date
        ).pipe(apply_rounding, rounding if rounding is not None else self._rounding)

        if show_input_info:
            self._show_two_asset_input_info(second, inputs)

        return exchange_option_df

    @handle_errors
    def get_spread_option(
        self,
        start_date: str | None = None,
        second_ticker: str | None = None,
        put_option: bool = False,
        method: Literal["bjerksund_stensland", "kirk"] = "bjerksund_stensland",
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the price of a European spread option of each ticker over a second asset (the benchmark by
        default): what an option on how much the ticker gains over the second asset is worth, with the closed-form
        approximation of Bjerksund and Stensland (2011) or of Kirk (1995).

        The option is on one share of the ticker less the units of the second asset worth the same today, so the
        spread starts at 0 and the call pays max(S1 - S2 - K, 0) at expiration, where S1 - S2 is what the ticker
        gained over the second asset per share. The strikes run from the strike price range below to the strike
        price range above a spread of 0, as a share of the ticker's price.

        The spread of two lognormal prices has no closed-form distribution. Kirk takes F2 + K as one lognormal
        price, and Bjerksund and Stensland exercise where F1 exceeds a boundary fitted to the exercise region,
        which gives a lower bound that is closer to the exact price. At a strike of 0 both are Margrabe's exact
        exchange option (get_exchange_option).

        The formula is as follows:

        - F1 = S1 * e^((r - q1) * T), F2 = S2 * e^((r - q2) * T), with S2 = S1 today
        - Kirk Call = e^(-rT) * (F1 * N(d1) - (F2 + K) * N(d1 - σK√T)), σK² = σ1² - 2ρσ1σ2 * b + σ2² * b², b = F2 / (F2 + K)
        - d1 = (ln(F1 / (F2 + K)) + σK² * T / 2) / (σK√T)
        - Bjerksund-Stensland Call = e^(-rT) * (F1 * N(e1) - F2 * N(e2) - K * N(e3))
        - e1 = (ln(F1 / (F2 + K)) + (σ1² / 2 - bρσ1σ2 + b²σ2² / 2) * T) / (σK√T)
        - e2 = (ln(F1 / (F2 + K)) + (-σ1² / 2 + ρσ1σ2 + b²σ2² / 2 - bσ2²) * T) / (σK√T)
        - e3 = (ln(F1 / (F2 + K)) + (-σ1² / 2 + b²σ2² / 2) * T) / (σK√T)
        - Put = Call - e^(-rT) * (F1 - F2 - K)

        Where S1 is the ticker's price (the traded close on the pricing date), K the strike price, r the
        continuously compounded risk-free rate, q1 and q2 the dividend yields (ln(1 + dividend yield),
        continuously compounded), σ1 and σ2 the annualized volatilities (the standard deviation of the daily
        returns over the trailing 252 trading days of each times √252), ρ the correlation of the daily returns of
        the two over the trailing 252 days on which both traded, T the time to expiration in years (days / 365)
        and N the cumulative standard normal distribution.

        Spread options on prices, strikes and correlations of your own (such as a crack or a calendar spread of
        futures), are priced by the model function
        financetoolkit.options.multiasset.multi_asset_model.get_spread_option.

        See: Kirk, E. (1995), "Correlation in the Energy Markets", in Managing Energy Price Risk, Risk
        Publications, 71-78; Bjerksund, P. & Stensland, G. (2014), "Closed Form Spread Option Valuation",
        Quantitative Finance, 14(10), 1785-1794; Haug, E. G. (2007), The Complete Guide to Option Pricing
        Formulas, 2nd ed., chapter 5.

        Also known as: spread option, relative performance option, Kirk's approximation.

        Referenced against:
        - QuantLib, BjerksundStenslandSpreadEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_bjerksund_stensland_spread_engine.html
        - QuantLib, KirkEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_kirk_engine.html

        Args:
            start_date (str | None, optional): The date whose prices, volatilities, correlation and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price or a month of returns shared with the second asset is left out with a warning.
            Defaults to None, which uses the last date of the historical data.
            second_ticker (str | None, optional): The second asset, "Benchmark" or one of the tickers (which is
            then left out of the rows). Defaults to None, which uses the benchmark.
            put_option (bool, optional): Whether the option is a put. Defaults to False.
            method (str, optional): The approximation. Options are: "bjerksund_stensland" (Bjerksund and Stensland,
            2011, the more accurate), "kirk" (Kirk, 1995). Defaults to "bjerksund_stensland".
            strike_price_range (float): The range of the strike prices around a spread of 0, as a share of the
            ticker's price, at most 0.5 (both approximations take the second asset's forward price plus the strike
            as a lognormal price, which lower strikes bring close to zero). Defaults to 0.25, which gives strikes from -25 to 25 for a price of 100.
            strike_step_size (float): The step size of the strike prices. Defaults to 5, which gives the strikes
            -25, -20, ..., 20 and 25 for a price of 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Spread option values per share of the ticker, with the tickers and strike prices as the
            index and the expiration dates as the columns.

        Raises:
            InvalidArgumentError: If strike_price_range is not between 0 and 0.5, strike_step_size is not above 0 or
            the method is not one of the two.
            ValueError: If the second asset is not in the historical data.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        spread_option = toolkit.options.get_spread_option(
            strike_price_range=0.1, expiration_time_range=91
        )

        spread_option.loc["AAPL", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            -30 |      29.995  |      30.5915 |      31.4131 |
        |            -25 |      25.2477 |      26.2471 |      27.3611 |
        |            -20 |      20.7127 |      22.1749 |      23.58   |
        |            -15 |      16.4979 |      18.431  |      20.1017 |
        |            -10 |      12.7141 |      15.0607 |      16.9489 |
        |             -5 |       9.4529 |      12.0938 |      14.1342 |
        |              0 |       6.7664 |       9.5415 |      11.659  |
        |              5 |       4.6569 |       7.3963 |       9.5148 |
        |             10 |       3.0798 |       5.6345 |       7.6844 |
        |             15 |       1.9575 |       4.2199 |       6.1438 |
        |             20 |       1.1965 |       3.1087 |       4.8645 |
        |             25 |       0.7041 |       2.2539 |       3.8161 |
        |             30 |       0.3994 |       1.6094 |       2.9672 |
        """
        check_option("method", method, ("bjerksund_stensland", "kirk"))
        # Tighter than the 0 to 1 of the other methods: both approximations take F2 + K as a lognormal price,
        # which a strike near minus the price would bring to zero or below, where the formula has no value.
        if not 0 <= strike_price_range <= MAXIMUM_SPREAD_STRIKE_RANGE:
            raise InvalidArgumentError(
                f"strike_price_range must be between 0 and {MAXIMUM_SPREAD_STRIKE_RANGE} for a spread option, not "
                f"{strike_price_range!r}: the approximations take the second asset's forward price plus the strike "
                "as a lognormal price, which strikes further below a spread of 0 bring close to zero or below. "
                f"Please pass a value between 0 and {MAXIMUM_SPREAD_STRIKE_RANGE} as strike_price_range."
            )
        if not strike_step_size > 0:
            raise InvalidArgumentError(
                f"strike_step_size is the distance between strike prices, above 0, not {strike_step_size!r}. "
                "Please pass a positive step such as 5 or 2.5 as strike_step_size."
            )

        start_date, second, inputs = self._two_asset_inputs(start_date, second_ticker)

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

        spread_option: dict[str, dict[float, dict[float, float]]] = {}

        for ticker, values in inputs.items():
            try:
                price = values["first_stock_price"]
                if (
                    2 * strike_price_range * price / strike_step_size
                    > helpers.MAX_STRIKES
                ):
                    raise InvalidArgumentError(
                        f"The strike prices of {ticker} would be more than {helpers.MAX_STRIKES:,} prices. Please use "
                        "a larger strike_step_size or a smaller strike_price_range."
                    )
                # The strikes are the multiples of the step from strike_price_range below to strike_price_range
                # above a spread of 0, so the grid covers the whole range and always holds a strike of 0.
                steps = np.arange(
                    np.floor(-strike_price_range * price / strike_step_size + 1e-9),
                    np.ceil(strike_price_range * price / strike_step_size - 1e-9) + 1,
                )
                strike_prices = (
                    [int(step * strike_step_size) for step in steps]
                    if float(strike_step_size).is_integer()
                    else [round(float(step * strike_step_size), 10) for step in steps]
                )
                # The units of the second asset are worth one share of the ticker today.
                pair = {**values, "second_stock_price": price}
                spread_option[ticker] = {
                    strike_price: {
                        time_to_expiration: multi_asset_model.get_spread_option(
                            **pair,
                            strike_price=float(strike_price),
                            risk_free_rate=float(risk_free_rate),
                            time_to_expiration=time_to_expiration,
                            put_option=put_option,
                            method=method,
                        )
                        for time_to_expiration in time_to_expiration_list
                    }
                    for strike_price in strike_prices
                }
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No spread option value for %s: %s", ticker, error)

        spread_option_df = helpers.create_greek_dataframe(
            greek_dictionary=spread_option, start_date=start_date
        ).pipe(apply_rounding, rounding if rounding is not None else self._rounding)

        if show_input_info:
            self._show_two_asset_input_info(second, inputs, risk_free_rate)

        return spread_option_df

    def _show_two_asset_input_info(
        self,
        second: str,
        inputs: dict[str, dict[str, float]],
        risk_free_rate: float | None = None,
    ) -> None:
        """Logs the prices, volatilities, dividend yields and correlations of the two-asset options."""
        for ticker, values in inputs.items():
            logger.info(
                "%s against %s: prices %.2f and %.2f, volatilities %.2f%% and %.2f%%, dividend yields %.2f%% "
                "and %.2f%%, correlation %.4f",
                ticker,
                second,
                values["first_stock_price"],
                values["second_stock_price"],
                values["first_volatility"] * 100,
                values["second_volatility"] * 100,
                values["first_dividend_yield"] * 100,
                values["second_dividend_yield"] * 100,
                values["correlation"],
            )
        if risk_free_rate is not None:
            logger.info("Risk Free Rate: %s%%", round(float(risk_free_rate) * 100, 2))
