"""Option Volatility Module"""

__docformat__ = "google"

from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.options import helpers, options_model
from financetoolkit.options.helpers import (
    get_days_to_expiration,
    get_time_to_expiration,
)
from financetoolkit.options.market import market_metrics_model
from financetoolkit.options.pricing import american_model, black_scholes_model
from financetoolkit.options.volatility import (
    garch_option_model,
    heston_model,
    local_volatility_model,
    risk_neutral_density_model,
    sabr_model,
    svi_model,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    handle_errors,
    validate_window,
)
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import (
    apply_rounding,
    calculate_standardization,
)

# ruff: noqa: E501

logger = logger_model.get_logger()

MINIMUM_OBSERVATIONS_FOR_SVI_FIT = 5

# The number of calendar days the last historical price may lag today before the spot price
# is treated as stale next to the live option chains (a weekend plus a holiday is four).
MAXIMUM_SPOT_AGE_DAYS = 5

MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK = 2

SMILE_METHODS = ("svi", "ssvi", "sabr")

# The Heston fit by default takes the listed expiration nearest to each of these times to
# expiration, in months, leaving out those under MINIMUM_DAYS_FOR_HESTON_FIT calendar days.
HESTON_EXPIRATION_MONTHS = (1, 2, 3, 6, 9, 12, 18)
MINIMUM_DAYS_FOR_HESTON_FIT = 14
HESTON_PARAMETERS = ("v0", "kappa", "theta", "sigma", "rho")


@filter_shown_columns
@track_sources
class VolatilityOptions:
    """
    Implied volatility, the SVI, SSVI and SABR volatility surfaces, the local volatility,
    the risk-neutral density, the Heston calibration and the Heston-Nandi GARCH prices.
    """

    def _warn_if_spot_is_stale(self) -> None:
        """
        Logs a warning when the last historical date lies more than MAXIMUM_SPOT_AGE_DAYS
        calendar days before today. The methods that work with the live option chains
        (implied volatility, the volatility surface and the risk-neutral density) take the
        quotes, the price of the underlying quoted with them and the time to expiration
        from the current time, but the risk free rate, the dividend yield and, when a chain quotes no
        price of the underlying, the stock price from the last historical date, so an
        end_date in the past or an outdated price history pairs today's option prices with
        old inputs. The history of an instance does not change, so the warning is shown once
        per instance rather than once for every expiration date a surface walks through.
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
                "The historical data ends on %s, %d days ago, while the option chains and the "
                "time to expiration are live: the risk free rate, the dividend yield and, when a "
                "chain quotes no price of the underlying, the stock price come from that date. "
                "Set end_date to today or leave it empty for current results.",
                last_timestamp.date(),
                spot_age,
            )

    @handle_errors
    def get_implied_volatility(
        self,
        expiration_date: str | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        american_option: bool = False,
        show_expiration_dates: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Implied Volatility (IV): the volatility at which the Black Scholes Model (or the American
        price) reproduces the actual option prices, for any of the available expiration dates.

        Implied Volatility (IV) is a measure of how much the market expects the price of the underlying asset to
        fluctuate in the future. It is a key component of options pricing and can also be used to calculate the
        theoretical value of an option.

        The price of the underlying quoted with the option chain and the most recent risk free rate and dividend
        yield are used, and the implied volatility is the volatility at which the Black Scholes price equals the
        option's mid price, the middle of its bid and ask (the last traded price when the option has no two-sided
        quote, for example outside market hours). A strike without a two-sided quote and without open interest or
        volume has no implied volatility: its last trade can be weeks old, from when the underlying stood elsewhere.

        The formulas are as follows:

        - d1 = (ln(S / K) + (r — q + (σ^2) / 2) * t) / (σ * sqrt(t))
        - d2 = d1 — σ * sqrt(t)
        - Call Option Price = S * e^(—q * t) * N(d1) — K * e^(—r * t) * N(d2)
        - Put Option Price = K * e^(—r * t) * N(—d2) — S * e^(—q * t) * N(—d1)

        Where S is the stock price, K is the strike price, r is the risk free rate, q is the dividend yield, σ is the
        volatility, t is the time to expiration (the minutes up to 16:00 New York time on the expiration date over
        the 525,600 minutes of a year, as the Cboe counts it), N(d1) is the cumulative normal distribution of d1 and N(d2) is the
        the cumulative normal distribution of d2.

        In which the Implied Volatility is then calculated as follows:

        - Implied Volatility = the σ for which Black Scholes Theoretical Price(σ) — Actual Option Price = 0

        To determine the Implied Volatility, the Black Scholes Model is used to calculate the theoretical option price in
        which sigma (σ) is the only unknown variable. Since that price rises strictly with σ, the implied volatility is
        the root of the difference between the theoretical and actual option price, found with Brent's method on the
        interval [1e-6, 5]. Prices outside the no-arbitrage bounds have no root and return NaN.

        Listed options on single stocks are American and can be exercised before expiration, which the Black Scholes
        price leaves out. For American options the implied volatility is the σ at which the American price of the QD+
        method (Li, 2010; Andersen, Lake and Offengenden, 2016, see ``get_american_option``) equals the mid price,
        found with Brent's method on [0.001, 5]. The European volatility attributes the value of exercising early to
        a higher volatility, so in-the-money puts and calls on stocks with a high dividend yield show a lower American
        implied volatility; for out-of-the-money options the two hardly differ. Index options such as those on the
        S&P 500 (^SPX) are European. The prices of QuantLib's QdPlusAmericanEngine invert to the volatility they
        were priced at.

        Mid prices and the underlying price quoted with the chain describe the same moment, unlike last traded
        prices, which can be hours or days old for illiquid strikes. Deep in-the-money and far out-of-the-money
        strikes still have wide quotes and can show extreme values (above 100%) that say more about the spread than
        about the market's view. Practitioners read the smile around the money, where trading is liquid, and use
        `get_volatility_surface`, which filters outliers and fits a smooth curve, when they need values across all
        strikes.

        The quotes, the underlying price quoted with them and the time to expiration are live, while the risk free
        rate and the dividend yield come from the last historical date, as does the stock price when a chain quotes
        no price of the underlying. When that date lies more than five days before today (for example with an
        end_date in the past), a warning is logged.

        Also known as: IV, option-implied volatility.

        Referenced against:
        - QuantLib, QdPlusAmericanEngine (American): https://www.quantlib.org/reference/class_quant_lib_1_1_qd_plus_american_engine.html

        Args:
            expiration_date (str | None, optional): The expiration date to use for the calculation. Defaults to None,
            which uses the first expiration date at least one calendar day away (an expiry of today has hardly any
            time left to price) that every ticker lists, if they share one. A ticker without options expiring on the
            date is left out with a warning.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            american_option (bool, optional): Whether to solve the volatility of the American price (QD+) instead of the
                European Black Scholes price. Defaults to False.
            show_expiration_dates (bool, optional): Whether to show the expiration dates: a list for a single ticker
                and a list per ticker (a dictionary) for several. Defaults to False.
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
            pd.Series | list[str]: Implied Volatility values indexed by ticker and strike price, with the expiration
            date as the Series name. If show_expiration_dates is True, it returns the expiration dates.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL"], api_key="FINANCIAL_MODELING_PREP_KEY")

        implied_volatility = toolkit.options.get_implied_volatility()

        implied_volatility.loc["AAPL"].tail(20)
        ```

        Which returns:

        |   Strike Price |   2026-11-18 |
        |---------------:|-------------:|
        |          347.5 |       0.3258 |
        |          352.5 |       0.2814 |
        |          357.5 |       0.2722 |
        |          362.5 |       0.2576 |
        |          367.5 |       0.2675 |
        |          372.5 |       0.2898 |
        |          377.5 |       0.3803 |
        """
        if show_expiration_dates:
            return self.get_option_chains(show_expiration_dates=True)

        # The dates of every ticker, so a date that only a later ticker lists is valid too.
        expiration_dates = self._all_expiration_dates()

        if expiration_date is None:
            # An expiry of today has hardly any time left to price, so the first expiry at
            # least a calendar day away is used, one that every ticker lists if they share any.
            now = self._current_time()
            expiration_date = next(
                (
                    date
                    for date in self._all_expiration_dates(shared_first=True)
                    if get_days_to_expiration(date, now) >= 1
                ),
                None,
            )
            if expiration_date is None:
                raise ValueError("No expiration date lies a day or more away.")
        elif expiration_date not in expiration_dates:
            raise ValueError(
                f"The expiration date {expiration_date} is not a valid date. Choose from "
                f"{', '.join(expiration_dates)}"
            )

        implied_volatility_df, stock_price, risk_free_rate, dividend_yield_value, _ = (
            self._solve_implied_volatility(
                expiration_date,
                put_option,
                risk_free_rate,
                dividend_yield,
                american_option,
            )
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
        implied_volatility_df.name = expiration_date

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=self._volatility.loc[self._daily_historical.index[-1]],
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield_value,
            )

        return implied_volatility_df

    def _solve_implied_volatility(
        self,
        expiration_date: str,
        put_option: bool,
        risk_free_rate: float | None,
        dividend_yield: float | None,
        american_option: bool = False,
    ) -> tuple[pd.Series, pd.Series, float, dict[str, float], float]:
        """
        The implied volatility of every strike of one expiry, unrounded, from the mid prices
        and the price of the underlying quoted with the chain (see _spot_price), with the
        inputs it used, so that the volatility surface and the risk-neutral density work
        with the same stock price and time to expiration.

        Args:
            expiration_date (str): The expiration date.
            put_option (bool): Whether to use the puts instead of the calls.
            risk_free_rate (float | None): The continuously compounded risk free rate, None
                for the rate on the last historical date.
            dividend_yield (float | None): The continuous dividend yield, None for the
                yield of each ticker on the last historical date.
            american_option (bool, optional): Whether to solve the volatility of the American (QD+)
                price instead of the Black Scholes price. Defaults to False.

        Returns:
            tuple: The implied volatility indexed by ticker and strike price, the stock price
            per ticker, the risk free rate, the dividend yield per ticker and the time to
            expiration in years.

        Raises:
            ValueError: If the options have expired.
        """
        time_to_expiration = get_time_to_expiration(
            expiration_date, self._current_time()
        )
        if time_to_expiration <= 0:
            raise ValueError(f"The options expiring on {expiration_date} have expired.")

        # The chains straight from the source, which keep the price of the underlying.
        option_chains = options_model.get_option_chains(
            tickers=self._tickers,
            expiration_date=expiration_date,
            put_option=put_option,
        )

        self._warn_if_spot_is_stale()

        current_period = self._daily_historical.index[-1]
        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[current_period]
        )

        tickers = option_chains.index.get_level_values(0).unique()
        stock_price: dict[str, float] = {}
        dividend_yield_value: dict[str, float] = {}
        implied_volatility: dict[str, dict[float, float]] = {}

        for ticker in tickers:
            implied_volatility[ticker] = {}
            option_chain = option_chains.loc[ticker]
            stock_price[ticker] = self._spot_price(ticker, option_chain)
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, current_period)
            )

            # The mid price and the quoted price of the underlying are of the same moment,
            # while a last traded price can be hours or days older than either; it is only
            # used for an option with open interest or volume (see get_mid_prices).
            option_prices = market_metrics_model.get_mid_prices(option_chain)

            for strike_price, option_price in option_prices.items():
                # Numerically finds the volatility matching the market option price.
                implied_volatility_value = (
                    american_model.get_american_implied_volatility
                    if american_option
                    else black_scholes_model.get_implied_volatility
                )(
                    market_price=float(option_price),
                    stock_price=float(stock_price[ticker]),
                    strike_price=float(strike_price),
                    risk_free_rate=float(risk_free_rate),
                    time_to_expiration=time_to_expiration,
                    dividend_yield=float(dividend_yield_value[ticker]),
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
        implied_volatility_df.name = expiration_date

        return (
            implied_volatility_df,
            pd.Series(stock_price),
            risk_free_rate,
            dividend_yield_value,
            time_to_expiration,
        )

    def _collect_smile_quotes(
        self,
        expiration_dates: list[str] | None,
        number_of_expirations: int,
        put_option: bool,
        risk_free_rate: float | None,
        dividend_yield: float | None,
        outlier_threshold: float,
    ) -> dict[str, list[dict]]:
        """
        Collect the market implied volatility of every ticker and expiry, without the quotes
        far from the median volatility, for the smile models and the Heston calibration.

        Args:
            expiration_dates (list[str] | None): The expiration dates, None for the first
                number_of_expirations dates (those every ticker lists, if they share any).
            number_of_expirations (int): The number of dates when expiration_dates is None.
            put_option (bool): Whether to use the puts instead of the calls.
            risk_free_rate (float | None): The continuously compounded risk free rate, None
                for the rate on the last historical date.
            dividend_yield (float | None): The continuous dividend yield, None for the yield
                of each ticker on the last historical date.
            outlier_threshold (float): The number of median absolute deviations beyond which
                a quote is dropped.

        Returns:
            dict[str, list[dict]]: Per ticker, the expiries in expiration order, each with the
            expiration date, time to expiration, stock and forward price, risk free rate,
            dividend yield and the market implied volatility by strike.

        Raises:
            ValueError: If an expiration date is not listed.
        """
        # The dates of every ticker; the nearest ones by default are those every ticker
        # lists, if they share any, so each expiry of the surface covers every ticker.
        all_expiration_dates = self._all_expiration_dates()

        if expiration_dates is None:
            # An expiry of today has hardly any time left, and its smile is dominated by the
            # bid-ask spread, so the default dates start a calendar day away, as the implied
            # volatility does.
            now = self._current_time()
            expiration_dates = [
                date
                for date in self._all_expiration_dates(shared_first=True)
                if get_days_to_expiration(date, now) >= 1
            ][:number_of_expirations]
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

        quotes: dict[str, list[dict]] = {}
        left_out: dict[str, str] = {}

        for expiration_date in sorted(expiration_dates):
            # The smile of every expiry is fitted with the stock price, rates and time to
            # expiration its implied volatilities were solved with. An expiry that has
            # expired or has no chain is left out.
            try:
                (
                    implied_volatility,
                    stock_price,
                    risk_free_rate_value,
                    dividend_yield_value_per_ticker,
                    time_to_expiration,
                ) = self._solve_implied_volatility(
                    expiration_date, put_option, risk_free_rate, dividend_yield
                )
            except (KeyError, IndexError, ValueError) as error:
                logger.debug("The %s expiry is left out: %s", expiration_date, error)
                left_out[expiration_date] = str(error)
                continue

            if implied_volatility.empty:
                continue

            for ticker in implied_volatility.index.get_level_values(0).unique():
                ticker_iv = implied_volatility.loc[ticker]

                # A single stale or wide quote can dominate a least-squares fit of the whole
                # expiry, so quotes far from the median volatility are dropped first.
                median_iv = ticker_iv.median()
                deviation = (ticker_iv - median_iv).abs()
                mad = deviation.median()
                if mad > 0:
                    ticker_iv = ticker_iv[deviation <= outlier_threshold * mad]

                if len(ticker_iv) < MINIMUM_OBSERVATIONS_FOR_SVI_FIT:
                    continue

                quotes.setdefault(ticker, []).append(
                    {
                        "expiration_date": expiration_date,
                        "time_to_expiration": time_to_expiration,
                        "stock_price": float(stock_price.loc[ticker]),
                        "forward_price": float(
                            stock_price.loc[ticker]
                            * np.exp(
                                (
                                    risk_free_rate_value
                                    - dividend_yield_value_per_ticker[ticker]
                                )
                                * time_to_expiration
                            )
                        ),
                        "risk_free_rate": float(risk_free_rate_value),
                        "dividend_yield": float(
                            dividend_yield_value_per_ticker[ticker]
                        ),
                        "implied_volatility": ticker_iv,
                    }
                )

        # Every expiry left out (all expired, or without a chain) leaves nothing to fit, said once.
        if not quotes and left_out:
            logger.warning(
                "None of the expiries %s can be fitted, so the result is empty: %s",
                ", ".join(left_out),
                next(iter(left_out.values())),
            )

        return quotes

    def _calibrate_smiles(
        self,
        expiration_dates: list[str] | None,
        number_of_expirations: int,
        put_option: bool,
        risk_free_rate: float | None,
        dividend_yield: float | None,
        outlier_threshold: float,
        method: str,
        sabr_beta: float = 1.0,
    ) -> dict[str, list[dict]]:
        """
        Fit the smile of every ticker and expiry with SVI, SSVI or SABR, check the SVI and SSVI
        surfaces for calendar-spread and butterfly arbitrage (logged, not corrected) and return
        the slices, so the volatility surface and the local volatility share one calibration.

        Args:
            expiration_dates (list[str] | None): The expiration dates, None for the first
                number_of_expirations dates (those every ticker lists, if they share any).
            number_of_expirations (int): The number of dates when expiration_dates is None.
            put_option (bool): Whether to use the puts instead of the calls.
            risk_free_rate (float | None): The continuously compounded risk free rate, None
                for the rate on the last historical date.
            dividend_yield (float | None): The continuous dividend yield, None for the yield
                of each ticker on the last historical date.
            outlier_threshold (float): The number of median absolute deviations beyond which
                a quote is dropped before fitting.
            method (str): "svi", "ssvi" or "sabr".
            sabr_beta (float, optional): The SABR elasticity β. Defaults to 1.0.

        Returns:
            dict[str, list[dict]]: Per ticker, the fitted slices in expiration order, each
            with the expiration date, time to expiration, stock and forward price, the market
            implied volatility it was fitted to (by strike), the fitted parameters, the
            fitted volatility at those strikes and, for SVI and SSVI, the raw SVI parameters.

        Raises:
            ValueError: If the method is unknown or an expiration date is not listed.
        """
        check_option("method", method, SMILE_METHODS)

        quotes = self._collect_smile_quotes(
            expiration_dates,
            number_of_expirations,
            put_option,
            risk_free_rate,
            dividend_yield,
            outlier_threshold,
        )

        smiles: dict[str, list[dict]] = {}

        for ticker, ticker_quotes in quotes.items():
            for quote in ticker_quotes:
                quote["log_moneyness"] = np.log(
                    quote["implied_volatility"].index.to_numpy(dtype=float)
                    / quote["forward_price"]
                )
                quote["total_variance"] = (
                    quote["implied_volatility"].to_numpy() ** 2
                ) * quote["time_to_expiration"]

            if method == "ssvi":
                # SSVI is fitted to every expiry at once, and needs the at-the-money total
                # variance of each, so an expiry whose quotes do not straddle the forward is
                # left out.
                straddling_quotes = [
                    quote
                    for quote in ticker_quotes
                    if quote["log_moneyness"].min() <= 0 <= quote["log_moneyness"].max()
                ]
                if not straddling_quotes:
                    continue
                try:
                    ssvi_parameters = svi_model.get_ssvi_parameters(
                        {
                            quote["time_to_expiration"]: quote["log_moneyness"]
                            for quote in straddling_quotes
                        },
                        {
                            quote["time_to_expiration"]: quote["total_variance"]
                            for quote in straddling_quotes
                        },
                    )
                except ValueError as error:
                    logger.warning("The SSVI fit of %s failed: %s", ticker, error)
                    continue
                for quote in straddling_quotes:
                    quote["parameters"] = ssvi_parameters[quote["time_to_expiration"]]
                    quote["svi_parameters"] = svi_model.convert_ssvi_to_svi(
                        **quote["parameters"]
                    )
            else:
                for quote in ticker_quotes:
                    try:
                        if method == "svi":
                            quote["parameters"] = svi_model.get_svi_parameters(
                                quote["log_moneyness"], quote["total_variance"]
                            )
                            quote["svi_parameters"] = quote["parameters"]
                        else:
                            quote["parameters"] = sabr_model.get_sabr_parameters(
                                quote["implied_volatility"].index.to_numpy(dtype=float),
                                quote["implied_volatility"].to_numpy(),
                                quote["forward_price"],
                                quote["time_to_expiration"],
                                beta=sabr_beta,
                            )
                    except ValueError as error:
                        logger.debug(
                            "The %s smile of %s is left out: %s",
                            quote["expiration_date"],
                            ticker,
                            error,
                        )

            fitted_quotes = [quote for quote in ticker_quotes if "parameters" in quote]

            for quote in fitted_quotes:
                if method == "sabr":
                    quote["fitted_volatility"] = sabr_model.get_sabr_volatility(
                        quote["implied_volatility"].index.to_numpy(dtype=float),
                        quote["forward_price"],
                        quote["time_to_expiration"],
                        **quote["parameters"],
                    )
                else:
                    quote["fitted_volatility"] = svi_model.get_svi_implied_volatility(
                        quote["log_moneyness"],
                        quote["time_to_expiration"],
                        **quote["svi_parameters"],
                    )

            if not fitted_quotes:
                continue

            smiles[ticker] = fitted_quotes

            if method == "sabr":
                continue

            svi_parameters = {
                quote["time_to_expiration"]: quote["svi_parameters"]
                for quote in fitted_quotes
            }

            if len(svi_parameters) >= MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK:
                calendar_violations = svi_model.check_calendar_arbitrage(svi_parameters)
                if not calendar_violations.empty:
                    logger.warning(
                        "The calibrated volatility surface for %s has %d calendar-spread "
                        "arbitrage violation(s) across the checked log-moneyness grid -- "
                        "this reflects genuine inconsistency in the underlying market "
                        "quotes across expiries, not a fitting error.",
                        ticker,
                        len(calendar_violations),
                    )

            butterfly_violations = svi_model.check_butterfly_arbitrage(svi_parameters)
            if not butterfly_violations.empty:
                logger.warning(
                    "The calibrated volatility surface for %s has butterfly arbitrage in %d "
                    "expiries (g(k) < 0 at %d points of the checked log-moneyness grid): the "
                    "implied density of the underlying is negative there.",
                    ticker,
                    butterfly_violations["Expiration"].nunique(),
                    len(butterfly_violations),
                )

        return smiles

    @handle_errors
    def get_volatility_surface(
        self,
        expiration_dates: list[str] | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        number_of_expirations: int = 6,
        outlier_threshold: float = 5.0,
        method: Literal["svi", "ssvi", "sabr"] = "svi",
        sabr_beta: float = 1.0,
        show_parameters: bool = False,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calibrate an implied volatility surface across multiple expiries: the smooth smile
        of each expiry that fits the market-implied smile (see `get_implied_volatility`), a
        raw SVI curve (Gatheral 2004) per expiry, an SSVI surface (Gatheral and Jacquier
        2014) across all expiries at once, or a SABR smile (Hagan et al. 2002) per expiry.

        A single per-expiry smile only tells you the shape of the market's volatility skew
        at that one maturity. Stitching several calibrated slices together instead gives a
        full surface, which is what is needed to price/interpolate options at maturities or
        strikes that don't trade directly, and to check for term-structure inconsistencies
        (see Notes).

        The formulas are as follows:

        - SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)² + sigma²)), per expiry
        - SSVI: w(k, θ_t) = θ_t / 2 * (1 + ρ * φ(θ_t) * k + sqrt((φ(θ_t) * k + ρ)² + 1 - ρ²))
        - SSVI curvature: φ(θ) = η / (θ^γ * (1 + θ)^(1 - γ))
        - SABR: σ(K) = α / ((F * K)^((1 - β) / 2) * D) * (z / x(z)) * (1 + I * T), per expiry

        Where w is the total implied variance (implied volatility squared times the time to
        expiration t), k the log-moneyness ln(K / F) with K the strike and F the forward
        price, θ_t the at-the-money total variance of the expiry and D, z, x(z) and I the
        terms of Hagan et al. (see `sabr_model.get_sabr_volatility`). The implied volatility
        is sqrt(w(k) / t) for SVI and SSVI.

        - SVI fits five parameters per expiry and follows each smile closely,
          but a slice can carry butterfly arbitrage and two slices can cross. The fit keeps
          the total variance non-negative through Gatheral's condition
          a + b * sigma * sqrt(1 - rho^2) >= 0.
        - SSVI takes θ_t from each expiry's at-the-money total variance and
          fits three parameters, ρ, η and γ, to all expiries together with
          η * (1 + |ρ|) <= 2 and 0 < γ <= 1/2, which keeps every slice free of butterfly
          arbitrage and the surface free of calendar-spread arbitrage when θ_t rises with
          the expiry. It follows each smile less closely than SVI.
        - SABR fits ρ and ν per expiry with β fixed and α set by
          the at-the-money volatility (West 2005), the smile model of the rates market and
          a common alternative for equities.

        Before fitting, strikes whose market-implied volatility is a statistical outlier
        relative to its neighbors (a common symptom of a stale or wide-bid/ask illiquid
        quote) are dropped via a median-absolute-deviation filter, since a single bad quote
        can otherwise dominate the least-squares fit for that whole expiry. Every fit is
        bounded and runs from a fixed grid of starting points, so the same quotes always
        give the same surface.

        The option chains, the price of the underlying quoted with them and the time to
        expiration (to 16:00 New York time on the expiration date) are live, while the risk
        free rate and the dividend yield come from the last historical date; a warning is
        logged when that date lies more than five days before today. The forward price is
        F = S * e^((r - q) * t) with the continuously compounded rate r and yield q.

        See: Gatheral, J. (2004), "A parsimonious arbitrage-free implied volatility
        parameterization with application to the valuation of volatility derivatives";
        Gatheral, J., & Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces",
        Quantitative Finance, 14(1), 59-71; Hagan, P. S., Kumar, D., Lesniewski, A. S., &
        Woodward, D. E. (2002), "Managing smile risk", Wilmott Magazine; West, G. (2005),
        "Calibration of the SABR model in illiquid markets", Applied Mathematical Finance.

        Also known as: SVI surface, SSVI surface, SABR smile, implied volatility surface.

        Notes:
            For SVI and SSVI a warning is logged (not raised) if the fitted surface has
            calendar-spread arbitrage, total implied variance decreasing with time to
            expiration at some log-moneyness, or butterfly arbitrage, g(k) < 0 for some
            slice (see `svi_model.check_butterfly_arbitrage`). These reflect inconsistent
            market quotes or a slice SVI cannot fit without arbitrage, and are only checked,
            not corrected; SSVI avoids the butterfly arbitrage by construction.

        Referenced against:
        - QuantLib, sabrVolatility (SABR): https://github.com/lballabio/QuantLib/blob/v1.44/ql/termstructures/volatility/sabr.cpp#L163
        - FinancePy, volatility_fns.g (butterfly arbitrage): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L247

        Shares similar logic:
        - QuantLib, SABRInterpolation (SABR): https://www.quantlib.org/reference/class_quant_lib_1_1_s_a_b_r_interpolation.html
        - FinancePy, SABR (SABR): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/sabr.py#L157
        - rateslib, IRSabrSmile (SABR): https://rateslib.com/py/en/2.7.x/api/rateslib.volatility.IRSabrSmile.html
        - FinancePy, volatility_fns.ssvi (SSVI): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L148

        Args:
            expiration_dates (list[str] | None, optional): The expiration dates to
                fit the surface over. Defaults to None, meaning the first
                `number_of_expirations` expiration dates at least one calendar day away
                (those every ticker lists, if they share any).
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
            method (str, optional): The smile model. Options are: "svi" (raw SVI per expiry), "ssvi"
                (SSVI across the expiries), "sabr" (SABR per expiry). Defaults to "svi".
            sabr_beta (float, optional): The SABR elasticity β between 0 and 1, only used
                for the SABR method: 1 is a lognormal backbone, the usual choice for equities.
                Defaults to 1.0.
            show_parameters (bool, optional): Whether to return the fitted parameters per
                ticker and expiration date instead of the surface. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The fitted implied volatility, indexed by (ticker, strike price),
            with one column per expiration date, NaN where a given strike wasn't part of
            that expiry's calibration. With show_parameters, the time to expiration (in
            years), the forward price and the fitted parameters (a, b, rho, m and sigma for
            SVI; theta, rho, eta and gamma for SSVI; alpha, beta, rho and nu for SABR),
            indexed by (ticker, expiration date).

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["JPM"], api_key="FINANCIAL_MODELING_PREP_KEY")

        volatility_surface = toolkit.options.get_volatility_surface(number_of_expirations=4)

        volatility_surface.loc["JPM"]
        ```

        Which returns:

        |   Strike Price |   2026-11-27 |   2026-12-04 |   2026-12-11 |   2026-12-18 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|
        |          160   |     nan      |     nan      |     nan      |       1.7956 |
        |          190   |     nan      |     nan      |     nan      |       1.5848 |
        |          210   |     nan      |     nan      |     nan      |       1.448  |
        |          220   |     nan      |     nan      |     nan      |       1.3798 |
        |          235   |     nan      |     nan      |     nan      |       1.2768 |
        |          240   |     nan      |     nan      |     nan      |       1.2422 |
        |          245   |     nan      |     nan      |     nan      |       1.2073 |
        |          250   |     nan      |     nan      |     nan      |       1.1721 |
        |          255   |     nan      |     nan      |     nan      |       1.1365 |
        |          260   |     nan      |     nan      |     nan      |       1.1005 |
        |          265   |       0.9504 |     nan      |     nan      |     nan      |
        |          270   |     nan      |     nan      |     nan      |       1.0269 |
        |          275   |     nan      |     nan      |     nan      |       0.9892 |
        |          280   |     nan      |     nan      |     nan      |       0.9507 |
        |          285   |       0.8015 |     nan      |     nan      |       0.9113 |
        |          287.5 |       0.7829 |     nan      |     nan      |     nan      |
        |          290   |       0.7643 |     nan      |     nan      |       0.8708 |
        |          292.5 |       0.7459 |     nan      |     nan      |     nan      |
        |          295   |       0.7274 |     nan      |     nan      |       0.8291 |
        |          297.5 |       0.7091 |     nan      |     nan      |     nan      |
        |          300   |       0.6909 |     nan      |     nan      |       0.786  |
        |          302.5 |     nan      |       0.4209 |     nan      |     nan      |
        |          305   |       0.6549 |       0.4109 |     nan      |       0.7411 |
        |          307.5 |       0.6372 |     nan      |     nan      |     nan      |
        |          310   |       0.6198 |       0.3907 |       0.3787 |       0.6941 |
        |          312.5 |       0.6027 |       0.3804 |     nan      |     nan      |
        |          315   |       0.5859 |       0.3701 |       0.3506 |       0.6445 |
        |          317.5 |       0.5697 |       0.3598 |     nan      |     nan      |
        |          320   |       0.5539 |       0.3494 |       0.3262 |       0.5917 |
        |          322.5 |       0.5388 |       0.3391 |     nan      |     nan      |
        |          325   |       0.5245 |       0.3291 |       0.3061 |       0.5346 |
        |          327.5 |       0.5111 |       0.3194 |     nan      |     nan      |
        |          330   |       0.4987 |       0.3104 |       0.2908 |       0.4717 |
        |          332.5 |       0.4876 |       0.3025 |     nan      |     nan      |
        |          335   |       0.4778 |       0.2966 |       0.2806 |       0.4002 |
        |          337.5 |       0.4697 |     nan      |     nan      |     nan      |
        |          340   |       0.4634 |       0.2938 |       0.2758 |       0.3143 |
        |          342.5 |       0.4592 |       0.2983 |     nan      |     nan      |
        |          345   |       0.4573 |       0.3062 |       0.2762 |       0.1955 |
        |          347.5 |       0.4579 |       0.3166 |     nan      |     nan      |
        |          350   |       0.4613 |       0.3285 |       0.2816 |       0.1748 |
        |          352.5 |       0.4676 |     nan      |     nan      |     nan      |
        |          355   |       0.477  |       0.3543 |       0.2913 |       0.2102 |
        |          357.5 |       0.4897 |     nan      |     nan      |     nan      |
        |          360   |       0.5056 |       0.3802 |       0.3047 |       0.2399 |
        |          362.5 |       0.5251 |     nan      |     nan      |     nan      |
        |          365   |       0.548  |       0.4052 |       0.321  |       0.2661 |
        |          367.5 |       0.5745 |       0.4173 |     nan      |     nan      |
        |          370   |       0.6047 |       0.429  |       0.3397 |       0.2895 |
        |          375   |       0.6765 |       0.4516 |       0.3601 |       0.311  |
        |          380   |       0.7646 |       0.4731 |       0.3819 |       0.3308 |
        |          385   |       0.8712 |     nan      |       0.4046 |       0.3492 |
        |          390   |       0.9999 |     nan      |     nan      |       0.3665 |
        |          395   |       1.1562 |     nan      |     nan      |       0.3828 |
        |          400   |     nan      |       0.5492 |     nan      |       0.3982 |
        |          405   |     nan      |     nan      |     nan      |       0.4129 |
        |          410   |     nan      |     nan      |     nan      |       0.4269 |
        |          415   |     nan      |     nan      |     nan      |       0.4403 |
        |          420   |     nan      |     nan      |     nan      |       0.4532 |
        |          425   |     nan      |     nan      |     nan      |       0.4655 |
        |          430   |     nan      |     nan      |     nan      |       0.4774 |
        |          435   |     nan      |     nan      |     nan      |       0.4889 |
        |          440   |     nan      |     nan      |     nan      |       0.4999 |
        |          445   |     nan      |     nan      |     nan      |       0.5107 |
        |          450   |     nan      |     nan      |     nan      |       0.521  |
        |          455   |     nan      |     nan      |     nan      |       0.5311 |
        |          460   |     nan      |     nan      |     nan      |       0.5409 |
        |          465   |     nan      |     nan      |     nan      |       0.5504 |
        |          470   |     nan      |     nan      |     nan      |       0.5596 |
        |          480   |     nan      |     nan      |     nan      |       0.5774 |
        |          490   |     nan      |     nan      |     nan      |       0.5942 |
        """
        smiles = self._calibrate_smiles(
            expiration_dates,
            number_of_expirations,
            put_option,
            risk_free_rate,
            dividend_yield,
            outlier_threshold,
            method,
            sabr_beta,
        )

        if not smiles:
            return pd.DataFrame()

        if show_parameters:
            parameters = pd.DataFrame(
                {
                    (ticker, smile["expiration_date"]): {
                        "Time to Expiration": smile["time_to_expiration"],
                        "Forward Price": smile["forward_price"],
                        **smile["parameters"],
                    }
                    for ticker, ticker_smiles in smiles.items()
                    for smile in ticker_smiles
                }
            ).T
            parameters.index.names = ["Ticker", "Expiration Date"]

            return apply_rounding(
                parameters, rounding if rounding is not None else self._rounding
            )

        volatility_surface = pd.concat(
            {
                ticker: pd.DataFrame(
                    {
                        smile["expiration_date"]: pd.Series(
                            smile["fitted_volatility"],
                            index=smile["implied_volatility"].index,
                        )
                        for smile in ticker_smiles
                    }
                ).sort_index()
                for ticker, ticker_smiles in smiles.items()
            }
        )
        volatility_surface.index.names = ["Ticker", "Strike Price"]
        # In expiration order, whatever order the tickers' expiries were collected in.
        volatility_surface = volatility_surface[sorted(volatility_surface.columns)]

        return apply_rounding(
            volatility_surface, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_local_volatility_surface(
        self,
        expiration_dates: list[str] | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        number_of_expirations: int = 6,
        outlier_threshold: float = 5.0,
        method: Literal["svi", "ssvi"] = "svi",
        strike_price_range: float = 0.25,
        number_of_strikes: int = 11,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Dupire (1994) local volatility surface: the volatility of the stock at
        each price and time that reproduces the SVI or SSVI implied volatility surface
        fitted to the option chains (see `get_volatility_surface`).

        The local volatility σ_loc(K, T) is the volatility, as a function of the price of
        the underlying and the time, of the only diffusion dS / S = (r - q) dt + σ_loc(S, t) dW
        that reprices every European option of the surface. Where the implied volatility at a
        strike is an average of the volatility along the paths to expiration, the local
        volatility is the instantaneous volatility the market prices for the moment the
        underlying stands at K at time T. It is the input of local volatility models for
        pricing path-dependent options consistently with the smile, and it shows the skew
        more steeply than the implied volatility (about twice as steep near the money).

        The formula is as follows:

        - σ_loc²(k, T) = (∂w / ∂T) / g(k)
        - g(k) = 1 - k / w * ∂w/∂k + 1/4 * (-1/4 - 1/w + k² / w²) * (∂w/∂k)² + 1/2 * ∂²w/∂k²

        Where w(k, T) is the total implied variance (implied volatility squared times the time
        to expiration T) of the fitted surface, k = ln(K / F_T) the log-moneyness with F_T the
        forward price of the expiry. The derivatives in k are analytic per slice and ∂w/∂T is
        taken at fixed k by finite differences between the expiries (three-point for unequal
        spacing at the inner expiries, one-sided at the first and the last), so the surface
        needs at least two expiries. The local volatility exists where g(k) > 0 (no butterfly
        arbitrage) and ∂w/∂T >= 0 (no calendar-spread arbitrage) and is NaN elsewhere, which
        SSVI avoids by construction where the at-the-money variance rises with
        the expiry.

        The strikes are an evenly spaced grid around the price of the underlying quoted with
        the chains, from (1 - strike_price_range) to (1 + strike_price_range) times it, the
        same for every expiry. The forward price is F_T = S * e^((r - q) * T) with the
        continuously compounded rate r and yield q.

        See: Dupire, B. (1994), "Pricing with a smile", Risk, 7(1), 18-20; Gatheral, J. (2006),
        "The Volatility Surface: A Practitioner's Guide", Wiley, eq. 1.10; and Gatheral, J., &
        Jacquier, A. (2014), "Arbitrage-free SVI volatility surfaces", Quantitative Finance,
        14(1), 59-71.

        Referenced against:
        - QuantLib, LocalVolSurface: https://www.quantlib.org/reference/class_quant_lib_1_1_local_vol_surface.html
        - FinancePy, volatility_fns.ssvi_local_varg: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/volatility_fns.py#L318

        Also known as: Dupire local volatility, local vol surface.

        Args:
            expiration_dates (list[str] | None, optional): The expiration dates to build the
                surface over, at least two. Defaults to None, meaning the first
                `number_of_expirations` expiration dates at least one calendar day away (those
                every ticker lists, if they share any).
            put_option (bool, optional): Whether to fit the implied volatility of the put
                options instead of the call options. Defaults to False.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
                uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
                risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
                quote to continuous compounding. A rate passed here is used as given, as a continuously compounded
                rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None,
                which uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative
                to its close, as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            number_of_expirations (int, optional): The number of near-term expiration dates
                when `expiration_dates` is not given. Defaults to 6.
            outlier_threshold (float, optional): The number of median absolute deviations
                from the median implied volatility beyond which a quote is treated as an
                outlier and dropped before fitting. Defaults to 5.0.
            method (str, optional): The implied volatility surface to start from. Options are: "svi"
                (raw SVI per expiry), "ssvi" (SSVI across the expiries). Defaults to "svi".
            strike_price_range (float, optional): The range of strikes, as a fraction of the
                price of the underlying in each direction. Defaults to 0.25, from 75% to 125%.
            number_of_strikes (int, optional): The number of strikes in the grid. Defaults to
                11.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set
                otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The local volatility, indexed by (ticker, strike price), with one
            column per expiration date; NaN where the fitted surface has arbitrage.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["JPM"], api_key="FINANCIAL_MODELING_PREP_KEY")

        local_volatility = toolkit.options.get_local_volatility_surface(
            method="ssvi", number_of_expirations=4
        )

        local_volatility.loc["JPM"]
        ```

        Which returns:

        |   Strike Price |   2026-11-27 |   2026-12-04 |   2026-12-11 |   2026-12-18 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|
        |         232.9  |       1.1416 |       1.182  |       0.9117 |       0.4401 |
        |         248.42 |       1.0243 |       1.0581 |       0.8129 |       0.3938 |
        |         263.95 |       0.8972 |       0.922  |       0.7033 |       0.3424 |
        |         279.48 |       0.7504 |       0.7622 |       0.5753 |       0.2822 |
        |         295    |       0.5552 |       0.5563 |       0.421  |       0.2093 |
        |         310.53 |       0.2814 |       0.3193 |       0.2673 |       0.1359 |
        |         326.06 |       0.3109 |       0.3004 |       0.2297 |       0.1165 |
        |         341.58 |       0.4285 |       0.4135 |       0.2958 |       0.1467 |
        |         357.11 |       0.5112 |       0.5059 |       0.3678 |       0.1803 |
        |         372.64 |       0.5768 |       0.5786 |       0.4277 |       0.2083 |
        |         388.16 |       0.6323 |       0.6391 |       0.4775 |       0.2316 |
        """
        check_option("method", method, ("svi", "ssvi"))

        smiles = self._calibrate_smiles(
            expiration_dates,
            number_of_expirations,
            put_option,
            risk_free_rate,
            dividend_yield,
            outlier_threshold,
            method,
        )

        surface: dict[str, pd.DataFrame] = {}

        for ticker, ticker_smiles in smiles.items():
            if (
                len(ticker_smiles)
                < local_volatility_model.MINIMUM_EXPIRATIONS_FOR_LOCAL_VOLATILITY
            ):
                logger.warning(
                    "The local volatility of %s needs at least two fitted expiries, and %d "
                    "could be fitted.",
                    ticker,
                    len(ticker_smiles),
                )
                continue

            stock_price = ticker_smiles[0]["stock_price"]
            strike_prices = np.round(
                np.linspace(
                    stock_price * (1 - strike_price_range),
                    stock_price * (1 + strike_price_range),
                    number_of_strikes,
                ),
                2,
            )
            svi_parameters = {
                smile["time_to_expiration"]: smile["svi_parameters"]
                for smile in ticker_smiles
            }

            # Every expiry has its own forward, so the same strikes are another
            # log-moneyness at each expiry.
            surface[ticker] = pd.DataFrame(
                {
                    smile[
                        "expiration_date"
                    ]: local_volatility_model.get_local_volatility(
                        np.log(strike_prices / smile["forward_price"]), svi_parameters
                    )[
                        smile["time_to_expiration"]
                    ].to_numpy()
                    for smile in ticker_smiles
                },
                index=strike_prices,
            )

        if not surface:
            return pd.DataFrame()

        local_volatility_surface = pd.concat(surface)
        local_volatility_surface.index.names = ["Ticker", "Strike Price"]

        return apply_rounding(
            local_volatility_surface,
            rounding if rounding is not None else self._rounding,
        )

    @handle_errors
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the market-implied risk-neutral probability density of the underlying's
        price at expiration: how likely the market prices each price at expiration, via the
        Breeden-Litzenberger (1978) theorem, applied to a volatility smile calibrated to
        real market option prices (see `get_implied_volatility`) rather than a single flat
        assumed volatility.

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

        The option chains, the price of the underlying quoted with them and the time to
        expiration (to 16:00 New York time on the expiration date) are live, while the
        risk free rate and the dividend yield come from the last historical date; a
        warning is logged when that date lies more than five days before today.

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
                evaluation grid, at least 3. Defaults to 200.
            outlier_threshold (float, optional): The number of median absolute
                deviations from the median implied volatility beyond which a quote
                is treated as an outlier and dropped before fitting. Defaults to
                5.0.
            rounding (int | None, optional): The number of decimals to round the
                results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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
        |        667.867 | 0      |
        |        669.717 | 0      |
        |        671.567 | 0      |
        |        673.417 | 0      |
        |        675.267 | 0      |
        |        677.117 | 0.0001 |
        |        678.968 | 0.0001 |
        |        680.818 | 0.0001 |
        |        682.668 | 0.0001 |
        |        684.518 | 0.0001 |
        |        686.368 | 0.0001 |
        |        688.218 | 0.0001 |
        |        690.068 | 0.0001 |
        |        691.918 | 0.0002 |
        |        693.768 | 0.0002 |
        |        695.618 | 0.0002 |
        |        697.468 | 0.0003 |
        |        699.318 | 0.0003 |
        |        701.168 | 0.0004 |
        |        703.018 | 0.0004 |
        |        704.868 | 0.0005 |
        |        706.718 | 0.0006 |
        |        708.568 | 0.0008 |
        |        710.418 | 0.001  |
        |        712.268 | 0.0012 |
        |        714.118 | 0.0015 |
        |        715.968 | 0.002  |
        |        717.819 | 0.0025 |
        |        719.669 | 0.0033 |
        |        721.519 | 0.0043 |
        |        723.369 | 0.0056 |
        |        725.219 | 0.0075 |
        |        727.069 | 0.01   |
        |        728.919 | 0.0133 |
        |        730.769 | 0.0177 |
        |        732.619 | 0.0233 |
        |        734.469 | 0.0302 |
        |        736.319 | 0.0378 |
        |        738.169 | 0.0455 |
        |        740.019 | 0.0518 |
        |        741.869 | 0.0549 |
        |        743.719 | 0.0537 |
        |        745.569 | 0.0479 |
        |        747.419 | 0.0389 |
        |        749.269 | 0.0289 |
        |        751.119 | 0.0198 |
        |        752.969 | 0.0127 |
        |        754.82  | 0.0078 |
        |        756.67  | 0.0046 |
        |        758.52  | 0.0028 |
        |        760.37  | 0.0016 |
        |        762.22  | 0.001  |
        |        764.07  | 0.0006 |
        |        765.92  | 0.0004 |
        |        767.77  | 0.0003 |
        |        769.62  | 0.0002 |
        |        771.47  | 0.0001 |
        |        773.32  | 0.0001 |
        |        775.17  | 0.0001 |
        |        777.02  | 0      |
        |        778.87  | 0      |
        |        780.72  | 0      |
        |        782.57  | 0      |
        |        784.42  | 0      |
        |        786.27  | 0      |
        |        788.12  | 0      |
        |        789.97  | 0      |
        |        791.82  | 0      |
        |        793.671 | 0      |
        |        795.521 | 0      |
        |        797.371 | 0      |
        |        799.221 | 0      |
        |        801.071 | 0      |
        |        802.921 | 0      |
        |        804.771 | 0      |
        |        806.621 | 0      |
        |        808.471 | 0      |
        |        810.321 | 0      |
        |        812.171 | 0      |
        """
        # The density at a strike is a second difference, which needs a strike on either side.
        validate_window("number_of_strikes", number_of_strikes, minimum=3)

        if expiration_date is not None:
            candidate_dates = [expiration_date]
        else:
            # Illiquid near-term expiries resolve nothing, so pick the first usable date,
            # one that every ticker lists if they share any.
            candidate_dates = self._all_expiration_dates(shared_first=True)

        for candidate_date in candidate_dates:
            try:
                (
                    implied_volatility,
                    stock_price,
                    risk_free_rate_value,
                    dividend_yield_value_per_ticker,
                    time_to_expiration,
                ) = self._solve_implied_volatility(
                    candidate_date, put_option, risk_free_rate, dividend_yield
                )
            except (KeyError, IndexError, ValueError):
                if expiration_date is not None:
                    raise
                continue

            if not implied_volatility.empty:
                expiration_date = candidate_date
                break
        else:
            raise ValueError(
                f"No implied volatility could be determined for expiration date(s) "
                f"{', '.join(candidate_dates)}."
            )

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

            dividend_yield_value = dividend_yield_value_per_ticker[ticker]

            forward_price = stock_price.loc[ticker] * np.exp(
                (risk_free_rate_value - dividend_yield_value) * time_to_expiration
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
                risk_free_rate=risk_free_rate_value,
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

    @handle_errors
    def get_heston_parameters(
        self,
        expiration_dates: list[str] | None = None,
        expiration_months: list[float] | None = None,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        outlier_threshold: float = 5.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calibrate the Heston (1993) stochastic volatility model to each ticker's option
        chain: the one set of parameters per ticker that reproduces the market's implied
        volatility across the term structure, from about a month to a year and a half, at
        once.

        In the Heston model the variance of the stock follows a mean-reverting square-root
        process correlated with the stock, dS = (r - q)S dt + √v S dW_1 and
        dv = κ(θ - v) dt + σ√v dW_2 with dW_1 dW_2 = ρ dt. Its five parameters read directly:
        √v0 is today's volatility, √θ the volatility the market expects in the long run, κ how
        fast it gets there (a half-life of ln(2) / κ years), σ the volatility of the variance
        (the curvature of the smile) and ρ the correlation of the stock with its variance (the
        skew: negative for equities, whose volatility rises as they fall).

        By default the fit uses the listed expiration nearest to each of 1, 2, 3, 6, 9, 12 and
        18 months (expiration_months), leaving out those less than 14 calendar days away, so
        that κ and θ, which describe how the variance moves across the term structure, are
        pinned down by expirations far apart. Each expiration contributes its out-of-the-money
        quotes: the puts below the forward price and the calls above it. Those are the liquid
        side of the chain; an in-the-money option is mostly its exercise value, so its implied
        volatility is dominated by the bid-ask spread. The implied volatility of each side (see
        `get_implied_volatility`) is cleaned of outliers with the same median-absolute-deviation
        filter as `get_volatility_surface`.

        The parameters minimize the vega-weighted price errors, to first order the errors in
        implied volatility, with a soft-l1 loss so that a few stale quotes cannot dominate.
        Quotes whose vega is below 5% of the at-the-money vega of their expiration are left
        out. The fit is a bounded least squares from two fixed starting points (see
        `heston_model.get_heston_parameters`), with every expiration priced by the COS method of
        Fang and Oosterlee (2008) with as many terms as the fitted parameters need, so the same
        quotes always give the same parameters.

        The formulas are as follows:

        - φ(u) = exp(C + D * v0), the characteristic function of ln(S_T / F) in the form of Albrecher et al. (2007)
        - Call = S * e^(-qT) - √(F * K) * e^(-rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ(u - i/2)] / (u² + 1/4) du
        - minimize Σ_i soft-l1([Call_Heston(K_i, T_i) - Call_BS(K_i, T_i, σ_i)] / Vega_BS(K_i, T_i, σ_i))
        - Feller Ratio = 2κθ / σ²

        Where S is the price of the underlying quoted with the nearest chain, K the strike, F the
        forward price, r the continuously compounded risk-free rate, q the continuous dividend
        yield, T the time to expiration in years (to 16:00 New York time on the expiration
        date) and σ_i the market implied volatility of quote i (a put and a call of the same
        strike have the same implied volatility error, as both models satisfy put-call parity).
        A Feller ratio of 1 or more keeps the variance above zero. It is reported rather than
        imposed: fits to equity smiles commonly break it, because the steep skew asks for a high
        σ.

        The RMSE is the root mean squared difference between the implied volatility of the
        fitted Heston prices, from a COS series converged to 1e-10 per unit of the strike, and
        the market's, over the quotes fitted.

        See: Heston, S. L. (1993), "A Closed-Form Solution for Options with Stochastic Volatility
        with Applications to Bond and Currency Options", The Review of Financial Studies, 6(2),
        327-343; Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007), "The Little
        Heston Trap", Wilmott Magazine; Fang, F. and Oosterlee, C. W. (2008), "A Novel Pricing
        Method for European Options Based on Fourier-Cosine Series Expansions", SIAM Journal on
        Scientific Computing, 31(2), 826-848.

        Also known as: Heston calibration, stochastic volatility calibration, Heston model fit.

        Notes:
            Heston is calibrated across the term structure because its parameters describe it:
            with one expiration, or several within a few weeks, κ and θ are hardly identified
            separately and κ and σ run to their bounds. The expirations under 14 days are left
            out by default: their smiles are dominated by jump risk (earnings and other events)
            and by the bid-ask spread, which a diffusion like Heston cannot produce without an
            implausibly fast mean reversion and volatile variance. The At Bound column names
            the parameters that end at a bound of the search (v0 and θ from 0.0001 to 4, κ from
            0.001 to 20, a half-life of under two weeks at 20, σ from 0.01 to 5 and ρ from
            -0.999 to 0.999), and a warning is logged with it. A parameter at its bound is a
            sign that the quotes ask for more than the model has, such as jumps or a humped
            term structure, and the other parameters then compensate for it, so they are less
            meaningful. Even within the bounds κ and σ trade off: a faster mean reversion with a
            more volatile variance gives much the same smile, so on a real surface the fit is
            flat along that direction and two similar smiles can give a quite different κ. The
            long-run level θ, the Feller ratio and the at-the-money term structure are far
            better determined than κ and σ separately. Listed options on single stocks are American; the fit treats
            their implied volatility as European, as `get_volatility_surface` does. The Heston
            prices themselves are available as `heston_model.get_heston_price`.

        Referenced against:
        - QuantLib, HestonModelHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_heston_model_helper.html
        - QuantLib, AnalyticHestonEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_heston_engine.html

        Shares similar logic:
        - QuantLib, COSHestonEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_c_o_s_heston_engine.html

        Args:
            expiration_dates (list[str] | None, optional): The expiration dates to fit over, used
                as given (also those under 14 days). Defaults to None, which picks them with
                expiration_months.
            expiration_months (list[float] | None, optional): The times to expiration, in months,
                whose nearest listed expiration (at least 14 days away, among each ticker's own
                dates) is fitted; an expiration nearest to several of them is fitted once. Defaults to None, which uses 1, 2, 3, 6, 9, 12 and 18 months.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
                uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
                risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
                quote to continuous compounding. A rate passed here is used as given, as a continuously compounded
                rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
                uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its
                close, as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            outlier_threshold (float, optional): The number of median absolute deviations from the
                median implied volatility of an expiration beyond which a quote is dropped before
                fitting. Defaults to 5.0.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per ticker, v0, kappa, theta, sigma and rho, the Feller Ratio, the RMSE
            in implied volatility, the number of quotes and the number of expirations fitted, and
            At Bound, the parameters that end at a bound of the search (empty when none does).

        Raises:
            ValueError: If no ticker lists an expiration 14 days or more away.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "MSFT"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.options.get_heston_parameters()
        ```

        Which returns:

        | Ticker   |     v0 |   kappa |   theta |   sigma |     rho |   Feller Ratio |   RMSE |   Quotes |   Expirations | At Bound   |
        |:---------|-------:|--------:|--------:|--------:|--------:|---------------:|-------:|---------:|--------------:|:-----------|
        | AAPL     | 0.0611 |  6.2395 |  0.0974 |  1.5007 | -0.5683 |         0.5398 | 0.0196 |      364 |             7 |            |
        | MSFT     | 0.1404 | 11.815  |  0.1373 |  2.5277 | -0.5619 |         0.5077 | 0.0277 |      375 |             7 |            |
        """
        # The expirations of each ticker: those given, or its own listed expiration nearest to
        # each of the expiration_months, as tickers list different dates.
        if expiration_dates is not None:
            dates_per_ticker = {
                ticker: set(expiration_dates) for ticker in self._tickers
            }
        else:
            now = self._current_time()
            dates_per_ticker = {}
            for ticker, listed_dates in self._expiration_dates().items():
                candidates = [
                    date
                    for date in listed_dates
                    if get_days_to_expiration(date, now) >= MINIMUM_DAYS_FOR_HESTON_FIT
                ]
                if not candidates:
                    logger.warning(
                        "%s lists no expiration date %d days or more away.",
                        ticker,
                        MINIMUM_DAYS_FOR_HESTON_FIT,
                    )
                    continue
                dates_per_ticker[ticker] = {
                    min(
                        candidates,
                        key=lambda date, months=months: abs(
                            get_time_to_expiration(date, now) - months / 12
                        ),
                    )
                    for months in (
                        expiration_months
                        if expiration_months is not None
                        else HESTON_EXPIRATION_MONTHS
                    )
                }
            if not dates_per_ticker:
                raise ValueError(
                    f"No expiration date lies {MINIMUM_DAYS_FOR_HESTON_FIT} days or more away."
                )
            expiration_dates = sorted(set().union(*dates_per_ticker.values()))

        # The out-of-the-money side of every expiration: the puts below the forward and the
        # calls above it.
        quotes_per_side = {
            put_option: self._collect_smile_quotes(
                expiration_dates,
                len(expiration_dates),
                put_option,
                risk_free_rate,
                dividend_yield,
                outlier_threshold,
            )
            for put_option in (False, True)
        }

        parameters: dict[str, dict[str, float]] = {}
        lower_bounds, upper_bounds = heston_model.HESTON_BOUNDS

        for ticker in sorted(set(quotes_per_side[False]) | set(quotes_per_side[True])):
            puts = {
                quote["expiration_date"]: quote
                for quote in quotes_per_side[True].get(ticker, [])
            }
            calls = {
                quote["expiration_date"]: quote
                for quote in quotes_per_side[False].get(ticker, [])
            }
            ticker_quotes = []
            for expiration_date in sorted(
                (set(puts) | set(calls)) & dates_per_ticker.get(ticker, set())
            ):
                quote = dict(calls.get(expiration_date) or puts[expiration_date])
                forward_price = quote["forward_price"]
                sides = []
                if expiration_date in puts:
                    put_volatility = puts[expiration_date]["implied_volatility"]
                    sides.append(put_volatility[put_volatility.index < forward_price])
                if expiration_date in calls:
                    call_volatility = calls[expiration_date]["implied_volatility"]
                    sides.append(
                        call_volatility[call_volatility.index >= forward_price]
                    )
                quote["implied_volatility"] = pd.concat(sides).sort_index()
                ticker_quotes.append(quote)

            # The quotes are collected for every ticker's dates together, so a ticker can have
            # chains only on dates it did not select.
            if not ticker_quotes:
                logger.warning(
                    "No expiration of %s selected for the Heston fit has enough quotes.",
                    ticker,
                )
                continue

            # One spot price, rate and yield for the whole fit: those of the nearest chain, as
            # the chains of a ticker are fetched within moments of each other.
            nearest = ticker_quotes[0]
            try:
                fit = heston_model.get_heston_parameters(
                    stock_price=nearest["stock_price"],
                    strike_price=np.concatenate(
                        [
                            quote["implied_volatility"].index.to_numpy(dtype=float)
                            for quote in ticker_quotes
                        ]
                    ),
                    time_to_expiration=np.concatenate(
                        [
                            np.full(
                                len(quote["implied_volatility"]),
                                quote["time_to_expiration"],
                            )
                            for quote in ticker_quotes
                        ]
                    ),
                    implied_volatility=np.concatenate(
                        [
                            quote["implied_volatility"].to_numpy()
                            for quote in ticker_quotes
                        ]
                    ),
                    risk_free_rate=nearest["risk_free_rate"],
                    dividend_yield=nearest["dividend_yield"],
                )
            except ValueError as error:
                logger.warning("The Heston fit of %s failed: %s", ticker, error)
                continue

            fitted = np.array([fit[name] for name in HESTON_PARAMETERS])
            at_bound = [
                name
                for name, value, lower, upper in zip(
                    HESTON_PARAMETERS, fitted, lower_bounds, upper_bounds, strict=True
                )
                if np.isclose(value, lower, rtol=1e-4, atol=0)
                or np.isclose(value, upper, rtol=1e-4, atol=0)
            ]
            if at_bound:
                logger.warning(
                    "The Heston fit of %s ends at the bound of %s: the quotes ask for more "
                    "than the model has (such as jumps or a humped term structure), so "
                    "these parameters are less meaningful.",
                    ticker,
                    ", ".join(at_bound),
                )

            parameters[ticker] = {
                **dict(zip(HESTON_PARAMETERS, fitted, strict=True)),
                "Feller Ratio": fit["feller_ratio"],
                "RMSE": fit["rmse"],
                "Quotes": fit["quotes"],
                "Expirations": len(ticker_quotes),
                "At Bound": ", ".join(at_bound),
            }

        if not parameters:
            return pd.DataFrame()

        # From the records per ticker, so the numbers keep a numeric type next to the text of
        # the At Bound column.
        parameters_df = pd.DataFrame.from_dict(parameters, orient="index")
        parameters_df.index.name = "Ticker"

        return apply_rounding(
            parameters_df, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_heston_nandi_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_parameters: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate European option prices in the GARCH model of Heston and Nandi (2000): what
        calls and puts on each ticker are worth when its variance follows a GARCH process
        with a leverage effect, with the GARCH parameters fitted on each ticker's own daily
        returns.

        Where the Black-Scholes model takes one volatility for the whole life of the option, the
        Heston-Nandi model lets the daily variance follow a GARCH(1, 1) process with a leverage
        effect: it starts from the variance the GARCH filter forecasts for tomorrow, reverts to
        its long-run level at the speed the returns show, and rises more after a fall than after
        a rise. The prices therefore follow the current volatility regime of the stock and carry
        a skew, without any option prices as input. It is the GARCH option pricing model with a
        closed form.

        The parameters are fitted by maximum likelihood on the daily log returns (from the
        Adj Close) over the risk-free rate of the historical data up to the pricing date (see
        `garch_option_model.get_heston_nandi_parameters`), so a past pricing date only uses what
        was known then. Under the risk-neutral measure the price of risk λ becomes -½ and the
        asymmetry γ becomes γ + λ + ½, and the price follows from the moment generating
        function of the log price by Fourier inversion.

        The formulas are as follows:

        - R_t - r_t / 252 = λ * h_t + √h_t * z_t
        - h_(t+1) = ω + β * h_t + α * (z_t - γ * √h_t)²
        - E[S_T^φ] = S^φ * exp(A + B * h), with A and B from a backward recursion over the trading days
        - Call = S * e^(-qT) - √(F * K) * e^(-rT) / π * ∫_0^∞ Re[e^(iu * ln(F / K)) * φ_X(u - i/2)] / (u² + 1/4) du

        Where R_t is the daily log return, r_t the risk-free rate of that day, h_t the
        conditional variance, z_t the standard normal shock, ω, α, β, γ and λ the fitted
        parameters, S the traded close on the pricing date, K the strike, h the variance
        forecast for the next trading day, r the continuously compounded risk-free rate, q the
        continuous dividend yield (ln(1 + dividend yield)), T the number of trading days to
        expiration over 252 and φ_X the characteristic function of ln(S_T / F) with F the
        forward price. The trading days to an expiration are the weekdays after the pricing date
        up to and including it, so an expiration on a weekend has the value of the Friday before
        it and one on the next weekday is one GARCH step away. The prices are computed with the
        COS method of Fang and Oosterlee (2008), which agrees with the Fourier integral within
        1e-10 here and prices all strikes of an expiration at once.

        See: Heston, S. L. and Nandi, S. (2000), "A Closed-Form GARCH Option Valuation Model",
        The Review of Financial Studies, 13(3), 585-625. Checked against a Monte Carlo
        simulation of the risk-neutral process and the Black-Scholes price at α = β = 0.

        Also known as: Heston-Nandi model, GARCH option pricing, HN-GARCH.

        Args:
            start_date (str | None, optional): The date whose price, rates and fitted variance the
            values start from; the GARCH parameters are fitted on the returns up to it. A date
            without prices (a weekend or a holiday) uses the last trading day before it. Defaults
            to None, which uses the last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which
            equals 25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that
            the strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 calendar days after the start date, so at least 2. Defaults to 30,
            which gives expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the pricing. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded
            rate. The excess returns of the fit always use the Toolkit's daily rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None,
            which uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to
            its close, as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_parameters (bool, optional): Whether to return the fitted GARCH parameters per ticker instead of
            the prices: omega, alpha, beta, gamma and lambda (daily), the persistence β + αγ², the annualized
            long-run and next-day volatility (√(252 * variance)) and the log-likelihood. Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None,
                which uses the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Heston-Nandi option values with the tickers and strike prices as the index and the
            expiration dates as the columns, or with show_parameters the fitted parameters per ticker.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2016-01-01",
            end_date="2025-12-31",
        )

        heston_nandi_put = toolkit.options.get_heston_nandi_model(
            put_option=True, strike_price_range=0.15, strike_step_size=10, expiration_time_range=91
        )

        heston_nandi_put.loc["AAPL", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            230 |       0.3272 |       1.3428 |       2.6223 |
        |            240 |       0.7636 |       2.367  |       4.1069 |
        |            250 |       1.6753 |       3.9993 |       6.216  |
        |            260 |       3.4452 |       6.477  |       9.1009 |
        |            270 |       6.6014 |      10.0485 |      12.8997 |
        |            280 |      11.6765 |      14.9192 |      17.7135 |
        |            290 |      18.8495 |      21.1766 |      23.5828 |
        |            300 |      27.6358 |      28.7298 |      30.4716 |
        |            310 |      37.2321 |      37.313  |      38.2669 |
        |            320 |      47.1053 |      46.5792 |      46.7986 |
        """
        validate_window("expiration_time_range", expiration_time_range, minimum=2)

        start_date = self._pricing_date(start_date)
        stock_price = self._prices.loc[start_date]

        # The excess returns of the fit take the risk-free rate of each day, known on that day.
        daily_rate = self._risk_free_rate.reindex(self._daily_historical.index).ffill()
        returns = np.log1p(self._daily_historical["Return"].loc[:start_date])

        pricing_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        fits: dict[str, dict[str, float]] = {}
        for ticker in self._tickers:
            ticker_returns = returns[ticker].dropna()
            try:
                fits[ticker] = garch_option_model.get_heston_nandi_parameters(
                    ticker_returns.to_numpy(),
                    risk_free_rate=daily_rate.loc[ticker_returns.index].to_numpy(),
                )
            except ValueError as error:
                logger.warning("The Heston-Nandi fit of %s failed: %s", ticker, error)

        if not fits:
            return pd.DataFrame()

        if show_parameters:
            parameters_df = pd.DataFrame(
                {
                    ticker: {
                        "omega": fit["omega"],
                        "alpha": fit["alpha"],
                        "beta": fit["beta"],
                        "gamma": fit["gamma"],
                        "lambda": fit["lambda"],
                        "Persistence": fit["persistence"],
                        "Long-Run Volatility": np.sqrt(
                            garch_option_model.TRADING_DAYS_PER_YEAR
                            * fit["unconditional_variance"]
                        ),
                        "Next-Day Volatility": np.sqrt(
                            garch_option_model.TRADING_DAYS_PER_YEAR
                            * fit["next_variance"]
                        ),
                        "Log-Likelihood": fit["log_likelihood"],
                    }
                    for ticker, fit in fits.items()
                }
            ).T
            parameters_df.index.name = "Ticker"

            # The parameters are of very different sizes (ω about 1e-7, γ about 100), so they
            # are rounded to significant digits rather than decimals. The log-likelihood keeps
            # its decimals, as fits are compared by its differences, which significant digits
            # of a number in the thousands would round away.
            digits = rounding if rounding is not None else self._rounding
            return parameters_df.apply(
                lambda column: (
                    column.round(digits)
                    if column.name == "Log-Likelihood"
                    else column.map(lambda value: float(f"{value:.{digits}g}"))
                )
            )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=list(fits),
            stock_price=stock_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
            values_per_strike=expiration_time_range,
        )

        # The weekdays after the pricing date up to and including each expiration.
        first_day = start_date.to_timestamp()
        trading_days = [
            int(
                np.busday_count(
                    (first_day + pd.Timedelta(days=1)).date(),
                    (first_day + pd.Timedelta(days=days + 1)).date(),
                )
            )
            for days in range(expiration_time_range)
        ]

        heston_nandi_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )
            fit = fits[ticker]
            strikes = np.asarray(strike_prices, dtype=float)

            values = {
                days: np.atleast_1d(
                    garch_option_model.get_heston_nandi_price(
                        stock_price=float(stock_price.loc[ticker]),
                        strike_price=strikes,
                        risk_free_rate=float(pricing_rate),
                        periods_to_expiration=periods,
                        next_variance=fit["next_variance"],
                        omega=fit["omega"],
                        alpha=fit["alpha"],
                        beta=fit["beta"],
                        gamma=fit["gamma"],
                        risk_premium=fit["lambda"],
                        dividend_yield=float(dividend_yield_value[ticker]),
                        put_option=put_option,
                        method="cos",
                    )
                )
                for days, periods in enumerate(trading_days)
            }

            heston_nandi_option[ticker] = {
                strike_price: {
                    days / 365: float(values[days][position]) for days in values
                }
                for position, strike_price in enumerate(strike_prices)
            }

        heston_nandi_df = helpers.create_greek_dataframe(
            greek_dictionary=heston_nandi_option,
            start_date=start_date,
        )

        heston_nandi_df = heston_nandi_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            heston_nandi_df = calculate_standardization(
                dataset=heston_nandi_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=stock_price,
                volatility=pd.Series(
                    {
                        ticker: np.sqrt(
                            garch_option_model.TRADING_DAYS_PER_YEAR
                            * fit["next_variance"]
                        )
                        for ticker, fit in fits.items()
                    }
                ),
                risk_free_rate=pricing_rate,
                dividend_yield=dividend_yield_value,
            )

        return heston_nandi_df
