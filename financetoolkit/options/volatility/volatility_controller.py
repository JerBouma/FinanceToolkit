"""Option Volatility Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import helpers, options_model
from financetoolkit.options.helpers import (
    get_days_to_expiration,
    get_time_to_expiration,
)
from financetoolkit.options.market import market_metrics_model
from financetoolkit.options.pricing import black_scholes_model
from financetoolkit.options.volatility import risk_neutral_density_model, svi_model
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

MINIMUM_OBSERVATIONS_FOR_SVI_FIT = 5

# The number of calendar days the last historical price may lag today before the spot price
# is treated as stale next to the live option chains (a weekend plus a holiday is four).
MAXIMUM_SPOT_AGE_DAYS = 5

MINIMUM_EXPIRATIONS_FOR_CALENDAR_CHECK = 2


@filter_shown_columns
@track_sources
class VolatilityOptions:
    """
    Implied volatility, the SVI volatility surface and the risk-neutral density.
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
        show_expiration_dates: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Implied Volatility (IV) based on the Black Scholes Model and the actual option prices for
        any of the available expiration dates.

        Implied Volatility (IV) is a measure of how much the market expects the price of the underlying asset to
        fluctuate in the future. It is a key component of options pricing and can also be used to calculate the
        theoretical value of an option.

        The price of the underlying quoted with the option chain and the most recent risk free rate and dividend
        yield are used, and the implied volatility is the volatility at which the Black Scholes price equals the
        option's mid price, the middle of its bid and ask (the last traded price when the option has no two-sided
        quote, for example outside market hours).

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

        Args:
            expiration_date (str | None, optional): The expiration date to use for the calculation. Defaults to None,
            which uses the first expiration date at least one calendar day away (an expiry of today has hardly any
            time left to price).
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
            date as the Series name. If show_expiration_dates is True, it returns the list of expiration dates.

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
        expiration_dates = self.get_option_chains(show_expiration_dates=True)

        if show_expiration_dates:
            return expiration_dates

        if expiration_date is None:
            # An expiry of today has hardly any time left to price, so the first expiry at
            # least a calendar day away is used.
            now = self._current_time()
            expiration_date = next(
                (
                    date
                    for date in expiration_dates
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
                expiration_date, put_option, risk_free_rate, dividend_yield
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
            # while a last traded price can be hours or days older than either.
            option_prices = market_metrics_model.get_mid_prices(option_chain)

            for strike_price, option_price in option_prices.items():
                # Numerically finds the volatility matching the market option price.
                implied_volatility_value = black_scholes_model.get_implied_volatility(
                    market_price=option_price,
                    stock_price=stock_price[ticker],
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    time_to_expiration=time_to_expiration,
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
        implied_volatility_df.name = expiration_date

        return (
            implied_volatility_df,
            pd.Series(stock_price),
            risk_free_rate,
            dividend_yield_value,
            time_to_expiration,
        )

    @handle_errors
    def get_volatility_surface(
        self,
        expiration_dates: list[str] | None = None,
        put_option: bool = False,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        number_of_expirations: int = 6,
        outlier_threshold: float = 5.0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
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

        The option chains, the price of the underlying quoted with them and the time to
        expiration (to 16:00 New York time on the expiration date) are live, while the
        risk free rate and the dividend yield come from the last historical date; a
        warning is logged when that date lies more than five days before today.

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
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        |   Strike Price |   2026-11-20 |   2026-11-27 |   2026-12-04 |   2026-12-11 |
        |---------------:|-------------:|-------------:|-------------:|-------------:|
        |          322.5 |       0.377  |     nan      |          nan |          nan |
        |          327.5 |       0.3611 |     nan      |          nan |          nan |
        |          332.5 |       0.3459 |       0.2733 |          nan |          nan |
        |          337.5 |       0.3317 |       0.263  |          nan |          nan |
        |          342.5 |       0.3192 |       0.2532 |          nan |          nan |
        |          347.5 |       0.3093 |       0.2444 |          nan |          nan |
        |          352.5 |       0.3033 |       0.2374 |          nan |          nan |
        |          357.5 |       0.303  |       0.2331 |          nan |          nan |
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

        surface: dict[str, dict[float, dict[str, float]]] = {}
        svi_parameters_per_ticker: dict[str, dict[float, dict[str, float]]] = {
            ticker: {} for ticker in self._tickers
        }

        for expiration_date in expiration_dates:
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
            except (KeyError, IndexError, ValueError):
                continue

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

                forward_price = stock_price.loc[ticker] * np.exp(
                    (risk_free_rate_value - dividend_yield_value_per_ticker[ticker])
                    * time_to_expiration
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
                evaluation grid. Defaults to 200.
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
        if expiration_date is not None:
            candidate_dates = [expiration_date]
        else:
            # Illiquid near-term expiries resolve nothing, so pick the first usable date.
            candidate_dates = self.get_option_chains(show_expiration_dates=True)

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
