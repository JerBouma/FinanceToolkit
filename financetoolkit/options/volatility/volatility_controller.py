"""Option Volatility Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import helpers
from financetoolkit.options.base.base_controller import OptionsBase, _days_to_expiration
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
class VolatilityOptions(OptionsBase):
    """
    Implied volatility, the SVI volatility surface and the risk-neutral density.
    """

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
