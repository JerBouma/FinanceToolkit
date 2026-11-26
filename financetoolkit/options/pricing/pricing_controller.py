"""Option Pricing Module"""

__docformat__ = "google"

import functools
import numbers
from typing import Literal

import numpy as np
import pandas as pd

from financetoolkit.options import helpers, options_model
from financetoolkit.options.numerical import finite_difference_model, monte_carlo_model
from financetoolkit.options.pricing import (
    american_model,
    binomial_trees_model,
    black_scholes_model,
    cev_model,
    compound_model,
    exotics_model,
    path_dependent_model,
)
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
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


@filter_shown_columns
@track_sources
class PricingOptions:
    """
    Option pricing: Black-Scholes (also with cash dividends), binomial trees (Cox-Ross-Rubinstein, Jarrow-Rudd, Tian, Trigeorgis, Leisen-Reimer, exact inversion), Monte Carlo, Garman-Kohlhagen, Black-76 on futures, the CEV model, American approximations (Bjerksund-Stensland, Barone-Adesi-Whaley, Ju-Zhong, QD+) and the QD fixed point, binary, barrier (single, double, partial-time, soft and discretely watched), Asian (geometric and arithmetic), lookback and touch options, chooser, forward-start and cliquet options, put-call parity and strategy payoffs.
    """

    @handle_errors
    def get_black_scholes_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        dividends: Literal["yield", "escrowed", "haug_haug_lewis"] = "yield",
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Black Scholes Model: what European calls and puts on each ticker are worth today, for a range
        of strike prices and times to expiration, when the stock follows a lognormal process with a constant
        volatility.

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

        Dividends are paid in cash on their ex-dividend dates, which the dividend yield q spreads evenly over the
        year. For short options that difference matters: an option that expires just after an ex-date loses the
        whole dividend, one that expires just before it none of it. With the escrowed or the Haug-Haug-Lewis
        dividend model the dividend yield is replaced by the projected cash dividends: the latest
        dividend per share before the pricing date, repeated at the median interval between its last ex-dates
        (see options_model.get_projected_dividends), so a past pricing date only uses what was known on that date.

        - Escrowed (Hull, 2022, chapter 17): the Black-Scholes formula with q = 0 on S* = S — Σ D_i * e^(—r * t_i),
          the stock price net of the present value of the dividends up to expiration.
        - Haug-Haug-Lewis (Haug, Haug and Lewis, 2003): the stock price drops by D_i at t_i and the option value
          is integrated over the lognormal stock price at each ex-date, from the Black-Scholes value after the
          last dividend backwards: V_i(x) = e^(—r * (t_(i+1) — t_i)) * E[V_(i+1)(S_(t_(i+1)) — D_(i+1))].

        Where D_i is the cash dividend with ex-date t_i years after the pricing date. The escrowed model is the
        market convention; it applies the volatility to S* instead of S, which the Haug-Haug-Lewis method does not,
        so the latter gives slightly higher prices when the dividends are large.

        Also known as: BSM, Black-Scholes-Merton, option pricing model, escrowed dividend model, discrete dividends.

        Referenced against:
        - QuantLib, AnalyticDividendEuropeanEngine (escrowed): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_dividend_european_engine.html
        - QuantLib, CashDividendEuropeanEngine (Haug-Haug-Lewis): https://www.quantlib.org/reference/class_quant_lib_1_1_cash_dividend_european_engine.html

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
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
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield. Only used with
            dividends="yield".
            dividends (str, optional): How the dividends enter the price. Options are: "yield" (a continuous
            dividend yield), "escrowed" (the projected cash dividends, escrowed model), "haug_haug_lewis" (the
            projected cash dividends, stock price dropping on each ex-date). Defaults to "yield".
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
        |            160 |      28.7163 |      31.7618 |      34.5225 |
        |            165 |      24.6254 |      28.1297 |      31.1413 |
        |            170 |      20.8379 |      24.7547 |      27.9827 |
        |            175 |      17.3918 |      21.6473 |      25.0494 |
        |            180 |      14.313  |      18.8122 |      22.3412 |
        |            185 |      11.6133 |      16.2484 |      19.855  |
        |            190 |       9.29   |      13.9503 |      17.5848 |
        |            195 |       7.3277 |      11.9077 |      15.5227 |
        |            200 |       5.7004 |      10.1071 |      13.6589 |
        |            205 |       4.3748 |       8.5321 |      11.9824 |
        |            210 |       3.3136 |       7.1649 |      10.4811 |
        """
        check_option("dividends", dividends, ("yield", "escrowed", "haug_haug_lewis"))
        if dividends != "yield" and dividend_yield is not None:
            raise InvalidArgumentError(
                'A dividend_yield is only used with dividends="yield"; the other methods use the cash dividends.'
            )

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

        black_scholes: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        cash_dividends: dict[str, pd.Series] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            black_scholes[ticker] = {}

            if dividends != "yield":
                cash_dividends[ticker] = self._projected_dividends(
                    ticker, start_date, expiration_time_range
                )
                dividend_yield_value[ticker] = 0.0
                black_scholes[ticker] = helpers.evaluate_on_grid(
                    lambda strike_price, time_to_expiration: black_scholes_model.get_black_scholes_cash_dividends(
                        stock_price=stock_price.loc[ticker],
                        strike_price=strike_price,
                        risk_free_rate=risk_free_rate,
                        volatility=volatility.loc[ticker],
                        time_to_expiration=time_to_expiration,
                        dividend_times=cash_dividends[ticker].index.to_numpy() / 365,
                        dividend_amounts=cash_dividends[ticker].to_numpy(),
                        put_option=put_option,
                        method=dividends,
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
                continue

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
            for ticker, projected in cash_dividends.items():
                logger.info(
                    "Cash dividends of %s (days after %s: dividend): %s",
                    ticker,
                    start_date,
                    ", ".join(f"{days}: {amount}" for days, amount in projected.items())
                    or "none",
                )

        return black_scholes_df

    @handle_errors
    def get_binomial_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        american_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        time_to_expiration: float = 1,
        time_steps: int = 10,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
        method: Literal[
            "cox_ross_rubinstein",
            "jarrow_rudd",
            "tian",
            "trigeorgis",
            "leisen_reimer",
            "exact_inversion",
        ] = "cox_ross_rubinstein",
    ):
        """
        Calculate the Binomial Option Pricing Model: what European and American calls and puts on each ticker are
        worth at every node of a binomial tree of stock prices, from the payoffs at expiration back to today.

        It creates a binomial tree of price paths for the underlying asset and works backwards through the tree to
        determine the price of the option at each node.

        The tree of Cox, Ross and Rubinstein (1979) is the default. The variants choose the up and down movements and
        the probability differently, which changes how fast and how smoothly the price converges as the number of
        time steps grows:

        - Jarrow-Rudd: equal probabilities of 1/2, with the moves centred on the drift.
        - Tian: matches the first three moments of the lognormal distribution in each step.
        - Trigeorgis: equal steps in the logarithm of the price with the drift in the
          probability, a tree that keeps its probability between 0 and 1 longer than Cox-Ross-Rubinstein.
        - Leisen-Reimer: centred on the strike so the price converges smoothly at a rate of
          1 / n² instead of oscillating at 1 / n, with the probabilities from the Peizer-Pratt inversion of the
          normal distribution.
        - Exact inversion: the Leisen-Reimer tree with the exact inverse of the binomial distribution, of which Joshi
          (2010) takes a fourth order expansion. It prices a European option at its Black-Scholes value at
          every number of steps and converges fastest for American options.

        The Leisen-Reimer and exact inversion trees need an odd number of time steps; an even number is raised by one (the
        default of 10 becomes 11).

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
        - Jarrow-Rudd: u = e^(νΔt + σ√Δt), d = e^(νΔt — σ√Δt), p = 1/2
        - Tian: v = e^(σ²Δt), u = e^((r — q)Δt) * v / 2 * (v + 1 + √(v² + 2v — 3)), d = e^((r — q)Δt) * v / 2 *
          (v + 1 — √(v² + 2v — 3)), p = (e^((r — q)Δt) — d) / (u — d)
        - Trigeorgis: Δx = √(σ²Δt + ν²Δt²), u = e^(Δx), d = 1 / u, p = 1/2 + νΔt / (2Δx)
        - Leisen-Reimer and exact inversion: p = h(d2), u = e^((r — q)Δt) * h(d1) / h(d2), d = (e^((r — q)Δt) — p * u) / (1 — p)
        - Peizer-Pratt (Leisen-Reimer): h(z) = 1/2 + sign(z) / 2 * √(1 — e^(—(z / (n + 1/3 + 0.1 / (n + 1)))² *
          (n + 1/6)))
        - exact inversion: h(z) is the p at which I_p((n + 1) / 2, (n + 1) / 2) = N(z)
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
        ν = r — q — σ² / 2, d1 = (ln(S / K) + (r — q + σ² / 2) * T) / (σ√T), d2 = d1 — σ√T, I_p is the regularized
        incomplete beta function, n is the number of time steps, C_u is the call option price at the up
        movement, C_d is the call option price at the down movement, P_u is the put option price at the up movement and
        P_d is the put option price at the down movement.

        The resulting output is a DataFrame containing the tickers, strike prices and movements as the index and the
        time to expiration as the columns. The movements index contains the number of up movements and the number of
        down movements. The output is the binomial tree displayed in a table. E.g. when using 10 time steps, the
        table for each strike price from each company will contain the actual binomial tree as also depicted
        in the image found here: https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method

        See: Cox, J. C., Ross, S. A., & Rubinstein, M. (1979), "Option Pricing: A Simplified Approach", Journal of
        Financial Economics, 7(3), 229-263; Jarrow, R., & Rudd, A. (1983), "Option Pricing", Irwin; Trigeorgis, L.
        (1991), "A Log-Transformed Binomial Numerical Analysis Method for Valuing Complex Multi-Option Investments",
        Journal of Financial and Quantitative Analysis, 26(3), 309-326; Tian, Y. (1993), "A Modified Lattice Approach
        to Option Pricing", Journal of Futures Markets, 13(5), 563-577; Leisen, D. P. J., & Reimer, M. (1996),
        "Binomial Models for Option Valuation - Examining and Improving Convergence", Applied Mathematical Finance,
        3(4), 319-346; Joshi, M. S. (2010), "Achieving Higher Order Convergence for the Prices of European Options in
        Binomial Trees", Mathematical Finance, 20(1), 89-103. The Jarrow-Rudd, Tian, Trigeorgis and
        Leisen-Reimer trees agree with QuantLib's within 1e-9 and its J4 tree, Joshi's expansion, lies within 1e-6 of
        the exact inverse at 101 steps. An equal probabilities tree that matches the
        mean and variance of the log price is the Jarrow-Rudd tree.

        Also known as: binomial tree, lattice model, option pricing, CRR, Leisen-Reimer, Jarrow-Rudd, Tian tree.

        Referenced against:
        - QuantLib, BinomialJRVanillaEngine (Jarrow-Rudd): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
        - QuantLib, BinomialTianVanillaEngine (Tian): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
        - QuantLib, BinomialTrigeorgisVanillaEngine (Trigeorgis): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
        - QuantLib, BinomialLRVanillaEngine (Leisen-Reimer): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
        - QuantLib, BinomialJ4VanillaEngine (exact inversion): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            american_option (bool, optional): Whether to value an American option, which can be exercised at every
                node, instead of a European option. Defaults to False.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            time_to_expiration (float): The number of years to use for the time to expiration, above 0. Defaults to 1 which equals
            one year.
            time_steps (int): The number of time steps to use for the binomial tree, from 1 to 1,000. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            show_input_info (bool, optional): Whether to show the input information, with the up and down movement
            and the risk neutral probability per ticker, or per ticker and strike for the Leisen-Reimer and exact
            inversion trees, which are centred on the strike. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.
            method (str, optional): The tree. Options are: "cox_ross_rubinstein" (Cox-Ross-Rubinstein),
                "jarrow_rudd" (Jarrow-Rudd), "tian" (Tian), "trigeorgis" (Trigeorgis), "leisen_reimer"
                (Leisen-Reimer), "exact_inversion" (exact inversion of the binomial distribution). Defaults to
                "cox_ross_rubinstein".

        Returns:
            pd.DataFrame: Binomial Trees values containing the tickers, strike prices and movements as the index and the
            time steps as the columns.

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
            strike_price_range=0, strike_step_size=1, time_steps=5
        )

        binomial_model.loc["KO"]
        ```

        Which returns:

        |               |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:--------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | (70, 'UUUUU') |       5.3445 |       8.4014 |      12.7664 |      18.5956 |      25.7101 |      33.495  |
        | (70, 'DUUUU') |     nan      |       2.4212 |       4.24   |       7.2343 |      11.8932 |      18.4642 |
        | (70, 'DDUUU') |     nan      |     nan      |       0.6695 |       1.3603 |       2.764  |       5.6163 |
        | (70, 'DDDUU') |     nan      |     nan      |     nan      |       0      |       0      |       0      |
        | (70, 'DDDDU') |     nan      |     nan      |     nan      |     nan      |       0      |       0      |
        | (70, 'DDDDD') |     nan      |     nan      |     nan      |     nan      |     nan      |       0      |
        """
        check_option("method", method, binomial_trees_model.BINOMIAL_TREE_METHODS)
        helpers.validate_time_to_expiration(time_to_expiration)
        validate_window("time_steps", time_steps)
        # Checked once for every ticker rather than reported per ticker by the tree.
        if time_steps > binomial_trees_model.MAX_TIMESTEPS:
            raise InvalidArgumentError(
                f"A tree of {time_steps:,} time steps is too large: time_steps must be at most "
                f"{binomial_trees_model.MAX_TIMESTEPS:,}. Please pass fewer time_steps."
            )

        if method in ("leisen_reimer", "exact_inversion") and time_steps % 2 == 0:
            # Both trees put the strike between the two middle nodes at expiration, which needs an odd count.
            logger.info(
                "The %s tree needs an odd number of time steps, so %s steps are used instead of %s.",
                method,
                time_steps + 1,
                time_steps,
            )
            time_steps += 1

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
            values_per_strike=(time_steps + 1) ** 2,
        )

        binomial_trees: dict[str, tuple[list[float], np.ndarray]] = {}
        binomial_trees_statistics: dict[str, dict[float, dict[str, float]]] = {
            "Up Movement": {},
            "Down Movement": {},
            "Risk Neutral Probability": {},
        }

        logger.info("Calculating Binomial Trees")
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # The trees of every strike at once, as a strikes × nodes × steps array.
            try:
                trees, up_movements, down_movements, probabilities = (
                    binomial_trees_model.get_option_payoff_trees(
                        stock_price=float(stock_price.loc[ticker]),
                        strike_prices=np.asarray(strike_prices, dtype=float),
                        years=time_to_expiration,
                        time_steps=time_steps,
                        risk_free_rate=float(risk_free_rate),
                        volatility=float(volatility.loc[ticker]),
                        dividend_yield=float(dividend_yield_value[ticker]),
                        put_option=put_option,
                        american_option=american_option,
                        method=method,
                    )
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker without a usable volatility (such as constant prices) leaves out
                # only that ticker.
                logger.warning("No binomial tree for %s: %s", ticker, error)
                continue
            binomial_trees[ticker] = (strike_prices, trees)

            if show_input_info:
                # The Leisen-Reimer and exact inversion trees are centred on the strike, so their movements
                # and probability differ per strike and are shown for each.
                for position, strike_price in enumerate(strike_prices):
                    label = (
                        f"{ticker} at a strike of {strike_price:g}"
                        if method in ("leisen_reimer", "exact_inversion")
                        else ticker
                    )
                    binomial_trees_statistics["Up Movement"][label] = float(
                        up_movements[position]
                    )
                    binomial_trees_statistics["Down Movement"][label] = float(
                        down_movements[position]
                    )
                    binomial_trees_statistics["Risk Neutral Probability"][label] = (
                        float(probabilities[position])
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

    @handle_errors
    def get_stock_price_simulation(
        self,
        start_date: str | None = None,
        time_to_expiration: float = 1,
        time_steps: int = 10,
        risk_free_rate: float | None = None,
        show_unique_combinations: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Simulate the Stock Price based on the Binomial Model: the prices each ticker can reach after every
        combination of up and down movements up to expiration.

        The binomial model is a mathematical model used to estimate the price of European and American style
        options. It creates a binomial tree of price paths for the underlying asset based on the stock price, its
        volatility and the time to expiration: each step multiplies the price by u = e^(σ * sqrt(Δt)) or by
        d = 1 / u. The stock price is then simulated based on these up and down movements.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            time_to_expiration (float): The number of years to use for the time to expiration, above 0. Defaults to 1 which equals
            one year.
            time_steps (int): The number of time steps to use for the binomial tree, from 1 to 1,000, and at
            most 16 when listing every path (show_unique_combinations=False). Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
            risk_free_rate (float, optional): The risk free rate shown with show_input_info; it does not affect the
            simulated prices. Defaults to None, which uses the Toolkit's risk-free rate on the pricing date: the
            10-year Treasury yield unless the Toolkit's risk_free_rate parameter selects '13w', '5y' or '30y',
            converted from its semi-annual bond-equivalent quote to continuous compounding. A rate passed here is used
            as given, as a continuously compounded rate.
            show_unique_combinations (bool, optional): Whether to show the unique combinations of the stock prices.
            Defaults to False.
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
            pd.DataFrame: Simulated Stock Price values containing the tickers and movements as the index and the
            time steps as the columns.

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
            time_steps=5, show_unique_combinations=True
        )

        stock_price_simulation.loc["AMD"]
        ```

        Which returns:

        | Movement   |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUUU      |       214.16 |      280.99  |      368.676 |     483.724  |     634.674  |     832.73   |
        | DUUUU      |       214.16 |      163.225 |      214.16  |     280.99   |     368.676  |     483.724  |
        | DDUUU      |       214.16 |      163.225 |      124.403 |     163.225  |     214.16   |     280.99   |
        | DDDUU      |       214.16 |      163.225 |      124.403 |      94.8154 |     124.403  |     163.225  |
        | DDDDU      |       214.16 |      163.225 |      124.403 |      94.8154 |      72.2646 |      94.8154 |
        | DDDDD      |       214.16 |      163.225 |      124.403 |      94.8154 |      72.2646 |      55.0773 |
        """
        helpers.validate_time_to_expiration(time_to_expiration)
        validate_window("time_steps", time_steps)
        # Checked once for every ticker rather than reported per ticker by the tree.
        if time_steps > binomial_trees_model.MAX_TIMESTEPS:
            raise InvalidArgumentError(
                f"A tree of {time_steps:,} time steps is too large: time_steps must be at most "
                f"{binomial_trees_model.MAX_TIMESTEPS:,}. Please pass fewer time_steps."
            )
        if (
            not show_unique_combinations
            and time_steps > binomial_trees_model.MAX_PATH_TIMESTEPS
        ):
            raise InvalidArgumentError(
                f"Listing every path of {time_steps:,} time steps gives 2^{time_steps} paths: "
                f"time_steps must be at most {binomial_trees_model.MAX_PATH_TIMESTEPS}. Please "
                "pass fewer time_steps, or show_unique_combinations=True for the distinct "
                "prices per step."
            )

        start_date = self._pricing_date(start_date)
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
            try:
                (
                    up_movement,
                    down_movement,
                ) = binomial_trees_model.calculate_up_and_down_movements(
                    volatility=volatility.loc[ticker],
                    time_delta=time_to_expiration / time_steps,
                )

                stock_price_simulation[ticker] = (
                    binomial_trees_model.calculate_stock_prices(
                        stock_price=stock_price.loc[ticker],
                        up_movement=up_movement,
                        down_movement=down_movement,
                        period_length=time_steps,
                        show_unique_combinations=show_unique_combinations,
                    )
                )
            except ValueError as error:
                # One ticker without a usable volatility (such as constant prices) leaves out
                # only that ticker.
                logger.warning("No stock price simulation for %s: %s", ticker, error)
                continue

            stock_price_statistics["Up Movement"][ticker] = up_movement
            stock_price_statistics["Down Movement"][ticker] = down_movement

        stock_price_simulation_df = helpers.create_stock_simulation_dataframe(
            stock_simulation_dictionary=stock_price_simulation,
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

    @handle_errors
    def get_put_call_parity(
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

        parity_gap: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            parity_gap[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            parity_gap[ticker] = helpers.evaluate_on_grid(
                functools.partial(
                    black_scholes_model.get_put_call_parity_gap,
                    stock_price=float(stock_price.loc[ticker]),
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    dividend_yield=dividend_yield_value[ticker],
                ),
                strike_prices,
                time_to_expiration_list,
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
            if not parity_gap_df.empty and parity_gap_df.isna().all().all():
                logger.warning(
                    "The put-call parity gaps do not vary within any ticker and strike price (Black-Scholes "
                    "prices meet parity exactly, so every gap rounds to 0), which leaves no Z-score to "
                    "calculate. Use standardize=False to see the gaps."
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

    @handle_errors
    def get_garman_kohlhagen(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        foreign_risk_free_rate: float = 0.0,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Garman-Kohlhagen Model: what European calls and puts on each currency pair are worth, the
        Black-Scholes model with the foreign interest rate in the role of a dividend yield.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date, so at least 2. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The domestic risk free rate to use for the calculation. Defaults to
            None, which uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the
            Toolkit's risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual
            bond-equivalent quote to continuous compounding. A rate passed here is used as given, as a continuously
            compounded rate.
            foreign_risk_free_rate (float, optional): The foreign risk free rate to use for the calculation, which
            plays the role of the dividend yield in the standard Black-Scholes model. Defaults to 0.0.
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
        |            148 |       8.4856 |       8.2657 |       8.1457 |
        |            150 |       6.555  |       6.4899 |       6.4972 |
        |            152 |       4.7419 |       4.8787 |       5.018  |
        |            154 |       3.15   |       3.4876 |       3.7399 |
        |            156 |       1.8838 |       2.3562 |       2.6819 |
        |            158 |       0.9965 |       1.4964 |       1.8457 |
        |            160 |       0.4596 |       0.8896 |       1.2167 |
        |            162 |       0.183  |       0.4935 |       0.767  |
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

        garman_kohlhagen: dict[str, dict[float, dict[float, float]]] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            # Every strike price and time to expiration in one call on arrays.
            garman_kohlhagen[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: black_scholes_model.get_garman_kohlhagen(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    foreign_risk_free_rate=foreign_risk_free_rate,
                    volatility=volatility.loc[ticker],
                    time_to_expiration=time_to_expiration,
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
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

    @handle_errors
    def get_binary_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        option_type: str = "cash-or-nothing",
        cash_payout: float = 1.0,
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
        Calculate the price of a Binary (Digital) Option: what a payoff of all or nothing at expiration is worth, a
        fixed cash amount or the stock itself when the option ends in the money, using the Black-Scholes framework.

        A binary option pays out a fixed amount if the option expires in-the-money and
        nothing otherwise. Two variants are supported: the cash-or-nothing option (the
        default) pays a fixed cash amount and the asset-or-nothing option pays the value
        of the underlying asset if the option expires
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            option_type (str, optional): The payoff. Options are: "cash-or-nothing" (a fixed cash amount),
            "asset-or-nothing" (the value of the stock). Defaults to "cash-or-nothing".
            cash_payout (float, optional): The fixed cash amount paid out by a cash-or-nothing option when it
            expires in-the-money. Ignored for asset-or-nothing options. Defaults to 1.0.
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

        binary_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            binary_option[ticker] = {}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )

            # Every strike price and time to expiration in one call on arrays.
            binary_option[ticker] = helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration: black_scholes_model.get_binary_option(
                    stock_price=stock_price.loc[ticker],
                    strike_price=strike_price,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility.loc[ticker],
                    time_to_expiration=time_to_expiration,
                    dividend_yield=dividend_yield_value[ticker],
                    put_option=put_option,
                    option_type=option_type,
                    cash_payout=cash_payout,
                ),
                strike_prices,
                time_to_expiration_list,
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

    @handle_errors
    def get_american_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        method: Literal[
            "qd_plus",
            "barone_adesi_whaley",
            "ju_zhong",
            "qd_fixed_point",
            "finite_difference",
            "bjerksund_stensland",
            "bjerksund_stensland_1993",
        ] = "qd_plus",
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
        Calculate American option prices: what calls and puts on each ticker that may be exercised at any time up to
        expiration are worth, with the method of choice.

        The methods are the QD+ method of Li (2010) and Andersen, Lake and Offengenden (2016), the quadratic
        approximations of Barone-Adesi and Whaley (1987) and Ju and Zhong (1999), the closed-form approximations of
        Bjerksund and Stensland (2002 and 1993), finite differences, and the exact value from the exercise boundary
        solved as a fixed point (Andersen, Lake and Offengenden, 2016).

        Listed options on single stocks are American: they can be exercised at any time up to expiration, which
        makes an in-the-money put, or a call on a stock with a high dividend yield, worth more than the European
        option the Black-Scholes model prices. These methods value that early exercise right without a tree:

        - QD+ (the default): the European price plus the early exercise premium integrated over the exercise
          boundary, with the boundary at each time to expiration from the QD+ equation of Li (2010). The most
          accurate of the approximations, within 0.3% of the exact value across typical inputs, and slower.
        - Barone-Adesi-Whaley: the early exercise premium is a power of S, fitted to the exercise value at the
          critical stock price S*, found numerically. Fast, and within about 1% of the exact value for most options,
          but several percent off for maturities of a year or more.
        - Ju-Zhong: Barone-Adesi-Whaley with a correction for the term it drops, which divides the premium by
          1 - χ(S), a quadratic in ln(S / S*). As fast and within about 0.3% for most options.
        - QD fixed point: the QD+ integral with the boundary that solves the integral equation of Kim (1990),
          iterated as a fixed point from the QD+ boundary until it changes less than 1e-12. It converges to the
          exact American value, within about 1e-6 (1e-8 for most options), at about 30 to 150 ms per price, so it
          is the slowest and best suited to a few strikes and expirations.
        - Finite differences: the Black-Scholes equation solved backwards from expiration with Crank-Nicolson
          (Rannacher start-up) on a grid of 1,000 steps in ln(S / K) and at least 400 time steps (4 per expiration
          day beyond 100 days), with the exercise constraint solved at every step by Brennan and Schwartz (1977). As
          V(S, K, τ) = K * v(S / K, τ), one solve gives every strike, and the expirations are solved in groups whose
          longest is at most four times their shortest, each on a grid sized for it, so that an expiration a day
          away is as precise as one months away: within about 1e-3 of the exact value for a stock price of 100
          (7.3e-4 at most from a day to a quarter at volatilities up to 100%; the error scales with the price), see
          ``finite_difference_model.get_finite_difference_option_values``.
        - Bjerksund-Stensland (2002): a closed form that assumes the early exercise boundary is flat in steps:
          once the stock price crosses the trigger price, immediate exercise is assumed optimal. The option's life
          is split at t1 = ½(√5 - 1) * T into two periods, each with its own flat trigger price, which brings the
          price closer to the exact American value than a single flat trigger.
        - Bjerksund-Stensland (1993): the original version, with one flat trigger price over the option's whole
          life, which is faster and slightly further below the exact American value.

        The formulas are as follows, with φ = 1 for a call and -1 for a put:

        - α = 2r / σ², β = 2(r - q) / σ², h = 1 - e^(-rT)
        - λ = [-(β - 1) + φ * √((β - 1)² + 4α / h)] / 2
        - S* solves φ * (S* - K) = V_E(S*) + φ * (1 - e^(-qT) * N(φ * d1(S*))) * S* / λ
        - Barone-Adesi-Whaley = V_E(S) + φ * (1 - e^(-qT) * N(φ * d1(S*))) * S* / λ * (S / S*)^λ
        - Ju-Zhong = V_E(S) + (φ * (S* - K) - V_E(S*)) * (S / S*)^λ / (1 - b * ln(S / S*)² - c * ln(S / S*))
        - QD+ Put = p(S) + ∫_0^T [r * K * e^(-rt) * N(-d2(t, S / B(T - t))) - q * S * e^(-qt) * N(-d1(t, S / B(T - t)))] dt
        - QD fixed point: B(τ) = K * e^(-(r - q)τ) * N(τ, B) / D(τ, B), iterated at 25 Chebyshev nodes in √τ, with N
          and D the integrals of the FP-A equation of Andersen, Lake and Offengenden (2016), and the put priced with
          the integral of QD+
        - QD+ Call (S, K, r, q) = QD+ Put (K, S, q, r), the put-call symmetry of McDonald and Schroder (1998)
        - Finite differences: V_τ = σ²/2 * V_xx + (r - q - σ²/2) * V_x - r * V on x = ln(S / K), stepped with
          (I - Δτ/2 * L) * V(τ + Δτ) = (I + Δτ/2 * L) * V(τ) subject to V ≥ φ * (S - K)

        The Bjerksund-Stensland formulas are as follows (for a call, with cost of carry b = r - q smaller than r):

        - β = (0.5 - b / σ²) + √((b / σ² - 0.5)² + 2r / σ²)
        - B∞ = β / (β - 1) * K
        - B0 = MAX(K, r / (r - b) * K)
        - h(t) = -(b * t + 2σ√t) * K² / ((B∞ - B0) * B0)
        - I(t) = B0 + (B∞ - B0) * (1 - e^h(t)), with I1 = I(t1) and I2 = I(T)
        - Call (2002) = MAX(Approximation, European Black-Scholes Call, MAX(S - K, 0)), where the Approximation is
          S - K when S ≥ I2 and otherwise combines univariate (first period) and bivariate (both periods)
          normal terms, see ``black_scholes_model.get_bjerksund_stensland``
        - Put (S, K, T, r, b, σ) = Call (K, S, T, r - b, -b, σ), the put-call transformation

        The 1993 version keeps one flat trigger price over the option's whole life:

        - h(T) = -(b * T + 2σ√T) * B0 / (B∞ - B0)
        - I = B0 + (B∞ - B0) * (1 - e^h(T)), α = (I - K) * I^(-β)
        - Call (1993) = α * S^β - α * φ(S, T, β, I, I) + φ(S, T, 1, I, I) - φ(S, T, 1, K, I) - K * φ(S, T, 0, I, I) +
          K * φ(S, T, 0, K, I), with φ the univariate knock-out claim of ``black_scholes_model._phi``

        Where each formula holds on the side of S* (or of the boundary B(T)) where holding the option is optimal,
        and the option is worth its exercise value φ * (S - K) beyond it. S is the stock price (the traded close on
        the pricing date), K is the strike price, r is the continuously compounded risk-free rate, q is the
        continuous dividend yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252), T is the time to expiration in years
        (days / 365), V_E and p are the European Black-Scholes prices, d1 and d2 are those of the Black-Scholes
        model, b and c are the Ju-Zhong coefficients (see ``american_model.get_ju_zhong``) and B(τ) is the QD+
        exercise boundary of a put with τ years to expiration (see ``american_model.get_qd_plus``).

        A call on a stock without dividends is never exercised early and a put at a non-positive rate neither;
        these are worth the Black-Scholes price, floored at the exercise value. With Bjerksund-Stensland, a call
        whose cost of carry b is at least r is likewise worth the Black-Scholes price, and where the boundary I2
        falls to or below the strike, a 1,000-step binomial tree prices the option instead. At zero days to
        expiration the value is the exercise value.

        Bjerksund-Stensland can undervalue an option when the risk-free rate or the cost of carry is negative: its
        flat boundaries can fall back to the European value while early exercise is worth more (with r = -2%,
        q = -3%, σ = 1% and T = 1, an at-the-money put of 0.0854 against 0.1669 for a 10,000-step binomial tree).
        A warning names the tickers this applies to; QD+ prices them accurately.

        See: Barone-Adesi, G., & Whaley, R. E. (1987), "Efficient Analytic Approximation of American Option
        Values", The Journal of Finance, 42(2), 301-320; Ju, N., & Zhong, R. (1999), "An Approximate Formula for
        Pricing American Options", The Journal of Derivatives, 7(2), 31-40; Li, M. (2010), "Analytical
        Approximations for the Critical Stock Prices of American Options: A Performance Comparison", Review of
        Derivatives Research, 13(1), 75-99; Andersen, L., Lake, M., & Offengenden, D. (2016), "High-Performance
        American Option Pricing", Journal of Computational Finance, 20(1), 39-87; Kim, I. J. (1990), "The Analytic
        Valuation of American Options", The Review of Financial Studies, 3(4), 547-572; Brennan, M. J., & Schwartz,
        E. S. (1977), "The Valuation of American Put Options", The Journal of Finance, 32(2), 449-462; Bjerksund,
        P., & Stensland, G. (2002), "Closed Form Valuation of American Options", Working paper, NHH; Bjerksund, P.,
        & Stensland, G. (1993), "Closed-Form Approximation of American Options", Scandinavian Journal of
        Management, 9, S87-S99, both as presented in Haug, E. G. (2007), "The Complete Guide to Option Pricing
        Formulas" (2nd ed.), sections 3.1.4 and 3.1.3.

        Also known as: BAW, quadratic approximation, American option price, early exercise premium, QD+, QdFp,
        BS2002, Bjerksund-Stensland approximation.

        Referenced against:
        - QuantLib, QdPlusAmericanEngine (QD+): https://www.quantlib.org/reference/class_quant_lib_1_1_qd_plus_american_engine.html
        - QuantLib, BaroneAdesiWhaleyApproximationEngine (Barone-Adesi-Whaley): https://www.quantlib.org/reference/class_quant_lib_1_1_barone_adesi_whaley_approximation_engine.html
        - QuantLib, JuQuadraticApproximationEngine (Ju-Zhong): https://www.quantlib.org/reference/class_quant_lib_1_1_ju_quadratic_approximation_engine.html
        - QuantLib, QdFpAmericanEngine (QD fixed point): https://www.quantlib.org/reference/class_quant_lib_1_1_qd_fp_american_engine.html
        - QuantLib, FdBlackScholesVanillaEngine (QD fixed point): https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html
        - QuantLib, FdBlackScholesVanillaEngine (Bjerksund-Stensland 2002): https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html
        - QuantLib, BjerksundStenslandApproximationEngine (Bjerksund-Stensland 1993): https://www.quantlib.org/reference/class_quant_lib_1_1_bjerksund_stensland_approximation_engine.html

        Shares similar logic:
        - QuantLib, FdBlackScholesVanillaEngine (finite differences): https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html
        - FinancePy, black_scholes_fd (finite differences): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/finite_difference.py#L318
        - FinancePy, baw_value (Barone-Adesi-Whaley): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black_scholes_analytic.py#L682

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            method (str, optional): The pricing method. Options are: "qd_plus" (QD+), "barone_adesi_whaley"
            (Barone-Adesi-Whaley), "ju_zhong" (Ju-Zhong), "qd_fixed_point" (QD fixed point), "finite_difference"
            (Crank-Nicolson finite differences), "bjerksund_stensland" (Bjerksund-Stensland 2002, two flat trigger
            prices) and "bjerksund_stensland_1993" (Bjerksund-Stensland 1993, one flat trigger price). Defaults to
            "qd_plus".
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
            pd.DataFrame: American option values containing the tickers and strike prices as the index and the time
            to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        american_put = toolkit.options.get_american_option(
            put_option=True, strike_price_range=0.15, strike_step_size=1, expiration_time_range=91
        )

        american_put.loc["VZ", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             34 |       0.001  |       0.0226 |       0.0764 |
        |             35 |       0.005  |       0.0551 |       0.1464 |
        |             36 |       0.0193 |       0.1202 |       0.2605 |
        |             37 |       0.0606 |       0.2369 |       0.4335 |
        |             38 |       0.1584 |       0.4268 |       0.6798 |
        |             39 |       0.3523 |       0.71   |       1.0107 |
        |             40 |       0.6807 |       1.1012 |       1.4332 |
        |             41 |       1.1664 |       1.6063 |       1.9493 |
        |             42 |       1.8081 |       2.2221 |       2.5554 |
        |             43 |       2.5816 |       2.9372 |       3.2441 |
        |             44 |       3.4518 |       3.735  |       4.0046 |
        |             45 |       4.3838 |       4.5974 |       4.8251 |
        |             46 |       5.3505 |       5.507  |       5.6935 |
        |             47 |       6.3345 |       6.449  |       6.5986 |
        """
        check_option(
            "method",
            method,
            (
                "qd_plus",
                "barone_adesi_whaley",
                "ju_zhong",
                "qd_fixed_point",
                "finite_difference",
                "bjerksund_stensland",
                "bjerksund_stensland_1993",
            ),
        )

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

        american_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            try:
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )

                if method == "finite_difference":
                    # One solve per group of expirations gives every strike (the value is
                    # homogeneous in the stock and strike price), with four time steps per
                    # expiration day.
                    values = (
                        finite_difference_model.get_finite_difference_option_values(
                            stock_price=float(stock_price.loc[ticker]),
                            strike_prices=np.asarray(strike_prices, dtype=float),
                            times_to_expiration=np.asarray(time_to_expiration_list),
                            risk_free_rate=float(risk_free_rate),
                            volatility=float(volatility.loc[ticker]),
                            dividend_yield=float(dividend_yield_value[ticker]),
                            put_option=put_option,
                            american_option=True,
                            time_steps=max(400, 4 * expiration_time_range),
                        )
                    )
                    american_option[ticker] = {
                        strike_price: dict(
                            zip(time_to_expiration_list, map(float, row), strict=True)
                        )
                        for strike_price, row in zip(strike_prices, values, strict=True)
                    }
                    continue

                if method == "qd_plus":
                    # Every strike price and expiration at once: the boundary of each expiration
                    # is solved once, for all expirations together, and scaled to every strike.
                    american_option[ticker] = helpers.evaluate_on_grid(
                        lambda strike_price, time_to_expiration: american_model.get_qd_plus(
                            stock_price=float(stock_price.loc[ticker]),
                            strike_price=strike_price,
                            risk_free_rate=float(risk_free_rate),
                            volatility=float(volatility.loc[ticker]),
                            time_to_expiration=time_to_expiration,
                            dividend_yield=float(dividend_yield_value[ticker]),
                            put_option=put_option,
                        ),
                        strike_prices,
                        time_to_expiration_list,
                    )
                    continue

                if method in ("bjerksund_stensland", "bjerksund_stensland_1993"):
                    # The closed forms evaluate every strike price and expiration on arrays.
                    american_option[ticker] = helpers.evaluate_on_grid(
                        lambda strike_price, time_to_expiration, ticker=ticker: (
                            black_scholes_model.get_bjerksund_stensland(
                                stock_price=float(stock_price.loc[ticker]),
                                strike_price=strike_price,
                                risk_free_rate=float(risk_free_rate),
                                volatility=float(volatility.loc[ticker]),
                                time_to_expiration=time_to_expiration,
                                dividend_yield=float(dividend_yield_value[ticker]),
                                put_option=put_option,
                                method=(
                                    "1993"
                                    if method == "bjerksund_stensland_1993"
                                    else "2002"
                                ),
                            )
                        ),
                        strike_prices,
                        time_to_expiration_list,
                    )
                    continue

                # The quadratic approximations solve their critical price once per expiration and
                # evaluate every strike at once; the QD fixed point is priced per option.
                american_option[ticker] = helpers.evaluate_on_grid(
                    lambda strike_price, time_to_expiration, ticker=ticker: (
                        american_model.get_american_option(
                            stock_price=float(stock_price.loc[ticker]),
                            strike_price=strike_price,
                            risk_free_rate=float(risk_free_rate),
                            volatility=float(volatility.loc[ticker]),
                            time_to_expiration=time_to_expiration,
                            dividend_yield=float(dividend_yield_value[ticker]),
                            put_option=put_option,
                            method=method,
                        )
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No American option value for %s: %s", ticker, error)

        if method in ("bjerksund_stensland", "bjerksund_stensland_1993"):
            # The flat exercise boundaries are derived for a positive rate and cost of carry; below zero the
            # approximation can fall to the European value while early exercise is worth more.
            negative_carry = [
                ticker
                for ticker, ticker_dividend_yield in dividend_yield_value.items()
                if risk_free_rate < 0 or risk_free_rate - ticker_dividend_yield < 0
            ]
            if negative_carry:
                logger.warning(
                    "The Bjerksund-Stensland approximation can undervalue American options when the risk-free rate "
                    "or the cost of carry (the risk-free rate minus the dividend yield) is negative, as for %s. "
                    'Use method="qd_plus" for these.',
                    ", ".join(negative_carry),
                )

        american_option_df = helpers.create_greek_dataframe(
            greek_dictionary=american_option,
            start_date=start_date,
        )

        american_option_df = american_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            american_option_df = calculate_standardization(
                dataset=american_option_df,
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

        return american_option_df

    @handle_errors
    def get_cev_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        beta: float = 0.5,
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
        Calculate European option prices in the constant elasticity of variance (CEV) model of Cox (1975): what
        calls and puts on each ticker are worth when the volatility rises as the stock price falls, with the
        non-central chi-squared formula of Schroder (1989).

        In the CEV model the stock's volatility depends on its price, σ(S) = δ * S^(β - 1). With an elasticity β below
        1 the volatility rises as the price falls (the leverage effect), so out-of-the-money puts are worth more and
        out-of-the-money calls less than in the Black-Scholes model (β = 1): the downward sloping skew of equity
        options. β = 0.5 is the square-root process and β = 0 an absolute (normal) volatility. The stock is absorbed
        at zero.

        The historical volatility is taken as the local volatility at today's price, δ = σ * S^(1 - β), so the
        at-the-money options are priced close to Black-Scholes and β only adds the skew. β is a model choice rather
        than data: fitted to an option chain it is typically between 0 and 1 for stocks and indices.

        The formulas are as follows:

        - δ = σ * S^(1 - β), F = S * e^((r - q)T), ν = 1 / (1 - β)
        - v = δ² * (e^(2(r - q)(1 - β)T) - 1) / (2(r - q)(1 - β)), or δ² * T when r = q
        - x = F^(2(1 - β)) / ((1 - β)² * v), y = K^(2(1 - β)) / ((1 - β)² * v)
        - Call = e^(-rT) * [F * (1 - χ²(y; ν + 2, x)) - K * χ²(x; ν, y)]
        - Put = e^(-rT) * [K * (1 - χ²(x; ν, y)) - F * χ²(y; ν + 2, x)]

        Where S is the stock price (the traded close on the pricing date), K is the strike price, r is the
        continuously compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)), σ is the
        annualized volatility (the standard deviation of the daily returns over the trailing 252 trading days times
        √252), β is the elasticity, T is the time to expiration in years (days / 365), F is the futures price of the
        expiration and χ²(z; k, λ) is the distribution function of the non-central chi-squared distribution with k
        degrees of freedom and non-centrality λ. With β = 1 the values are those of the Black-Scholes model. At zero
        days to expiration the value is the exercise value.

        See: Cox, J. C. (1975), "Notes on Option Pricing I: Constant Elasticity of Variance Diffusions", working paper,
        Stanford University (reprinted in The Journal of Portfolio Management, 23(2), 1996); Schroder, M. (1989),
        "Computing the Constant Elasticity of Variance Option Pricing Formula", The Journal of Finance, 44(1),
        211-219.

        Also known as: CEV model, constant elasticity of variance, CEV option price, leverage effect skew.

        Referenced against:
        - QuantLib, AnalyticCEVEngine: https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/vanilla/analyticcevengine.hpp#L62

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            beta (float, optional): The elasticity β, between 0 and 1, where 1 is the Black-Scholes model. Defaults to
            0.5, the square-root process.
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
            pd.DataFrame: CEV option values containing the tickers and strike prices as the index and the time to
            expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["AAPL"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        cev_put = toolkit.options.get_cev_model(
            put_option=True, beta=0.5, strike_price_range=0.15, strike_step_size=10, expiration_time_range=91
        )

        cev_put.loc["AAPL", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            230 |       0.3865 |       1.6792 |       3.1579 |
        |            240 |       1.0437 |       3.082  |       5.0247 |
        |            250 |       2.4176 |       5.2461 |       7.6031 |
        |            260 |       4.8871 |       8.3465 |      10.9942 |
        |            270 |       8.7686 |      12.5031 |      15.2615 |
        |            280 |      14.1992 |      17.7584 |      20.424  |
        |            290 |      21.0882 |      24.072  |      26.4555 |
        |            300 |      29.1649 |      31.3332 |      33.2904 |
        |            310 |      38.084  |      39.386  |      40.8347 |
        |            320 |      47.5269 |      48.0578 |      48.9786 |
        """
        if not 0 <= beta <= 1:
            raise InvalidArgumentError(
                f"beta must be between 0 and 1, received {beta!r}."
            )

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

        cev_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            try:
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )

                # Every strike price and time to expiration in one call on arrays.
                cev_option[ticker] = helpers.evaluate_on_grid(
                    lambda strike_price, time_to_expiration: cev_model.get_cev(
                        stock_price=float(stock_price.loc[ticker]),
                        strike_price=strike_price,
                        risk_free_rate=float(risk_free_rate),
                        volatility=float(volatility.loc[ticker]),
                        time_to_expiration=time_to_expiration,
                        beta=float(beta),
                        dividend_yield=float(dividend_yield_value[ticker]),
                        put_option=put_option,
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No CEV option value for %s: %s", ticker, error)

        cev_option_df = helpers.create_greek_dataframe(
            greek_dictionary=cev_option,
            start_date=start_date,
        )

        cev_option_df = cev_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            cev_option_df = calculate_standardization(
                dataset=cev_option_df,
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

        return cev_option_df

    @handle_errors
    def get_black_76_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the Black (1976) model: what European calls and puts on futures and forwards are worth, the
        Black-Scholes model with each ticker's price taken as the futures price.

        A futures price already contains the cost of carry of the underlying (interest, storage, dividends or the
        convenience yield of a commodity), so it grows at no rate under the risk-neutral measure and the option only
        discounts its payoff. Use it for tickers that quote futures prices, such as commodities (GCUSD for gold,
        CLUSD for crude oil) and index futures. It assumes the option expires with the futures contract, so the
        futures price of each expiration is the one quoted today; for an option that expires before its futures
        contract the price is the same with the time to the option's expiration.

        The formulas are as follows:

        - d1 = (ln(F / K) + (σ^2 / 2) * t) / (σ * sqrt(t))
        - d2 = d1 - σ * sqrt(t)
        - Call Option Price = e^(-r * t) * (F * N(d1) - K * N(d2))
        - Put Option Price = e^(-r * t) * (K * N(-d2) - F * N(-d1))

        Where F is the futures price (the traded close on the pricing date), K is the strike price, r is the
        continuously compounded risk-free rate, σ is the annualized volatility of the futures price (the standard
        deviation of the daily returns over the trailing 252 trading days times √252), t is the time to expiration in
        years (days / 365) and N is the standard normal distribution. At zero days to expiration the value is the
        exercise value. The implied volatility of a quoted option on a futures contract is
        ``black_scholes_model.get_black_76_implied_volatility``.

        See: Black, F. (1976), "The Pricing of Commodity Contracts", Journal of Financial Economics, 3(1-2), 167-179.

        Also known as: Black-76, Black's model, options on futures, options on forwards, commodity options.

        Referenced against:
        - QuantLib, blackFormula: https://www.quantlib.org/reference/namespace_quant_lib.html#adfe71f9b0b157cae6e4bd53faaa676af

        Shares similar logic:
        - FinancePy, Black: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black.py#L18

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current futures price is 100.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current futures price is 100.
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
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Black-76 values containing the tickers and strike prices as the index and the time to
            expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["GCUSD"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        black_76 = toolkit.options.get_black_76_model(
            strike_price_range=0.1, strike_step_size=100, expiration_time_range=91
        )

        black_76.loc["GCUSD", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |           3900 |     442.905  |     454.105  |     468.105  |
        |           4000 |     349.15   |     368.535  |     388.087  |
        |           4100 |     262.153  |     290.585  |     315.343  |
        |           4200 |     185.691  |     222.011  |     250.855  |
        |           4300 |     123.028  |     163.997  |     195.201  |
        |           4400 |      75.7173 |     116.939  |     148.498  |
        |           4500 |      43.0764 |      80.4062 |     110.409  |
        |           4600 |      22.5863 |      53.2835 |      80.2235 |
        |           4700 |      10.9008 |      34.0278 |      56.973  |
        |           4800 |       4.843  |      20.9478 |      39.5581 |
        """
        start_date = self._pricing_date(start_date)
        futures_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        strike_prices_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=futures_price,
            strike_step_size=strike_step_size,
            strike_price_range=strike_price_range,
            values_per_strike=expiration_time_range,
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

        # Every strike price and time to expiration in one call on arrays.
        black_76: dict[str, dict[float, dict[float, float]]] = {
            ticker: helpers.evaluate_on_grid(
                lambda strike_price, time_to_expiration, ticker=ticker: black_scholes_model.get_black_76(
                    forward_price=float(futures_price.loc[ticker]),
                    strike_price=strike_price,
                    risk_free_rate=float(risk_free_rate),
                    volatility=float(volatility.loc[ticker]),
                    time_to_expiration=time_to_expiration,
                    put_option=put_option,
                ),
                strike_prices,
                time_to_expiration_list,
            )
            for ticker, strike_prices in strike_prices_per_ticker.items()
        }

        black_76_df = helpers.create_greek_dataframe(
            greek_dictionary=black_76,
            start_date=start_date,
        )

        black_76_df = black_76_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            black_76_df = calculate_standardization(
                dataset=black_76_df,
                rounding=rounding if rounding is not None else self._rounding,
                axis="columns",
            )

        if show_input_info:
            helpers.show_input_info(
                start_date=self._daily_historical.index[0],
                end_date=self._daily_historical.index[-1],
                stock_prices=futures_price,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
            )

        return black_76_df

    @handle_errors
    def get_monte_carlo_option_price(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        simulations: int = 10_000,
        time_steps: int = 100,
        seed: int | None = 0,
        show_standard_error: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
        american_option: bool = False,
        variance_reduction: Literal["antithetic", "control_variate"] | None = None,
        sequence: Literal["pseudo_random", "sobol"] = "pseudo_random",
    ):
        """
        Calculate option prices through Monte Carlo simulation: what European calls and puts on each ticker are
        worth on simulated Geometric Brownian Motion (GBM) stock price paths, or American ones by least-squares
        Monte Carlo (Longstaff and Schwartz, 2001).

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

        An American option can be exercised at each of the time steps: stepping
        back from expiration, the discounted cash flows of the paths in the money are regressed
        on a constant and the weighted Laguerre polynomials e^(-x/2) * L_n(x), n = 0, 1, 2, of
        x = S(t) / K, and a path is exercised where its exercise value beats the fitted
        continuation value (see ``monte_carlo_model.get_longstaff_schwartz_option``):

        - C(t) = β_0 + Σ β_n * e^(-x/2) * L_n(x), fitted on the in-the-money paths at t
        - Cash Flow(t) = φ * (S(t) - K) if φ * (S(t) - K) > C(t), else e^(-r * Δt) * Cash Flow(t + Δt)
        - American Price = MAX(φ * (S - K), MEAN(e^(-r * Δt) * Cash Flow(Δt)))

        Two variance reductions and quasi-random numbers narrow the standard error at the same
        number of paths (Glasserman, 2003, chapters 4 and 5):

        - Antithetic variates: every path Z is paired with -Z (simulations must be
          even) and the standard error comes from the averages of the pairs.
        - Control variate: Price = MEAN(Y - β * (X - E[X])), β = Cov(Y, X) /
          Var(X), with X the discounted stock price at expiration (E[X] = S * e^(-q * T)) for a
          European option and the discounted European payoff (E[X] its Black-Scholes price) for
          an American option.
        - Sobol sequence: scrambled Sobol points with a Brownian bridge in 16 independently
          scrambled replications, whose spread gives the standard error; the error shrinks
          close to 1 / simulations rather than 1 / √simulations.

        All strike prices of an expiry are priced on the same simulated paths (common
        random numbers), which keeps the prices consistent across strikes and means the
        paths are simulated once per expiry rather than once per strike. The European
        expiries share one set of standardised shocks, each scaled to its own time to
        expiration, which with a seed gives the prices of simulating every expiry with that
        seed.

        As it is a simulation, the result comes with sampling error. Showing the standard error
        additionally returns the standard error of each estimate — as a rule of thumb, the true
        price lies within plus or minus 2 times the standard error roughly 95% of the time. The
        seed makes the results reproducible; without one (None) every call draws new paths.

        See: Boyle, P. P. (1977), "Options: A Monte Carlo Approach", Journal of Financial
        Economics, 4(3), 323-338; Longstaff, F. A., & Schwartz, E. S. (2001), "Valuing American
        Options by Simulation: A Simple Least-Squares Approach", The Review of Financial Studies,
        14(1), 113-147; Glasserman, P. (2003), "Monte Carlo Methods in Financial Engineering",
        Springer. The libraries below draw their own random numbers,
        which give other paths.

        Also known as: Monte Carlo option pricing, simulation-based option pricing.

        Shares similar logic:
        - QuantLib, MCEuropeanEngine (European): https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_european_engine.html
        - QuantLib, MCAmericanEngine (American): https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_american_engine.html
        - FinancePy, equity_lsmc (American): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/equity_lsmc.py#L39

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
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
            simulations (int, optional): The number of simulated stock price paths, at least 1. Defaults to 10,000.
            time_steps (int, optional): The number of time steps used to build each simulated path, at least 1.
            Defaults to 100.
            seed (int | None, optional): The seed used to initialize the random number generator, ensuring
            reproducible results; None draws new paths on every call. Defaults to 0.
            show_standard_error (bool, optional): Whether to also return the standard error of each Monte Carlo
            estimate as a second DataFrame. Defaults to False.
            show_input_info (bool, optional): Whether to show the input information. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            standardize (bool, optional): Whether to standardize (Z-Score) the result across the
                time to expiration columns for each ticker and strike price. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.
            american_option (bool, optional): Whether to price American options by least-squares Monte
            Carlo, exercisable at each time step. Defaults to False.
            variance_reduction (str | None, optional): The variance reduction. Options are: None (none),
            "antithetic" (antithetic variates), "control_variate" (control variate). Defaults to None.
            sequence (str, optional): The random numbers. Options are: "pseudo_random" (pseudo-random numbers),
            "sobol" (scrambled Sobol points, at least 32 simulations). Defaults to "pseudo_random".

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
        |            770 |      92.8712 |      93.4178 |      95.1631 |
        |            780 |      82.8804 |      83.4928 |      85.5947 |
        |            790 |      72.8944 |      73.6473 |      76.2372 |
        |            800 |      62.9375 |      63.9363 |      67.1776 |
        |            810 |      53.0607 |      54.4486 |      58.49   |
        |            820 |      43.3666 |      45.3481 |      50.2664 |
        |            830 |      34.0541 |      36.7981 |      42.5944 |
        |            840 |      25.4471 |      28.9931 |      35.5504 |
        |            850 |      17.8995 |      22.0913 |      29.1762 |
        |            860 |      11.7123 |      16.2119 |      23.5432 |
        |            870 |       7.0923 |      11.4611 |      18.6827 |
        |            880 |       3.9767 |       7.8036 |      14.5838 |
        |            890 |       2.0394 |       5.1199 |      11.2047 |
        |            900 |       0.961  |       3.213  |       8.4756 |
        |            910 |       0.4225 |       1.9345 |       6.2838 |
        |            920 |       0.1698 |       1.1303 |       4.5777 |
        |            930 |       0.0582 |       0.6374 |       3.2774 |
        |            940 |       0.018  |       0.3432 |       2.3097 |
        |            950 |       0.0049 |       0.1773 |       1.6158 |
        """
        validate_window("simulations", simulations)
        validate_window("time_steps", time_steps)

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
            # A value from paths of the variance reductions or Sobol points evaluates every
            # path per strike: about 0.4 milliseconds a value at 10,000 paths, so it counts as
            # a value per 1,000 paths. The plain estimate sorts the paths once per expiration.
            values_per_strike=expiration_time_range
            * (
                1
                if variance_reduction is None and sequence == "pseudo_random"
                else -(-simulations // 1_000)
            ),
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

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

            inputs = {
                "stock_price": float(stock_price.loc[ticker]),
                "strike_prices": np.asarray(strike_prices, dtype=float),
                "risk_free_rate": float(risk_free_rate),
                "volatility": float(volatility.loc[ticker]),
                "dividend_yield": float(dividend_yield_value[ticker]),
                "put_option": put_option,
                "simulations": simulations,
                "time_steps": time_steps,
                "seed": seed,
                "variance_reduction": variance_reduction,
                "sequence": sequence,
            }

            if not american_option:
                # Every strike and expiry is priced on one set of simulated paths, scaled to
                # each expiry, which with a seed gives the same prices as simulating per
                # strike and expiry.
                prices, standard_errors = options_model.get_monte_carlo_option_prices(
                    **inputs, time_to_expiration=np.asarray(time_to_expiration_list)
                )
                monte_carlo_price[ticker] = {
                    strike_price: dict(
                        zip(time_to_expiration_list, map(float, row), strict=True)
                    )
                    for strike_price, row in zip(strike_prices, prices, strict=True)
                }
                monte_carlo_error[ticker] = {
                    strike_price: dict(
                        zip(time_to_expiration_list, map(float, row), strict=True)
                    )
                    for strike_price, row in zip(
                        strike_prices, standard_errors, strict=True
                    )
                }
                continue

            # Every strike is priced on the same simulated paths of an expiry, which with
            # a seed gives the same prices as simulating per strike, once per expiry.
            for time_to_expiration in time_to_expiration_list:
                prices, standard_errors = (
                    monte_carlo_model.get_longstaff_schwartz_option_prices(
                        **inputs, time_to_expiration=time_to_expiration
                    )
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

    @handle_errors
    def get_barrier_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        barrier_percentage: float = 0.9,
        barrier_direction: str = "down",
        knock_type: str = "out",
        rebate: float = 0.0,
        upper_barrier_percentage: float = 1.1,
        barrier_window: str = "full",
        window_share: float = 0.5,
        soft_barrier_percentage: float | None = None,
        monitoring: str = "continuous",
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
        Calculate the closed-form price of a barrier option: what a European option on each ticker is worth that
        knocks in or out when the stock price touches a barrier, single (up or down) with the Reiner & Rubinstein
        (1991) formulas, double, partial-time, soft or discretely watched.

        A barrier option is a path-dependent option whose payoff (and existence)
        depends on whether the underlying stock price touches a pre-specified barrier
        level at any point before expiration:

        - Knock-out: the option becomes worthless if the barrier is touched.
        - Knock-in: the option only comes into existence if the barrier is touched.
        - Down barrier: the barrier is below the current stock price.
        - Up barrier: the barrier is above the current stock price.

        The barrier level is defined relative to the current stock price as a percentage,
        e.g. a value of 0.9 sets the barrier at 90% of the
        current stock price (a sensible default for a down barrier).

        Besides the single barrier watched continuously over the whole life, the barrier can be:

        - Double: a lower barrier and an upper barrier, each a percentage of the stock price;
          the knock-out pays only when the stock price stays between the two, priced with the
          series of Ikeda and Kunitomo (1992) for flat barriers.
        - Partial-time: watched only during the first (a start window) or the last (an end
          window) share of the life (Heynen & Kat, 1994). With an end window the option is also
          knocked out (or in) when the stock price is beyond the barrier at the start of the
          window.
        - Soft: a band between the barrier and a second, soft barrier through which the option
          knocks out gradually (Hart & Ross, 1994): a soft down-and-out pays the vanilla payoff
          times (min S - L) / (U - L), the share of the band the lowest price did not reach,
          which is the average of the hard barrier prices over the band.
        - Discretely watched (daily, weekly or monthly): checked only at each close, which
          is priced by moving the barriers away from the stock price by e^(0.5826 * σ * √Δt), with Δt = 1/252,
          1/52 or 1/12 (Broadie, Glasserman & Kou, 1997). A daily barrier is crossed noticeably less often than
          a continuous one, so it is worth setting for a knock-out with a barrier close to the stock price.

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
        - Double Knock-out Call = S * e^((b - r)T) * Σ_n [(U^n / L^n)^μ' * (N(d1) - N(d2)) - (L^(n+1) / (U^n * S))^μ' * (N(d3) - N(d4))]
          - K * e^(-rT) * Σ_n [(U^n / L^n)^(μ' - 2) * (N(d1 - σ√T) - N(d2 - σ√T)) - (L^(n+1) / (U^n * S))^(μ' - 2) * (N(d3 - σ√T) - N(d4 - σ√T))],
          with μ' = 2b / σ² + 1, d1 = (ln(S * U^2n / (K' * L^2n)) + (b + σ²/2)T) / (σ√T), d2 the same with U in place of K',
          d3 = (ln(L^(2n+2) / (K' * S * U^2n)) + (b + σ²/2)T) / (σ√T), d4 the same with U in place of K', K' = max(K, L), and
          the put the same way over the prices between L and min(K, U)
        - Partial-time Knock-out, start window up to t1 = Window Share * T: e^(-r t1) * {E[BS(S(t1)) * 1{ηS(t1) > ηH}]
          - (H / S)^(2μ) * E[BS(S(t1) * (H / S)²) * 1{ηS(t1) * (H / S)² > ηH}]}, with BS the Black-Scholes price over T - t1
        - Partial-time Knock-out, end window from t1 = (1 - Window Share) * T: e^(-r t1) * E[KO(S(t1)) * 1{ηS(t1) > ηH}], with
          KO the knock-out price above over T - t1; both expectations are closed form with the bivariate normal distribution
        - Soft Knock-out = 1 / (U - L) * ∫_L^U Knock-out(H) dH over the band [L, U]
        - Discrete monitoring: H → H * e^(0.5826 * σ * √Δt) for a barrier above and H * e^(-0.5826 * σ * √Δt) for one below S
        - Every Knock-in of the double, partial-time and soft barriers = Black-Scholes price - Knock-out

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

        See: Reiner, E., & Rubinstein, M. (1991), "Breaking Down the Barriers", Risk, 4(8), 28-35; Ikeda, M., &
        Kunitomo, N. (1992), "Pricing Options with Curved Boundaries", Mathematical Finance, 2(4), 275-298; Heynen,
        R. C., & Kat, H. M. (1994), "Partial Barrier Options", The Journal of Financial Engineering, 3(3/4),
        253-274; Hart, I., & Ross, M. (1994), "Striking Continuity", Risk, 7(6), 51-56; Broadie, M., Glasserman,
        P., & Kou, S. (1997), "A Continuity Correction for Discrete Barrier Options", Mathematical Finance, 7(4),
        325-349; Haug, E. G. (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., section 4.17.

        Also known as: knock-in option, knock-out option, down-and-out, down-and-in,
        up-and-out, up-and-in option, double barrier option, partial barrier option, window barrier option,
        soft barrier option, discrete barrier option.

        Referenced against:
        - QuantLib, AnalyticDoubleBarrierEngine (double): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_double_barrier_engine.html
        - QuantLib, AnalyticPartialTimeBarrierOptionEngine (partial-time): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_partial_time_barrier_option_engine.html
        - QuantLib, AnalyticSoftBarrierEngine (soft): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_soft_barrier_engine.html

        Shares similar logic:
        - QuantLib, AnalyticBarrierEngine (single): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_barrier_engine.html
        - FinancePy, EquityBarrierOption (discretely watched): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_barrier_option.py#L28

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            barrier_percentage (float, optional): The barrier level as a percentage of the current stock price,
            above 0. Defaults to 0.9 which equals 90% of the current stock price. A down barrier needs a value below 1
            and an up barrier a value above 1; otherwise the barrier is already breached.
            barrier_direction (str, optional): The side of the barrier. Options are: "down" (below the stock
            price), "up" (above the stock price), "double" (barrier_percentage is then the lower and
            upper_barrier_percentage the upper barrier). Defaults to "down".
            knock_type (str, optional): The knock. Options are: "in" (knock-in), "out" (knock-out). Defaults to
            "out".
            rebate (float, optional): The fixed cash amount paid out if the option knocks out (or fails to knock
            in), for a single barrier watched over the whole life only. Defaults to 0.0.
            upper_barrier_percentage (float, optional): The upper barrier of a double barrier option as a
            percentage of the current stock price. Defaults to 1.1, 110% of the current stock price.
            barrier_window (str, optional): When the barrier is watched. Options are: "full" (the whole life),
            "start" (the first window_share of the life), "end" (the last window_share of the life). Defaults to
            "full".
            window_share (float, optional): The share of the life a "start" or "end" window covers, above 0 and at
            most 1. Defaults to 0.5.
            soft_barrier_percentage (float | None, optional): The other edge of a soft barrier band (the first is
            barrier_percentage), on the same side of the stock price. Defaults to None, a hard barrier.
            monitoring (str, optional): How often the barrier is checked. Options are: "continuous"
            (continuously), "daily" (252 times a year), "weekly" (52 times a year), "monthly" (12 times a year).
            Defaults to "continuous".
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
            pd.DataFrame: Barrier option values containing the tickers and strike prices as the index and the
            time to expiration as the columns.

        Raises:
            InvalidArgumentError: If barrier_percentage is not above 0 or puts the barrier at or beyond the current stock price for the
                given barrier_direction (at or below 1 for "up", at or above 1 for "down", or outside the lower and
                upper barrier for "double"), since such a barrier is breached at the start and every price would
                be the rebate (knock-out) or the vanilla price (knock-in); or when a rebate is combined with a
                double, partial-time or soft barrier, or two of those are combined, or monitoring is not one of its values.

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
        |            590 |      70.2113 |      70.5597 |      71.0164 |
        |            600 |      63.6597 |      65.6622 |      66.9713 |
        |            610 |      57.1768 |      60.7954 |      62.9442 |
        |            620 |      50.8683 |      56.0068 |      58.9631 |
        |            630 |      44.827  |      51.3397 |      55.0541 |
        |            640 |      39.1295 |      46.8319 |      51.2401 |
        |            650 |      33.8345 |      42.5151 |      47.5413 |
        |            660 |      28.9823 |      38.4145 |      43.9749 |
        |            670 |      24.5957 |      34.5495 |      40.5549 |
        |            680 |      20.6814 |      30.9334 |      37.2922 |
        |            690 |      17.2323 |      27.5738 |      34.1949 |
        |            700 |      14.23   |      24.4735 |      31.2686 |
        |            710 |      11.6473 |      21.6308 |      28.5162 |
        |            720 |       9.4508 |      19.0403 |      25.9385 |
        """
        check_option("barrier_direction", barrier_direction, ("down", "up", "double"))
        check_option("barrier_window", barrier_window, ("full", "start", "end"))
        check_option(
            "monitoring",
            monitoring,
            ("continuous", *path_dependent_model.MONITORING_INTERVALS),
        )
        if not barrier_percentage > 0:
            raise InvalidArgumentError(
                "barrier_percentage is the barrier as a fraction of the stock price, above 0 (0.9 for a barrier "
                f"10% below it), not {barrier_percentage!r}. Please pass a positive barrier_percentage."
            )
        variants = [
            name
            for name, chosen in (
                ("a double barrier", barrier_direction == "double"),
                ("a partial-time barrier", barrier_window != "full"),
                ("a soft barrier", soft_barrier_percentage is not None),
            )
            if chosen
        ]
        if len(variants) > 1:
            raise InvalidArgumentError(
                f"Choose one of {' and '.join(variants)}; they cannot be combined."
            )
        if variants and rebate:
            raise InvalidArgumentError(
                f"A rebate is only available for a single hard barrier, not for {variants[0]}."
            )
        if (
            barrier_direction == "double"
            and not barrier_percentage < 1 < upper_barrier_percentage
        ):
            raise InvalidArgumentError(
                f"A double barrier needs barrier_percentage below 1 and upper_barrier_percentage above 1 "
                f"(received {barrier_percentage} and {upper_barrier_percentage})."
            )
        if (
            soft_barrier_percentage is not None
            and (soft_barrier_percentage - 1) * (barrier_percentage - 1) <= 0
        ):
            raise InvalidArgumentError(
                f"Both edges of a soft barrier must lie on the same side of the current stock price "
                f"(received {barrier_percentage} and {soft_barrier_percentage})."
            )

        # The barrier is set relative to the spot, so a barrier on the wrong side of it is breached
        # before the option starts. The prices would then be a flat rebate or the vanilla price,
        # which hides a mistaken barrier_percentage or barrier_direction instead of reporting it.
        if barrier_direction == "up" and barrier_percentage <= 1:
            raise InvalidArgumentError(
                f"An up barrier must lie above the current stock price, so barrier_percentage must be above 1 "
                f"(received {barrier_percentage}). Use barrier_direction='down' for a barrier below the stock price."
            )
        if barrier_direction == "down" and barrier_percentage >= 1:
            raise InvalidArgumentError(
                f"A down barrier must lie below the current stock price, so barrier_percentage must be below 1 "
                f"(received {barrier_percentage}). Use barrier_direction='up' for a barrier above the stock price."
            )

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

        barrier_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            try:
                barrier_option[ticker] = {}
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )
                ticker_price = float(stock_price.loc[ticker])
                ticker_volatility = float(volatility.loc[ticker])
                interval = path_dependent_model.MONITORING_INTERVALS.get(
                    monitoring, 0.0
                )
                # A discretely watched barrier prices as a continuous one moved away from the stock price.
                barriers = [
                    path_dependent_model.get_discrete_monitoring_barrier(
                        ticker_price * percentage,
                        ticker_price,
                        ticker_volatility,
                        interval,
                    )
                    for percentage in (
                        barrier_percentage,
                        upper_barrier_percentage,
                        (
                            soft_barrier_percentage
                            if soft_barrier_percentage is not None
                            else barrier_percentage
                        ),
                    )
                ]
                barrier_option[ticker] = helpers.evaluate_on_grid(
                    functools.partial(
                        path_dependent_model.get_barrier_option_value,
                        stock_price=ticker_price,
                        risk_free_rate=float(risk_free_rate),
                        volatility=ticker_volatility,
                        dividend_yield=float(dividend_yield_value[ticker]),
                        put_option=put_option,
                        knock_type=knock_type,
                        barrier_direction=barrier_direction,
                        barrier=barriers[0],
                        upper_barrier=barriers[1],
                        second_barrier=(
                            barriers[2] if soft_barrier_percentage is not None else None
                        ),
                        barrier_window=barrier_window,
                        window_share=window_share,
                        rebate=rebate,
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No barrier option value for %s: %s", ticker, error)

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

    @handle_errors
    def get_asian_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        method: Literal[
            "geometric", "turnbull_wakeman", "curran", "monte_carlo"
        ] = "geometric",
        fixings: int | None = None,
        strike_price_range: float = 0.25,
        strike_step_size: float = 5,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        simulations: int = 10_000,
        time_steps: int = 100,
        seed: int | None = 0,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the price of an average price Asian option: what an option on each ticker is worth whose payoff
        depends on the average stock price over its life rather than the price at expiration.

        The geometric average is priced in closed form (Kemna & Vorst, 1990) and the arithmetic average, the kind
        that is traded, with the approximations of Turnbull and Wakeman (1991) or Curran (1994), or a Monte Carlo
        simulation with the geometric option as control variate.

        An Asian option's payoff, max(A - K, 0) for a call and max(K - A, 0) for a put, depends on the average
        A of the stock price rather than its price at expiration, which typically makes it cheaper than the
        equivalent vanilla option (the averaging reduces variance). The average is taken continuously, or over
        a number of equally spaced fixings, the first a 1/n-th of the life from now and the
        last at expiration. The methods are:

        - Geometric: the geometric average of a lognormal price is lognormal, so this is
          a Black-Scholes option on it. It never exceeds the arithmetic average, so it is a lower bound for
          the arithmetic call and an upper bound for the arithmetic put.
        - Turnbull-Wakeman: takes the arithmetic average to be lognormal with its
          exact first two moments. Fast and close at low volatilities and short maturities, but it loses
          accuracy as both grow (off by up to 0.8 on a stock price of 100 at 60% volatility over two years).
          For a continuous average that starts today, Levy (1992) matches the same two moments with a
          lognormal and gives the same price; the two differ only for an average that has already started.
        - Curran: conditions on the geometric average, which is exact for the paths on which
          the geometric average ends above the strike (there A ≥ G ≥ K) and approximates the others. The most
          accurate of the approximations: within 0.05 of a simulation with 200,000 paths on a stock price of 100
          up to 60% volatility and two years, and within 0.002 at 15% volatility.
        - Monte Carlo: simulates the arithmetic average along each path and corrects the
          simulation error with the geometric option, whose price is known, which makes it about 50 to 100
          times more precise than a plain simulation. A continuous average is approximated by a number of
          fixings.

        The formula is as follows:

        - Geometric: E[ln G] = ln S + (b - σ²/2) * T / 2 and Var[ln G] = σ² * T / 3 for a continuous average,
          E[ln G] = ln S + (b - σ²/2) * T * (n + 1) / (2n) and Var[ln G] = σ² * T * (n + 1)(2n + 1) / (6n²) for
          n fixings, Call = e^(-rT) * (e^(E[ln G] + Var[ln G] / 2) * N(d1) - K * N(d2)),
          d1 = (E[ln G] - ln K + Var[ln G]) / √Var[ln G], d2 = d1 - √Var[ln G]
        - Turnbull-Wakeman: M1 = E[A] / S, M2 = E[A²] / S², F_A = S * M1, v = ln(M2 / M1²),
          Call = e^(-rT) * (F_A * N(d1) - K * N(d2)), d1 = (ln(F_A / K) + v / 2) / √v, d2 = d1 - √v,
          with M1 = (e^(bT) - 1) / (bT) and M2 = 2 / T² * ∫_0^T e^(bt) * (e^((b + σ²)t) - 1) / (b + σ²) dt for a
          continuous average and M1 = Σ e^(b t_i) / n, M2 = Σ_i Σ_j e^(b(t_i + t_j) + σ² min(t_i, t_j)) / n²
          for fixings t_i = iT / n
        - Curran: μ_i = ln S + (b - σ²/2) * t_i, σ_i² = σ² * t_i, μ_G = Σ μ_i / n,
          σ_G² = σ² * Σ_i Σ_j min(t_i, t_j) / n², σ_iG = σ² * Σ_j min(t_i, t_j) / n,
          K̂ = 2K - Σ e^(μ_i + σ_iG / σ_G² * (ln K - μ_G) + (σ_i² - σ_iG² / σ_G²) / 2) / n,
          Call = e^(-rT) * [Σ e^(μ_i + σ_i² / 2) * N((μ_G - ln K̂) / σ_G + σ_iG / σ_G) / n - K * N((μ_G - ln K̂) / σ_G)],
          with integrals over [0, T] in place of the sums for a continuous average
        - Monte Carlo: S(t_i) = S(t_i-1) * e^((b - σ²/2) * Δt + σ * √Δt * Z) at n fixings (time_steps for a
          continuous average), Y = e^(-rT) * max(A - K, 0) and X = e^(-rT) * max(G - K, 0) per path,
          Call = MEAN(Y) - β * (MEAN(X) - Geometric Call), β = Cov(Y, X) / Var(X) (ddof=1)
        - Put = Call - e^(-rT) * (E[A] - K) by put-call parity of the average (the geometric put and the
          Monte Carlo put are priced directly)

        Where S is the stock price (the traded close on the pricing date), K is the strike price, r is the
        continuously compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)),
        b = r - q is the cost of carry, σ is the annualized volatility (the standard deviation of the daily
        returns over the trailing 252 trading days times √252), T is the time to expiration in years
        (days / 365), n is the number of fixings, G and A are the geometric and arithmetic average and N is
        the cumulative standard normal distribution.

        The Monte Carlo prices are reproducible with a seed, and all strike prices of an expiry are priced
        on the same paths. Its standard error is around 0.1% of the price with the defaults.

        See: Kemna, A. G. Z., & Vorst, A. C. F. (1990), "A Pricing Method for Options Based on Average Asset
        Values", Journal of Banking & Finance, 14(1), 113-129; Turnbull, S. M., & Wakeman, L. M. (1991), "A
        Quick Algorithm for Pricing European Average Options", Journal of Financial and Quantitative Analysis,
        26(3), 377-389; Levy, E. (1992), "Pricing European Average Rate Currency Options", Journal of
        International Money and Finance, 11(5), 474-491; Curran, M. (1994), "Valuing Asian and Portfolio Options
        by Conditioning on the Geometric Mean Price", Management Science, 40(12), 1705-1711; Haug, E. G. (2007),
        The Complete Guide to Option Pricing Formulas, 2nd ed., section 4.20.

        Also known as: Asian option, average rate option, average price option, arithmetic Asian option,
        geometric Asian option.

        Referenced against:
        - QuantLib, AnalyticDiscreteGeometricAveragePriceAsianEngine (geometric, fixings): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_discrete_geometric_average_price_asian_engine.html
        - QuantLib, TurnbullWakemanAsianEngine (Turnbull-Wakeman): https://www.quantlib.org/reference/class_quant_lib_1_1_turnbull_wakeman_asian_engine.html
        - QuantLib, ContinuousArithmeticAsianLevyEngine (Turnbull-Wakeman, continuous): https://www.quantlib.org/reference/class_quant_lib_1_1_continuous_arithmetic_asian_levy_engine.html
        - QuantLib, MCDiscreteArithmeticAPEngine (Curran, Monte Carlo): https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_discrete_arithmetic_a_p_engine.html

        Shares similar logic:
        - QuantLib, AnalyticContinuousGeometricAveragePriceAsianEngine (geometric, continuous): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_geometric_average_price_asian_engine.html
        - FinancePy, EquityAsianOption.value_geometric (geometric): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_asian_option.py#L180
        - FinancePy, EquityAsianOption.value_turnbull_wakeman (Turnbull-Wakeman): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_asian_option.py#L332
        - FinancePy, EquityAsianOption.value_curran (Curran): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_asian_option.py#L256

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            method (str, optional): The pricing method. Options are: "geometric" (the geometric average),
            "turnbull_wakeman" (Turnbull-Wakeman, arithmetic average), "curran" (Curran, arithmetic average),
            "monte_carlo" (Monte Carlo, arithmetic average). Defaults to "geometric".
            fixings (int | None, optional): The number of equally spaced fixings of the average, at least 1, the last on the
            expiration date. Defaults to None, which averages continuously.
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
            simulations (int, optional): The number of simulated paths of the Monte Carlo method, at least 1. Defaults
            to 10,000.
            time_steps (int, optional): The fixings that approximate a continuous average with the Monte Carlo method,
            at least 1 (fixings, when given, take their place). Defaults to 100.
            seed (int | None, optional): The seed of the random number generator of the Monte Carlo method, which
            makes the results reproducible; None draws new paths on every call. Defaults to 0.
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
            pd.DataFrame: Asian option values containing the tickers and strike prices as the index and the time to
            expiration as the columns.

        Raises:
            InvalidArgumentError: If method is not one of the methods or fixings, simulations or time_steps is not a whole
            number of at least 1.

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
            method="curran", fixings=21, strike_price_range=0.1, strike_step_size=2, expiration_time_range=91
        )

        asian_option.loc["XOM", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            108 |      12.3465 |      12.4086 |      12.5336 |
        |            110 |      10.3676 |      10.5055 |      10.7123 |
        |            112 |       8.4205 |       8.678  |       8.9842 |
        |            114 |       6.5487 |       6.9662 |       7.3786 |
        |            116 |       4.8214 |       5.4133 |       5.9236 |
        |            118 |       3.3215 |       4.0579 |       4.6412 |
        |            120 |       2.1184 |       2.9257 |       3.5445 |
        |            122 |       1.2399 |       2.0242 |       2.6359 |
        |            124 |       0.6617 |       1.3417 |       1.9077 |
        |            126 |       0.3207 |       0.8511 |       1.3431 |
        |            128 |       0.1408 |       0.5164 |       0.9198 |
        |            130 |       0.056  |       0.2997 |       0.6128 |
        |            132 |       0.0202 |       0.1665 |       0.3973 |
        |            134 |       0.0066 |       0.0885 |       0.2507 |
        """
        methods = ("geometric", "turnbull_wakeman", "curran", "monte_carlo")
        check_option("method", method, methods)
        validate_window("simulations", simulations)
        validate_window("time_steps", time_steps)
        validate_window("fixings", fixings)

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
            # A Monte Carlo value evaluates every path with its control variate per strike:
            # about 1 millisecond a value at 10,000 paths, so it counts as a value per 500
            # paths.
            values_per_strike=expiration_time_range
            * (-(-simulations // 500) if method == "monte_carlo" else 1),
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

        asian_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            asian_option[ticker] = {strike_price: {} for strike_price in strike_prices}
            dividend_yield_value[ticker] = (
                dividend_yield
                if dividend_yield is not None
                else self._dividend_yield_on(ticker, start_date)
            )
            inputs = {
                "stock_price": float(stock_price.loc[ticker]),
                "risk_free_rate": float(risk_free_rate),
                "volatility": float(volatility.loc[ticker]),
                "dividend_yield": float(dividend_yield_value[ticker]),
                "put_option": put_option,
            }

            if method == "geometric":
                # Every strike price and time to expiration in one call on arrays.
                asian_option[ticker] = helpers.evaluate_on_grid(
                    lambda strike_price, time_to_expiration, inputs=inputs: (
                        exotics_model.get_asian_option(
                            **inputs,
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                        )
                        if fixings is None
                        else path_dependent_model.get_geometric_asian_option(
                            **inputs,
                            strike_price=strike_price,
                            time_to_expiration=time_to_expiration,
                            fixings=fixings,
                        )
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
                continue

            for time_to_expiration in time_to_expiration_list:
                if method == "monte_carlo":
                    # Every strike is priced on the same simulated paths of an expiry.
                    prices, _ = path_dependent_model.get_monte_carlo_asian_options(
                        **inputs,
                        strike_prices=np.asarray(strike_prices, dtype=float),
                        time_to_expiration=time_to_expiration,
                        fixings=fixings if fixings is not None else time_steps,
                        simulations=simulations,
                        seed=seed,
                    )
                    for strike_price, price in zip(strike_prices, prices, strict=True):
                        asian_option[ticker][strike_price][time_to_expiration] = float(
                            price
                        )
                    continue

                # The moments (Turnbull-Wakeman) and the conditioning on the geometric
                # average (Curran) of an expiration are shared by all its strikes.
                values = {
                    "turnbull_wakeman": path_dependent_model.get_turnbull_wakeman,
                    "curran": path_dependent_model.get_curran,
                }[method](
                    **inputs,
                    strike_price=np.asarray(strike_prices, dtype=float),
                    time_to_expiration=time_to_expiration,
                    fixings=fixings,
                )
                for strike_price, value in zip(strike_prices, values, strict=True):
                    asian_option[ticker][strike_price][time_to_expiration] = float(
                        value
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

    @handle_errors
    def get_lookback_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        strike_type: str = "fixed",
        lookback_period: float = 1.0,
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
        Calculate the closed-form price of a lookback option: what an option on each ticker is worth whose payoff
        depends on the highest or lowest stock price over its life.

        The formulas are those of Conze and Viswanathan (1991) for a fixed strike, Goldman, Sosin and Gatto (1979)
        for a floating strike and Heynen and Kat (1994) for a partial-time lookback.

        - Fixed strike: the call pays max(S_max - K, 0) and the put max(K - S_min, 0), the
          best price reached against the strike.
        - Floating strike: the call pays S(T) - S_min, buying at the lowest price, and the
          put S_max - S(T), selling at the highest. It has no strike, so the result has a row per ticker.
        - Partial-time (a lookback period below 1): the extreme is only watched during part of the life, the
          first part for a floating strike and the last part for a fixed strike, which makes
          the option cheaper.

        The options start on the pricing date, so the highest and lowest price so far are the stock price.

        The formula is as follows:

        - x = (ln(S / E) + (b + σ² / 2) * T) / (σ√T), with E the extreme so far: E = max(S, K) for a fixed call,
          min(S, K) for a fixed put and S for a floating strike
        - Fixed Call = e^(-rT) * (E - K) + S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T)
          + S * e^(-rT) * σ² / (2b) * [-(S / E)^(-2b / σ²) * N(x - 2b√T / σ) + e^(bT) * N(x)]
        - Fixed Put = e^(-rT) * (K - E) - S * e^((b - r)T) * N(-x) + E * e^(-rT) * N(-x + σ√T)
          + S * e^(-rT) * σ² / (2b) * [(S / E)^(-2b / σ²) * N(-x + 2b√T / σ) - e^(bT) * N(-x)]
        - Floating Call = S * e^((b - r)T) * N(x) - E * e^(-rT) * N(x - σ√T)
          + S * e^(-rT) * σ² / (2b) * [(S / E)^(-2b / σ²) * N(-x + 2b√T / σ) - e^(bT) * N(-x)]
        - Floating Put = E * e^(-rT) * N(-x + σ√T) - S * e^((b - r)T) * N(-x)
          + S * e^(-rT) * σ² / (2b) * [-(S / E)^(-2b / σ²) * N(x - 2b√T / σ) + e^(bT) * N(x)]
        - Partial Fixed, watched from t1 = (1 - Lookback Period) * T: e^(-r t1) * E[Fixed(S(t1), T - t1)], in closed
          form with the bivariate normal distribution
        - Partial Floating, watched up to t1 = Lookback Period * T: S * e^((b - r) t1) * E'[c(e^(-X), T - t1)] for a call,
          with c the Black-Scholes call on a unit stock price and X = ln(S(t1) / S_min(t1)), which is distributed as the
          running maximum of a Brownian motion with drift b + σ² / 2 (time reversal), integrated over its density

        Where S is the stock price (the traded close on the pricing date), K is the strike price, r is the
        continuously compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)),
        b = r - q is the cost of carry, σ is the annualized volatility (the standard deviation of the daily returns
        over the trailing 252 trading days times √252), T is the time to expiration in years (days / 365) and N is
        the cumulative standard normal distribution. Where b = 0 the formulas are taken at b = ±0.00001 and
        averaged, which equals their limit to within O(b²).

        See: Goldman, M. B., Sosin, H. B., & Gatto, M. A. (1979), "Path Dependent Options: Buy at the Low, Sell at
        the High", The Journal of Finance, 34(5), 1111-1127; Conze, A., & Viswanathan (1991), "Path Dependent
        Options: The Case of Lookback Options", The Journal of Finance, 46(5), 1893-1907; Heynen, R. C., & Kat, H.
        M. (1994), "Selective Memory", Risk, 7(11), 73-76; Haug, E. G. (2007), The Complete Guide to Option Pricing
        Formulas, 2nd ed., section 4.15.

        Also known as: lookback option, fixed strike lookback, floating strike lookback, partial lookback, hindsight
        option, no-regret option.

        Referenced against:
        - QuantLib, AnalyticContinuousFixedLookbackEngine (fixed strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_fixed_lookback_engine.html
        - QuantLib, AnalyticContinuousFloatingLookbackEngine (floating strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_floating_lookback_engine.html
        - QuantLib, AnalyticContinuousPartialFixedLookbackEngine (partial fixed strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_partial_fixed_lookback_engine.html
        - QuantLib, AnalyticContinuousPartialFloatingLookbackEngine (partial floating strike): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_continuous_partial_floating_lookback_engine.html

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_type (str, optional): The strike. Options are: "fixed" (fixed strike), "floating" (floating
            strike). Defaults to "fixed".
            lookback_period (float, optional): The share of the life over which the extreme is watched, above 0 and
            at most 1: the start of the life for a floating strike and the end for a fixed strike. Defaults to 1,
            the whole life.
            strike_price_range (float): The percentage range to use for the strike prices, between 0 and 1. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100. Not used for a
            floating strike.
            strike_step_size (float): The step size to use for the strike prices, above 0. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, ..., 120 and 125 if the current stock price is 100. Not used for a
            floating strike.
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
                time to expiration columns for each ticker (and strike price). Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Lookback option values containing the tickers and strike prices (only the tickers for a
            floating strike) as the index and the time to expiration as the columns.

        Raises:
            InvalidArgumentError: If strike_type is not "fixed" or "floating" or lookback_period is not above 0 and at most 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        lookback_option = toolkit.options.get_lookback_option(
            strike_price_range=0.1, strike_step_size=5, expiration_time_range=91
        )

        lookback_option.loc["NVDA", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            165 |      43.8427 |      53.7637 |      61.6312 |
        |            170 |      38.8596 |      48.7974 |      56.6817 |
        |            175 |      33.8765 |      43.8312 |      51.7322 |
        |            180 |      28.8934 |      38.8649 |      46.7827 |
        |            185 |      23.9103 |      33.8986 |      41.8332 |
        |            190 |      19.1196 |      29.0713 |      36.9989 |
        |            195 |      15.0553 |      24.7714 |      32.6025 |
        |            200 |      11.6939 |      20.9915 |      28.6406 |
        |            205 |       8.9622 |      17.694  |      25.0864 |
        |            210 |       6.7797 |      14.8381 |      21.9116 |
        TABLE_LOOKBACK
        """
        check_option("strike_type", strike_type, ("fixed", "floating"))
        if not 0 < lookback_period <= 1:
            raise InvalidArgumentError(
                f"lookback_period must be above 0 and at most 1, received {lookback_period!r}."
            )

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

        lookback_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            try:
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )
                inputs = {
                    "stock_price": float(stock_price.loc[ticker]),
                    "risk_free_rate": float(risk_free_rate),
                    "volatility": float(volatility.loc[ticker]),
                    "dividend_yield": float(dividend_yield_value[ticker]),
                    "put_option": put_option,
                    "strike_type": strike_type,
                    "lookback_period": float(lookback_period),
                }

                # A floating strike has no strike price, so it is priced once per expiration.
                lookback_option[ticker] = helpers.evaluate_on_grid(
                    lambda strike, time, inputs=inputs: path_dependent_model.get_lookback_option(
                        **inputs, strike_price=strike, time_to_expiration=time
                    ),
                    (
                        strike_prices
                        if strike_type == "fixed"
                        else [inputs["stock_price"]]
                    ),
                    time_to_expiration_list,
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No lookback option value for %s: %s", ticker, error)

        if strike_type == "fixed":
            lookback_option_df = helpers.create_greek_dataframe(
                greek_dictionary=lookback_option,
                start_date=start_date,
            )
        else:
            lookback_option_df = helpers.create_expiration_dataframe(
                value_dictionary={
                    ticker: next(iter(values.values()))
                    for ticker, values in lookback_option.items()
                },
                start_date=start_date,
            )

        lookback_option_df = lookback_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            lookback_option_df = calculate_standardization(
                dataset=lookback_option_df,
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

        return lookback_option_df

    @handle_errors
    def get_touch_option(
        self,
        start_date: str | None = None,
        option_type: str = "one-touch",
        barrier_percentage: float = 0.9,
        upper_barrier_percentage: float = 1.1,
        payment: str = "expiration",
        cash_payout: float = 1.0,
        monitoring: str = "continuous",
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the closed-form price of a one-touch, no-touch, double one-touch or double no-touch option: what a
        binary option on whether the stock price of each ticker touches a barrier before expiration is worth.

        The formulas are those of Reiner and Rubinstein (1991) for a single barrier and Hui (1996) for two barriers.

        A touch option is a binary option on whether the stock price touches a barrier before expiration:

        - One-touch: pays the cash payout once the barrier at barrier_percentage of the
          stock price is touched, at that moment or at expiration (see payment).
        - No-touch: pays the cash payout at expiration when the barrier is never touched.
        - Double one-touch: pays at expiration when either the lower barrier
          (barrier_percentage) or the upper barrier (upper_barrier_percentage) is touched.
        - Double no-touch: pays at expiration when the stock price stays between
          the two barriers.

        A barrier below the stock price (barrier_percentage below 1) is touched when the price falls to it, one
        above (barrier_percentage above 1) when the price rises to it. The price of a one-touch paid at expiration
        is the discounted risk-neutral probability of touching the barrier, so with a cash payout of 1 and the
        discounting removed it reads as that probability.

        The formula is as follows:

        - P(touch) = N(η(h - νT) / (σ√T)) + (H / S)^(2μ) * N(η(h + νT) / (σ√T))
        - One-touch, paid at expiration = Cash * e^(-rT) * P(touch), with Cash defaulting to 1
        - One-touch, paid at the hit = Cash * [(H / S)^(μ + λ) * N(ηz) + (H / S)^(μ - λ) * N(ηz - 2ηλσ√T)],
          z = h / (σ√T) + λσ√T
        - No-touch = Cash * e^(-rT) * (1 - P(touch))
        - P(stay) = Σ_n [e^(2nZν / σ²) * (N((u - 2nZ - νT) / (σ√T)) - N((l - 2nZ - νT) / (σ√T)))
          - e^(2ν(l - nZ) / σ²) * (N((u - 2l + 2nZ - νT) / (σ√T)) - N((-l + 2nZ - νT) / (σ√T)))]
        - Double no-touch = Cash * e^(-rT) * P(stay)
        - Double one-touch = Cash * e^(-rT) * (1 - P(stay))
        - Discrete monitoring: H → H * e^(0.5826 * σ * √Δt) for a barrier above and H * e^(-0.5826 * σ * √Δt) for
          one below the stock price, with Δt = 1/252, 1/52 or 1/12 (Broadie, Glasserman & Kou, 1997)

        Where S is the stock price (the traded close on the pricing date), H is the barrier, h = ln(H / S),
        l = ln(L / S) and u = ln(U / S) with L and U the lower and upper barrier, Z = u - l, r is the continuously
        compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)), ν = r - q - σ² / 2,
        μ = ν / σ², λ = √(μ² + 2r / σ²), η = 1 for a barrier below and -1 for one above the stock price, σ is the
        annualized volatility (the standard deviation of the daily returns over the trailing 252 trading days times
        √252), T is the time to expiration in years (days / 365) and N is the cumulative standard normal
        distribution. P(stay) is the probability of staying between the barriers from the method of images,
        summed until its terms are below machine precision; Hui (1996) writes the same probability as a sine
        series.

        See: Reiner, E., & Rubinstein, M. (1991), "Unscrambling the Binary Code", Risk, 4(9), 75-83; Hui, C. H.
        (1996), "One-Touch Double Barrier Binary Option Values", Applied Financial Economics, 6(4), 343-346;
        Broadie, M., Glasserman, P., & Kou, S. (1997), "A Continuity Correction for Discrete Barrier Options",
        Mathematical Finance, 7(4), 325-349; Haug, E. G. (2007), The Complete Guide to Option Pricing Formulas, 2nd
        ed., section 4.19.

        Also known as: one-touch option, no-touch option, double one-touch, double no-touch, American digital
        option, barrier binary option, touch probability.

        Referenced against:
        - QuantLib, AnalyticDigitalAmericanEngine (one-touch): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_digital_american_engine.html
        - QuantLib, AnalyticBinaryBarrierEngine (no-touch): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_binary_barrier_engine.html
        - QuantLib, AnalyticDoubleBarrierBinaryEngine (double one-touch, double no-touch): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_double_barrier_binary_engine.html

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            option_type (str, optional): The kind of touch option. Options are: "one-touch" (one-touch),
            "no-touch" (no-touch), "double-one-touch" (double one-touch), "double-no-touch" (double no-touch).
            Defaults to "one-touch".
            barrier_percentage (float, optional): The barrier (the lower barrier of a double touch option) as a
            percentage of the current stock price, above 0. Defaults to 0.9, 90% of the current stock price.
            upper_barrier_percentage (float, optional): The upper barrier of a double touch option as a percentage of
            the current stock price. Defaults to 1.1, 110% of the current stock price.
            payment (str, optional): When a one-touch option pays. Options are: "expiration" (at expiration),
            "hit" (when the barrier is touched). The other touch options pay at expiration. Defaults to
            "expiration".
            cash_payout (float, optional): The cash amount paid. Defaults to 1.0.
            monitoring (str, optional): How often the barrier is checked. Options are: "continuous"
            (continuously), "daily" (252 times a year), "weekly" (52 times a year), "monthly" (12 times a year).
            Defaults to "continuous".
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
                time to expiration columns for each ticker. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Touch option values containing the tickers as the index and the time to expiration as the
            columns.

        Raises:
            InvalidArgumentError: If option_type, payment or monitoring is not one of its values, a touch option other
                than the one-touch is asked to pay at the hit, barrier_percentage is not above 0, the barrier is at the
                current stock price or a double touch option's barriers do not lie on either side of it.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["TSLA", "KO"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        touch_option = toolkit.options.get_touch_option(
            option_type="double-no-touch", barrier_percentage=0.9, upper_barrier_percentage=1.1,
            expiration_time_range=91,
        )

        touch_option[["2026-01-07", "2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        | Ticker   |   2026-01-07 |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |:---------|-------------:|-------------:|-------------:|-------------:|
        | TSLA     |       0.4951 |       0.0223 |       0.0004 |       0      |
        | KO       |       0.9991 |       0.9028 |       0.6768 |       0.4958 |
        TABLE_TOUCH
        """
        option_types = ("one-touch", "no-touch", "double-one-touch", "double-no-touch")
        check_option("option_type", option_type, option_types)
        check_option("payment", payment, ("expiration", "hit"))
        if payment == "hit" and option_type != "one-touch":
            raise InvalidArgumentError(
                f"Only a one-touch option pays at the hit; a {option_type} option pays at expiration."
            )
        check_option(
            "monitoring",
            monitoring,
            ("continuous", *path_dependent_model.MONITORING_INTERVALS),
        )
        if not barrier_percentage > 0:
            raise InvalidArgumentError(
                "barrier_percentage is the barrier as a fraction of the stock price, above 0 (0.9 for a barrier "
                f"10% below it), not {barrier_percentage!r}. Please pass a positive barrier_percentage."
            )
        double = option_type.startswith("double")
        if double and not barrier_percentage < 1 < upper_barrier_percentage:
            raise InvalidArgumentError(
                f"A double touch option needs barrier_percentage below 1 and upper_barrier_percentage above 1 "
                f"(received {barrier_percentage} and {upper_barrier_percentage})."
            )
        if not double and barrier_percentage == 1:
            raise InvalidArgumentError(
                "The barrier is at the current stock price, so it is touched already; use a barrier_percentage "
                "below 1 for a barrier below the stock price or above 1 for one above it."
            )

        start_date = self._pricing_date(start_date)
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )
        interval = path_dependent_model.MONITORING_INTERVALS.get(monitoring, 0.0)

        touch_option: dict[str, dict[float, float]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker in self._tickers:
            try:
                # A ticker without a price on the date is left out, as on the strike price grids.
                if pd.isna(stock_price.get(ticker)):
                    logger.warning(
                        "%s has no price on %s, so it is left out of the option values.",
                        ticker,
                        stock_price.name,
                    )
                    continue
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )
                ticker_price = float(stock_price.loc[ticker])
                ticker_volatility = float(volatility.loc[ticker])
                lower_barrier, upper_barrier = (
                    path_dependent_model.get_discrete_monitoring_barrier(
                        ticker_price * percentage,
                        ticker_price,
                        ticker_volatility,
                        interval,
                    )
                    for percentage in (barrier_percentage, upper_barrier_percentage)
                )
                inputs = {
                    "stock_price": ticker_price,
                    "risk_free_rate": float(risk_free_rate),
                    "volatility": ticker_volatility,
                    "dividend_yield": float(dividend_yield_value[ticker]),
                    "option_type": option_type,
                    "cash_payout": float(cash_payout),
                }

                touch_option[ticker] = {
                    time_to_expiration: (
                        path_dependent_model.get_double_touch_option(
                            **inputs,
                            lower_barrier=lower_barrier,
                            upper_barrier=upper_barrier,
                            time_to_expiration=time_to_expiration,
                        )
                        if double
                        else path_dependent_model.get_touch_option(
                            **inputs,
                            barrier=lower_barrier,
                            time_to_expiration=time_to_expiration,
                            payment=payment,
                        )
                    )
                    for time_to_expiration in time_to_expiration_list
                }
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No touch option value for %s: %s", ticker, error)

        touch_option_df = helpers.create_expiration_dataframe(
            value_dictionary=touch_option,
            start_date=start_date,
        )

        touch_option_df = touch_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            touch_option_df = calculate_standardization(
                dataset=touch_option_df,
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

        return touch_option_df

    @handle_errors
    def get_chooser_option(
        self,
        start_date: str | None = None,
        choice_period: float = 0.5,
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
        Calculate the closed-form price of a simple chooser option: what the right to decide at the choice time
        whether the option on each ticker becomes a European call or a European put is worth, with the formula of
        Rubinstein (1991).

        The call and the put share the strike price and expiration, so the chooser is worth at least as much as
        either and at most as much as the straddle of both.

        The choice is made after a share of the life, the choice period: at a choice period of 0 the chooser is
        the larger of the call and the put today, at 1 a straddle. By put-call parity at the choice time t the chooser
        is a call on (K, T) plus a put on (K * e^(-(r - q)(T - t)), t), scaled by e^(-q(T - t)).

        The formula is as follows:

        - t = Choice Period * T
        - d = (ln(S / K) + (r - q + σ² / 2) * T) / (σ√T)
        - y = (ln(S / K) + (r - q) * T + σ² * t / 2) / (σ√t)
        - Chooser = S * e^(-qT) * N(d) - K * e^(-rT) * N(d - σ√T) - S * e^(-qT) * N(-y) + K * e^(-rT) * N(-y + σ√t)

        Where S is the stock price (the traded close on the pricing date), K is the strike price, r is the
        continuously compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)), σ is the
        annualized volatility (the standard deviation of the daily returns over the trailing 252 trading days times
        √252), T is the time to expiration in years (days / 365) and N is the cumulative standard normal
        distribution.

        A complex chooser, which chooses between a call and a put with their own strike and expiration, is priced
        by the model function financetoolkit.options.pricing.compound_model.get_chooser_option with
        put_strike_price and put_time_to_expiration.

        See: Rubinstein, M. (1991), "Options for the Undecided", Risk, 4(4), 43; Haug, E. G. (2007), The Complete
        Guide to Option Pricing Formulas, 2nd ed., section 4.10.

        Also known as: chooser option, as-you-like-it option, pay now choose later option.

        Referenced against:
        - QuantLib, AnalyticSimpleChooserEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_simple_chooser_engine.html

        Shares similar logic:
        - FinancePy, EquityChooserOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_chooser_option.py#L67

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            choice_period (float, optional): The share of the life after which the holder chooses between the call
            and the put, at least 0 and at most 1. Defaults to 0.5, halfway.
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
            pd.DataFrame: Chooser option values containing the tickers and strike prices as the index and the time
            to expiration as the columns.

        Raises:
            InvalidArgumentError: If choice_period is not at least 0 and at most 1.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        chooser_option = toolkit.options.get_chooser_option(
            strike_price_range=0.1, strike_step_size=5, expiration_time_range=91
        )

        chooser_option.loc["NVDA", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            165 |      25.515  |      30.5881 |      35.0495 |
        |            170 |      22.4621 |      28.3685 |      33.27   |
        |            175 |      20.1357 |      26.7519 |      32.0089 |
        |            180 |      18.6412 |      25.7699 |      31.2783 |
        |            185 |      18.0413 |      25.4352 |      31.0788 |
        |            190 |      18.3475 |      25.7412 |      31.4009 |
        |            195 |      19.5216 |      26.6638 |      32.226  |
        |            200 |      21.485  |      28.1649 |      33.5281 |
        |            205 |      24.1331 |      30.1962 |      35.2753 |
        |            210 |      27.3507 |      32.7032 |      37.4324 |
        """
        if not 0 <= choice_period <= 1:
            raise InvalidArgumentError(
                f"choice_period must be at least 0 and at most 1, received {choice_period!r}."
            )

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

        chooser_option: dict[str, dict[float, dict[float, float]]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker, strike_prices in strike_prices_per_ticker.items():
            try:
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )
                inputs = {
                    "stock_price": float(stock_price.loc[ticker]),
                    "risk_free_rate": float(risk_free_rate),
                    "volatility": float(volatility.loc[ticker]),
                    "dividend_yield": float(dividend_yield_value[ticker]),
                }

                # Every strike price and time to expiration in one call on arrays.
                chooser_option[ticker] = helpers.evaluate_on_grid(
                    lambda strike_price, time_to_expiration, inputs=inputs: compound_model.get_chooser_option(
                        **inputs,
                        strike_price=strike_price,
                        time_to_expiration=time_to_expiration,
                        choice_time=choice_period * time_to_expiration,
                    ),
                    strike_prices,
                    time_to_expiration_list,
                )
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning("No chooser option value for %s: %s", ticker, error)

        chooser_option_df = helpers.create_greek_dataframe(
            greek_dictionary=chooser_option,
            start_date=start_date,
        )

        chooser_option_df = chooser_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            chooser_option_df = calculate_standardization(
                dataset=chooser_option_df,
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

        return chooser_option_df

    @handle_errors
    def get_forward_start_option(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        start_period: float = 0.5,
        moneyness: float = 1.0,
        periods: int = 1,
        performance: bool = False,
        expiration_time_range: int = 30,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the closed-form price of a forward-start option: what a European option on each ticker is worth
        whose strike is set later at a share of the stock price then, with the formula of Rubinstein (1990), or of a
        cliquet (ratchet) option, a series of them, or of their performance versions.

        - Forward-start (one period): a European option whose strike is set after the start period of the life to
          moneyness times the stock price then, paying max(S(T) - α * S(t1), 0) for a call.
        - Cliquet (more than one period): the time from the start to expiration is split into equal periods, and each
          period is a forward-start option whose strike is reset to moneyness times the stock price at its start and
          whose payoff is paid at its end. With a start period of 0 the first period starts today.
        - Performance: each period pays the return S(t(i+1)) / S(ti) beyond the moneyness
          instead, max(S(t(i+1)) / S(ti) - α, 0) for a call, a decimal per unit of notional.

        The options have no strike today, so the result has a row per ticker.

        The formula is as follows:

        - t1 = Start Period * T, ti = t1 + (i - 1) * (T - t1) / Periods and t(Periods + 1) = T
        - Forward Start = Σ S * e^(-q ti) * BS(1, α, t(i+1) - ti)
        - Performance = Σ e^(-r ti) * BS(1, α, t(i+1) - ti)
        - BS(1, α, τ) = e^(-qτ) * N(d1) - α * e^(-rτ) * N(d1 - σ√τ) for a call and α * e^(-rτ) * N(-d1 + σ√τ)
          - e^(-qτ) * N(-d1) for a put, d1 = (ln(1 / α) + (r - q + σ² / 2) * τ) / (σ√τ)

        Where S is the stock price (the traded close on the pricing date), α is the moneyness, r is the continuously
        compounded risk-free rate, q is the continuous dividend yield (ln(1 + dividend yield)), σ is the annualized
        volatility (the standard deviation of the daily returns over the trailing 252 trading days times √252), T is
        the time to expiration in years (days / 365) and N is the cumulative standard normal distribution. A
        performance option is a forward-start option divided by S only when r = q.

        See: Rubinstein, M. (1990), "Pay Now, Choose Later", Risk, 3(2); Haug, E. G. (2007), The Complete Guide to
        Option Pricing Formulas, 2nd ed., section 4.6.

        Also known as: forward-start option, delayed option, cliquet option, ratchet option, performance option.

        Referenced against:
        - QuantLib, AnalyticCliquetEngine (cliquet): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_cliquet_engine.html
        - QuantLib, AnalyticPerformanceEngine (performance): https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_performance_engine.html

        Shares similar logic:
        - QuantLib, ForwardEuropeanEngine (forward-start): https://www.quantlib.org/reference/class_quant_lib_1_1_forward_vanilla_engine.html

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            start_period (float, optional): The share of the life after which the (first) strike is set, at least
            0 and below 1. Defaults to 0.5, halfway.
            moneyness (float, optional): The strike as a share of the stock price at each start, above 0. Defaults
            to 1, at the money.
            periods (int, optional): The number of equal periods from the start to expiration, 1 for a
            forward-start option and more for a cliquet. Defaults to 1.
            performance (bool, optional): Whether each period pays its return beyond the moneyness instead of the
            price difference. Defaults to False.
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
                time to expiration columns for each ticker. Defaults to False.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Forward-start option values containing the tickers as the index and the time to expiration
            as the columns.

        Raises:
            InvalidArgumentError: If start_period is not at least 0 and below 1, moneyness is not positive or periods is not
            a positive whole number.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["NVDA", "AAPL", "MSFT"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        forward_start_option = toolkit.options.get_forward_start_option(
            expiration_time_range=91
        )

        forward_start_option[["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        | Ticker   |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |:---------|-------------:|-------------:|-------------:|
        | NVDA     |       7.6351 |      10.8773 |      13.393  |
        | AAPL     |       7.3395 |      10.4915 |      12.9514 |
        | MSFT     |       9.834  |      14.0922 |      17.4287 |
        """
        if not 0 <= start_period < 1:
            raise InvalidArgumentError(
                f"start_period must be at least 0 and below 1, received {start_period!r}."
            )
        if moneyness <= 0:
            raise InvalidArgumentError(
                f"moneyness must be positive, received {moneyness!r}."
            )
        if (
            isinstance(periods, bool)
            or not isinstance(periods, numbers.Integral)
            or periods < 1
        ):
            raise InvalidArgumentError(
                f"periods must be a positive whole number, received {periods!r}."
            )

        start_date = self._pricing_date(start_date)
        stock_price = self._prices.loc[start_date]
        volatility = self._volatility.loc[start_date]

        risk_free_rate = (
            risk_free_rate
            if risk_free_rate is not None
            else self._risk_free_rate.loc[start_date]
        )

        time_to_expiration_list = helpers.define_times_to_expiration(
            expiration_time_range
        )

        forward_start_option: dict[str, dict[float, float]] = {}
        dividend_yield_value: dict[str, float] = {}

        for ticker in self._tickers:
            try:
                # A ticker without a price on the date is left out, as on the strike price grids.
                if pd.isna(stock_price.get(ticker)):
                    logger.warning(
                        "%s has no price on %s, so it is left out of the option values.",
                        ticker,
                        stock_price.name,
                    )
                    continue
                dividend_yield_value[ticker] = (
                    dividend_yield
                    if dividend_yield is not None
                    else self._dividend_yield_on(ticker, start_date)
                )
                inputs = {
                    "stock_price": float(stock_price.loc[ticker]),
                    "risk_free_rate": float(risk_free_rate),
                    "volatility": float(volatility.loc[ticker]),
                    "moneyness": float(moneyness),
                    "dividend_yield": float(dividend_yield_value[ticker]),
                    "put_option": put_option,
                    "performance": performance,
                }

                # The pricing date itself (no time to expiration) has no periods and is left out of the result.
                forward_start_option[ticker] = {
                    time_to_expiration: (
                        compound_model.get_cliquet_option(
                            **inputs,
                            time_to_expiration=time_to_expiration,
                            reset_times=[
                                time_to_expiration
                                * (start_period + (1 - start_period) * period / periods)
                                for period in range(periods)
                            ],
                        )
                        if time_to_expiration > 0
                        else np.nan
                    )
                    for time_to_expiration in time_to_expiration_list
                }
            except InvalidArgumentError:
                raise
            except ValueError as error:
                # One ticker the formula cannot value (such as a volatility of zero from constant
                # prices) is left out rather than emptying the result of every ticker.
                logger.warning(
                    "No forward start option value for %s: %s", ticker, error
                )

        forward_start_option_df = helpers.create_expiration_dataframe(
            value_dictionary=forward_start_option,
            start_date=start_date,
        )

        forward_start_option_df = forward_start_option_df.pipe(
            apply_rounding, rounding if rounding is not None else self._rounding
        )

        if standardize:
            forward_start_option_df = calculate_standardization(
                dataset=forward_start_option_df,
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

        return forward_start_option_df

    @handle_errors
    def get_strategy_payoff(
        self,
        legs: list[dict[str, float | bool | str]] | None = None,
        start_date: str | None = None,
        stock_price_range: float = 0.5,
        stock_price_step_size: int = 1,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the net expiration profit and loss (P&L) profile of a multi-leg option (and, optionally, stock)
        strategy: what the combined position earns or loses at expiration across a range of stock prices.

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
            legs (list[dict] | None, optional): A list of leg dictionaries as described above. Must
            contain at least one leg. The same legs are applied to every ticker, so
            strike prices should be chosen with the relevant tickers' price levels in
            mind. Defaults to None, which buys an at-the-money straddle on each ticker: a
            call and a put with the stock price on the start date as strike and 30 days to
            expiration, priced with the Black Scholes model on the volatility, risk free rate
            and dividend yield of that date, as `get_black_scholes_model` does.
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            stock_price_range (float): The percentage range to use for the stock prices at expiration. Defaults
            to 0.5 which equals 50% and thus results in stock prices from 50 to 150 if the current stock price is
            100.
            stock_price_step_size (int): The step size to use for the stock prices at expiration. Defaults to 1.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None, which uses
                the rounding of the Toolkit (4 unless set otherwise).
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

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

        toolkit.options.get_strategy_payoff(stock_price_range=0.25, stock_price_step_size=5)
        ```

        Which returns:

        |   Stock Price |     TSLA |
        |--------------:|---------:|
        |           335 |  49.7822 |
        |           340 |  44.7822 |
        |           345 |  39.7822 |
        |           350 |  34.7822 |
        |           355 |  29.7822 |
        |           360 |  24.7822 |
        |           365 |  19.7822 |
        |           370 |  14.7822 |
        |           375 |   9.7822 |
        |           380 |   4.7822 |
        |           385 |  -0.2178 |
        |           390 |  -5.2178 |
        |           395 | -10.2178 |
        |           400 | -15.2178 |
        |           405 | -20.2178 |
        |           410 | -25.2178 |
        |           415 | -30.2178 |
        |           420 | -35.2178 |
        |           425 | -40.2178 |
        |           430 | -45.2178 |
        |           435 | -50.2178 |
        |           440 | -55.2178 |
        |           445 | -60.2178 |
        |           450 | -64.6578 |
        |           455 | -59.6578 |
        |           460 | -54.6578 |
        |           465 | -49.6578 |
        |           470 | -44.6578 |
        |           475 | -39.6578 |
        |           480 | -34.6578 |
        |           485 | -29.6578 |
        |           490 | -24.6578 |
        |           495 | -19.6578 |
        |           500 | -14.6578 |
        |           505 |  -9.6578 |
        |           510 |  -4.6578 |
        |           515 |   0.3422 |
        |           520 |   5.3422 |
        |           525 |  10.3422 |
        |           530 |  15.3422 |
        |           535 |  20.3422 |
        |           540 |  25.3422 |
        |           545 |  30.3422 |
        |           550 |  35.3422 |
        |           555 |  40.3422 |
        """
        # Checked here, as define_strike_prices names its own strike parameters.
        if not 0 <= stock_price_range <= 1:
            raise InvalidArgumentError(
                "stock_price_range is a fraction of the stock price between 0 and 1, e.g. 0.25 "
                f"for prices from 75% to 125% of the price, not {stock_price_range!r}. Please "
                "pass a value between 0 and 1 as stock_price_range."
            )
        if not stock_price_step_size > 0:
            raise InvalidArgumentError(
                "stock_price_step_size is the distance between stock prices, above 0, not "
                f"{stock_price_step_size!r}. Please pass a positive step such as 5 or 2.5 as "
                "stock_price_step_size."
            )

        start_date = self._pricing_date(start_date)
        stock_price = self._prices.loc[start_date]

        stock_price_range_per_ticker = helpers.define_strike_prices(
            tickers=self._tickers,
            stock_price=stock_price,
            strike_step_size=stock_price_step_size,
            strike_price_range=stock_price_range,
        )

        strategy_payoff: dict[str, pd.Series] = {}
        if legs is None:
            # The default straddle is priced at each ticker's volatility.
            self._report_short_history()

        for ticker, price_range in stock_price_range_per_ticker.items():
            price_range_series = pd.Series(price_range, index=price_range, name=ticker)

            strategy_payoff[ticker] = binomial_trees_model.get_strategy_payoff(
                stock_price=price_range_series,
                legs=(
                    legs
                    if legs is not None
                    else self._at_the_money_straddle(ticker, start_date)
                ),
            )

        strategy_payoff_df = pd.DataFrame(strategy_payoff)
        strategy_payoff_df.index.name = "Stock Price"

        strategy_payoff_df = apply_rounding(
            strategy_payoff_df, rounding if rounding is not None else self._rounding
        )

        return strategy_payoff_df

    def _at_the_money_straddle(
        self, ticker: str, date
    ) -> list[dict[str, float | bool | str]]:
        """
        The legs of a long straddle bought on a date: a call and a put struck at that day's
        stock price with 30 days to expiration, at their Black Scholes prices.
        """
        stock_price = float(self._prices.loc[date, ticker])

        return [
            {
                "strike_price": stock_price,
                "put_option": put_option,
                "position": "long",
                "premium": float(
                    black_scholes_model.get_black_scholes(
                        stock_price=stock_price,
                        strike_price=stock_price,
                        risk_free_rate=float(self._risk_free_rate.loc[date]),
                        volatility=float(self._trailing_volatility.loc[date, ticker]),
                        time_to_expiration=30 / 365,
                        dividend_yield=self._dividend_yield_on(ticker, date),
                        put_option=put_option,
                    )
                ),
            }
            for put_option in (False, True)
        ]
