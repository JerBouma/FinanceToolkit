"""Option Pricing Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options import helpers, options_model
from financetoolkit.options.base.base_controller import OptionsBase
from financetoolkit.options.pricing import (
    binomial_trees_model,
    black_scholes_model,
    exotics_model,
)
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
class PricingOptions(OptionsBase):
    """
    Option pricing: Black-Scholes, binomial trees, Monte Carlo, Garman-Kohlhagen, Bjerksund-Stensland, binary, barrier and Asian options, put-call parity and strategy payoffs.
    """

    @handle_errors
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
        show_columns: str | list[str] | None = None,
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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
        |            160 |      28.7159 |      31.761  |      34.5215 |
        |            165 |      24.6249 |      28.1288 |      31.1403 |
        |            170 |      20.8373 |      24.7538 |      27.9815 |
        |            175 |      17.3911 |      21.6463 |      25.0482 |
        |            180 |      14.3123 |      18.8111 |      22.34   |
        |            185 |      11.6125 |      16.2474 |      19.8537 |
        |            190 |       9.2892 |      13.9492 |      17.5835 |
        |            195 |       7.3269 |      11.9066 |      15.5214 |
        |            200 |       5.6997 |      10.106  |      13.6576 |
        |            205 |       4.3742 |       8.5311 |      11.9811 |
        |            210 |       3.313  |       7.1639 |      10.4798 |
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

    @handle_errors
    def get_binomial_model(
        self,
        start_date: str | None = None,
        put_option: bool = False,
        american_option: bool = False,
        strike_price_range: float = 0.25,
        strike_step_size: int = 5,
        time_to_expiration: float = 1,
        timesteps: int = 10,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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

        - time step (Δt) = T / n
        - up movement (u) = e^(σ * sqrt(Δt))
        - down movement (d) = 1 / u
        - risk neutral probability (p) = (e^((r — q) * Δt) — d) / (u — d)
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
        n is the number of time steps, C_u is the call option price at the up
        movement, C_d is the call option price at the down movement, P_u is the put option price at the up movement and
        P_d is the put option price at the down movement.

        The resulting output is a DataFrame containing the tickers, strike prices and movements as the index and the
        time to expiration as the columns. The movements index contains the number of up movements and the number of
        down movements. The output is the binomial tree displayed in a table. E.g. when using 10 time steps, the
        table for each strike price from each company will contain the actual binomial tree as also depicted
        in the image found here: https://en.wikipedia.org/wiki/Binomial_options_pricing_model#Method

        Also known as: binomial tree, lattice model, option pricing.

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            american_option (bool, optional): Whether to value an American option, which can be exercised at every
                node, instead of a European option. Defaults to False.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            time_to_expiration (float): The number of years to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
            time steps. This will be evenly distributed over the time to expiration.
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
            pd.DataFrame: Binomial Trees values containing the tickers, strike prices and movements as the index and the
            timesteps as the columns.

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
            strike_price_range=0, strike_step_size=1, timesteps=5
        )

        binomial_model.loc["KO"]
        ```

        Which returns:

        |               |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:--------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | (69, 'UUUUU') |        5.821 |       9.067  |      13.6268 |      19.594  |      26.7232 |      34.524  |
        | (69, 'DUUUU') |      nan     |       2.7196 |       4.7251 |       7.9735 |      12.8942 |      19.479  |
        | (69, 'DDUUU') |      nan     |     nan      |       0.7891 |       1.6034 |       3.2581 |       6.6205 |
        | (69, 'DDDUU') |      nan     |     nan      |     nan      |       0      |       0      |       0      |
        | (69, 'DDDDU') |      nan     |     nan      |     nan      |     nan      |       0      |       0      |
        | (69, 'DDDDD') |      nan     |     nan      |     nan      |     nan      |     nan      |       0      |
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

    @handle_errors
    def get_stock_price_simulation(
        self,
        start_date: str | None = None,
        time_to_expiration: float = 1,
        timesteps: int = 10,
        risk_free_rate: float | None = None,
        show_unique_combinations: bool = False,
        show_input_info: bool = False,
        rounding: int | None = None,
        standardize: bool = False,
        show_columns: str | list[str] | None = None,
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
        For example for below's example use `stock_price_simulation.loc['AMD'].T.plot(legend=False)`

        Also known as: binomial tree, stock price tree, stock price path, lattice.

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            time_to_expiration (float): The number of years to use for the time to expiration. Defaults to 1 which equals
            one year.
            timesteps (int): The number of time steps to use for the binomial tree. Defaults to 10 which equals 10
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
            timesteps as the columns.

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
            timesteps=5, show_unique_combinations=True
        )

        stock_price_simulation.loc["AMD"]
        ```

        Which returns:

        | Movement   |   2025-12-31 |   2026-03-14 |   2026-05-26 |   2026-08-07 |   2026-10-19 |   2026-12-31 |
        |:-----------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
        | UUUUU      |       214.16 |      280.988 |      368.67  |     483.712  |      634.654 |     832.696  |
        | DUUUU      |       214.16 |      163.226 |      214.16  |     280.988  |      368.67  |     483.712  |
        | DDUUU      |       214.16 |      163.226 |      124.405 |     163.226  |      214.16  |     280.988  |
        | DDDUU      |       214.16 |      163.226 |      124.405 |      94.8177 |      124.405 |     163.226  |
        | DDDDU      |       214.16 |      163.226 |      124.405 |      94.8177 |       72.267 |      94.8177 |
        | DDDDD      |       214.16 |      163.226 |      124.405 |      94.8177 |       72.267 |      55.0796 |
        """
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
        strike_step_size: int = 5,
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
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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
        |            148 |       8.4856 |       8.266  |       8.1462 |
        |            150 |       6.5551 |       6.4903 |       6.498  |
        |            152 |       4.7422 |       4.8793 |       5.0189 |
        |            154 |       3.1505 |       3.4884 |       3.741  |
        |            156 |       1.8845 |       2.3571 |       2.683  |
        |            158 |       0.9971 |       1.4973 |       1.8467 |
        |            160 |       0.4601 |       0.8903 |       1.2176 |
        |            162 |       0.1832 |       0.494  |       0.7678 |
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the price of a Binary (Digital) Option using the Black-Scholes
        framework.

        A binary option pays out a fixed amount if the option expires in-the-money and
        nothing otherwise. Two variants are supported through the ``option_type``
        parameter: "cash-or-nothing" (the default) pays a fixed cash amount and
        "asset-or-nothing" pays the value of the underlying asset if the option expires
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
            option_type (str, optional): Either "cash-or-nothing" or "asset-or-nothing". Defaults to
            "cash-or-nothing".
            cash_payout (float, optional): The fixed cash amount paid out by a cash-or-nothing option when it
            expires in-the-money. Ignored for asset-or-nothing options. Defaults to 1.0.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate American option prices using the Bjerksund-Stensland (2002)
        closed-form analytical approximation.

        Unlike European options, American options can be exercised at any time up to
        and including expiration, which normally requires a numerical approach such as
        the Binomial Tree model (see ``get_binomial_model``). The Bjerksund-Stensland
        model instead derives a closed-form approximation by assuming the early-exercise
        boundary is flat in steps: once the stock price crosses the trigger price,
        immediate exercise is assumed optimal. The option's life is split at
        t1 = ½(√5 - 1) * T into two periods, each with its own flat trigger price, which
        brings the price closer to the exact American value than a single flat trigger.

        The formula is as follows (for a call, with cost of carry b = r - q smaller than r):

        - β = (0.5 - b / σ²) + √((b / σ² - 0.5)² + 2r / σ²)
        - B∞ = β / (β - 1) * K
        - B0 = MAX(K, r / (r - b) * K)
        - h(t) = -(b * t + 2σ√t) * K² / ((B∞ - B0) * B0)
        - I(t) = B0 + (B∞ - B0) * (1 - e^h(t)), with I1 = I(t1) and I2 = I(T)
        - Call = MAX(Approximation, European Black-Scholes Call, MAX(S - K, 0)), where the Approximation is
          S - K when S ≥ I2 and otherwise combines univariate (first period) and bivariate (both periods)
          normal terms, see ``black_scholes_model.get_bjerksund_stensland``
        - Put (S, K, T, r, b, σ) = Call (K, S, T, r - b, -b, σ), the put-call transformation

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, r is the continuously compounded risk-free rate, q is the continuous dividend
        yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252) and T is the time
        to expiration in years (days / 365).

        When b ≥ r early exercise of a call is not optimal and the value is the Black-Scholes
        price (floored at the exercise value). Where the boundary I2 falls to or below the
        strike, a 1,000-step binomial tree prices the option instead. At zero days to
        expiration the value is the exercise value.

        See: Bjerksund, P., & Stensland, G. (2002), "Closed Form Valuation of American
        Options", Working paper, NHH, as presented in Haug, E. G. (2007), "The Complete
        Guide to Option Pricing Formulas" (2nd ed.), section 3.1.4.

        Also known as: BS2002, Bjerksund-Stensland approximation, American option
        approximation.

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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
            pd.DataFrame: Bjerksund-Stensland American option values containing the tickers and strike prices as
            the index and the time to expiration as the columns.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(
            ["VZ"],
            api_key="FINANCIAL_MODELING_PREP_KEY",
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        bjerksund_stensland = toolkit.options.get_bjerksund_stensland(
            strike_price_range=0.15, strike_step_size=1, expiration_time_range=91
        )

        bjerksund_stensland.loc["VZ", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |             34 |       6.73   |       6.73   |       6.73   |
        |             35 |       5.73   |       5.73   |       5.7417 |
        |             36 |       4.73   |       4.753  |       4.8234 |
        |             37 |       3.7409 |       3.8477 |       3.9719 |
        |             38 |       2.8269 |       3.0217 |       3.202  |
        |             39 |       2.0118 |       2.2958 |       2.5242 |
        |             40 |       1.3361 |       1.6834 |       1.9439 |
        |             41 |       0.8214 |       1.189  |       1.4615 |
        |             42 |       0.4646 |       0.8079 |       1.0723 |
        |             43 |       0.241  |       0.5278 |       0.7678 |
        |             44 |       0.1144 |       0.3314 |       0.5365 |
        |             45 |       0.0497 |       0.2001 |       0.366  |
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate European option prices through Monte Carlo simulation of Geometric
        Brownian Motion (GBM) stock price paths.

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
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
            expirations of 1 to 29 days.
            risk_free_rate (float, optional): The risk free rate to use for the calculation. Defaults to None, which
            uses the Toolkit's risk-free rate on the pricing date: the 10-year Treasury yield unless the Toolkit's
            risk_free_rate parameter selects '13w', '5y' or '30y', converted from its semi-annual bond-equivalent
            quote to continuous compounding. A rate passed here is used as given, as a continuously compounded rate.
            dividend_yield (float, optional): The dividend yield to use for the calculation. Defaults to None, which
            uses the dividend yield on the pricing date (the dividends of the 365 days up to it relative to its close,
            as the continuously compounded ln(1 + q)). A given value is used as a continuous yield.
            simulations (int, optional): The number of simulated stock price paths. Defaults to 10,000.
            time_steps (int, optional): The number of time steps used to build each simulated path. Defaults to 100.
            seed (int | None, optional): The seed used to initialize the random number generator, ensuring
            reproducible results. Defaults to None, which means the results will not be reproducible.
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
        |            780 |      82.8804 |      83.4929 |      85.5951 |
        |            790 |      72.8944 |      73.6474 |      76.2377 |
        |            800 |      62.9375 |      63.9365 |      67.1784 |
        |            810 |      53.0608 |      54.449  |      58.4909 |
        |            820 |      43.3668 |      45.3486 |      50.2675 |
        |            830 |      34.0544 |      36.7989 |      42.5958 |
        |            840 |      25.4476 |      28.9941 |      35.5519 |
        |            850 |      17.9002 |      22.0923 |      29.1778 |
        |            860 |      11.7131 |      16.2131 |      23.5449 |
        |            870 |       7.0931 |      11.4623 |      18.6844 |
        |            880 |       3.9774 |       7.8047 |      14.5854 |
        |            890 |       2.0399 |       5.1208 |      11.2062 |
        |            900 |       0.9613 |       3.2137 |       8.477  |
        |            910 |       0.4226 |       1.935  |       6.285  |
        |            920 |       0.1699 |       1.1307 |       4.5788 |
        |            930 |       0.0583 |       0.6377 |       3.2783 |
        |            940 |       0.018  |       0.3433 |       2.3104 |
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
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

        Also known as: knock-in option, knock-out option, down-and-out, down-and-in,
        up-and-out, up-and-in option.

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            barrier_percentage (float, optional): The barrier level as a percentage of the current stock price.
            Defaults to 0.9 which equals 90% of the current stock price. A down barrier needs a value below 1
            and an up barrier a value above 1; otherwise the barrier is already breached.
            barrier_direction (str, optional): Either "down" or "up". Defaults to "down".
            knock_type (str, optional): Either "in" or "out". Defaults to "out".
            rebate (float, optional): The fixed cash amount paid out if the option knocks out (or fails to knock
            in). Defaults to 0.0.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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
            ValueError: If barrier_percentage puts the barrier at or beyond the current stock price for the
                given barrier_direction (at or below 1 for "up", at or above 1 for "down"), since such a
                barrier is breached at the start and every price would be the rebate (knock-out) or the
                vanilla price (knock-in).

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
        |            590 |      70.211  |      70.5594 |      71.0161 |
        |            600 |      63.6599 |      65.6623 |      66.9713 |
        |            610 |      57.1775 |      60.7959 |      62.9446 |
        |            620 |      50.8693 |      56.0077 |      58.9639 |
        |            630 |      44.8284 |      51.341  |      55.0551 |
        |            640 |      39.1312 |      46.8335 |      51.2414 |
        |            650 |      33.8365 |      42.5169 |      47.543  |
        |            660 |      28.9845 |      38.4166 |      43.9768 |
        |            670 |      24.5979 |      34.5518 |      40.557  |
        |            680 |      20.6836 |      30.9358 |      37.2945 |
        |            690 |      17.2345 |      27.5763 |      34.1974 |
        |            700 |      14.2322 |      24.4761 |      31.2713 |
        |            710 |      11.6493 |      21.6334 |      28.519  |
        |            720 |       9.4527 |      19.043  |      25.9413 |
        """
        # The barrier is set relative to the spot, so a barrier on the wrong side of it is breached
        # before the option starts. The prices would then be a flat rebate or the vanilla price,
        # which hides a mistaken barrier_percentage or barrier_direction instead of reporting it.
        if barrier_direction == "up" and barrier_percentage <= 1:
            raise ValueError(
                f"An up barrier must lie above the current stock price, so barrier_percentage must be above 1 "
                f"(received {barrier_percentage}). Use barrier_direction='down' for a barrier below the stock price."
            )
        if barrier_direction == "down" and barrier_percentage >= 1:
            raise ValueError(
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

    @handle_errors
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
        show_columns: str | list[str] | None = None,
    ):
        """
        Calculate the closed-form price of a geometric-average Asian option using the
        Kemna & Vorst (1990) formula.

        An Asian option's payoff depends on the average price of the underlying stock
        over the option's life, rather than the price at a single point in time, which
        typically makes it cheaper than the equivalent vanilla option (the averaging
        reduces variance). This prices the continuously sampled geometric-average Asian
        option only; the arithmetic-average version has no closed form and is not
        offered. The geometric-average version has a closed-form solution based on an
        adjusted volatility and cost of carry.

        The formula is as follows:

        - σ_A = σ / √3
        - b_A = 0.5 * (b - σ² / 6), where b = r - q is the cost of carry
        - d1 = (ln(S / K) + (b_A + σ_A² / 2) * t) / (σ_A * √t)
        - d2 = d1 - σ_A * √t
        - Call Price = S * e^((b_A - r) * t) * N(d1) - K * e^(-r * t) * N(d2)
        - Put Price = K * e^(-r * t) * N(-d2) - S * e^((b_A - r) * t) * N(-d1)

        Where S is the stock price (the traded close on the pricing date), K is the strike
        price, r is the continuously compounded risk-free rate, q is the continuous dividend
        yield (ln(1 + dividend yield)), σ is the annualized volatility (the standard deviation
        of the daily returns over the trailing 252 trading days times √252), t is the time to
        expiration in years (days / 365) and N is the cumulative standard normal distribution.

        Also known as: geometric Asian option, average rate option, average price
        option.

        Args:
            start_date (str | None, optional): The date whose price, volatility and rates the values
            start from; a date without prices (a weekend or a holiday) uses the last trading day before it, and a
            ticker without a price on that day is left out with a warning. Defaults to None, which uses the
            last date of the historical data.
            put_option (bool, optional): Whether to calculate the put option price. Defaults to False which means
            it will calculate the call option price.
            strike_price_range (float): The percentage range to use for the strike prices. Defaults to 0.25 which equals
            25% and thus results in strike prices from 75 to 125 if the current stock price is 100.
            strike_step_size (int): The step size to use for the strike prices. Defaults to 5 which means that the
            strike prices will be 75, 80, 85, 90, 95, 100, 105, 110, 115 and 120 if the current stock price is 100.
            expiration_time_range (int): The number of days to use for the time to expiration: the columns are the
            expirations 1 up to expiration_time_range - 1 days after the start date. Defaults to 30, which gives
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
            pd.DataFrame: Geometric-average Asian option values containing the tickers and strike prices as the
            index and the time to expiration as the columns.

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
            strike_price_range=0.1, strike_step_size=2, expiration_time_range=91
        )

        asian_option.loc["XOM", ["2026-01-30", "2026-03-01", "2026-03-31"]]
        ```

        Which returns:

        |   Strike Price |   2026-01-30 |   2026-03-01 |   2026-03-31 |
        |---------------:|-------------:|-------------:|-------------:|
        |            108 |      12.2984 |      12.3072 |      12.3784 |
        |            110 |      10.3173 |      10.3976 |      10.5491 |
        |            112 |       8.365  |       8.5609 |       8.8116 |
        |            114 |       6.4839 |       6.8383 |       7.197  |
        |            116 |       4.7442 |       5.2756 |       5.7352 |
        |            118 |       3.2336 |       3.9144 |       4.4502 |
        |            120 |       2.0276 |       2.7832 |       3.3568 |
        |            122 |       1.1573 |       1.8907 |       2.4581 |
        |            124 |       0.5964 |       1.2243 |       1.7458 |
        |            126 |       0.276  |       0.7545 |       1.2016 |
        |            128 |       0.1143 |       0.4421 |       0.8013 |
        |            130 |       0.0423 |       0.2462 |       0.5176 |
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
        |           335 |  49.7821 |
        |           340 |  44.7821 |
        |           345 |  39.7821 |
        |           350 |  34.7821 |
        |           355 |  29.7821 |
        |           360 |  24.7821 |
        |           365 |  19.7821 |
        |           370 |  14.7821 |
        |           375 |   9.7821 |
        |           380 |   4.7821 |
        |           385 |  -0.2179 |
        |           390 |  -5.2179 |
        |           395 | -10.2179 |
        |           400 | -15.2179 |
        |           405 | -20.2179 |
        |           410 | -25.2179 |
        |           415 | -30.2179 |
        |           420 | -35.2179 |
        |           425 | -40.2179 |
        |           430 | -45.2179 |
        |           435 | -50.2179 |
        |           440 | -55.2179 |
        |           445 | -60.2179 |
        |           450 | -64.6579 |
        |           455 | -59.6579 |
        |           460 | -54.6579 |
        |           465 | -49.6579 |
        |           470 | -44.6579 |
        |           475 | -39.6579 |
        |           480 | -34.6579 |
        |           485 | -29.6579 |
        |           490 | -24.6579 |
        |           495 | -19.6579 |
        |           500 | -14.6579 |
        |           505 |  -9.6579 |
        |           510 |  -4.6579 |
        |           515 |   0.3421 |
        |           520 |   5.3421 |
        |           525 |  10.3421 |
        |           530 |  15.3421 |
        |           535 |  20.3421 |
        |           540 |  25.3421 |
        |           545 |  30.3421 |
        |           550 |  35.3421 |
        |           555 |  40.3421 |
        """
        start_date = self._pricing_date(start_date)
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
                        volatility=float(self._volatility.loc[date, ticker]),
                        time_to_expiration=30 / 365,
                        dividend_yield=self._dividend_yield_on(ticker, date),
                        put_option=put_option,
                    )
                ),
            }
            for put_option in (False, True)
        ]
