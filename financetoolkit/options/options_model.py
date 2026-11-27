"""Options Model"""

import numbers

import numpy as np
import pandas as pd

from financetoolkit import yfinance_model
from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.options.helpers import (
    VARIANCE_REDUCTIONS,
    control_variate,
    draw_standard_normals,
    monte_carlo_estimate,
    simulation_groups,
    validate_numeric_inputs,
)
from financetoolkit.utilities.error_model import check_option
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()


def get_option_expiry_dates(ticker: str) -> list[str]:
    """
    Retrieve the available option expiry dates of a ticker: the dates on which its listed
    options expire.

    Args:
        ticker (str): The ticker symbol for which to fetch option expiry dates.

    Returns:
        list[str]: A list of option expiry dates in 'YYYY-MM-DD' format.
    """
    cache = get_active_cache()

    if cache is not None:
        cached_dates = cache.get(
            source=policy_model.YAHOO_FINANCE, dataset="option_expiries", entity=ticker
        )

        if cached_dates is not None:
            return cached_dates

    expiry_dates = yfinance_model.get_option_expiry_dates(ticker)

    if cache is not None and expiry_dates:
        cache.set(
            source=policy_model.YAHOO_FINANCE,
            dataset="option_expiries",
            entity=ticker,
            data=expiry_dates,
        )

    return expiry_dates


def get_option_chains(
    tickers: list[str], expiration_date: str, put_option: bool = False
) -> pd.DataFrame:
    """
    Retrieve the option chains (calls or puts) of a list of tickers for one expiration date: the
    strike prices, quotes, volume, open interest and implied volatility of every listed option.

    Option quotes move continuously while the market is open, so a cached chain is
    only served for a few minutes (see the yfinance.option_chain policy). That is
    long enough to stop a single analysis from requesting the same chain repeatedly
    without ever presenting a materially outdated quote as current.

    Args:
        tickers (list[str]): List of ticker symbols.
        expiration_date (str): The expiration date for the options (format: 'YYYY-MM-DD').
        put_option (bool, optional): If True, fetch put options; otherwise, fetch call options. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the option chains for the specified tickers and expiration date,
        with the price of the underlying quoted with the chain in the "Underlying Price" column (NaN when the
        source does not provide it).
    """
    cache = get_active_cache()
    cache_parameters = {
        "expiration_date": expiration_date,
        "put_option": put_option,
    }
    result_dict = {}

    for ticker in tickers:
        if cache is not None:
            cached_chain = cache.get(
                source=policy_model.YAHOO_FINANCE,
                dataset="option_chain",
                entity=ticker,
                parameters=cache_parameters,
            )

            if cached_chain is not None:
                result_dict[ticker] = cached_chain
                continue

        options_df, underlying_price = yfinance_model.get_option_chain(
            ticker, expiration_date, put_option
        )

        if options_df.empty:
            continue

        options_df = options_df.rename(
            columns={
                "contractSymbol": "Contract Symbol",
                "strike": "Strike",
                "currency": "Currency",
                "lastPrice": "Last Price",
                "change": "Change",
                "percentChange": "Percent Change",
                "volume": "Volume",
                "openInterest": "Open Interest",
                "bid": "Bid",
                "ask": "Ask",
                "contractSize": "Contract Size",
                "expiration": "Expiration",
                "lastTradeDate": "Last Trade Date",
                "impliedVolatility": "Implied Volatility",
                "inTheMoney": "In The Money",
            }
        )

        if "Contract Size" in options_df.columns:
            options_df = options_df.drop(columns="Contract Size")

        options_df["Underlying Price"] = underlying_price

        options_df = options_df.set_index("Strike")
        result_dict[ticker] = options_df

        if cache is not None and not options_df.empty:
            cache.set(
                source=policy_model.YAHOO_FINANCE,
                dataset="option_chain",
                entity=ticker,
                data=options_df,
                parameters=cache_parameters,
            )

    if not result_dict:
        return pd.DataFrame()

    result_final = pd.concat(result_dict)
    if "Last Trade Date" in result_final.columns:
        result_final["Last Trade Date"] = pd.to_datetime(
            result_final["Last Trade Date"], unit="s", errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    result_final.index.names = ["Ticker", "Strike Price"]

    return result_final


# Treasury yields are quoted on a bond-equivalent basis: an annual rate compounded twice a year.
TREASURY_COMPOUNDING_PER_YEAR = 2


def get_continuously_compounded_rate(
    bond_equivalent_yield: pd.Series | pd.DataFrame | float,
    compounding_per_year: int = TREASURY_COMPOUNDING_PER_YEAR,
) -> pd.Series | pd.DataFrame | float:
    """
    Calculate the continuously compounded rate of a yield quoted with periodic compounding:
    the rate that grows an investment by the same amount over a year, as the option pricing
    formulas need.

    A yield such as the bond-equivalent yield of a US Treasury is converted into the
    continuously compounded rate that the option pricing formulas (Black-Scholes, binomial
    trees, Bjerksund-Stensland, barrier, Asian and binary options and the Greeks) discount
    with through e^(-r * t). Both rates grow an investment by the same amount over a year:

        - (1 + y / m)^m = e^r, so r = m * ln(1 + y / m)

    Where y is the quoted yield, m the number of compounding periods per year (2 for a
    semi-annual bond-equivalent yield) and r the continuously compounded rate. A yield of 4.00%
    compounded semi-annually equals a continuously compounded rate of 3.96%. Using the quoted
    yield directly in e^(-r * t) would overstate the discounting and the forward drift.

    The 13-week Treasury bill rate (^IRX) is quoted as a bank discount rate d instead, whose
    bond-equivalent yield is 365 * d / (360 — d * days). The Toolkit's rate series do not carry
    their maturity, so every series is treated as a semi-annual bond-equivalent yield; for the
    13-week rate this understates the yield slightly (a 5.00% discount rate is a 5.13%
    bond-equivalent yield), making it an approximation.

    Args:
        bond_equivalent_yield (pd.Series | pd.DataFrame | float): The quoted yield as a decimal.
        compounding_per_year (int, optional): The number of compounding periods per year.
            Defaults to 2, the semi-annual compounding of a bond-equivalent Treasury yield.

    Returns:
        pd.Series | pd.DataFrame | float: The continuously compounded rate as a decimal, in the
        same shape as the input.
    """
    return compounding_per_year * np.log1p(bond_equivalent_yield / compounding_per_year)


# The trailing window of the dividend yield, and the history a ticker needs before a full window is available.  # noqa: E501
DIVIDEND_YIELD_WINDOW = "365D"
DIVIDEND_YIELD_DAYS = 365
# The recovered yield below which a window counts as showing no dividends: cent rounding of
# the adjusted close leaves a few millionths where no dividend was adjusted for.
MINIMUM_RECOVERED_DIVIDEND_YIELD = 1e-6


def get_trailing_dividend_yield(
    close: pd.DataFrame,
    adjusted_close: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate the trailing dividend yield on every date: the dividends per share that went
    ex-dividend in the 365 days up to and including that date, divided by the close of that
    date.

    This is the dividend yield q of the Black-Scholes model on a pricing date, built only
    from what was known on that date.

    The dividends are recovered from the dividend adjustment of the prices rather than taken
    from a list of dividends per share. The adjusted close is the close multiplied by a factor
    f = Adj Close / Close that rises by each dividend on its ex-date: the close before the
    ex-date is scaled down by (1 - Dividend / Close before the ex-date). The dividend of a date
    therefore follows as:

        - Dividend_t = (1 - f_(t-1) / f_t) * Close_(t-1)
        - Dividend Yield_t = SUM(Dividend over the 365 days up to t) / Close_t

    Because the close and the adjusted close are split-adjusted alike, the factor and so the
    recovered dividends are in the units of today's shares, which a list of dividends per
    share as paid at the time is not. The daily changes are summed with their sign: prices
    rounded to the cent move the factor by a few hundredths of a percent from day to day,
    which cancels in a sum over a year but would add up to a yield of several tenths of a
    percent if only the increases were kept.

    A date without a full year of history before it (the first year of a ticker's prices)
    takes the yield of the first date that has one. That yield includes dividends paid after
    the date, so within the first year the yield is not limited to what was known at the time.

    Some price histories carry dividends without adjusting the close for them (an Adj Close
    equal to the Close). The recovered dividends of such a window are zero, and when a
    dividends per share column is given, the dividends it lists over the same 365 days are
    used instead. These are as paid at the time, so a split within the window mixes share
    units, which is why they only fill windows the adjustment leaves empty.

    The result is a simple annual yield; the option formulas carry the yield continuously
    and take ln(1 + q), see options_controller.

    Args:
        close (pd.DataFrame): The daily (split-adjusted) close per ticker, with a
            DatetimeIndex or a daily PeriodIndex.
        adjusted_close (pd.DataFrame): The daily close adjusted for dividends and splits,
            shaped like `close`.
        dividends (pd.DataFrame | None, optional): The dividends per share on their ex-dates,
            shaped like `close`, used where the adjustment shows no dividends over the window.
            Defaults to None (only the recovered dividends).

    Returns:
        pd.DataFrame: The trailing dividend yield as a decimal, shaped like `close`. Dates
        after a ticker's last close are NaN.
    """
    index = close.index
    dates = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index

    # Carried forward so a date without a price (between two trading days) moves the
    # dividend to the next trading day instead of losing it.
    filled_close = close.ffill()
    adjustment = (adjusted_close / close).ffill()

    recovered_dividends = (1 - adjustment.shift(1) / adjustment) * filled_close.shift(1)
    trailing_dividends = (
        recovered_dividends.set_axis(dates)
        .rolling(DIVIDEND_YIELD_WINDOW)
        .sum()
        .set_axis(index)
    )

    dividend_yield = trailing_dividends / close

    if dividends is not None:
        listed_dividends = (
            dividends.reindex(index=index, columns=close.columns)
            .fillna(0)
            .set_axis(dates)
            .rolling(DIVIDEND_YIELD_WINDOW)
            .sum()
            .set_axis(index)
        )

        dividend_yield = dividend_yield.mask(
            (dividend_yield.fillna(0) < MINIMUM_RECOVERED_DIVIDEND_YIELD)
            & (listed_dividends > 0),
            listed_dividends / close,
        )

    # A window of a full year only exists from a year after a ticker's first close.
    first_dates = pd.Series(
        [dates[close[column].notna().to_numpy()].min() for column in close.columns],
        index=close.columns,
    )
    full_year = pd.DataFrame(
        dates.to_numpy()[:, None]
        >= (first_dates + pd.Timedelta(days=DIVIDEND_YIELD_DAYS)).to_numpy()[None, :],
        index=index,
        columns=close.columns,
    )

    return dividend_yield.where(full_year).clip(lower=0).bfill().where(close.notna())


# The most random numbers a Monte Carlo price draws (simulations times time steps), which bounds
# its run time; the paths are simulated in chunks, so the memory stays small regardless.
MAX_SIMULATED_VALUES = 50_000_000
# The paths simulated at once, so one chunk of shocks holds at most this many rows.
SIMULATION_CHUNK_SIZE = 10_000


def get_monte_carlo_option_price(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    simulations: int = 10_000,
    time_steps: int = 100,
    seed: int | None = 0,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
) -> tuple[float, float]:
    """
    Calculate the price of a European option through Monte Carlo simulation: what the option
    is worth as the discounted average payoff over simulated Geometric Brownian Motion (GBM)
    stock price paths.

    Also known as: Monte Carlo option pricing, simulation-based option pricing.

    The Monte Carlo method prices an option by simulating a large number of possible
    future paths for the underlying stock price under the risk-neutral measure,
    computing the option's payoff at expiration for each simulated path, and then
    discounting the average payoff back to the present. As the number of simulations
    grows, the Monte Carlo estimate converges (by the law of large numbers) to the
    true (risk-neutral) expected discounted payoff — for a plain-vanilla European
    option this converges to the Black-Scholes price.

    Each simulated stock price path follows Geometric Brownian Motion:

    - S(t + Δt) = S(t) * e^((r — q — σ²/2) * Δt + σ * √Δt * Z)

    Where S(t) is the stock price at time t, r is the risk-free rate, q is the dividend
    yield, σ is the volatility, Δt is the length of a single time step and Z is a
    standard normal random variable.

    The option is then priced as the discounted average of the terminal payoffs:

    - Call Price = e^(—r * T) * mean(max(S(T) — K, 0))
    - Put Price = e^(—r * T) * mean(max(K — S(T), 0))

    Because it is an estimate, the price comes with sampling (Monte Carlo) error. The
    standard error of the estimate is returned alongside the price and shrinks
    proportionally to 1 / √simulations — quadrupling the number of simulations halves
    the standard error.

    Two variance reductions and quasi-random numbers narrow that error at the same number of
    paths (Glasserman, 2003, chapters 4 and 5):

    - Antithetic variates: every path Z is paired with its mirror
      image -Z, and the standard error comes from the averages of the pairs.
    - Control variate: the discounted stock price at
      expiration X = e^(—r * T) * S(T), whose expected value S * e^(—q * T) is known, takes
      out the simulation error it shares with the payoff Y: Price = MEAN(Y — β * (X —
      S * e^(—q * T))), β = Cov(Y, X) / Var(X).
    - Sobol sequence: scrambled Sobol points with a Brownian bridge, whose first
      coordinate sets the stock price at expiration, in 16 independently scrambled
      replications; the standard error is that of the 16 replication means. The error
      shrinks close to 1 / simulations rather than 1 / √simulations.

    Shares similar logic:
    - QuantLib, MCEuropeanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_european_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The option's strike price.
        risk_free_rate (float): The risk-free interest rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration of the option (in years).
        dividend_yield (float): The dividend yield of the stock. Defaults to 0.
        put_option (bool): Whether the option is a put option or not.
        simulations (int): The number of simulated stock price paths. Defaults to 10,000.
        time_steps (int): The number of time steps used to build each simulated path.
            Defaults to 100. Since only the terminal stock price affects a European
            option's payoff, this mainly affects how faithfully the path itself is
            simulated rather than the resulting price.
        seed (int | None): The seed used to initialize the random number generator via
            `np.random.default_rng`, ensuring reproducible results; None draws new paths
            on every call. Defaults to 0.
        variance_reduction (str | None): The variance reduction. Options are: None (none),
            "antithetic" (antithetic variates, needs an even number of simulations),
            "control_variate" (control variate). Defaults to None.
        sequence (str): The random numbers. Options are: "pseudo_random" (pseudo-random
            numbers), "sobol" (scrambled Sobol points, at least 32 simulations). Defaults to
            "pseudo_random".

    Returns:
        tuple[float, float]: a tuple of (option_price, standard_error) where option_price
        is the Monte Carlo estimate of the option's fair value and standard_error is the
        standard error of that estimate.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if simulations or time_steps is not a positive integer, if
            simulations times time_steps exceeds MAX_SIMULATED_VALUES (50 million), or if
            variance_reduction or sequence is not one of the choices.

    Notes:
        References: Boyle, P. P. (1977). "Options: A Monte Carlo Approach." Journal of
        Financial Economics, 4(3), 323-338. Glasserman, P. (2003). "Monte Carlo Methods in
        Financial Engineering." Springer. Sobol, I. M. (1967). "On the Distribution of
        Points in a Cube and the Approximate Evaluation of Integrals." USSR Computational
        Mathematics and Mathematical Physics, 7(4), 86-112. Owen, A. B. (1997). "Scrambled
        Net Variance for Integrals of Smooth Functions." Annals of Statistics, 25(4),
        1541-1562.
        - QuantLib's MCEuropeanEngine has pseudo-random and low-discrepancy traits; its own
          random numbers give other paths.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )

    if not isinstance(simulations, numbers.Integral) or simulations <= 0:
        raise ValueError(
            f"simulations must be a positive integer, received {simulations!r}."
        )

    if not isinstance(time_steps, numbers.Integral) or time_steps <= 0:
        raise ValueError(
            f"time_steps must be a positive integer, received {time_steps!r}."
        )

    prices, standard_errors = get_monte_carlo_option_prices(
        stock_price=stock_price,
        strike_prices=np.array([strike_price], dtype=float),
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
        put_option=put_option,
        simulations=simulations,
        time_steps=time_steps,
        seed=seed,
        variance_reduction=variance_reduction,
        sequence=sequence,
    )

    return float(prices[0]), float(standard_errors[0])


def _simulate_terminal_shocks(
    simulations: int,
    time_steps: int,
    seed: int | None,
    antithetic: bool = False,
    sequence: str = "pseudo_random",
) -> np.ndarray:
    """
    Simulates the standardised terminal shock of Geometric Brownian Motion paths: the sum of
    the time_steps standard normal shocks of every path, divided by √time_steps, which is
    standard normal. The stock price at expiration T of a path is then:

    - S(T) = S * e^((r — q — σ²/2) * T + σ * √T * Z)

    So one set of paths prices every expiration, each by scaling the same shocks (common
    random numbers across expirations). Every path is built from time_steps shocks, so a
    seed gives the same terminal prices whatever the number of paths simulated at once; only
    the end of each path is kept, since a European payoff needs nothing else. The shocks come
    from draw_standard_normals.

    Args:
        simulations (int): The number of paths.
        time_steps (int): The number of steps per path.
        seed (int | None): The seed of the random number generator.
        antithetic (bool, optional): Whether the paths come in antithetic pairs. Defaults
            to False.
        sequence (str, optional): "pseudo_random" or "sobol". Defaults to "pseudo_random".

    Returns:
        np.ndarray: The standardised terminal shock of every path.
    """
    if simulations * time_steps > MAX_SIMULATED_VALUES:
        raise ValueError(
            f"{simulations:,} simulations of {time_steps:,} time steps is too large; keep "
            f"simulations times time steps under {MAX_SIMULATED_VALUES:,}."
        )

    shocks = np.empty(simulations)
    first = 0

    # The shocks come in chunks of paths, which holds only a chunk in memory.
    for random_shocks in draw_standard_normals(
        simulations,
        time_steps,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
        chunk_size=SIMULATION_CHUNK_SIZE,
    ):
        rows = random_shocks.shape[0]
        shocks[first : first + rows] = random_shocks[:, :, 0].sum(axis=1)
        first += rows

    return shocks / np.sqrt(time_steps)


def _plain_monte_carlo_estimates(
    terminal_stock_prices: np.ndarray,
    strike_prices: np.ndarray,
    discount_factor: float,
    put_option: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The Monte Carlo price and standard error of every strike from the terminal stock prices
    of independent paths, without a payoff per path and strike: with the prices sorted, the
    count, sum and sum of squares of the prices beyond each strike are cumulative sums, and:

    - Call = e^(—rT) * Σ_{S(T) > K} (S(T) — K) / n
    - Standard error = √(Σ payoff² / n — (Σ payoff / n)²) * √(n / (n — 1)) / √n

    With Σ_{S(T) > K} (S(T) — K)² = Σ S(T)² — 2K * Σ S(T) + K² * count over the same paths,
    and the put alike over the paths below the strike.
    """
    simulations = len(terminal_stock_prices)
    ordered = np.sort(terminal_stock_prices)
    sums = np.concatenate([[0.0], np.cumsum(ordered)])
    squares = np.concatenate([[0.0], np.cumsum(ordered**2)])
    # The paths below each strike are the first `below` of the ordered prices.
    below = np.searchsorted(ordered, strike_prices, side="right")

    if put_option:
        count, total, total_squares = below, sums[below], squares[below]
        sign = -1.0
    else:
        count = simulations - below
        total = sums[-1] - sums[below]
        total_squares = squares[-1] - squares[below]
        sign = 1.0

    payoff_sum = sign * (total - strike_prices * count)
    payoff_squares = (
        total_squares - 2 * strike_prices * total + strike_prices**2 * count
    )
    mean = payoff_sum / simulations
    variance = (
        np.maximum(payoff_squares / simulations - mean**2, 0.0)
        * simulations
        / max(simulations - 1, 1)
    )

    return (
        discount_factor * mean,
        discount_factor * np.sqrt(variance / simulations),
    )


def get_monte_carlo_option_prices(
    stock_price: float,
    strike_prices: np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float | np.ndarray,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    simulations: int = 10_000,
    time_steps: int = 100,
    seed: int | None = 0,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Monte Carlo price of European options with several strike prices and
    expirations from one set of simulated paths: what each option is worth as its discounted
    average payoff (see get_monte_carlo_option_price).

    Pricing every strike on the same paths (common random numbers) gives the same prices as
    simulating per strike with the same seed, keeps the prices consistent across strikes and
    simulates once rather than once per strike. The expirations scale the same standardised
    shocks (see _simulate_terminal_shocks), which with a seed gives the prices of simulating
    each expiration with that seed. Every estimate still comes from independent paths, so
    its standard error is that of one expiration simulated on its own.

    Shares similar logic:
    - QuantLib, MCEuropeanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_european_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_prices (np.ndarray): The strike prices.
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float | np.ndarray): The time to expiration in years, or an
            array of them.
        dividend_yield (float, optional): The dividend yield. Defaults to 0.0.
        put_option (bool, optional): Whether to price puts instead of calls. Defaults to
            False.
        simulations (int, optional): The number of simulated paths. Defaults to 10,000.
        time_steps (int, optional): The number of steps per path. Defaults to 100.
        seed (int | None, optional): The seed of the random number generator, None for new
            paths on every call. Defaults to 0.
        variance_reduction (str | None, optional): The variance reduction. Options are: None
            (none), "antithetic" (antithetic variates), "control_variate" (control variate).
            Defaults to None.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".

    Returns:
        tuple[np.ndarray, np.ndarray]: The price and its standard error per strike price,
        and per expiration (strikes by expirations) when time_to_expiration is an array.
    """
    check_option("variance_reduction", variance_reduction, VARIANCE_REDUCTIONS)

    shocks = _simulate_terminal_shocks(
        simulations,
        time_steps,
        seed,
        antithetic=variance_reduction == "antithetic",
        sequence=sequence,
    )
    strike_prices = np.asarray(strike_prices, dtype=float)
    times = np.atleast_1d(np.asarray(time_to_expiration, dtype=float))
    option_prices = np.empty((len(strike_prices), len(times)))
    standard_errors = np.empty((len(strike_prices), len(times)))
    groups = simulation_groups(
        simulations, sequence, variance_reduction == "antithetic"
    )
    # Independent paths give the estimates of all strikes from sorted prices; pairs, Sobol
    # replications and the control variate take a payoff per path and strike.
    independent = sequence == "pseudo_random" and variance_reduction is None
    # The payoffs of this many strikes at a time, so a block holds about a million values.
    block = max(1, 1_000_000 // max(simulations, 1))

    for column, time in enumerate(times):
        terminal_stock_prices = stock_price * np.exp(
            (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time
            + volatility * np.sqrt(time) * shocks
        )
        discount_factor = np.exp(-risk_free_rate * time)

        if independent:
            option_prices[:, column], standard_errors[:, column] = (
                _plain_monte_carlo_estimates(
                    terminal_stock_prices, strike_prices, discount_factor, put_option
                )
            )
            continue

        for first in range(0, len(strike_prices), block):
            strikes = strike_prices[first : first + block, None]
            payoffs = np.maximum(
                (
                    (strikes - terminal_stock_prices)
                    if put_option
                    else (terminal_stock_prices - strikes)
                ),
                0,
            )
            discounted_payoffs = discount_factor * payoffs
            if variance_reduction == "control_variate":
                discounted_payoffs = control_variate(
                    discounted_payoffs,
                    discount_factor * terminal_stock_prices,
                    stock_price * np.exp(-dividend_yield * time),
                )
            (
                option_prices[first : first + block, column],
                standard_errors[first : first + block, column],
            ) = monte_carlo_estimate(discounted_payoffs, groups)

    if np.ndim(time_to_expiration) == 0:
        return option_prices[:, 0], standard_errors[:, 0]

    return option_prices, standard_errors


def get_projected_dividends(
    dividends: pd.Series,
    pricing_date: pd.Timestamp,
    horizon_days: int,
    history: int = 4,
) -> pd.Series:
    """
    Calculate the projected cash dividends of a stock after a pricing date: the dividends it
    may pay up to expiration, from the dividends it paid up to that date.

    The latest dividend is repeated at the usual interval between its ex-dividend dates, so
    a past pricing date only uses what was known on that date.

    - Interval = the median number of days between the last `history` + 1 ex-dividend dates
      (365 days with a single dividend)
    - Next ex-date = last ex-date + Interval, moved on by the Interval until it lies after
      the pricing date, and every Interval after it up to the horizon
    - Amount = the latest dividend per share

    A stock whose latest dividend lies more than twice the interval before the pricing date
    has stopped paying and projects no dividends, as does a stock without dividends.

    Args:
        dividends (pd.Series): The dividends per share of one stock on their ex-dividend
            dates (0 or NaN on the other dates), with a DatetimeIndex or a daily PeriodIndex.
        pricing_date (pd.Timestamp): The date the dividends are projected from.
        horizon_days (int): The number of calendar days after the pricing date to project.
        history (int, optional): The number of intervals between ex-dates the usual
            interval is the median of. Defaults to 4, a year of quarterly dividends.

    Returns:
        pd.Series: The projected dividend per share, indexed by the number of calendar days
        after the pricing date of each ex-dividend date. Empty without projected dividends.
    """
    index = dividends.index
    dates = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
    paid = pd.Series(dividends.to_numpy(dtype=float), index=dates)
    pricing_date = pd.Timestamp(pricing_date)
    paid = paid[(paid > 0) & (paid.index <= pricing_date)]

    if paid.empty:
        return pd.Series(dtype=float, name="Dividend")

    ex_dates = paid.index[-(history + 1) :]
    interval = (
        int(np.median(np.diff(ex_dates).astype("timedelta64[D]").astype(int)))
        if len(ex_dates) > 1
        else 365
    )
    last_date = paid.index[-1]

    if interval <= 0 or (pricing_date - last_date).days > 2 * interval:
        return pd.Series(dtype=float, name="Dividend")

    next_date = last_date + pd.Timedelta(days=interval)
    while next_date <= pricing_date:
        next_date += pd.Timedelta(days=interval)

    days = []
    while (next_date - pricing_date).days <= horizon_days:
        days.append((next_date - pricing_date).days)
        next_date += pd.Timedelta(days=interval)

    return pd.Series(
        float(paid.iloc[-1]),
        index=pd.Index(days, name="Days", dtype=int),
        name="Dividend",
    )
