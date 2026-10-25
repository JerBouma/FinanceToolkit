"""Options Model"""

import numpy as np
import pandas as pd
import yfinance as yf

from financetoolkit.cache import policy_model
from financetoolkit.cache.cache_controller import get_active_cache
from financetoolkit.economics import cboe_model
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# pylint: disable=too-many-arguments,too-many-locals


def get_option_expiry_dates(ticker: str) -> list[str]:
    """
    Retrieve available option expiry dates for a given ticker symbol.

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

    try:
        expiry_dates = list(yf.Ticker(ticker).options)
    except Exception as error:  # noqa: BLE001 - Yahoo Finance fails in many ways
        logger.info("Yahoo Finance has no options of %s (%s).", ticker, error)
        expiry_dates = []

    # Cboe's delayed quotes list every US-listed option when Yahoo Finance has none.
    if not expiry_dates:
        chain = cboe_model.get_option_chain(ticker)
        expiry_dates = sorted(chain["Expiration"].unique()) if not chain.empty else []

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
    Retrieve option chains (calls or puts) for a list of tickers and a specific expiration date.

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

        underlying_price = np.nan
        try:
            option_chain = yf.Ticker(ticker).option_chain(expiration_date)
            options_df = option_chain.puts if put_option else option_chain.calls
            # The price of the underlying quoted with the chain, so metrics that combine
            # the two use prices of the same moment rather than the last historical close.
            underlying_price = (option_chain.underlying or {}).get(
                "regularMarketPrice", np.nan
            )
        except Exception as error:  # noqa: BLE001 - Yahoo Finance fails in many ways
            logger.info("Yahoo Finance has no option chain of %s (%s).", ticker, error)
            options_df = pd.DataFrame()

        if options_df.empty:
            options_df = _get_cboe_option_chain(ticker, expiration_date, put_option)
            if options_df.empty:
                continue
            result_dict[ticker] = options_df
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


def _get_cboe_option_chain(
    ticker: str, expiration_date: str, put_option: bool
) -> pd.DataFrame:
    """
    Retrieves the calls or puts of one expiry from Cboe's delayed quotes, in the layout of
    the Yahoo Finance option chain.

    Args:
        ticker (str): The ticker, e.g. "AAPL" or "^SPX".
        expiration_date (str): The expiry (YYYY-MM-DD).
        put_option (bool): Whether to return the puts instead of the calls.

    Returns:
        pd.DataFrame: The options, indexed by strike price, with Cboe's current price of the
        underlying in the "Underlying Price" column.
    """
    chain = cboe_model.get_option_chain(ticker)

    if chain.empty:
        return chain

    chain = chain[
        (chain["Expiration"] == expiration_date) & (chain["Put"] == put_option)
    ]

    # Index options can list a monthly and a weekly contract (SPX and SPXW) on the same
    # strike and expiry; the one more widely held is kept.
    chain = chain.sort_values("Open Interest").drop_duplicates("Strike", keep="last")
    in_the_money = (
        chain["Strike"] > chain["Underlying Price"]
        if put_option
        else chain["Strike"] < chain["Underlying Price"]
    )

    return (
        chain.assign(
            Currency="USD",
            **{"Last Trade Date": None, "In The Money": in_the_money},
        )[
            [
                "Contract Symbol",
                "Strike",
                "Currency",
                "Last Price",
                "Change",
                "Percent Change",
                "Volume",
                "Open Interest",
                "Bid",
                "Ask",
                "Expiration",
                "Last Trade Date",
                "Implied Volatility",
                "In The Money",
                "Underlying Price",
            ]
        ]
        .set_index("Strike")
        .sort_index()
    )


# Treasury yields are quoted on a bond-equivalent basis: an annual rate compounded twice a year.
TREASURY_COMPOUNDING_PER_YEAR = 2


def get_continuously_compounded_rate(
    bond_equivalent_yield: pd.Series | pd.DataFrame | float,
    compounding_per_year: int = TREASURY_COMPOUNDING_PER_YEAR,
) -> pd.Series | pd.DataFrame | float:
    """
    Converts a yield quoted with periodic compounding, such as the bond-equivalent yield of a
    US Treasury, into the continuously compounded rate that the option pricing formulas
    (Black-Scholes, binomial trees, Bjerksund-Stensland, barrier, Asian and binary options and
    the Greeks) discount with through e^(-r * t). Both rates grow an investment by the same
    amount over a year:

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
    Calculates the trailing dividend yield on every date: the dividends per share that went
    ex-dividend in the 365 days up to and including that date, divided by the close of that
    date. This is the dividend yield q of the Black-Scholes model on a pricing date, built
    only from what was known on that date.

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
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Calculate the price of a European option through Monte Carlo simulation of
    Geometric Brownian Motion (GBM) stock price paths.

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
            `np.random.default_rng`, ensuring reproducible results. Defaults to None,
            which means the results will not be reproducible.

    Returns:
        tuple[float, float]: a tuple of (option_price, standard_error) where option_price
        is the Monte Carlo estimate of the option's fair value and standard_error is the
        standard error of that estimate.

    Raises:
        TypeError: if any of the numeric inputs is not a float or int.
        ValueError: if simulations or time_steps is not a positive integer.

    Notes:
        Reference: Boyle, P. P. (1977). "Options: A Monte Carlo Approach." Journal of
        Financial Economics, 4(3), 323-338.
    """
    for name, value in {
        "stock_price": stock_price,
        "strike_price": strike_price,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "time_to_expiration": time_to_expiration,
        "dividend_yield": dividend_yield,
    }.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a float or int, received {type(value).__name__}."
            )

    if not isinstance(simulations, int) or simulations <= 0:
        raise ValueError(
            f"simulations must be a positive integer, received {simulations!r}."
        )

    if not isinstance(time_steps, int) or time_steps <= 0:
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
    )

    return float(prices[0]), float(standard_errors[0])


def _simulate_terminal_stock_prices(
    stock_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
    simulations: int,
    time_steps: int,
    seed: int | None,
) -> np.ndarray:
    """
    Simulates the stock price at expiration of Geometric Brownian Motion paths under the
    risk-neutral measure.

    Every path is built from time_steps shocks as before, so a seed gives the same terminal
    prices; only the end of each path is exponentiated, since a European payoff needs
    nothing else.

    Args:
        stock_price (float): The current stock price.
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The dividend yield.
        simulations (int): The number of paths.
        time_steps (int): The number of steps per path.
        seed (int | None): The seed of the random number generator.

    Returns:
        np.ndarray: The stock price at expiration of every path.
    """
    random_number_generator = np.random.default_rng(seed)

    time_delta = time_to_expiration / time_steps
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_delta
    diffusion = volatility * np.sqrt(time_delta)

    random_shocks = random_number_generator.standard_normal((simulations, time_steps))
    log_returns = drift + diffusion * random_shocks
    log_paths = np.cumsum(log_returns, axis=1)

    return stock_price * np.exp(log_paths[:, -1])


def get_monte_carlo_option_prices(
    stock_price: float,
    strike_prices: np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    simulations: int = 10_000,
    time_steps: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Monte Carlo price of European options with several strike prices from
    one set of simulated paths, see get_monte_carlo_option_price. Pricing every strike on
    the same paths (common random numbers) gives the same prices as simulating per strike
    with the same seed, keeps the prices consistent across strikes and simulates once
    rather than once per strike.

    Args:
        stock_price (float): The current stock price.
        strike_prices (np.ndarray): The strike prices.
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The dividend yield. Defaults to 0.0.
        put_option (bool, optional): Whether to price puts instead of calls. Defaults to
            False.
        simulations (int, optional): The number of simulated paths. Defaults to 10,000.
        time_steps (int, optional): The number of steps per path. Defaults to 100.
        seed (int | None, optional): The seed of the random number generator. Defaults to
            None.

    Returns:
        tuple[np.ndarray, np.ndarray]: The price and its standard error per strike price.
    """
    terminal_stock_prices = _simulate_terminal_stock_prices(
        stock_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        simulations,
        time_steps,
        seed,
    )
    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    option_prices = np.empty(len(strike_prices))
    standard_errors = np.empty(len(strike_prices))

    # One strike at a time keeps the memory to one payoff per path.
    for position, strike_price in enumerate(strike_prices):
        if put_option:
            payoffs = np.maximum(strike_price - terminal_stock_prices, 0)
        else:
            payoffs = np.maximum(terminal_stock_prices - strike_price, 0)

        discounted_payoffs = discount_factor * payoffs
        option_prices[position] = discounted_payoffs.mean()
        standard_errors[position] = discounted_payoffs.std(ddof=1) / np.sqrt(
            simulations
        )

    return option_prices, standard_errors
