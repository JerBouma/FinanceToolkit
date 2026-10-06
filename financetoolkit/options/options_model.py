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
        pd.DataFrame: A DataFrame containing the option chains for the specified tickers and expiration date.
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

        try:
            option_chain = yf.Ticker(ticker).option_chain(expiration_date)
            options_df = option_chain.puts if put_option else option_chain.calls
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
        pd.DataFrame: The options, indexed by strike price.
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
            ]
        ]
        .set_index("Strike")
        .sort_index()
    )


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

    random_number_generator = np.random.default_rng(seed)

    time_delta = time_to_expiration / time_steps
    drift = (risk_free_rate - dividend_yield - 0.5 * volatility**2) * time_delta
    diffusion = volatility * np.sqrt(time_delta)

    random_shocks = random_number_generator.standard_normal((simulations, time_steps))
    log_returns = drift + diffusion * random_shocks
    log_paths = np.cumsum(log_returns, axis=1)
    stock_price_paths = stock_price * np.exp(log_paths)

    terminal_stock_prices = stock_price_paths[:, -1]

    if put_option:
        payoffs = np.maximum(strike_price - terminal_stock_prices, 0)
    else:
        payoffs = np.maximum(terminal_stock_prices - strike_price, 0)

    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    discounted_payoffs = discount_factor * payoffs

    option_price = float(discounted_payoffs.mean())
    standard_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(simulations))

    return option_price, standard_error
