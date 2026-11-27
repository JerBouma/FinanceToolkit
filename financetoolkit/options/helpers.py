"""Option Helpers Module"""

__docformat__ = "google"

import numbers
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri, owens_t
from scipy.stats import qmc

from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import (
    InvalidArgumentError,
    check_option,
    validate_window,
)

logger = logger_model.get_logger()


# The most strike prices per ticker: every one is priced for each expiration (and in a
# binomial model has a tree of its own), so a grid of a high share price in small steps
# would not fit in memory.
MAX_STRIKES = 10_000
# The most option values per ticker, strike prices times expirations (or the nodes of a
# binomial tree): the closed forms take about a microsecond a value on arrays, but QD+, the
# Monte Carlo price and the table of the answer grow with every value. A server that answers
# within a time limit sets a lower value at startup.
MAX_OPTION_VALUES = 10_000_000


def define_strike_prices(
    tickers: list[str],
    stock_price: pd.DataFrame,
    strike_step_size: float,
    strike_price_range: float,
    values_per_strike: int = 1,
):
    """
    Define strike prices for each ticker in the tickers list. The strike prices run from
    strike_price_range below the current stock price, rounded down to a multiple of
    strike_step_size, up to and including strike_price_range above it, rounded up, so the
    whole range is covered (a price of 100 with a range of 0.25 and a step of 5 gives 75, 80,
    ..., 125). Strikes are positive: a range that reaches zero starts at the first step, and a
    price below the step size still gets the strike of one step. With a range of 0 the only
    strike is the multiple of strike_step_size nearest to the price.

    Args:
        tickers (list[str]): the list of tickers.
        stock_price (pd.DataFrame): the current stock price for each ticker.
        strike_step_size (float): the strike step size above 0, a whole number (5) or a
            fraction (2.5).
        strike_price_range (float): the strike price range as a fraction of the price,
            between 0 and 1.
        values_per_strike (int, optional): the values calculated per strike price, such as
            the number of expirations (times the nodes of a tree, or a weight for the Monte
            Carlo values that evaluate every path), which with the strikes may be at most
            MAX_OPTION_VALUES per ticker. Defaults to 1.

    Returns:
        dict: a dictionary of strike prices for each ticker, without the tickers that have no
        price. The strikes are integers when the step size is an integer.

    Raises:
        InvalidArgumentError: If the range is not between 0 and 1, the step is not above 0, or
            the strikes or option values exceed MAX_STRIKES or MAX_OPTION_VALUES for a ticker.
    """
    # A range above 1 would put strikes at or below zero (they are clamped to one step anyway),
    # and the negated comparisons also refuse NaN.
    if not 0 <= strike_price_range <= 1:
        raise InvalidArgumentError(
            "strike_price_range is a fraction of the stock price between 0 and 1, e.g. 0.25 for "
            f"strikes from 75% to 125% of the price, not {strike_price_range!r}. Please pass a "
            "value between 0 and 1 as strike_price_range."
        )
    if not strike_step_size > 0:
        raise InvalidArgumentError(
            f"strike_step_size is the distance between strike prices, above 0, not {strike_step_size!r}. "
            "Please pass a positive step such as 5 or 2.5 as strike_step_size."
        )

    # Integer steps give integer strikes (100 rather than 100.0); fractional steps are
    # rounded to remove the floating point noise of step multiples such as 3 * 0.1.
    whole_steps = float(strike_step_size).is_integer()

    strike_prices_per_ticker = {}

    for ticker in tickers:
        # A ticker without a price on the date (not listed yet, delisted or suspended) has
        # no strikes around it, and is left out rather than failing the other tickers.
        if pd.isna(stock_price.get(ticker)):
            logger.warning(
                "%s has no price on %s, so it is left out of the option values.",
                ticker,
                stock_price.name,
            )
            continue

        price = float(stock_price.loc[ticker])

        if (
            strike_count := 2 * strike_price_range * price / strike_step_size
        ) > MAX_STRIKES:
            raise InvalidArgumentError(
                f"The strike prices of {ticker} would be about {strike_count:,.0f} prices; at "
                f"most {MAX_STRIKES:,} can be calculated. Please use a larger "
                "strike_step_size or a smaller strike_price_range."
            )

        # Strikes are counted in steps, so the grid is exact whatever the step size: the
        # lowest is the step at or below the bottom of the range (at least one step, since a
        # strike of zero or below has no option), the highest the step at or above its top.
        # The tolerance keeps a bound that is a multiple of the step (100 * 1.25 / 5) from
        # moving a step out through floating point error.
        lowest_step = max(
            np.floor(price * (1 - strike_price_range) / strike_step_size + 1e-9), 1
        )
        highest_step = max(
            np.ceil(price * (1 + strike_price_range) / strike_step_size - 1e-9),
            lowest_step,
        )
        if strike_price_range == 0:
            lowest_step = highest_step = max(round(price / strike_step_size), 1)

        strikes = np.arange(lowest_step, highest_step + 1) * strike_step_size

        if (option_values := len(strikes) * values_per_strike) > MAX_OPTION_VALUES:
            raise InvalidArgumentError(
                f"{ticker} would have {len(strikes):,} strike prices with {values_per_strike:,} "
                f"values each, {option_values:,} option values; at most {MAX_OPTION_VALUES:,} "
                "can be calculated per ticker. Please use a larger strike_step_size, a smaller "
                "strike_price_range, or fewer expirations (expiration_time_range), tree "
                "steps (time_steps) or simulations."
            )

        strike_prices_per_ticker[ticker] = (
            [int(round(strike)) for strike in strikes]
            if whole_steps
            else [round(float(strike), 10) for strike in strikes]
        )

    return strike_prices_per_ticker


def validate_time_to_expiration(time_to_expiration: float) -> None:
    """
    Check that a time to expiration in years is above 0.

    Args:
        time_to_expiration (float): The time to expiration in years.

    Raises:
        InvalidArgumentError: If it is not a number above 0 (a ValueError).
    """
    if not time_to_expiration > 0:
        raise InvalidArgumentError(
            f"time_to_expiration is a number of years above 0, e.g. 1 for one year, not "
            f"{time_to_expiration!r}. Please pass a positive time_to_expiration."
        )


def define_times_to_expiration(expiration_time_range: int) -> list[float]:
    """
    Define the times to expiration of an option grid: 0 up to expiration_time_range - 1 days,
    in years. The first is the start date itself, which the tables leave out, so the range is
    at least 2 to give one expiration.

    Args:
        expiration_time_range (int): the number of days, at least 2.

    Returns:
        list[float]: the times to expiration in years (days / 365).

    Raises:
        InvalidArgumentError: if expiration_time_range is not a whole number of at least 2.
    """
    validate_window("expiration_time_range", expiration_time_range, minimum=2)

    return [time / 365 for time in range(expiration_time_range)]


def create_greek_dataframe(
    greek_dictionary: dict[str, dict[float, dict[float, float]]], start_date: str
):
    """
    Creates the DataFrame that correctly displays the greeks for each ticker and strike price.
    over time (the time of expiration).

    Args:
        greek_dictionary (dict[str, dict[float, dict[float, float]]]): a dictionary
        containing the greeks for each ticker, strike price and expiration date.
        start_date (str): the start date that should be excluded out of the DataFrame.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the greeks for each ticker
        and strike price.
    """
    if not greek_dictionary:
        # Every ticker left out (each with a warning of why) leaves nothing to show.
        return pd.DataFrame()

    greek_dataframe = pd.DataFrame.from_dict(
        {
            (ticker, strike_price): values
            for ticker, strike_prices in greek_dictionary.items()
            for strike_price, values in strike_prices.items()
        },
        orient="index",
    )

    greek_dataframe.columns = pd.period_range(
        start=start_date,
        periods=len(greek_dataframe.columns),
        freq="D",
    )

    greek_dataframe = greek_dataframe.drop(start_date, axis=1)

    greek_dataframe.index.names = ["Ticker", "Strike Price"]

    return greek_dataframe


def create_expiration_dataframe(
    value_dictionary: dict[str, dict[float, float]], start_date: str
) -> pd.DataFrame:
    """
    Creates the DataFrame of an option without a strike price (such as a floating strike
    lookback or a touch option), with the tickers as the index and the expiration dates as
    the columns.

    Args:
        value_dictionary (dict[str, dict[float, float]]): The value per ticker and time to
            expiration.
        start_date (str): The start date, which is left out of the DataFrame.

    Returns:
        pd.DataFrame: The values with the tickers as the index and the expiration dates as the
        columns.
    """
    if not value_dictionary:
        # Every ticker left out (each with a warning of why) leaves nothing to show.
        return pd.DataFrame()

    dataframe = pd.DataFrame.from_dict(value_dictionary, orient="index")

    dataframe.columns = pd.period_range(
        start=start_date,
        periods=len(dataframe.columns),
        freq="D",
    )

    dataframe = dataframe.drop(start_date, axis=1)

    dataframe.index.name = "Ticker"

    return dataframe


def create_binomial_tree_dataframe(
    binomial_tree_dictionary: dict[str, tuple[list[float], np.ndarray]],
    start_date: str | pd.Period,
    time_to_expiration: float,
):
    """
    Creates the DataFrame that correctly displays the binomial tree for each ticker and strike price
    over time (up to the time of expiration).

    Args:
        binomial_tree_dictionary (dict[str, tuple[list[float], np.ndarray]]): the strike
        prices of each ticker and their binomial trees, as an array of strikes × nodes ×
        steps in which node k has k down movements (see
        binomial_trees_model.get_option_payoff_trees).
        start_date (str | pd.Period): the date the tree starts from, which labels the
        first column.
        time_to_expiration (float): the time to expiration in years, which sets the date
        of the last column.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the binomial tree for each ticker
        and strike price.
    """
    if not binomial_tree_dictionary:
        # Every ticker left out (each with a warning of why) leaves nothing to show.
        return pd.DataFrame()

    frames = []
    for ticker, (strike_prices, trees) in binomial_tree_dictionary.items():
        steps = trees.shape[1] - 1
        movements = ["D" * downs + "U" * (steps - downs) for downs in range(steps + 1)]
        frames.append(
            pd.DataFrame(
                trees.reshape(-1, trees.shape[2]),
                index=pd.MultiIndex.from_product([[ticker], strike_prices, movements]),
            )
        )
    binomial_tree_dataframe = pd.concat(frames)

    start_date_string = pd.to_datetime(str(start_date))
    end_date_string = pd.to_datetime(
        str(start_date_string + pd.Timedelta(days=time_to_expiration * 365))
    )

    binomial_tree_dataframe.columns = pd.date_range(
        start=start_date_string,
        end=end_date_string,
        periods=len(binomial_tree_dataframe.columns),
    )

    binomial_tree_dataframe.columns = pd.PeriodIndex(
        binomial_tree_dataframe.columns, freq="D"
    )

    binomial_tree_dataframe.index.names = ["Ticker", "Strike Price", "Movement"]

    return binomial_tree_dataframe


def create_stock_simulation_dataframe(
    stock_simulation_dictionary: dict[str, pd.DataFrame],
    start_date: str | pd.Period,
    time_to_expiration: float,
):
    """
    Creates the DataFrame that correctly displays the simulated stock prices for each ticker over
    time (up to the time of expiration).

    Args:
        stock_simulation_dictionary (dict[str, pd.DataFrame]): the simulated stock prices
        per ticker, one DataFrame of price paths per ticker.
        start_date (str | pd.Period): the date the simulation starts from, which labels
        the first column.
        time_to_expiration (float): the time to expiration in years.

    Returns:
        pd.DataFrame: the DataFrame that correctly displays the simulated stock prices for
        each ticker over time.
    """
    if not stock_simulation_dictionary:
        # Every ticker left out (each with a warning of why) leaves nothing to show.
        return pd.DataFrame()

    stock_simulation_dataframe = pd.concat(stock_simulation_dictionary)

    start_date_string = pd.to_datetime(str(start_date))
    end_date_string = pd.to_datetime(
        str(start_date_string + pd.Timedelta(days=time_to_expiration * 365))
    )

    stock_simulation_dataframe.columns = pd.date_range(
        start=start_date_string,
        end=end_date_string,
        periods=len(stock_simulation_dataframe.columns),
    )

    stock_simulation_dataframe.columns = pd.PeriodIndex(
        stock_simulation_dataframe.columns, freq="D"
    )

    stock_simulation_dataframe.index.names = ["Ticker", "Movement"]

    return stock_simulation_dataframe


def show_input_info(
    start_date: str,
    end_date: str,
    stock_prices: pd.Series,
    volatility: pd.Series,
    risk_free_rate: float,
    dividend_yield: dict[str, float] | None = None,
    up_movement_dict: dict | None = None,
    down_movement_dict: dict | None = None,
    risk_neutral_probability_dict: dict | None = None,
):
    """
    Based on the input parameters, log the input information.

    Args:
        start_date (str): the start date.
        end_date (str): the end date.
        stock_prices (pd.Series): the stock price for each ticker.
        volatility (pd.Series): the volatility for each ticker.
        risk_free_rate (float): the risk free rate.
        dividend_yield (dict[str, float] | None): the dividend yield for each ticker. Defaults
        to None, which leaves it out.
        up_movement_dict (dict | None): the up movement of the binomial tree for each ticker.
        Defaults to None, which leaves it out.
        down_movement_dict (dict | None): the down movement of the binomial tree for each
        ticker. Defaults to None, which leaves it out.
        risk_neutral_probability_dict (dict | None): the risk neutral probability of an up
        movement for each ticker. Defaults to None, which leaves it out.

    Returns:
        None: the input information is logged.
    """
    stock_price_list = [
        (f"{ticker} ({round(stock_price_value, 2)})")
        for ticker, stock_price_value in stock_prices.items()
    ]
    volatility_list = [
        (f"{ticker} ({round(volatility_value * 100, 2)}%)")
        for ticker, volatility_value in volatility.items()
    ]

    stock_price_str = ", ".join(stock_price_list)
    volatility_str = ", ".join(volatility_list)

    logger.info(
        "Based on the period %s to %s the following parameters were used:\n"
        "Stock Price: %s\n"
        "Volatility: %s",
        start_date,
        end_date,
        stock_price_str,
        volatility_str,
    )

    if dividend_yield is not None:
        dividend_yield_list = [
            (f"{ticker} ({round(dividend_yield_value * 100, 2)}%)")
            for ticker, dividend_yield_value in dividend_yield.items()
        ]
        logger.info("Dividend Yield: %s", ", ".join(dividend_yield_list))

    if up_movement_dict is not None:
        up_movement_list = [
            (f"{ticker} ({round(up_movement * 100, 2)}%)")
            for ticker, up_movement in up_movement_dict.items()
        ]
        logger.info("Up Movement: %s", ", ".join(up_movement_list))

    if down_movement_dict is not None:
        down_movement_list = [
            (f"{ticker} ({round(down_movement * 100, 2)}%)")
            for ticker, down_movement in down_movement_dict.items()
        ]
        logger.info("Down Movement: %s", ", ".join(down_movement_list))

    if risk_neutral_probability_dict is not None:
        risk_neutral_probability_list = [
            (f"{ticker} ({round(risk_neutral_probability * 100, 2)}%)")
            for ticker, risk_neutral_probability in risk_neutral_probability_dict.items()
        ]
        logger.info(
            "Risk Neutral Probability: %s", ", ".join(risk_neutral_probability_list)
        )

    logger.info("Risk Free Rate: %s%%", round(risk_free_rate * 100, 2))


# The most combinations of strike price and expiration evaluate_on_grid passes to a formula at
# once.
GRID_CHUNK_SIZE = 100_000


def evaluate_on_grid(
    function: Callable[[Any, Any], Any],
    strike_prices: list[float],
    times_to_expiration: list[float],
) -> dict[float, dict[float, float]]:
    """
    Evaluates an option formula for every strike price and time to expiration in one call
    on arrays, instead of once per combination, which is as exact and many times faster
    since the formulas are numpy operations. A formula that does not accept arrays is
    evaluated per combination instead.

    Args:
        function (Callable): The formula, a function of the strike price and the time to
            expiration.
        strike_prices (list[float]): The strike prices.
        times_to_expiration (list[float]): The times to expiration in years.

    Returns:
        dict[float, dict[float, float]]: The value per strike price and time to expiration.
    """
    strikes = list(strike_prices)
    times = list(times_to_expiration)
    grid_strikes = np.repeat(np.asarray(strikes, dtype=float), len(times))
    grid_times = np.tile(np.asarray(times, dtype=float), len(strikes))

    try:
        # In chunks, so the intermediate arrays of a formula (such as the images of a double
        # barrier) stay small on the widest grids.
        values = np.concatenate(
            [
                np.asarray(
                    function(
                        grid_strikes[start : start + GRID_CHUNK_SIZE],
                        grid_times[start : start + GRID_CHUNK_SIZE],
                    ),
                    dtype=float,
                ).reshape(-1)
                for start in range(0, len(grid_strikes), GRID_CHUNK_SIZE)
            ]
            or [np.empty(0)]
        )
        if values.shape != grid_strikes.shape:
            raise ValueError("The formula does not return a value per combination.")
    except (ValueError, TypeError):
        return {
            strike: {time: function(strike, time) for time in times}
            for strike in strikes
        }

    values = values.reshape(len(strikes), len(times))

    return {
        strike: {time: float(values[row, column]) for column, time in enumerate(times)}
        for row, strike in enumerate(strikes)
    }


# The random numbers of a Monte Carlo price: pseudo-random draws of numpy's default generator
# (PCG64) or scrambled Sobol points, and the variance reductions applied to them.
MONTE_CARLO_SEQUENCES = ("pseudo_random", "sobol")
VARIANCE_REDUCTIONS = (None, "antithetic", "control_variate")
# Sobol prices are averaged over this many independently scrambled sequences, whose spread
# gives the standard error (randomized quasi-Monte Carlo, Owen 1997): the points of a single
# sequence are not independent, so their own spread overstates the error.
SOBOL_REPLICATIONS = 16
# The most dimensions (time steps times assets) of scipy's Sobol direction numbers (Joe and
# Kuo, 2008).
MAX_SOBOL_DIMENSIONS = 21_201


def _brownian_bridge_order(time_steps: int) -> list[tuple[int, int, int]]:
    """
    Orders the points of a Brownian path on time_steps equal steps as a Brownian bridge
    fills them: the end first, then the midpoint of every interval, level by level.

    Args:
        time_steps (int): The number of steps of the path.

    Returns:
        list[tuple[int, int, int]]: Per point the step it sits at and the steps of the two
        points it lies between, already set (the start, 0, for the end point).
    """
    order = [(time_steps, 0, 0)]
    intervals = [(0, time_steps)]

    while intervals:
        left, right = intervals.pop(0)
        if right - left > 1:
            middle = (left + right) // 2
            order.append((middle, left, right))
            intervals += [(left, middle), (middle, right)]

    return order


def _brownian_bridge(
    normals: np.ndarray, time_steps: int, dimensions: int
) -> np.ndarray:
    """
    Builds Brownian paths with unit steps from standard normal numbers through a Brownian
    bridge and returns their increments, which are independent standard normal numbers in
    distribution. The first numbers of every row set the end of the path and the later ones
    ever finer detail, so the most important coordinates of a Sobol point drive the parts
    of the path that matter most for the price (Moskowitz and Caflisch, 1996):

    - W(m) = ((r - m) * W(l) + (m - l) * W(r)) / (r - l) + √((m - l) * (r - m) / (r - l)) * Z

    Args:
        normals (np.ndarray): The standard normal numbers, rows by time steps times
            dimensions, in the order of _brownian_bridge_order with the dimensions within.
        time_steps (int): The number of steps of every path.
        dimensions (int): The number of Brownian motions per path.

    Returns:
        np.ndarray: The increments, rows by time steps by dimensions.
    """
    rows = normals.shape[0]
    paths = np.zeros((rows, time_steps + 1, dimensions))

    for position, (point, left, right) in enumerate(_brownian_bridge_order(time_steps)):
        draws = normals[:, position * dimensions : (position + 1) * dimensions]
        if left == right:
            paths[:, point] = np.sqrt(point) * draws
        else:
            paths[:, point] = (
                (right - point) * paths[:, left] + (point - left) * paths[:, right]
            ) / (right - left) + np.sqrt(
                (point - left) * (right - point) / (right - left)
            ) * draws

    return np.diff(paths, axis=1)


def _validate_monte_carlo(
    simulations: int, time_steps: int, dimensions: int, sequence: str, antithetic: bool
) -> None:
    """
    Raises a ValueError for random numbers that can not be drawn as asked.

    Args:
        simulations (int): The number of paths.
        time_steps (int): The number of steps per path.
        dimensions (int): The number of Brownian motions per path.
        sequence (str): The numbers drawn. Options are: "pseudo_random" (pseudo random
            numbers), "sobol" (scrambled Sobol points).
        antithetic (bool): Whether the paths come in antithetic pairs.
    """
    check_option("sequence", sequence, MONTE_CARLO_SEQUENCES)
    if not isinstance(simulations, numbers.Integral) or isinstance(simulations, bool):
        raise ValueError(f"simulations must be an integer, received {simulations!r}.")
    if not isinstance(time_steps, numbers.Integral) or time_steps <= 0:
        raise ValueError(
            f"time_steps must be a positive integer, received {time_steps!r}."
        )
    if antithetic and simulations % 2:
        raise ValueError(
            f"Antithetic paths come in pairs, so simulations must be even, received "
            f"{simulations!r}."
        )
    minimum = 2 * SOBOL_REPLICATIONS if sequence == "sobol" else 2 if antithetic else 1
    if simulations < minimum:
        raise ValueError(
            f"simulations must be at least {minimum} with sequence={sequence!r}, received "
            f"{simulations!r}."
        )
    if sequence == "sobol" and time_steps * dimensions > MAX_SOBOL_DIMENSIONS:
        raise ValueError(
            f"Sobol points have at most {MAX_SOBOL_DIMENSIONS:,} dimensions, received "
            f"{time_steps:,} time steps of {dimensions} assets."
        )


def _replication_sizes(simulations: int, antithetic: bool) -> np.ndarray:
    """
    Splits the paths over the Sobol replications as evenly as possible, in whole pairs for
    antithetic paths.

    Args:
        simulations (int): The number of paths.
        antithetic (bool): Whether the paths come in antithetic pairs.

    Returns:
        np.ndarray: The number of paths of every replication.
    """
    pair = 2 if antithetic else 1
    units = simulations // pair
    sizes = np.full(SOBOL_REPLICATIONS, units // SOBOL_REPLICATIONS)
    sizes[: units % SOBOL_REPLICATIONS] += 1
    return sizes * pair


def draw_standard_normals(
    simulations: int,
    time_steps: int,
    dimensions: int = 1,
    sequence: str = "pseudo_random",
    antithetic: bool = False,
    seed: int | None = None,
    chunk_size: int = 10_000,
) -> Iterator[np.ndarray]:
    """
    Draws the independent standard normal increments of Monte Carlo paths in chunks of
    paths, so a simulation holds one chunk at a time.

    - pseudo_random: numpy's default generator (PCG64) seeded with seed fills the numbers
      row by row, so the chunks give the same numbers as one draw of all paths.
    - sobol: scrambled Sobol points (Sobol 1967, direction numbers of Joe and Kuo 2008,
      scipy.stats.qmc.Sobol) turned into normal numbers with the inverse normal distribution
      and into paths with a Brownian bridge. The paths are split over SOBOL_REPLICATIONS
      independently scrambled sequences (seeded from seed), so the spread of their means
      gives the standard error.
    - antithetic: every path is followed by its mirror image, the same numbers with the
      opposite sign (Boyle, 1977).

    Args:
        simulations (int): The number of paths.
        time_steps (int): The number of steps per path.
        dimensions (int, optional): The number of Brownian motions per path. Defaults to 1.
        sequence (str, optional): The numbers to draw. Options are: "pseudo_random" (pseudo
            random numbers), "sobol" (scrambled Sobol points). Defaults to "pseudo_random".
        antithetic (bool, optional): Whether to draw antithetic pairs. Defaults to False.
        seed (int | None, optional): The seed of the numbers. Defaults to None.
        chunk_size (int, optional): The most paths per chunk; an even number. Defaults to
            10,000.

    Yields:
        np.ndarray: The increments of a chunk of paths, paths by time steps by dimensions.

    Raises:
        ValueError: If the sequence is unknown, simulations is odd for antithetic paths or
            too small, or a Sobol point would need too many dimensions.
    """
    _validate_monte_carlo(simulations, time_steps, dimensions, sequence, antithetic)
    shape = (time_steps, dimensions)

    def mirrored(normals: np.ndarray) -> np.ndarray:
        if not antithetic:
            return normals
        pairs = np.empty((2 * normals.shape[0], *normals.shape[1:]))
        pairs[0::2] = normals
        pairs[1::2] = -normals
        return pairs

    pair = 2 if antithetic else 1

    if sequence == "pseudo_random":
        random_number_generator = np.random.default_rng(seed)
        for first in range(0, simulations, chunk_size):
            rows = min(chunk_size, simulations - first) // pair
            yield mirrored(random_number_generator.standard_normal((rows, *shape)))
        return

    seeds = np.random.SeedSequence(seed).spawn(SOBOL_REPLICATIONS)

    for size, replication_seed in zip(
        _replication_sizes(simulations, antithetic), seeds, strict=True
    ):
        engine = qmc.Sobol(
            d=time_steps * dimensions,
            scramble=True,
            seed=np.random.default_rng(replication_seed),
        )
        for first in range(0, int(size), chunk_size):
            rows = min(chunk_size, int(size) - first) // pair
            # Scipy warns when the number of points is not a power of two, the sizes at
            # which a Sobol sequence is best balanced; any number of points remains valid.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                uniforms = engine.random(rows)
            normals = ndtri(np.clip(uniforms, 1e-16, 1 - 1e-16))
            yield mirrored(_brownian_bridge(normals, time_steps, dimensions))


def simulation_groups(simulations: int, sequence: str, antithetic: bool) -> np.ndarray:
    """
    Labels the independent groups of the paths of draw_standard_normals, whose means give
    the standard error of a Monte Carlo price: every path on its own, every antithetic pair,
    or every Sobol replication.

    Args:
        simulations (int): The number of paths.
        sequence (str): The numbers drawn. Options are: "pseudo_random" (pseudo random
            numbers), "sobol" (scrambled Sobol points).
        antithetic (bool): Whether the paths come in antithetic pairs.

    Returns:
        np.ndarray: The group of every path.
    """
    if sequence == "sobol":
        return np.repeat(
            np.arange(SOBOL_REPLICATIONS), _replication_sizes(simulations, antithetic)
        )
    return np.arange(simulations) // (2 if antithetic else 1)


def monte_carlo_estimate(values: np.ndarray, groups: np.ndarray):
    """
    Returns the Monte Carlo estimate, the average of the values of the paths, and its
    standard error from the spread of the means of the independent groups of paths
    (ddof=1).

    Args:
        values (np.ndarray): The discounted payoff of every path, or a matrix with a row of
            them per option priced on the same paths.
        groups (np.ndarray): The group of every path (simulation_groups), in order.

    Returns:
        tuple[float, float] | tuple[np.ndarray, np.ndarray]: The estimate and its standard
        error, per row for a matrix.
    """
    counts = np.bincount(groups)
    if values.ndim == 2:  # noqa: PLR2004
        starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        means = np.add.reduceat(values, starts, axis=1) / counts
        return values.mean(axis=1), means.std(axis=1, ddof=1) / np.sqrt(counts.size)
    means = np.bincount(groups, weights=values) / counts
    return float(values.mean()), float(means.std(ddof=1) / np.sqrt(means.size))


def control_variate(
    values: np.ndarray, control: np.ndarray, expected_control: float
) -> np.ndarray:
    """
    Subtracts the simulation error of a control whose expected value is known from the
    values of the paths (Glasserman, 2003, section 4.1):

    - Y_i - β * (X_i - E[X]), β = Cov(Y, X) / Var(X)

    Args:
        values (np.ndarray): The discounted payoff of every path, Y, or a matrix with a row
            of them per option priced on the same paths (each with its own β).
        control (np.ndarray): The control of every path, X.
        expected_control (float): The expected value of the control, E[X].

    Returns:
        np.ndarray: The controlled values; the values themselves when the control does not
        vary.
    """
    variance = control.var(ddof=1)
    if variance <= 0:
        return values
    if values.ndim == 2:  # noqa: PLR2004
        deviations = control - control.mean()
        beta = (
            (values - values.mean(axis=1, keepdims=True))
            @ deviations
            / (control.size - 1)
            / variance
        )
        return values - beta[:, None] * (control - expected_control)
    beta = np.cov(values, control, ddof=1)[0, 1] / variance
    return values - beta * (control - expected_control)


def bivariate_normal_cdf(first, second, correlation):
    """
    The bivariate standard normal distribution function M(h, k; ρ) = P(X ≤ h, Y ≤ k) of two
    standard normal variables with correlation ρ, from Owen's T function (Owen, 1956):

    - M(h, k; ρ) = N(h) / 2 + N(k) / 2 - T(h, (k - ρh) / (h√(1 - ρ²))) - T(k, (h - ρk) / (k√(1 - ρ²))) - β
    - β = 0 when hk > 0, or hk = 0 and h + k ≥ 0, and 1/2 otherwise

    This is exact to machine precision, where a numerical integration of the density (such
    as scipy's multivariate normal distribution function) stops at about 1e-6.

    Args:
        first (float | np.ndarray): The upper limit h of the first variable.
        second (float | np.ndarray): The upper limit k of the second variable.
        correlation (float | np.ndarray): The correlation ρ, between -1 and 1.

    Returns:
        float | np.ndarray: The probability M(h, k; ρ).
    """
    h, k, rho = np.broadcast_arrays(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
        np.asarray(correlation, dtype=float),
    )
    # T(0, a) has a limit as a goes to infinity, so a limit of zero is moved off zero by an
    # amount far below the precision of the result.
    h = np.where(h == 0, 1e-15, h)
    k = np.where(k == 0, 1e-15, k)
    root = np.sqrt(np.maximum(1 - rho**2, 1e-300))

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = (
            0.5 * ndtr(h)
            + 0.5 * ndtr(k)
            - owens_t(h, (k - rho * h) / (h * root))
            - owens_t(k, (h - rho * k) / (k * root))
            - np.where(h * k > 0, 0.0, 0.5)
        )

    # A limit at infinity leaves the distribution of the other variable, and a correlation of
    # one or minus one makes the two variables the same or opposite.
    value = np.where(np.isposinf(h), ndtr(k), value)
    value = np.where(np.isposinf(k), ndtr(h), value)
    value = np.where(np.isneginf(h) | np.isneginf(k), 0.0, value)
    value = np.where(rho >= 1, ndtr(np.minimum(h, k)), value)
    value = np.where(rho <= -1, np.maximum(ndtr(h) + ndtr(k) - 1, 0.0), value)
    value = np.clip(value, 0.0, 1.0)

    return value if value.ndim else float(value)


def validate_numeric_inputs(arrays: bool = False, **kwargs) -> None:
    """
    Raises a TypeError for an option input that is not a number. A boolean is not taken as a
    number, even though Python counts it as an int.

    Args:
        arrays (bool, optional): Whether an np.ndarray or pd.Series is accepted as well, for
            the formulas that price a whole range of inputs at once. An input of None (an
            optional input left out) is then skipped. Defaults to False.
        **kwargs: The inputs to validate, keyed by argument name.

    Raises:
        TypeError: If any of the inputs is not a float or int (or, when arrays are
            accepted, an np.ndarray or pd.Series).
    """
    for name, value in kwargs.items():
        if arrays and value is None:
            continue
        # numbers.Real also takes numpy's numbers, such as an np.int64 from a DataFrame.
        if not isinstance(
            value,
            (numbers.Real, np.ndarray, pd.Series) if arrays else numbers.Real,
        ) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a float, int, np.ndarray or pd.Series, "
                f"received {type(value).__name__}."
                if arrays
                else f"{name} must be a float or int, received {type(value).__name__}."
            )


def validate_positive_inputs(allow_zero: bool = False, **kwargs) -> None:
    """
    Raises a ValueError for an option input that is below zero, or at zero unless allow_zero,
    such as a negative volatility or a strike price of zero that a formula divides by. NaN
    and None (an optional input left out) pass, as they give NaN rather than a wrong number.

    Args:
        allow_zero (bool, optional): Whether zero is accepted. Defaults to False.
        **kwargs: The inputs to validate, keyed by argument name: numbers or arrays.

    Raises:
        ValueError: If any value of an input is below zero, or at zero unless allow_zero.
    """
    for name, value in kwargs.items():
        if value is None:
            continue
        values = np.asarray(value, dtype=float)
        invalid = values < 0 if allow_zero else values <= 0
        if np.any(invalid):
            raise ValueError(
                f"{name} must be {'0 or above' if allow_zero else 'above 0'}, received "
                f"{float(values[invalid].flat[0])!r}."
            )


# Listed options expire at the close of trading on their expiration date, 16:00 in New York,
# and the time to expiration is counted in minutes over the 525,600 minutes of a 365-day year,
# as the Cboe does for the VIX. The AM-settled monthly index options (such as the standard SPX
# expiry) settle at the open instead, which this close slightly overstates.
EXPIRATION_TIMEZONE = "America/New_York"
EXPIRATION_HOUR = 16
MINUTES_PER_YEAR = 525_600


def get_current_time() -> pd.Timestamp:
    """
    The current time in New York, the time zone of the expiration times.

    Returns:
        pd.Timestamp: The current time, aware of the New York time zone.
    """
    return pd.Timestamp.now(tz=EXPIRATION_TIMEZONE)


def _in_new_york(current_time: pd.Timestamp | None) -> pd.Timestamp:
    """The given time (New York time when it has no time zone), or the current time."""
    if current_time is None:
        return get_current_time()
    current_time = pd.Timestamp(current_time)
    if current_time.tzinfo is None:
        # A wall-clock time that a daylight saving change skips or repeats takes the later one.
        return current_time.tz_localize(
            EXPIRATION_TIMEZONE, ambiguous=False, nonexistent="shift_forward"
        )
    return current_time.tz_convert(EXPIRATION_TIMEZONE)


def get_time_to_expiration(
    expiration_date: str, current_time: pd.Timestamp | None = None
) -> float:
    """
    The time from the current time to an option's expiration in years: the minutes up to 16:00 New York
    time on the expiration date over the 525,600 minutes of a year. Counting to the time of
    expiration, rather than in whole days, keeps an option that expires this afternoon from
    being priced as if a full day remained.

    Args:
        expiration_date (str): The expiration date (YYYY-MM-DD).
        current_time (pd.Timestamp | None, optional): The time to count from, New York time
            when it has no time zone. Defaults to None, which uses the current time.

    Returns:
        float: The time to expiration in years, zero or negative once the option expired.
    """
    expiration_time = (
        pd.Timestamp(expiration_date).normalize() + pd.Timedelta(hours=EXPIRATION_HOUR)
    ).tz_localize(EXPIRATION_TIMEZONE)
    minutes = (expiration_time - _in_new_york(current_time)).total_seconds() / 60
    return minutes / MINUTES_PER_YEAR


def get_days_to_expiration(
    expiration_date: str, current_time: pd.Timestamp | None = None
) -> int:
    """
    Counts the calendar days from today in New York to an option's expiration date, which
    labels and selects expiries. Prices use get_time_to_expiration, the exact time left.

    Args:
        expiration_date (str): The expiration date (YYYY-MM-DD).
        current_time (pd.Timestamp | None, optional): The time to count from, New York time
            when it has no time zone. Defaults to None, which uses the current time.

    Returns:
        int: The number of days, zero for an option that expires today.
    """
    today = _in_new_york(current_time).tz_localize(None).normalize()
    return (pd.Timestamp(expiration_date).normalize() - today).days


def apply_expiration_limit(
    value,
    limit,
    stock_price,
    strike_price,
    time_to_expiration,
    risk_free_rate,
    volatility,
    dividend_yield=0.0,
):
    """
    Replace a Greek by its limit where no uncertainty is left, at expiration or without
    volatility, and the forward lies off the strike. The closed forms then divide zero by
    zero (or infinity by infinity), while the option is simply worth its discounted
    intrinsic value: the Greeks of the curvature (Gamma, Speed, Vanna, Color and the like)
    are 0 and those of time follow from the intrinsic value. At the strike itself the limit
    is infinite or undefined, so the value of the closed form is kept there.

    Args:
        value: The Greek from its closed form (float, np.ndarray, pd.Series or pd.DataFrame).
        limit: The limit where no uncertainty is left, a number or an array of the same shape.
        stock_price: The stock price.
        strike_price: The strike price.
        time_to_expiration: The time to expiration in years.
        risk_free_rate: The risk free rate.
        volatility: The volatility.
        dividend_yield: The dividend yield. Defaults to 0.0.

    Returns:
        The Greek, of the same type as value, with the limit where no uncertainty is left.
    """
    spread = volatility * np.sqrt(time_to_expiration)
    riskless = np.asarray(spread) == 0
    if not np.any(riskless):
        return value

    # The forward against the strike, as in d1: above it a call ends in the money.
    log_forward_moneyness = np.asarray(
        np.log(stock_price / strike_price)
        + (risk_free_rate - dividend_yield) * time_to_expiration,
        dtype=float,
    )
    values = np.where(
        riskless & (log_forward_moneyness != 0),
        np.asarray(limit, dtype=float),
        np.asarray(value, dtype=float),
    )
    if isinstance(value, pd.Series):
        return pd.Series(values, index=value.index, name=value.name)
    if isinstance(value, pd.DataFrame):
        return pd.DataFrame(values, index=value.index, columns=value.columns)
    return float(values) if np.ndim(values) == 0 else values
