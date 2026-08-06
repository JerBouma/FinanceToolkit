"""Binomial Trees Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

# pylint: disable=too-many-locals


def calculate_up_and_down_movements(volatility: float, time_delta: float):
    """
    Calculates the up and down movements in the binomial tree.

    Args:
        volatility (float): the volatility of a stock.
        time_delta (int): the time delta.

    Returns:
        tuple: Up and down movements.
    """
    up_movement = np.exp(volatility * np.sqrt(time_delta))
    down_movement = 1 / up_movement

    return up_movement, down_movement


def calculate_risk_neutral_probability(
    risk_free_rate: float,
    dividend_yield: float,
    timestep: float,
    up_movement: float,
    down_movement: float,
):
    """
    Calculates the risk-neutral probability.

    Args:
        risk_free_rate (float): The risk-free interest rate.
        dividend_yield (float): The dividend yield.
        timestep (float): The time step.
        up_movement (float): The up movement.
        down_movement (float): The down movement.

    Returns:
        float: The risk-neutral probability.
    """
    risk_neutral_probability = (
        np.exp((risk_free_rate - dividend_yield) * timestep) - down_movement
    ) / (up_movement - down_movement)

    return risk_neutral_probability


def calculate_stock_prices(
    stock_price: float,
    up_movement: float,
    down_movement: float,
    period_length: int,
    show_unique_combinations: bool = True,
):
    """
    Calculates stock prices at each node.

    Args:
        stock_price (float): Current stock price.
        up_movement (float): Up movement.
        down_movement (float): Down movement.
        period_length (int): Period length.
        show_unique_combinations (bool, optional): Whether to show unique
        combinations. Defaults to True.

    Returns:
        pd.DataFrame: Stock prices at each node.
    """
    combinations_dict: dict = {}
    unique_combinations = set()

    # Generate combinations of maximum period length
    combination = [up_movement] * period_length

    while combination.count(up_movement) > 0:
        combination_key = "".join("U" if x == up_movement else "D" for x in combination)

        # Check if combination_key is unique
        if combination_key not in unique_combinations:
            combinations_dict[combination_key] = []
            value = stock_price

            # Add the current stock price to Period 0
            combinations_dict[combination_key].append(value)

            for movement in combination:
                value = value * movement
                combinations_dict[combination_key].append(value)

            unique_combinations.add(combination_key)

        # Generate next combination
        for i in range(period_length - 1, -1, -1):
            if combination[i] == up_movement:
                combination[i] = down_movement
                break

            combination[i] = up_movement

    combination_key = "".join("U" if x == up_movement else "D" for x in combination)
    combinations_dict[combination_key] = []
    value = stock_price

    combinations_dict[combination_key].append(value)

    for movement in combination:
        value = value * movement
        combinations_dict[combination_key].append(value)

    unique_combinations.add(combination_key)

    # Create a DataFrame from the Series
    combinations_df = pd.DataFrame.from_dict(combinations_dict, orient="index")

    if show_unique_combinations:
        # Add a scoring system in a separate dictionary
        scoring_system = {"U": 1, "D": -1}

        scoring = pd.Series(
            {
                combination: sum(scoring_system[movement] for movement in combination)
                for combination in combinations_df.index
            }
        )

        # keep='last' picks the downs-first path, so row k is a real node at every t >= k.
        unique_movements = scoring.drop_duplicates(keep="last")

        combinations_df = combinations_df.loc[unique_movements.index]

    return combinations_df


def get_call_option_payoffs(stock_price: float, strike_price: float):
    """
    Calculates call option payoffs at maturity.

    Args:
        stock_price (float): Stock price
        strike_price (float): Option strike price

    Returns:
        float: Call option payoff at maturity
    """
    call_option_payoff = np.maximum(stock_price - strike_price, 0)

    return call_option_payoff


def get_put_option_payoffs(stock_price: float, strike_price: float):
    """
    Calculates put option payoffs at maturity.

    Args:
        stock_price (float): Stock price
        strike_price (float): Option strike price

    Returns:
        float: Put option payoff at maturity
    """
    put_option_payoff = np.maximum(strike_price - stock_price, 0)

    return put_option_payoff


def calculate_option_value(
    up_option_payoff: float,
    down_option_payoff: float,
    risk_free_rate: float,
    time_to_expiration: int,
    timestep: int | float,
    risk_neutral_probability: float,
):
    """
    Calculates option values at each node.

    Args:
    up_option_payoff (float): Option payoff in the up state
    down_option_payoff (float): Option payoff in the down state
    risk_free_rate (float): Risk-free interest rate
    time_to_expiration (int): Time to expiration
    timestep (int): Number of time steps
    risk_neutral_probability (float): Risk-neutral probability

    Returns:
    float: Option value
    """
    option_value = np.exp(-risk_free_rate * time_to_expiration / timestep) * (
        risk_neutral_probability * up_option_payoff
        + (1 - risk_neutral_probability) * down_option_payoff
    )

    return option_value


def get_option_payoffs(
    stock_price: float,
    strike_price: float,
    years: int,
    timesteps: int,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0,
    put_option: bool = False,
    american_option: bool = False,
    show_input_info: bool = False,
):
    """
    Calculates the price of a European call option using the CRR binomial tree model.

    Args:
        stock_price (float): Current stock price
        strike_price (float): Option strike price
        years (int): Time to expiration in years
        timesteps (int): Number of time steps
        risk_free_rate (float): Risk-free interest rate
        volatility (float): Volatility of the underlying stock
        dividend_yield (float, optional): Dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put option. Defaults to False.
        american_option (bool, optional): Whether the option is an American option. Defaults to False.
        show_input_info (bool, optional): Whether to show input info. Defaults to False.

    Returns:
        pd.DataFrame: Option payoffs
    """
    up_movement, down_movement = calculate_up_and_down_movements(
        volatility=volatility, time_delta=years / timesteps
    )

    risk_neutral_probability = calculate_risk_neutral_probability(
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        timestep=years / timesteps,
        up_movement=up_movement,
        down_movement=down_movement,
    )

    stock_prices = calculate_stock_prices(
        stock_price=stock_price,
        up_movement=up_movement,
        down_movement=down_movement,
        period_length=timesteps,
    )

    option_payoffs = pd.DataFrame(
        np.nan, index=stock_prices.index, columns=stock_prices.columns
    )

    if put_option:
        option_payoffs.iloc[:, -1] = [
            get_put_option_payoffs(stock_price=stock_price, strike_price=strike_price)
            for stock_price in stock_prices.iloc[:, -1]
        ]
    else:
        option_payoffs.iloc[:, -1] = [
            get_call_option_payoffs(stock_price=stock_price, strike_price=strike_price)
            for stock_price in stock_prices.iloc[:, -1]
        ]

    periods = len(option_payoffs.columns)

    # Row k at column i has (i - k) ups and k downs; shift(-1) gives the down-child.
    for i in range(periods - 2, -1, -1):
        option_value = calculate_option_value(
            up_option_payoff=option_payoffs.iloc[:, i + 1],
            down_option_payoff=option_payoffs.iloc[:, i + 1].shift(-1),
            risk_free_rate=risk_free_rate,
            time_to_expiration=years,
            timestep=timesteps,
            risk_neutral_probability=risk_neutral_probability,
        )

        if american_option:
            # Compare early exercise against this node's price, not the price at t=0.
            if put_option:
                exercise_value = get_put_option_payoffs(
                    stock_price=stock_prices.iloc[:, i], strike_price=strike_price
                )
            else:
                exercise_value = get_call_option_payoffs(
                    stock_price=stock_prices.iloc[:, i], strike_price=strike_price
                )

            option_payoffs.iloc[:, i] = np.maximum(option_value, exercise_value)
        else:
            option_payoffs.iloc[:, i] = option_value

    if show_input_info:
        return option_payoffs, up_movement, down_movement, risk_neutral_probability

    return option_payoffs


def get_strategy_payoff(
    stock_price: float | pd.Series | np.ndarray,
    legs: list[dict[str, float | bool | str]],
) -> float | pd.Series:
    """
    Calculate the net expiration profit and loss (P&L) of a multi-leg option (and,
    optionally, stock) strategy.

    Also known as: option strategy payoff diagram, P&L profile.

    A strategy is expressed as a list of "legs". Each leg is a dictionary describing
    either an option position or a stock position:

    - For an option leg:
        - "instrument" (str): "option" (this is the default if omitted).
        - "strike_price" (float): the option's strike price.
        - "put_option" (bool): whether the leg is a put option. Defaults to False (call).
        - "position" (str): "long" or "short". Defaults to "long".
        - "premium" (float): the premium paid (long) or received (short) per option.
          Defaults to 0.
    - For a stock leg:
        - "instrument" (str): "stock".
        - "position" (str): "long" or "short". Defaults to "long".
        - "premium" (float): the entry price paid (long) or received (short) per share.
          Defaults to 0.

    The net P&L at expiration is the sum, across all legs, of:

    - Long option: max(payoff, 0) — premium
    - Short option: premium — max(payoff, 0)
    - Long stock: stock_price — entry_price
    - Short stock: entry_price — stock_price

    Where payoff is max(S — K, 0) for a call and max(K — S, 0) for a put (see
    `get_call_option_payoffs` and `get_put_option_payoffs`).

    This single, generic building block can express many common strategies by
    combining option (and stock) legs, for example:

    - Straddle: long call + long put, same strike.
    - Strangle: long call + long put, different (OTM) strikes.
    - Bull call spread: long call (lower strike) + short call (higher strike).
    - Bear put spread: long put (higher strike) + short put (lower strike).
    - Covered call: long stock + short call.
    - Protective put: long stock + long put.
    - Iron condor: short put + long put (lower strikes) + short call + long call
      (higher strikes).

    Args:
        stock_price (float | pd.Series | np.ndarray): The stock price(s) at expiration
            for which to evaluate the strategy, e.g. a range of prices to build a
            payoff diagram.
        legs (list[dict]): A list of leg dictionaries as described above. Must contain
            at least one leg.

    Returns:
        float | pd.Series: The net strategy P&L for the given stock price(s).

    Raises:
        TypeError: if legs is not a list of dictionaries or if a leg's numeric fields
        are not numeric.
        ValueError: if legs is empty, or a leg's "position" or "instrument" is invalid,
        or an option leg is missing "strike_price".
    """
    if not isinstance(legs, list):
        raise TypeError(
            f"legs must be a list of dictionaries, received {type(legs).__name__}."
        )

    if not legs:
        raise ValueError("legs must contain at least one leg.")

    total_payoff: float | pd.Series | np.ndarray = 0.0

    for index, leg in enumerate(legs):
        if not isinstance(leg, dict):
            raise TypeError(
                f"Leg {index} must be a dictionary, received {type(leg).__name__}."
            )

        instrument = leg.get("instrument", "option")
        if instrument not in ("option", "stock"):
            raise ValueError(
                f"Leg {index} 'instrument' must be 'option' or 'stock', received {instrument!r}."
            )

        position = leg.get("position", "long")
        if position not in ("long", "short"):
            raise ValueError(
                f"Leg {index} 'position' must be 'long' or 'short', received {position!r}."
            )

        premium = leg.get("premium", 0.0)
        if not isinstance(premium, (int, float)) or isinstance(premium, bool):
            raise TypeError(
                f"Leg {index} 'premium' must be a float or int, received {type(premium).__name__}."
            )

        if instrument == "stock":
            leg_value = stock_price
        else:
            strike_price = leg.get("strike_price")
            if (
                strike_price is None
                or not isinstance(strike_price, (int, float))
                or isinstance(strike_price, bool)
            ):
                raise TypeError(
                    f"Leg {index} must define a numeric 'strike_price' for an option instrument."
                )

            put_option = leg.get("put_option", False)
            if not isinstance(put_option, bool):
                raise TypeError(
                    f"Leg {index} 'put_option' must be a bool, received {type(put_option).__name__}."
                )

            if put_option:
                leg_value = get_put_option_payoffs(
                    stock_price=stock_price, strike_price=strike_price
                )
            else:
                leg_value = get_call_option_payoffs(
                    stock_price=stock_price, strike_price=strike_price
                )

        leg_pnl = leg_value - premium if position == "long" else premium - leg_value

        total_payoff = total_payoff + leg_pnl

    return total_payoff
