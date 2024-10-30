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

        unique_movements = scoring.drop_duplicates(keep="first")

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
    option_value = np.exp(-risk_free_rate * time_to_expiration / (timestep - 1)) * (
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

    # Calculate put option prices for earlier periods
    for i in range(periods - 2, -1, -1):
        option_value = calculate_option_value(
            up_option_payoff=option_payoffs.iloc[:, i + 1].shift(-1),
            down_option_payoff=option_payoffs.iloc[:, i + 1],
            risk_free_rate=risk_free_rate,
            time_to_expiration=years,
            timestep=timesteps,
            risk_neutral_probability=risk_neutral_probability,
        )

        if american_option:
            if put_option:
                exercise_value = [
                    get_put_option_payoffs(
                        stock_price=stock_price, strike_price=strike_price
                    )
                ]
            else:
                exercise_value = [
                    get_call_option_payoffs(
                        stock_price=stock_price, strike_price=strike_price
                    )
                ]

            option_payoffs.iloc[:, i] = np.maximum(option_value, exercise_value)
        else:
            option_payoffs.iloc[:, i] = option_value

    if show_input_info:
        return option_payoffs, up_movement, down_movement, risk_neutral_probability

    return option_payoffs
