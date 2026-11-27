"""Delta Hedging Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.options.greeks import greeks_model
from financetoolkit.options.pricing import black_scholes_model


def get_delta_hedging_profit_and_loss(
    stock_prices: np.ndarray,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    transaction_cost: float = 0.0,
    rebalancing_interval: int = 1,
    hedging_volatility: float | None = None,
    dividends: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate the profit and loss of selling a European option at its Black-Scholes value
    and delta hedging it until expiration: how much the hedged seller gains or loses on one
    or more price paths (Hull, 2022, chapter 19).

    The seller receives the option value, buys the delta in shares, keeps the rest in cash
    at the risk free rate and rebalances to the new delta every `rebalancing_interval`
    observations:

        - V_0 = Black-Scholes value at the volatility σ
        - Δ_i = Black-Scholes delta at t_i, with the hedging volatility (σ unless given)
        - Cash_0 = V_0 - Δ_0 S_0 - k |Δ_0| S_0
        - Cash_(i+1) = Cash_i e^(r Δt) + Δ_i D_(i+1) - (Δ_(i+1) - Δ_i) S_(i+1) - k |Δ_(i+1) - Δ_i| S_(i+1)
        - P&L = e^(-rT) [Cash_T + Δ_T S_T - Payoff(S_T)]

    with the proportional transaction cost k, paid on the value of every trade, and the
    dividend D per share paid while the shares are held. At expiration the hedge moves to the
    position the option settles with, one share for a call that expires in the money (minus
    one for a put) and none otherwise, as in Hull's example, so that trade costs as well.

    With continuous rebalancing and no costs the profit and loss is zero when the stock
    moves with the volatility σ. Discrete rebalancing leaves a hedging error with a mean
    near zero and a standard deviation that falls with the square root of the number of
    rebalances (see get_hedging_error_approximation). A realized volatility above σ makes the
    seller lose about ½ Γ S² (σ²_realized - σ²) per unit of time (Wilmott, 2006), and costs
    make the seller lose about k sqrt(2 / π) σ S² Γ sqrt(Δt) per rebalance (Leland, 1985).

    For more information, see the following sources:

    - Hull, J. C. (2022). "Options, Futures, and Other Derivatives." 11th edition, chapter 19.
    - Leland, H. E. (1985). "Option Pricing and Replication with Transactions Costs." The
      Journal of Finance, 40(5), 1283-1301.
    - Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance." 2nd edition, chapter 12.

    Args:
        stock_prices (np.ndarray): The prices at equally spaced observations from the sale to
            expiration, one row per path (or one path as a single row or a 1-D array).
        strike_price (float): The strike price.
        time_to_expiration (float): The time from the first to the last observation in years.
        risk_free_rate (float): The continuously compounded risk free rate.
        volatility (float): The volatility the option is sold at.
        dividend_yield (float, optional): The continuously compounded dividend yield the option
            is valued with and, without `dividends`, the stock pays. Defaults to 0.0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        transaction_cost (float, optional): The cost of a trade as a fraction of its value
            (half the bid-ask spread plus commissions; 0.001 is 10 basis points). Defaults to 0.0.
        rebalancing_interval (int, optional): The number of observations between rebalances.
            Defaults to 1, every observation.
        hedging_volatility (float | None, optional): The volatility of the hedge's delta.
            Defaults to None, the volatility the option is sold at.
        dividends (np.ndarray | None, optional): The dividend per share paid at each
            observation, with the shape of `stock_prices`. Defaults to None, the continuous
            dividend yield, S_(i+1) (e^(q Δt) - 1) per share.

    Returns:
        pd.DataFrame: Per path the Option Value V_0, the Payoff at expiration, the Transaction
        Costs and the Profit and Loss after costs, both discounted to the sale.
    """
    prices = np.atleast_2d(np.asarray(stock_prices, dtype=float))
    if prices.shape[1] < 2:  # noqa: PLR2004
        raise ValueError(
            "The hedge needs at least a price at the sale and at expiration."
        )
    if transaction_cost < 0:
        raise ValueError("The transaction cost can not be negative.")
    if rebalancing_interval < 1:
        raise ValueError("The rebalancing interval must be at least one observation.")
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Every price of the hedge must be positive.")

    steps = prices.shape[1] - 1
    step = time_to_expiration / steps
    hedging_volatility = (
        volatility if hedging_volatility is None else hedging_volatility
    )
    if dividends is None:
        dividends = prices * np.expm1(dividend_yield * step)
    else:
        dividends = np.atleast_2d(np.asarray(dividends, dtype=float))

    option_value = float(
        black_scholes_model.get_black_scholes(
            stock_price=prices[0, 0],
            strike_price=strike_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_expiration=time_to_expiration,
            dividend_yield=dividend_yield,
            put_option=put_option,
        )
    )

    growth = np.exp(risk_free_rate * step)
    shares = np.zeros(prices.shape[0])
    cash = np.full(prices.shape[0], option_value)
    costs = np.zeros(prices.shape[0])

    def trade(target: np.ndarray, price: np.ndarray, discount: float) -> None:
        nonlocal shares, cash, costs
        traded = target - shares
        cost = transaction_cost * np.abs(traded) * price
        cash = cash - traded * price - cost
        costs = costs + cost * discount
        shares = target

    for index in range(steps):
        if index % rebalancing_interval == 0:
            trade(
                greeks_model.get_delta(
                    stock_price=prices[:, index],
                    strike_price=strike_price,
                    time_to_expiration=time_to_expiration - index * step,
                    risk_free_rate=risk_free_rate,
                    volatility=hedging_volatility,
                    dividend_yield=dividend_yield,
                    put_option=put_option,
                ),
                prices[:, index],
                np.exp(-risk_free_rate * index * step),
            )
        cash = cash * growth + shares * dividends[:, index + 1]

    final = prices[:, -1]
    in_the_money = final < strike_price if put_option else final > strike_price
    settlement = np.where(in_the_money, -1.0 if put_option else 1.0, 0.0)
    discount = np.exp(-risk_free_rate * time_to_expiration)
    trade(settlement, final, discount)
    payoff = (
        np.maximum(strike_price - final, 0)
        if put_option
        else np.maximum(final - strike_price, 0)
    )

    return pd.DataFrame(
        {
            "Option Value": option_value,
            "Payoff": payoff,
            "Transaction Costs": costs,
            "Profit and Loss": discount * (cash + shares * final - payoff),
        }
    )


def get_delta_hedging_simulation(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    implied_volatility: float | None = None,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    transaction_cost: float = 0.0,
    steps: int = 63,
    rebalancing_interval: int = 1,
    drift: float | None = None,
    simulations: int = 10_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """
    Calculate the distribution of the hedging error of a delta-hedged option: how much
    selling and delta hedging the option gains or loses across simulated Geometric Brownian
    Motion paths of the stock, each hedged with get_delta_hedging_profit_and_loss.

    The option is sold at the implied volatility and hedged with its delta, while the stock
    moves with the volatility given:

        - S(t + Δt) = S(t) e^((μ - q - σ² / 2) Δt + σ sqrt(Δt) Z), Z ~ N(0, 1)

    with the drift μ (the risk free rate unless given), observed `steps` times and rebalanced
    every `rebalancing_interval` observations. With the implied volatility equal to the
    volatility and no costs, the mean of the profit and loss is near zero and its standard
    deviation follows get_hedging_error_approximation; halving the time between rebalances
    shrinks it by about sqrt(2).

    Referenced against:
    - FinancePy, simulate_hedge_paths: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/black_scholes_hedging_sim.py#L592

    Args:
        stock_price (float): The stock price at the sale.
        strike_price (float): The strike price.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.
        volatility (float): The volatility the stock moves with, 0 or above.
        implied_volatility (float | None, optional): The volatility the option is sold and
            hedged at. Defaults to None, the volatility the stock moves with.
        dividend_yield (float, optional): The continuously compounded dividend yield.
            Defaults to 0.0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        transaction_cost (float, optional): The cost of a trade as a fraction of its value.
            Defaults to 0.0.
        steps (int, optional): The number of observations up to expiration. Defaults to 63,
            the trading days of a quarter.
        rebalancing_interval (int, optional): The number of observations between rebalances.
            Defaults to 1.
        drift (float | None, optional): The expected return μ of the stock. Defaults to None,
            the risk free rate.
        simulations (int, optional): The number of simulated paths. Defaults to 10,000.
        seed (int | None, optional): The seed of the random numbers, None for a different draw
            on every call. Defaults to 0.

    Returns:
        pd.DataFrame: Per path the Option Value, Payoff, Transaction Costs and Profit and
        Loss of get_delta_hedging_profit_and_loss, and the Final Stock Price.
    """
    if steps < 1 or simulations < 1:
        raise ValueError("The simulation needs at least one step and one path.")
    if volatility < 0:
        raise ValueError("The volatility can not be negative.")

    step = time_to_expiration / steps
    drift = risk_free_rate if drift is None else drift
    shocks = np.random.default_rng(seed).standard_normal((simulations, steps))
    log_returns = (
        drift - dividend_yield - volatility**2 / 2
    ) * step + volatility * np.sqrt(step) * shocks
    prices = stock_price * np.exp(
        np.concatenate(
            [np.zeros((simulations, 1)), np.cumsum(log_returns, axis=1)], axis=1
        )
    )

    result = get_delta_hedging_profit_and_loss(
        stock_prices=prices,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=volatility if implied_volatility is None else implied_volatility,
        dividend_yield=dividend_yield,
        put_option=put_option,
        transaction_cost=transaction_cost,
        rebalancing_interval=rebalancing_interval,
    )
    result["Final Stock Price"] = prices[:, -1]
    return result


def get_hedging_error_approximation(
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    rebalances: int,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate the approximate standard deviation of the hedging error of an option delta
    hedged at discrete times: how far the profit and loss of the hedge typically strays from
    zero, from Boyle and Emanuel (1980) and Derman and Kamal (1999):

        - σ_P&L ≈ sqrt(π / 4) × Vega × σ / sqrt(N)

    with the Black-Scholes vega (per unit of volatility) and N rebalances. Each rebalance
    leaves an error of ½ Γ S² (σ² Δt - (ΔS / S)²); summed over the life of an at-the-money
    option their variance gives this result, which holds for calls and puts alike. Away from
    the money the gamma is spread unevenly over the life, so the approximation is less close.

    For more information, see the following papers:

    - Boyle, P. P. and Emanuel, D. (1980). "Discretely Adjusted Option Hedges." Journal of
      Financial Economics, 8(3), 259-282.
    - Derman, E. and Kamal, M. (1999). "When You Cannot Hedge Continuously: The Corrections
      of Black-Scholes." Risk, 12(1), 82-85.

    Args:
        stock_price (float): The stock price.
        strike_price (float): The strike price.
        time_to_expiration (float): The time to expiration in years.
        risk_free_rate (float): The continuously compounded risk free rate.
        volatility (float): The volatility.
        rebalances (int): The number of rebalances N up to expiration.
        dividend_yield (float, optional): The continuously compounded dividend yield.
            Defaults to 0.0.

    Returns:
        float: The standard deviation of the profit and loss, in the currency of the price.
    """
    # get_vega is per percentage point of volatility, the approximation per unit.
    vega = 100 * greeks_model.get_vega(
        stock_price=stock_price,
        strike_price=strike_price,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    return float(np.sqrt(np.pi / 4) * vega * volatility / np.sqrt(rebalances))
