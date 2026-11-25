"""Binomial Trees Model"""

__docformat__ = "google"


import itertools

import numpy as np
import pandas as pd
from scipy.special import betaincinv, betaln, log_ndtr, ndtr

from financetoolkit.utilities.error_model import InvalidArgumentError, check_option

# The binomial tree variants of calculate_tree_movements.
BINOMIAL_TREE_METHODS = (
    "cox_ross_rubinstein",
    "jarrow_rudd",
    "tian",
    "trigeorgis",
    "leisen_reimer",
    "exact_inversion",
)

# The most time steps of a tree: the tree is returned as a table of steps by steps, so its size
# grows with the square of the steps.
MAX_TIMESTEPS = 1_000
# The most time steps whose paths are listed one by one (show_unique_combinations=False):
# there are 2 ** time_steps of them, 65,536 at 16 steps.
MAX_PATH_TIMESTEPS = 16


def calculate_up_and_down_movements(volatility: float, time_delta: float):
    """
    Calculate the up and down movements of the Cox-Ross-Rubinstein binomial tree: the factors by
    which the stock price rises or falls in one time step.

    Args:
        volatility (float): the volatility of a stock, above 0.
        time_delta (float): the length of a time step in years, above 0.

    Returns:
        tuple: Up and down movements.

    Raises:
        ValueError: If the volatility or the time delta is not a finite number above 0.
    """
    # A zero, negative or NaN input gives equal or NaN movements, from which no tree follows.
    if not (np.isfinite(volatility) and volatility > 0):
        raise ValueError(
            f"volatility must be a finite number above 0, received {volatility!r}."
        )
    if not (np.isfinite(time_delta) and time_delta > 0):
        raise ValueError(
            f"time_delta must be a finite number above 0, received {time_delta!r}."
        )

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
    Calculate the risk-neutral probability of an up movement: the probability under which the
    expected stock price grows at the risk-free rate less the dividend yield.

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

    # Outside 0 to 1 the tree weighs its branches with negative probabilities and
    # produces prices that allow arbitrage. It happens when a step's growth at the cost of
    # carry exceeds its up movement, i.e. a low volatility for the length of a step.
    if np.any((risk_neutral_probability < 0) | (risk_neutral_probability > 1)):
        raise ValueError(
            f"The risk-neutral probability ({np.round(risk_neutral_probability, 4)}) is "
            "not between 0 and 1, because the volatility is too low for the length of "
            "each step compared to the risk-free rate minus the dividend yield. Use more "
            "steps (shorter time steps) or check the volatility."
        )

    return risk_neutral_probability


def _peizer_pratt_inversion(z: float, time_steps: int) -> tuple[float, float]:
    """
    Helper function with the Peizer-Pratt (method 2) inversion of Leisen and Reimer (1996):
    the probability p for which a binomial distribution of n (odd) steps exceeds (n — 1) / 2
    up moves with probability about N(z), and its complement 1 — p.

    - h(z) = 1/2 + sign(z) * 1/2 * √(1 — e^(—(z / (n + 1/3 + 0.1 / (n + 1)))² * (n + 1/6)))

    The smaller of h and 1 — h is 1/2 * e^(—w) / (1 + √(1 — e^(—w))), with w the exponent above,
    so that it keeps its digits where h itself rounds to 0 or 1 (a large |z|).
    """
    exponent = (z / (time_steps + 1 / 3 + 0.1 / (time_steps + 1))) ** 2 * (
        time_steps + 1 / 6
    )
    tail = 0.5 * np.exp(-exponent) / (1 + np.sqrt(-np.expm1(-exponent)))
    return (1 - tail, tail) if z > 0 else (tail, 1 - tail)


def _exact_binomial_inversion(z: float, time_steps: int) -> tuple[float, float]:
    """
    Helper function with the exact inversion of the binomial distribution: the probability
    p for which a binomial distribution of n (odd) steps has at least (n + 1) / 2 up moves
    with probability N(z) exactly, and its complement 1 — p. That probability is the
    regularized incomplete beta function I_p((n + 1) / 2, (n + 1) / 2), so p is its inverse.

    I_p(a, a) = 1 — I_(1 — p)(a, a), so the smaller of p and 1 — p is the inverse at N(—|z|),
    which keeps its digits where N(z) itself rounds to 1 (a large z) and p with it.

    Below a tail of about 1e-100 with few steps scipy's inverse can give NaN; p is then so small
    that I_p(a, a) = p^a / (a * B(a, a)) * (1 + O(p)) gives it to machine precision.
    """
    half = (time_steps + 1) / 2
    tail = float(betaincinv(half, half, ndtr(-abs(z))))
    if not np.isfinite(tail):
        tail = float(
            np.exp((log_ndtr(-abs(z)) + np.log(half) + betaln(half, half)) / half)
        )
    return (1 - tail, tail) if z > 0 else (tail, 1 - tail)


# A degenerate tree (equal movements) divides 0 by 0; it is refused below with its reason.
@np.errstate(divide="ignore", invalid="ignore")
def calculate_tree_movements(
    method: str,
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    years: float,
    time_steps: int,
) -> tuple[float, float, float]:
    """
    Calculate the up movement, down movement and probability of an up movement of a binomial
    tree variant: how far the stock price moves in a time step and how likely a rise is.

    With Δt = T / n, b = r — q and ν = b — σ² / 2:

    - Cox-Ross-Rubinstein (1979): u = e^(σ√Δt), d = 1 / u, p = (e^(bΔt) — d) / (u — d)
    - Jarrow-Rudd (1983), equal probabilities: u = e^(νΔt + σ√Δt), d = e^(νΔt — σ√Δt), p = 1/2
    - Tian (1993), the first three moments matched: v = e^(σ²Δt),
      u = e^(bΔt) * v / 2 * (v + 1 + √(v² + 2v — 3)), d = e^(bΔt) * v / 2 * (v + 1 — √(v² + 2v — 3)),
      p = (e^(bΔt) — d) / (u — d)
    - Trigeorgis (1991), the log-transformed tree: Δx = √(σ²Δt + ν²Δt²), u = e^(Δx), d = 1 / u,
      p = 1/2 + νΔt / (2Δx)
    - Leisen-Reimer (1996), centred on the strike: p = h(d2), p' = h(d1), u = e^(bΔt) * p' / p,
      d = (e^(bΔt) — p * u) / (1 — p) = e^(bΔt) * (1 — p') / (1 — p), with h the Peizer-Pratt inversion
    - Exact inversion: the Leisen-Reimer tree with h the exact inversion of the binomial distribution,
      which Joshi (2010) expands to fourth order

    Where d1 = (ln(S / K) + (b + σ² / 2) * T) / (σ√T) and d2 = d1 — σ√T. The Leisen-Reimer
    and exact inversion trees need an odd number of steps and a positive strike. Where the
    option is so far in or out of the money that 1 — p' or p is 0 in floating point, the
    outcome is certain: p is 1 with u = e^(bΔt), or 0 with d = e^(bΔt), so the tree prices
    the option at its value on the forward.

    Args:
        method (str): The tree. Options are: "cox_ross_rubinstein" (Cox-Ross-Rubinstein),
            "jarrow_rudd" (Jarrow-Rudd), "tian" (Tian), "trigeorgis" (Trigeorgis),
            "leisen_reimer" (Leisen-Reimer), "exact_inversion" (exact inversion of the
            binomial distribution).
        stock_price (float): The current stock price.
        strike_price (float): The strike price, which the Leisen-Reimer and exact inversion trees are
            centred on.
        risk_free_rate (float): The continuously compounded risk-free rate.
        dividend_yield (float): The continuous dividend yield.
        volatility (float): The volatility of the stock.
        years (float): The time to expiration in years.
        time_steps (int): The number of time steps.

    Returns:
        tuple[float, float, float]: The up movement, the down movement and the probability of
            an up movement.

    Raises:
        ValueError: if the method is not one of the trees, the Leisen-Reimer or exact inversion tree
            gets an even number of steps or a strike that is not positive, or the probability is
            not between 0 and 1.
    """
    check_option("method", method, BINOMIAL_TREE_METHODS)

    time_delta = years / time_steps
    carry = risk_free_rate - dividend_yield
    growth = np.exp(carry * time_delta)
    drift = carry - 0.5 * volatility**2

    if method == "cox_ross_rubinstein":
        up_movement, down_movement = calculate_up_and_down_movements(
            volatility=volatility, time_delta=time_delta
        )
        probability = calculate_risk_neutral_probability(
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            timestep=time_delta,
            up_movement=up_movement,
            down_movement=down_movement,
        )
        return up_movement, down_movement, probability

    if method == "jarrow_rudd":
        up_movement = np.exp(drift * time_delta + volatility * np.sqrt(time_delta))
        down_movement = np.exp(drift * time_delta - volatility * np.sqrt(time_delta))
        probability = 0.5
    elif method == "tian":
        variance_factor = np.exp(volatility**2 * time_delta)
        root = np.sqrt(variance_factor**2 + 2 * variance_factor - 3)
        up_movement = 0.5 * growth * variance_factor * (variance_factor + 1 + root)
        down_movement = 0.5 * growth * variance_factor * (variance_factor + 1 - root)
        probability = (growth - down_movement) / (up_movement - down_movement)
    elif method == "trigeorgis":
        log_step = np.sqrt(volatility**2 * time_delta + (drift * time_delta) ** 2)
        up_movement = np.exp(log_step)
        down_movement = 1 / up_movement
        probability = 0.5 + 0.5 * drift * time_delta / log_step
    else:
        if time_steps % 2 == 0:
            raise ValueError(
                f"The {method} tree is centred on the strike and needs an odd number of time "
                f"steps, received {time_steps}; use {time_steps + 1}."
            )
        if strike_price <= 0:
            raise ValueError(
                f"The {method} tree is centred on the strike, which must be positive, "
                f"received {strike_price!r}."
            )
        d1 = (
            np.log(stock_price / strike_price) + (carry + 0.5 * volatility**2) * years
        ) / (volatility * np.sqrt(years))
        d2 = d1 - volatility * np.sqrt(years)
        inversion = (
            _peizer_pratt_inversion
            if method == "leisen_reimer"
            else _exact_binomial_inversion
        )
        probability, complement = inversion(d2, time_steps)
        probability_d1, complement_d1 = inversion(d1, time_steps)

        # u = e^(bΔt) * p' / p and d = e^(bΔt) * (1 — p') / (1 — p), each from the probability or its
        # complement as computed, so the move on the unlikely side keeps its digits. So far in or out of the
        # money (|d1| or |d2| in the tens) that the smaller one underflows to 0, the price ends at the forward
        # with certainty, so the tree only takes the likely move, at the cost of carry, and the option is worth
        # its value on that one path (the discounted forward less the strike, or 0). The other move only needs
        # to be a valid node.  # noqa: E501
        spread = np.exp(2 * volatility * np.sqrt(time_delta))
        if complement_d1 == 0:
            probability, up_movement = 1.0, growth
            down_movement = growth / spread
        elif probability == 0:
            probability, down_movement = 0.0, growth
            up_movement = growth * spread
        else:
            up_movement = growth * probability_d1 / probability
            down_movement = growth * complement_d1 / complement

    # The strike-centred trees take a probability of 0 or 1 for an outcome that is certain (above).
    centred = method in ("leisen_reimer", "exact_inversion")
    if not (0 <= probability <= 1 if centred else 0 < probability < 1) or not (
        down_movement > 0
    ):
        raise ValueError(
            f"The {method} tree has a probability of {probability:.4f} and a down movement of "
            f"{down_movement:.4f}, which does not give a valid tree for a volatility of "
            f"{volatility:.4f} over steps of {time_delta:.4f} years. Use more time steps or "
            "check the volatility."
        )

    return float(up_movement), float(down_movement), float(probability)


def calculate_stock_prices(
    stock_price: float,
    up_movement: float,
    down_movement: float,
    period_length: int,
    show_unique_combinations: bool = True,
) -> pd.DataFrame:
    """
    Calculate the stock prices at each node of a binomial tree: the prices the stock can reach
    after every combination of up and down movements.

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
    if period_length > MAX_TIMESTEPS:
        raise ValueError(
            f"A tree of {period_length:,} time steps is too large; use at most "
            f"{MAX_TIMESTEPS:,}."
        )

    if up_movement == down_movement:
        raise ValueError(
            "The up and down movement are equal, which happens when the volatility or the "
            "time to expiration is zero. The tree cannot be constructed in that case."
        )

    if not show_unique_combinations and period_length > MAX_PATH_TIMESTEPS:
        raise ValueError(
            f"Listing every path of {period_length:,} time steps gives 2^{period_length} "
            f"paths; use at most {MAX_PATH_TIMESTEPS} time steps, or "
            "show_unique_combinations=True for the distinct prices per step."
        )

    if show_unique_combinations:
        # The tree recombines, so only the net number of ups matters and there are just period_length + 1 distinct nodes; enumerating all 2^period_length paths first and deduplicating afterwards is exponential and unusable beyond ~20 steps.  # noqa: E501
        combinations_dict = {}

        for ups in range(period_length, -1, -1):
            downs = period_length - ups
            combination_key = "D" * downs + "U" * ups

            value = stock_price
            values = [value]

            for period in range(1, period_length + 1):
                value = value * (down_movement if period <= downs else up_movement)
                values.append(value)

            combinations_dict[combination_key] = values

        return pd.DataFrame.from_dict(combinations_dict, orient="index")

    # Every path of up and down movements, from all ups to all downs, one by one: a fixed number
    # of 2^period_length paths, so the listing ends whatever the movements are (even NaN).
    combinations_dict = {}

    for path in itertools.product("UD", repeat=period_length):
        values = [stock_price]

        for movement in path:
            values.append(
                values[-1] * (up_movement if movement == "U" else down_movement)
            )

        combinations_dict["".join(path)] = values

    return pd.DataFrame.from_dict(combinations_dict, orient="index")


def get_call_option_payoffs(
    stock_price: float | pd.Series | np.ndarray, strike_price: float
) -> float | pd.Series | np.ndarray:
    """
    Calculate the payoff of a call option at expiration: the amount by which the stock price
    exceeds the strike price, or zero.

    Args:
        stock_price (float | pd.Series | np.ndarray): Stock price, or a range of stock
            prices to evaluate the payoff over.
        strike_price (float): Option strike price

    Returns:
        float | pd.Series | np.ndarray: Call option payoff at maturity, in the
            shape of the stock price input.
    """
    call_option_payoff = np.maximum(stock_price - strike_price, 0)

    return call_option_payoff


def get_put_option_payoffs(
    stock_price: float | pd.Series | np.ndarray, strike_price: float
) -> float | pd.Series | np.ndarray:
    """
    Calculate the payoff of a put option at expiration: the amount by which the strike price
    exceeds the stock price, or zero.

    Args:
        stock_price (float | pd.Series | np.ndarray): Stock price, or a range of stock
            prices to evaluate the payoff over.
        strike_price (float): Option strike price

    Returns:
        float | pd.Series | np.ndarray: Put option payoff at maturity, in the
            shape of the stock price input.
    """
    put_option_payoff = np.maximum(strike_price - stock_price, 0)

    return put_option_payoff


def calculate_option_value(
    up_option_payoff: float,
    down_option_payoff: float,
    risk_free_rate: float,
    time_to_expiration: float,
    timestep: int | float,
    risk_neutral_probability: float,
):
    """
    Calculate the option value at a node of a binomial tree: the discounted expected value of
    the option in the up and down states that follow.

    Args:
        up_option_payoff (float): The option payoff in the up state.
        down_option_payoff (float): The option payoff in the down state.
        risk_free_rate (float): The risk-free rate, continuously compounded.
        time_to_expiration (float): The time to expiration in years.
        timestep (int | float): The number of time steps of the tree.
        risk_neutral_probability (float): The risk-neutral probability of the up state.

    Returns:
        float: The option value, the discounted expected payoff.
    """
    option_value = np.exp(-risk_free_rate * time_to_expiration / timestep) * (
        risk_neutral_probability * up_option_payoff
        + (1 - risk_neutral_probability) * down_option_payoff
    )

    return option_value


def get_option_payoffs(
    stock_price: float,
    strike_price: float,
    years: float,
    time_steps: int,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0,
    put_option: bool = False,
    american_option: bool = False,
    show_input_info: bool = False,
    method: str = "cox_ross_rubinstein",
):
    """
    Calculate the binomial tree of a European or American option: the value of the option at
    every node, from the payoffs at expiration back to today.

    The tree is that of Cox-Ross-Rubinstein (1979) or one of its variants (see
    calculate_tree_movements).

    Referenced against:
    - QuantLib, BinomialJRVanillaEngine (Jarrow-Rudd): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialTianVanillaEngine (Tian): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialTrigeorgisVanillaEngine (Trigeorgis): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialLRVanillaEngine (Leisen-Reimer): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialJ4VanillaEngine (exact inversion): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html

    Args:
        stock_price (float): Current stock price
        strike_price (float): Option strike price
        years (float): Time to expiration in years
        time_steps (int): Number of time steps
        risk_free_rate (float): Risk-free interest rate
        volatility (float): Volatility of the underlying stock
        dividend_yield (float, optional): Dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put option. Defaults to False.
        american_option (bool, optional): Whether the option is an American option. Defaults to False.
        show_input_info (bool, optional): Whether to show input info. Defaults to False.
        method (str, optional): The tree. Options are: "cox_ross_rubinstein"
            (Cox-Ross-Rubinstein), "jarrow_rudd" (Jarrow-Rudd), "tian" (Tian), "trigeorgis"
            (Trigeorgis), "leisen_reimer" (Leisen-Reimer), "exact_inversion" (exact
            inversion of the binomial distribution). The last two need an odd number of time
            steps. Defaults to "cox_ross_rubinstein".

    Returns:
        pd.DataFrame: Option payoffs

    Notes:
        - The Jarrow-Rudd, Tian, Trigeorgis and Leisen-Reimer trees agree with QuantLib's
          within 1e-9.
        - Joshi (2010) expands the inverse of the binomial distribution to fourth order (his
          J4 tree); this tree takes the exact inverse, which prices a European option at its
          Black-Scholes value and lies within 1e-6 of the J4 tree at 101 steps.
    """
    trees, up_movements, down_movements, probabilities = get_option_payoff_trees(
        stock_price=stock_price,
        strike_prices=np.array([strike_price], dtype=float),
        years=years,
        time_steps=time_steps,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        put_option=put_option,
        american_option=american_option,
        method=method,
    )

    option_payoffs = pd.DataFrame(
        trees[0],
        index=[
            "D" * downs + "U" * (time_steps - downs) for downs in range(time_steps + 1)
        ],
    )

    if show_input_info:
        return (
            option_payoffs,
            float(up_movements[0]),
            float(down_movements[0]),
            float(probabilities[0]),
        )

    return option_payoffs


def get_option_payoff_trees(
    stock_price: float,
    strike_prices: np.ndarray,
    years: float,
    time_steps: int,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0,
    put_option: bool = False,
    american_option: bool = False,
    method: str = "cox_ross_rubinstein",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the binomial trees of European or American options for an array of strike prices
    at once: the value of each option at every node, from the payoffs at expiration back to
    today.

    The trees use numpy backward induction over a strikes × nodes array (see get_option_payoffs
    for the trees).

    Row k of a tree holds the node with k down movements: at step i it is the price after
    min(i, k) downs followed by the ups, S * d^min(i, k) * u^max(i — k, 0), and the option
    value there is defined for k ≤ i (NaN beyond, where the node does not exist yet). Going
    back from expiration, the value at (k, i) is e^(—rΔt) * (p * V(k, i + 1) + (1 — p) *
    V(k + 1, i + 1)), and for an American option at least the exercise value at the node.

    Referenced against:
    - QuantLib, BinomialJRVanillaEngine (Jarrow-Rudd): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialTianVanillaEngine (Tian): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialTrigeorgisVanillaEngine (Trigeorgis): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialLRVanillaEngine (Leisen-Reimer): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html
    - QuantLib, BinomialJ4VanillaEngine (exact inversion): https://www.quantlib.org/reference/class_quant_lib_1_1_binomial_vanilla_engine.html

    Args:
        stock_price (float): Current stock price.
        strike_prices (np.ndarray): The option strike prices.
        years (float): Time to expiration in years.
        time_steps (int): Number of time steps.
        risk_free_rate (float): Risk-free interest rate.
        volatility (float): Volatility of the underlying stock.
        dividend_yield (float, optional): Dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the options are put options. Defaults to False.
        american_option (bool, optional): Whether the options are American options.
            Defaults to False.
        method (str, optional): The tree (see calculate_tree_movements). Options are:
            "cox_ross_rubinstein" (Cox-Ross-Rubinstein), "jarrow_rudd" (Jarrow-Rudd), "tian"
            (Tian), "trigeorgis" (Trigeorgis), "leisen_reimer" (Leisen-Reimer),
            "exact_inversion" (exact inversion of the binomial distribution). Defaults to
            "cox_ross_rubinstein".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The option values as an array
            of strikes × nodes × steps, and the up movement, down movement and risk-neutral
            probability per strike (the same for every strike, except for the trees centred
            on the strike).
    """
    strike_prices = np.asarray(strike_prices, dtype=float)

    # Only the Leisen-Reimer and exact inversion trees depend on the strike.
    centred = method in ("leisen_reimer", "exact_inversion")
    movements = np.array(
        [
            calculate_tree_movements(
                method=method,
                stock_price=stock_price,
                strike_price=strike_price,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=volatility,
                years=years,
                time_steps=time_steps,
            )
            for strike_price in (strike_prices if centred else strike_prices[:1])
        ]
    ).reshape(-1, 3)
    movements = np.broadcast_to(movements, (len(strike_prices), 3))
    up_movement, down_movement, probability = (
        movements[:, [0]],
        movements[:, [1]],
        movements[:, [2]],
    )

    # The checks of calculate_stock_prices.
    if time_steps > MAX_TIMESTEPS:
        raise ValueError(
            f"A tree of {time_steps:,} time steps is too large; use at most "
            f"{MAX_TIMESTEPS:,}."
        )
    if np.any(up_movement == down_movement):
        raise ValueError(
            "The up and down movement are equal, which happens when the volatility or the "
            "time to expiration is zero. The tree cannot be constructed in that case."
        )

    # The stock price at every node, multiplied step by step as calculate_stock_prices does:
    # row k moves down for its first k steps and up after.
    downs = np.arange(time_steps + 1)
    stock_prices = np.empty((len(strike_prices), time_steps + 1, time_steps + 1))
    stock_prices[:, :, 0] = stock_price
    for step in range(1, time_steps + 1):
        stock_prices[:, :, step] = stock_prices[:, :, step - 1] * np.where(
            step <= downs, down_movement, up_movement
        )

    sign = -1 if put_option else 1
    strikes = strike_prices[:, None]
    discount = np.exp(-risk_free_rate * years / time_steps)

    values = np.full(stock_prices.shape, np.nan)
    values[:, :, -1] = np.maximum(sign * (stock_prices[:, :, -1] - strikes), 0)

    for step in range(time_steps - 1, -1, -1):
        later = values[:, :, step + 1]
        # The down-child of row k is row k + 1, which the last row does not have.
        down_values = np.concatenate(
            [later[:, 1:], np.full((len(strike_prices), 1), np.nan)], axis=1
        )
        option_value = discount * (
            probability * later + (1 - probability) * down_values
        )
        if american_option:
            # Compare early exercise against this node's price, not the price at t=0.
            option_value = np.maximum(
                option_value,
                np.maximum(sign * (stock_prices[:, :, step] - strikes), 0),
            )
        values[:, :, step] = option_value

    return values, up_movement[:, 0], down_movement[:, 0], probability[:, 0]


# The keys a leg of get_strategy_payoff takes.
LEG_KEYS = ("instrument", "strike_price", "put_option", "position", "premium")


def get_strategy_payoff(
    stock_price: float | pd.Series | np.ndarray,
    legs: list[dict[str, float | bool | str]],
) -> float | pd.Series | np.ndarray:
    """
    Calculate the net expiration profit and loss (P&L) of a multi-leg option (and, optionally,
    stock) strategy: what the combined position earns or loses at each stock price at
    expiration.

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
        InvalidArgumentError: if legs is empty, a leg has a key other than those above, a
        leg's "position" or "instrument" is invalid, or an option leg is missing
        "strike_price" (a ValueError).
    """
    if not isinstance(legs, list):
        raise TypeError(
            f"legs must be a list of dictionaries, received {type(legs).__name__}."
        )

    if not legs:
        raise InvalidArgumentError("legs must contain at least one leg.")

    total_payoff: float | pd.Series | np.ndarray = 0.0

    for index, leg in enumerate(legs):
        if not isinstance(leg, dict):
            raise TypeError(
                f"Leg {index} must be a dictionary, received {type(leg).__name__}."
            )

        if unknown := [key for key in leg if key not in LEG_KEYS]:
            raise InvalidArgumentError(
                f"Leg {index} has the key(s) {', '.join(map(repr, unknown))}, which a leg does "
                f"not take. Please use the keys {', '.join(map(repr, LEG_KEYS))}."
            )

        instrument = leg.get("instrument", "option")
        if instrument not in ("option", "stock"):
            raise InvalidArgumentError(
                f"Leg {index} 'instrument' must be 'option' or 'stock', received {instrument!r}."
            )

        position = leg.get("position", "long")
        if position not in ("long", "short"):
            raise InvalidArgumentError(
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
            if strike_price is None:
                raise InvalidArgumentError(
                    f"Leg {index} is an option, which needs a 'strike_price'."
                )
            if not isinstance(strike_price, (int, float)) or isinstance(
                strike_price, bool
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
