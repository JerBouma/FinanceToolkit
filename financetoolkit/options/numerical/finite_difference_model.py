"""Finite Difference Model"""

__docformat__ = "google"

import numbers
from collections.abc import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.special import ndtr

from financetoolkit.options.helpers import validate_numeric_inputs
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.utilities.error_model import check_option

# The weight θ of the new time level in the θ-scheme of each finite difference scheme.
SCHEMES = {"explicit": 0.0, "implicit": 1.0, "crank_nicolson": 0.5}
EXERCISE_METHODS = ("brennan_schwartz", "psor")

# The grid reaches this many standard deviations of the log price at expiration beyond the
# stock prices (and strikes) priced, where the value of an option is its asymptote to within
# far less than the error of the scheme.
GRID_STANDARD_DEVIATIONS = 5.0

# The largest ratio of the longest to the shortest time to expiration solved on one grid by
# get_finite_difference_option_values: the grid spacing of the shortest expiration, in its
# own standard deviations, then stays within a factor of two of that of the longest. With
# 1,000 price steps that keeps every European value from a day to a quarter within 7.3e-4
# of Black-Scholes (stock price 100, volatility up to 100%), where one grid for the whole
# range is 2.6e-2 off a day from expiration.
EXPIRATION_RATIO_PER_SOLVE = 4

# Projected successive over-relaxation (Cryer, 1971) stops once a sweep changes the values
# by less than the tolerance (in units of the strike price), within the most sweeps.
PSOR_RELAXATION = 1.2
PSOR_TOLERANCE = 1e-12
PSOR_SWEEPS = 10_000


def _validate(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
    scheme: str,
    price_steps: int,
    time_steps: int,
    rannacher_steps: int,
    exercise_method: str,
) -> None:
    """
    Raises an error for inputs the finite difference schemes do not take.

    Args:
        stock_price (float): The stock price.
        strike_price (float): The strike price.
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The dividend yield.
        scheme (str): The scheme.
        price_steps (int): The steps of the price grid.
        time_steps (int): The time steps.
        rannacher_steps (int): The Rannacher start-up steps.
        exercise_method (str): The early exercise method.
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    if stock_price <= 0 or strike_price <= 0:
        raise ValueError("stock_price and strike_price must be above zero.")
    if volatility <= 0:
        raise ValueError(f"volatility must be above zero, received {volatility!r}.")
    if time_to_expiration < 0:
        raise ValueError(
            f"time_to_expiration must be at least zero, received {time_to_expiration!r}."
        )
    check_option("scheme", scheme, SCHEMES)
    check_option("exercise_method", exercise_method, EXERCISE_METHODS)
    for name, value, minimum in (
        ("price_steps", price_steps, 10),
        ("time_steps", time_steps, 1),
        ("rannacher_steps", rannacher_steps, 0),
    ):
        if (
            not isinstance(value, numbers.Integral)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer of at least {minimum}, received {value!r}."
            )


def _grid_width(
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
) -> float:
    """
    The distance in log price the grid reaches beyond the prices it values, five standard
    deviations of the log price at expiration plus its drift (at least that of a day).

    Args:
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float): The longest time to expiration in years.
        dividend_yield (float): The dividend yield.

    Returns:
        float: The width in log price.
    """
    time = max(time_to_expiration, 1 / 365)
    return (
        GRID_STANDARD_DEVIATIONS * volatility * np.sqrt(time)
        + abs(risk_free_rate - dividend_yield - 0.5 * volatility**2) * time
    )


def _time_steps(times: np.ndarray, time_steps: int) -> list[tuple[float, int | None]]:
    """
    Divides the time to the longest expiration into steps that end exactly at every
    expiration asked for, with about time_steps steps in all and at least one between two
    expirations.

    Args:
        times (np.ndarray): The times to expiration, sorted and above zero.
        time_steps (int): The number of time steps up to the longest time.

    Returns:
        list[tuple[float, int | None]]: Per step its length and the position of the time
        to expiration it ends on, or None.
    """
    steps: list[tuple[float, int | None]] = []
    previous = 0.0

    for position, time in enumerate(times):
        count = max(1, int(round(time_steps * (time - previous) / times[-1])))
        steps += [((time - previous) / count, None)] * (count - 1)
        steps.append(((time - previous) / count, position))
        previous = time

    return steps


def _brennan_schwartz(
    lower: float,
    diagonal: float,
    upper: float,
    right_hand_side: list[float],
    obstacle: list[float],
    exercise_low: bool,
) -> np.ndarray:
    """
    Solves the tridiagonal system with constant coefficients A * V = b subject to V ≥ g by
    the algorithm of Brennan and Schwartz (1977): Gaussian elimination towards the side
    where the option is exercised, then substitution from that side with the projection
    max(V, g). It is exact when the exercise region is one interval at one end of the grid,
    as for American puts (low prices) and calls (high prices).

    Args:
        lower (float): The coefficient of V(i-1) in row i.
        diagonal (float): The coefficient of V(i) in row i.
        upper (float): The coefficient of V(i+1) in row i.
        right_hand_side (list[float]): The right-hand side b.
        obstacle (list[float]): The exercise values g.
        exercise_low (bool): Whether the exercise region lies at the low prices.

    Returns:
        np.ndarray: The values V.
    """
    size = len(right_hand_side)
    pivots = [0.0] * size
    reduced = [0.0] * size
    values = [0.0] * size
    # Eliminating towards the exercise side leaves its first row with one unknown; the
    # elimination runs the other way for a call, so the arrays are read in reverse.
    order = range(size) if not exercise_low else range(size - 1, -1, -1)
    near, far = (lower, upper) if not exercise_low else (upper, lower)
    previous = None

    for index in order:
        if previous is None:
            pivots[index] = diagonal
            reduced[index] = right_hand_side[index]
        else:
            multiplier = near / pivots[previous]
            pivots[index] = diagonal - multiplier * far
            reduced[index] = right_hand_side[index] - multiplier * reduced[previous]
        previous = index

    following = None
    for index in reversed(order):
        value = (
            reduced[index]
            if following is None
            else reduced[index] - far * values[following]
        ) / pivots[index]
        values[index] = max(value, obstacle[index])
        following = index

    return np.asarray(values)


def _psor(
    lower: float,
    diagonal: float,
    upper: float,
    right_hand_side: list[float],
    obstacle: list[float],
    guess: list[float],
) -> np.ndarray:
    """
    Solves the linear complementarity problem A * V ≥ b, V ≥ g, (A * V - b) * (V - g) = 0
    of a tridiagonal A with constant coefficients by projected successive over-relaxation
    (Cryer, 1971): Gauss-Seidel sweeps over-relaxed by ω and projected onto V ≥ g.

    - V(i) = max(g(i), V(i) + ω * ((b(i) - l * V(i-1) - u * V(i+1)) / d - V(i)))

    Args:
        lower (float): The coefficient l of V(i-1) in row i.
        diagonal (float): The coefficient d of V(i) in row i.
        upper (float): The coefficient u of V(i+1) in row i.
        right_hand_side (list[float]): The right-hand side b.
        obstacle (list[float]): The exercise values g.
        guess (list[float]): The starting values.

    Returns:
        np.ndarray: The values V.

    Raises:
        ValueError: If the sweeps do not converge.
    """
    values = list(guess)
    size = len(values)

    for _ in range(PSOR_SWEEPS):
        change = 0.0
        for index in range(size):
            neighbours = (lower * values[index - 1] if index > 0 else 0.0) + (
                upper * values[index + 1] if index < size - 1 else 0.0
            )
            updated = values[index] + PSOR_RELAXATION * (
                (right_hand_side[index] - neighbours) / diagonal - values[index]
            )
            updated = max(updated, obstacle[index])
            change = max(change, abs(updated - values[index]))
            values[index] = updated
        if change < PSOR_TOLERANCE:
            return np.asarray(values)

    raise ValueError(f"PSOR did not converge within {PSOR_SWEEPS:,} sweeps.")


def _theta_scheme(
    grid: np.ndarray,
    payoff: np.ndarray,
    times: np.ndarray,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    scheme: str,
    time_steps: int,
    rannacher_steps: int,
    boundaries: Callable[[float], tuple[float, float]],
    obstacle: Callable[[float], np.ndarray] | None = None,
    exercise_low: bool = True,
    exercise_method: str = "brennan_schwartz",
) -> list[np.ndarray]:
    """
    Steps the Black-Scholes equation in the log price x from expiration back to today with
    the θ-scheme, on a uniform grid with central differences:

    - V_τ = a * V_xx + b * V_x - r * V, a = σ² / 2, b = r - q - σ² / 2
    - (I - θ * Δτ * L) * V(τ + Δτ) = (I + (1 - θ) * Δτ * L) * V(τ)
    - L * V(i) = (a / Δx² - b / (2Δx)) * V(i-1) - (2a / Δx² + r) * V(i) + (a / Δx² + b / (2Δx)) * V(i+1)

    With θ = 0 (explicit), 1 (implicit) or 1/2 (Crank-Nicolson). The first rannacher_steps
    Crank-Nicolson steps are each taken as two implicit half steps (Rannacher, 1984), which
    damp the oscillations the kink of the payoff sets off and keep the scheme second order.
    With an obstacle (an exercise value) every step solves V ≥ g by Brennan-Schwartz or PSOR
    (the explicit scheme projects with max(V, g)).

    Args:
        grid (np.ndarray): The uniform grid of log prices.
        payoff (np.ndarray): The value at expiration on the grid.
        times (np.ndarray): The times to expiration at which to keep the values, sorted
            and above zero.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility.
        dividend_yield (float): The continuous dividend yield.
        scheme (str): "explicit", "implicit" or "crank_nicolson".
        time_steps (int): The number of time steps up to the longest time.
        rannacher_steps (int): The Crank-Nicolson steps taken as implicit half steps.
        boundaries (Callable): The values at the lowest and highest grid point per time to
            expiration.
        obstacle (Callable | None, optional): The exercise value on the grid per time to
            expiration. Defaults to None (European).
        exercise_low (bool, optional): Whether the exercise region lies at low prices.
            Defaults to True.
        exercise_method (str, optional): "brennan_schwartz" or "psor". Defaults to
            "brennan_schwartz".

    Returns:
        list[np.ndarray]: The values on the grid at every time asked for.

    Raises:
        ValueError: If the explicit scheme is unstable at these steps.
    """
    theta = SCHEMES[scheme]
    spacing = grid[1] - grid[0]
    diffusion = 0.5 * volatility**2 / spacing**2
    convection = (risk_free_rate - dividend_yield - 0.5 * volatility**2) / (2 * spacing)
    below, centre, above = (
        diffusion - convection,
        -2 * diffusion - risk_free_rate,
        diffusion + convection,
    )
    steps = _time_steps(times, time_steps)

    if theta == 0.0:
        longest = max(step for step, _ in steps)
        if longest * (2 * diffusion + risk_free_rate) > 1:
            needed = int(np.ceil(times[-1] * (2 * diffusion + risk_free_rate)))
            raise ValueError(
                "The explicit scheme is unstable at these steps: it needs Δτ ≤ Δx² / σ², "
                f"at least {needed:,} time steps for {grid.size - 1:,} price steps."
            )

    # The Rannacher start-up replaces each of the first steps by two implicit half steps.
    substeps: list[tuple[float, float, int | None]] = []
    for number, (step, record) in enumerate(steps):
        if scheme == "crank_nicolson" and number < rannacher_steps:
            substeps += [(0.5 * step, 1.0, None), (0.5 * step, 1.0, record)]
        else:
            substeps.append((step, theta, record))

    values = payoff.copy()
    elapsed = 0.0
    kept: list[np.ndarray] = []

    for step, weight, record in substeps:
        elapsed += step
        interior = values[1:-1]
        right_hand_side = interior + (1 - weight) * step * (
            below * values[:-2] + centre * interior + above * values[2:]
        )
        low, high = boundaries(elapsed)
        exercise = obstacle(elapsed)[1:-1] if obstacle is not None else None

        if weight == 0.0:
            new = (
                right_hand_side
                if exercise is None
                else np.maximum(right_hand_side, exercise)
            )
        else:
            right_hand_side[0] += weight * step * below * low
            right_hand_side[-1] += weight * step * above * high
            lower, diagonal, upper = (
                -weight * step * below,
                1 - weight * step * centre,
                -weight * step * above,
            )
            if exercise is None:
                bands = np.empty((3, interior.size))
                bands[0], bands[1], bands[2] = upper, diagonal, lower
                new = solve_banded((1, 1), bands, right_hand_side)
            elif exercise_method == "brennan_schwartz":
                new = _brennan_schwartz(
                    lower,
                    diagonal,
                    upper,
                    right_hand_side.tolist(),
                    exercise.tolist(),
                    exercise_low,
                )
            else:
                new = _psor(
                    lower,
                    diagonal,
                    upper,
                    right_hand_side.tolist(),
                    exercise.tolist(),
                    np.maximum(interior, exercise).tolist(),
                )

        values = np.concatenate(([low], new, [high]))
        if record is not None:
            kept.append(values)

    return kept


def _log_grid(lowest: float, highest: float, price_steps: int) -> np.ndarray:
    """
    A uniform grid of log moneyness ln(S / K) from about lowest to highest in price_steps
    steps, with a grid point at 0 (the strike price), where the payoff has its kink.

    Args:
        lowest (float): The lowest log moneyness.
        highest (float): The highest log moneyness.
        price_steps (int): The number of steps.

    Returns:
        np.ndarray: The grid.
    """
    spacing = (highest - lowest) / price_steps
    first = int(np.floor(lowest / spacing))
    last = int(np.ceil(highest / spacing))
    return np.arange(first, last + 1) * spacing


def _vanilla_boundaries(
    grid: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float,
    put_option: bool,
    american_option: bool,
) -> Callable[[float], tuple[float, float]]:
    """
    The values of a vanilla option per unit of strike at the ends of the grid: the
    discounted forward intrinsic value, and for an American option at least the exercise
    value.

    Args:
        grid (np.ndarray): The grid of log moneyness.
        risk_free_rate (float): The risk-free rate.
        dividend_yield (float): The dividend yield.
        put_option (bool): Whether the option is a put.
        american_option (bool): Whether the option is American.

    Returns:
        Callable: The low and high boundary value per time to expiration.
    """
    sign = -1 if put_option else 1
    prices = np.exp(grid[[0, -1]])

    def boundaries(time: float) -> tuple[float, float]:
        value = np.maximum(
            sign
            * (
                prices * np.exp(-dividend_yield * time) - np.exp(-risk_free_rate * time)
            ),
            0.0,
        )
        if american_option:
            value = np.maximum(value, sign * (prices - 1))
        return float(value[0]), float(value[1])

    return boundaries


def get_finite_difference_option_values(
    stock_price: float,
    strike_prices: np.ndarray | list[float],
    times_to_expiration: np.ndarray | list[float],
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    american_option: bool = False,
    scheme: str = "crank_nicolson",
    price_steps: int = 1000,
    time_steps: int = 200,
    rannacher_steps: int = 2,
    exercise_method: str = "brennan_schwartz",
) -> np.ndarray:
    """
    Calculate the finite difference values of European or American options for several strike
    prices and times to expiration: what each option is worth, from one solve of the
    Black-Scholes equation (see get_finite_difference_option).

    The value of a vanilla option is homogeneous in the stock price and the strike price,
    V(S, K, τ) = K * v(ln(S / K), τ), so the equation is solved once for a strike of 1 on a
    grid of log moneyness that covers every strike, and stepping from expiration back to
    today passes every time to expiration on the way. The value of each strike is read off
    the grid at ln(S / K) with a cubic spline.

    A grid sized for the longest expiration is too coarse for one a day away, so the times
    to expiration are solved in groups whose longest is at most four times their shortest,
    each on a grid that reaches five standard deviations of its own longest expiration and
    with all time_steps. The values of a day to a quarter so stay within about 7e-4 of the
    exact European value for a stock price of 100 and a volatility up to 100%, where one
    grid for the whole range is off by up to 2.6e-2 a day from expiration.

    The formulas are as follows:

    - V(S, K, τ) = K * v(ln(S / K), τ), v solving v_τ = a * v_xx + b * v_x - r * v, a = σ² / 2, b = r - q - σ² / 2
    - (I - θ * Δτ * L) * v(τ + Δτ) = (I + (1 - θ) * Δτ * L) * v(τ), with L as in get_finite_difference_option
    - American: v(τ + Δτ) ≥ φ * (e^x - 1) at every step
    - Groups of times to expiration with T_max / T_min ≤ 4, each solved on its own grid

    Where τ is the time to expiration, x = ln(S / K), r the continuously compounded risk-free
    rate, q the continuous dividend yield, σ the volatility, φ = 1 for a call and -1 for a
    put and θ = 0, 1 or 1/2 for the explicit, implicit and Crank-Nicolson schemes.

    Shares similar logic:
    - QuantLib, FdBlackScholesVanillaEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html
    - FinancePy, black_scholes_fd: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/finite_difference.py#L318

    Args:
        stock_price (float): The current stock price.
        strike_prices (np.ndarray | list[float]): The strike prices.
        times_to_expiration (np.ndarray | list[float]): The times to expiration in years.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the options are puts. Defaults to False.
        american_option (bool, optional): Whether the options are American. Defaults to
            False.
        scheme (str, optional): The time stepping. Options are: "explicit" (explicit),
            "implicit" (implicit), "crank_nicolson" (Crank-Nicolson). Defaults to
            "crank_nicolson".
        price_steps (int, optional): The number of steps of the grid of log moneyness,
            which spans every strike price plus five standard deviations on either side.
            Defaults to 1,000.
        time_steps (int, optional): The number of time steps up to the longest time to
            expiration of each group solved together, with at least one between two times.
            Defaults to 200.
        rannacher_steps (int, optional): The number of Crank-Nicolson steps taken as two
            implicit half steps at the start. Defaults to 2.
        exercise_method (str, optional): The solver of the exercise constraint. Options are:
            "brennan_schwartz" (Brennan-Schwartz), "psor" (projected successive
            over-relaxation). Defaults to "brennan_schwartz".

    Returns:
        np.ndarray: The values, strike prices by times to expiration.

    Raises:
        ValueError: If a price or the volatility is not above zero, the scheme or the
            exercise method is unknown, or the explicit scheme is unstable at the steps.

    Notes:
        References: Crank, J., & Nicolson, P. (1947). "A Practical Method for Numerical
        Evaluation of Solutions of Partial Differential Equations of the Heat-Conduction
        Type." Mathematical Proceedings of the Cambridge Philosophical Society, 43(1),
        50-67. Brennan, M. J., & Schwartz, E. S. (1977). "The Valuation of American Put
        Options." The Journal of Finance, 32(2), 449-462.
        - QuantLib's FdBlackScholesVanillaEngine is used on a 4,000 by 4,000 grid.

    As an example:

    ```python
    from financetoolkit.options.numerical import finite_difference_model

    finite_difference_model.get_finite_difference_option_values(
        stock_price=100,
        strike_prices=[90, 100, 110],
        times_to_expiration=[1 / 365, 1 / 12, 1],
        risk_free_rate=0.04,
        volatility=0.3,
        dividend_yield=0.01,
        put_option=True,
        american_option=True,
    ).round(4).tolist()
    ```

    Which returns:

    ```
    [[0.0, 0.4196, 6.0368], [0.6226, 3.3391, 10.5385], [10.0, 10.4574, 16.4085]]
    ```
    """
    strikes = np.asarray(strike_prices, dtype=float)
    times = np.asarray(times_to_expiration, dtype=float)
    _validate(
        stock_price,
        float(strikes.min()) if strikes.size else 1.0,
        risk_free_rate,
        volatility,
        float(times.min()) if times.size else 0.0,
        dividend_yield,
        scheme,
        price_steps,
        time_steps,
        rannacher_steps,
        exercise_method,
    )
    sign = -1 if put_option else 1
    moneyness = np.log(stock_price / strikes)
    values = np.empty((strikes.size, times.size))
    values[:, times == 0] = np.maximum(sign * (stock_price - strikes), 0.0)[:, None]
    positive = np.unique(times[times > 0])

    if positive.size == 0 or strikes.size == 0:
        return values

    # One grid sized for the longest expiration spaces its points by a fraction of the
    # standard deviation of that expiration, far too coarse for one a day away, and gives
    # the first day a sliver of the time steps. So the expirations are solved in groups whose
    # longest is at most EXPIRATION_RATIO_PER_SOLVE times their shortest, each on its own grid
    # with all time_steps.
    groups = []
    for time in positive:
        if groups and time <= EXPIRATION_RATIO_PER_SOLVE * groups[-1][0]:
            groups[-1].append(time)
        else:
            groups.append([time])

    for group in groups:
        width = _grid_width(risk_free_rate, volatility, group[-1], dividend_yield)
        grid = _log_grid(
            min(moneyness.min(), 0.0) - width,
            max(moneyness.max(), 0.0) + width,
            price_steps,
        )
        payoff = np.maximum(sign * (np.exp(grid) - 1), 0.0)
        levels = _theta_scheme(
            grid,
            payoff,
            np.asarray(group),
            risk_free_rate,
            volatility,
            dividend_yield,
            scheme,
            time_steps,
            rannacher_steps,
            _vanilla_boundaries(
                grid, risk_free_rate, dividend_yield, put_option, american_option
            ),
            obstacle=(lambda _time, payoff=payoff: payoff) if american_option else None,
            exercise_low=put_option,
            exercise_method=exercise_method,
        )

        for time, level in zip(group, levels, strict=True):
            values[:, times == time] = (strikes * CubicSpline(grid, level)(moneyness))[
                :, None
            ]

    return values


def get_finite_difference_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    american_option: bool = False,
    scheme: str = "crank_nicolson",
    price_steps: int = 400,
    time_steps: int = 200,
    rannacher_steps: int = 2,
    exercise_method: str = "brennan_schwartz",
) -> float:
    """
    Calculate the value of a European or American option by solving the Black-Scholes equation
    with finite differences on the log stock price: what the option is worth on a grid of prices
    and times.

    Also known as: finite difference method, PDE pricing, Crank-Nicolson, θ-scheme.

    The equation is solved backwards from the payoff at expiration on a uniform grid of
    the log moneyness x = ln(S / K), with central differences in x and the θ-scheme in time.

    The formulas are as follows:

    - V_τ = a * V_xx + b * V_x - r * V, a = σ² / 2, b = r - q - σ² / 2
    - (I - θ * Δτ * L) * V(τ + Δτ) = (I + (1 - θ) * Δτ * L) * V(τ)
    - L * V(i) = (a / Δx² - b / (2Δx)) * V(i-1) - (2a / Δx² + r) * V(i) + (a / Δx² + b / (2Δx)) * V(i+1)
    - American: V(τ + Δτ) ≥ φ * (S - K) at every step

    Where τ is the time to expiration, r the continuously compounded risk-free rate, q the
    continuous dividend yield, σ the volatility, φ = 1 for a call and -1 for a put, and
    θ = 0 for the explicit scheme, 1 for the implicit scheme and 1/2 for Crank-Nicolson
    (1947). The explicit scheme is only stable for Δτ ≤ Δx² / σ² (and raises otherwise); the
    implicit scheme is first order in time and Crank-Nicolson second order, with its first
    steps taken as two implicit half steps each (Rannacher, 1984) so that the kink of the
    payoff does not set off oscillations. The grid reaches five standard deviations of the
    log price beyond the stock and strike price, with the discounted intrinsic value at its
    ends, and has a grid point at the strike price; the value at the stock price is read
    off with a cubic spline.

    American options are solved at every step as a linear complementarity problem, by the
    direct method of Brennan and Schwartz (1977), which eliminates towards the exercise
    region and substitutes back with the projection max(V, φ * (S - K)), or by projected
    successive over-relaxation (PSOR, Cryer 1971) with ω = 1.2 until a sweep changes less
    than 1e-12, which is far slower and gives the same values.

    Referenced against:
    - QuantLib, FdBlackScholesVanillaEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_vanilla_engine.html
    - FinancePy, black_scholes_fd: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/finite_difference.py#L318

    Shares similar logic:
    - FinancePy, black_scholes_fd_psor: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/finite_difference_psor.py#L13

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        american_option (bool, optional): Whether the option is American. Defaults to
            False.
        scheme (str, optional): The time stepping. Options are: "explicit" (explicit),
            "implicit" (implicit), "crank_nicolson" (Crank-Nicolson). Defaults to
            "crank_nicolson".
        price_steps (int, optional): The number of steps of the grid of log moneyness.
            Defaults to 400.
        time_steps (int, optional): The number of time steps. Defaults to 200.
        rannacher_steps (int, optional): The number of Crank-Nicolson steps taken as two
            implicit half steps at the start. Defaults to 2.
        exercise_method (str, optional): The solver of the exercise constraint of
            American options. Options are:
            "brennan_schwartz" (Brennan-Schwartz), "psor" (projected successive
            over-relaxation). Defaults to "brennan_schwartz".

    Returns:
        float: The value of the option.

    Raises:
        ValueError: If a price or the volatility is not above zero, the scheme or the
            exercise method is unknown, or the explicit scheme is unstable at the steps.

    Notes:
        References: Crank, J., & Nicolson, P. (1947). "A Practical Method for Numerical
        Evaluation of Solutions of Partial Differential Equations of the Heat-Conduction
        Type." Mathematical Proceedings of the Cambridge Philosophical Society, 43(1),
        50-67. Brennan, M. J., & Schwartz, E. S. (1977). "The Valuation of American Put
        Options." The Journal of Finance, 32(2), 449-462. Cryer, C. W. (1971). "The Solution
        of a Quadratic Programming Problem Using Systematic Overrelaxation." SIAM Journal on
        Control, 9(3), 385-392. Rannacher, R. (1984). "Finite Element Solution of Diffusion
        Problems with Irregular Data." Numerische Mathematik, 43(2), 309-327. Wilmott, P.,
        Howison, S., & Dewynne, J. (1995). "The Mathematics of Financial Derivatives."
        Cambridge University Press.

    As an example:

    ```python
    from financetoolkit.options.numerical import finite_difference_model

    finite_difference_model.get_finite_difference_option(
        stock_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiration=1,
        put_option=True,
        american_option=True,
    )
    ```

    Which returns:

    ```
    6.089271198367818
    ```
    """
    return float(
        get_finite_difference_option_values(
            stock_price,
            [strike_price],
            [time_to_expiration],
            risk_free_rate,
            volatility,
            dividend_yield,
            put_option,
            american_option,
            scheme,
            price_steps,
            time_steps,
            rannacher_steps,
            exercise_method,
        )[0, 0]
    )


def _barrier_solve(
    moneyness: float,
    barrier_moneyness: float,
    payoff: Callable[[np.ndarray], np.ndarray],
    far_boundary: Callable[[float, float], float],
    rebate: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float,
    scheme: str,
    price_steps: int,
    time_steps: int,
    rannacher_steps: int,
) -> float:
    """
    Solves a knock-out claim per unit of strike on a grid that ends at the barrier, with a
    grid point at the barrier and at the strike price, and the rebate paid on the barrier.

    Args:
        moneyness (float): The log moneyness ln(S / K) of the stock price.
        barrier_moneyness (float): The log moneyness ln(H / K) of the barrier.
        payoff (Callable): The payoff on the grid of log moneyness.
        far_boundary (Callable): The value at the far end of the grid per time.
        rebate (float): The rebate per unit of strike, paid when the barrier is hit.
        risk_free_rate (float): The risk-free rate.
        volatility (float): The volatility.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float): The dividend yield.
        scheme (str): The scheme.
        price_steps (int): The steps between the barrier and the far end of the grid.
        time_steps (int): The time steps.
        rannacher_steps (int): The Rannacher start-up steps.

    Returns:
        float: The value per unit of strike.
    """
    direction = 1 if barrier_moneyness < moneyness else -1
    width = _grid_width(risk_free_rate, volatility, time_to_expiration, dividend_yield)
    far = max(moneyness, 0.0) + width if direction == 1 else min(moneyness, 0.0) - width
    spacing = abs(far - barrier_moneyness) / price_steps
    # The barrier lies on a whole number of steps from the strike, so both are grid points.
    if barrier_moneyness != 0:
        spacing = abs(barrier_moneyness) / max(
            1, int(round(abs(barrier_moneyness) / spacing))
        )
    count = int(np.ceil(abs(far - barrier_moneyness) / spacing))
    grid = barrier_moneyness + direction * spacing * np.arange(count + 1)
    if direction == -1:
        grid = grid[::-1]

    values = payoff(grid)
    values[0 if direction == 1 else -1] = rebate

    def boundaries(time: float) -> tuple[float, float]:
        outer = far_boundary(grid[-1 if direction == 1 else 0], time)
        return (rebate, outer) if direction == 1 else (outer, rebate)

    level = _theta_scheme(
        grid,
        values,
        np.array([time_to_expiration]),
        risk_free_rate,
        volatility,
        dividend_yield,
        scheme,
        time_steps,
        rannacher_steps,
        boundaries,
    )[0]

    return float(CubicSpline(grid, level)(moneyness))


def get_finite_difference_barrier_option(
    stock_price: float,
    strike_price: float,
    barrier: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    barrier_direction: str = "down",
    knock_type: str = "out",
    rebate: float = 0.0,
    scheme: str = "crank_nicolson",
    price_steps: int = 400,
    time_steps: int = 200,
    rannacher_steps: int = 2,
) -> float:
    """
    Calculate the value of a continuously monitored single-barrier option with a rebate by
    solving the Black-Scholes equation with finite differences: what an option is worth that
    knocks in or out at the barrier (see get_finite_difference_option for the scheme).

    A knock-out option is solved on a grid that ends at the barrier, where its value is the
    rebate (paid when the barrier is hit), with grid points at the barrier and the strike
    price. A knock-in option is the European option less the knock-out option without
    rebate, plus the rebate paid at expiration when the barrier was never hit (a no-touch
    claim solved on the same grid).

    The formulas are as follows:

    - Knock-out: V_τ = a * V_xx + b * V_x - r * V on the side of the barrier, V(H, τ) = R
    - Knock-in = Black-Scholes - Knock-out (R = 0) + R * No-touch

    Where H is the barrier, R the rebate, and a and b as in get_finite_difference_option.
    The closed forms of Reiner and Rubinstein (1991) are in exotics_model.get_barrier_option.

    Shares similar logic:
    - QuantLib, FdBlackScholesBarrierEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_barrier_engine.html
    - QuantLib, FdBlackScholesRebateEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_fd_black_scholes_rebate_engine.html

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price.
        barrier (float): The barrier.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        barrier_direction (str, optional): The side of the barrier. Options are: "down"
            (below the stock price), "up" (above the stock price). Defaults to "down".
        knock_type (str, optional): The knock. Options are: "out" (knock-out), "in"
            (knock-in). Defaults to "out".
        rebate (float, optional): The rebate, paid when the barrier is hit for a knock-out
            option and at expiration when it is never hit for a knock-in option. Defaults
            to 0.
        scheme (str, optional): The time stepping. Options are: "explicit" (explicit),
            "implicit" (implicit), "crank_nicolson" (Crank-Nicolson). Defaults to
            "crank_nicolson".
        price_steps (int, optional): The number of steps between the barrier and the far
            end of the grid. Defaults to 400.
        time_steps (int, optional): The number of time steps. Defaults to 200.
        rannacher_steps (int, optional): The number of Crank-Nicolson steps taken as two
            implicit half steps at the start. Defaults to 2.

    Returns:
        float: The value of the barrier option. A stock price on or beyond the barrier has
        knocked the option out (worth the rebate) or in (worth the European option).

    Raises:
        ValueError: If the barrier direction or knock type is unknown, or as
            get_finite_difference_option.

    Notes:
        References: Reiner, E., & Rubinstein, M. (1991). "Breaking Down the Barriers."
        Risk, 4(8), 28-35. Wilmott, P., Howison, S., & Dewynne, J. (1995). "The Mathematics
        of Financial Derivatives." Cambridge University Press, chapter 11.

    As an example:

    ```python
    from financetoolkit.options.numerical import finite_difference_model

    finite_difference_model.get_finite_difference_barrier_option(
        stock_price=100,
        strike_price=100,
        barrier=90,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiration=1,
        rebate=3,
    )
    ```

    Which returns:

    ```
    10.290515246530084
    ```
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        scheme,
        price_steps,
        time_steps,
        rannacher_steps,
        "brennan_schwartz",
    )
    validate_numeric_inputs(barrier=barrier, rebate=rebate)
    check_option("barrier_direction", barrier_direction, ("down", "up"))
    check_option("knock_type", knock_type, ("out", "in"))
    if barrier <= 0:
        raise ValueError(f"barrier must be above zero, received {barrier!r}.")

    sign = -1 if put_option else 1
    european = float(
        get_black_scholes(
            stock_price,
            strike_price,
            risk_free_rate,
            volatility,
            time_to_expiration,
            dividend_yield,
            put_option,
        )
        if time_to_expiration > 0
        else max(sign * (stock_price - strike_price), 0.0)
    )
    hit = (
        stock_price <= barrier
        if barrier_direction == "down"
        else stock_price >= barrier
    )

    if hit:
        return float(rebate) if knock_type == "out" else european
    if time_to_expiration == 0:
        return european if knock_type == "out" else float(rebate)

    def vanilla(grid: np.ndarray) -> np.ndarray:
        return np.maximum(sign * (np.exp(grid) - 1), 0.0)

    def vanilla_far(point: float, time: float) -> float:
        return max(
            sign
            * (
                np.exp(point) * np.exp(-dividend_yield * time)
                - np.exp(-risk_free_rate * time)
            ),
            0.0,
        )

    inputs = (
        np.log(stock_price / strike_price),
        np.log(barrier / strike_price),
    )
    settings = (
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        scheme,
        price_steps,
        time_steps,
        rannacher_steps,
    )

    if knock_type == "out":
        return strike_price * _barrier_solve(
            *inputs, vanilla, vanilla_far, rebate / strike_price, *settings
        )

    knock_out = _barrier_solve(*inputs, vanilla, vanilla_far, 0.0, *settings)
    no_touch = (
        _barrier_solve(
            *inputs,
            np.ones_like,
            lambda _point, time: float(np.exp(-risk_free_rate * time)),
            0.0,
            *settings,
        )
        if rebate
        else 0.0
    )

    return european - strike_price * knock_out + rebate * no_touch


def get_finite_difference_shout_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    scheme: str = "crank_nicolson",
    price_steps: int = 400,
    time_steps: int = 200,
    rannacher_steps: int = 2,
    exercise_method: str = "brennan_schwartz",
) -> float:
    """
    Calculate the value of a shout option by solving the Black-Scholes equation with
    finite differences: what an option is worth whose holder may once lock in its intrinsic
    value before expiration and keep the upside (see get_finite_difference_option for the
    scheme).

    The holder of a shout option may once, at any time τ before expiration, lock in the
    intrinsic value S_τ - K (of a call; K - S_τ of a put) and keeps the upside after it: at
    expiration the call pays max(S_T - K, S_τ - K, 0) with a shout and max(S_T - K, 0)
    without. Shouting swaps the option for the locked amount, paid at expiration, plus an
    at-the-money option on the rest of the life, so the shout is an American-style right
    with that as its exercise value (Thomas, 1993).

    The formulas are as follows:

    - V(S, τ) ≥ φ * e^(-r * τ) * (S - K) + BS(S, S, τ)
    - BS(S, S, τ) = S * BS(1, 1, τ), the at-the-money Black-Scholes option

    Where τ is the time to expiration, φ = 1 for a call and -1 for a put and BS the
    Black-Scholes value of the same type. Every time step solves this constraint by
    Brennan-Schwartz or PSOR, as for an American option.

    Referenced against:
    - QuantLib, FdBlackScholesShoutEngine: https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/vanilla/fdblackscholesshoutengine.hpp#L35

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        scheme (str, optional): The time stepping. Options are: "explicit" (explicit),
            "implicit" (implicit), "crank_nicolson" (Crank-Nicolson). Defaults to
            "crank_nicolson".
        price_steps (int, optional): The number of steps of the grid of log moneyness.
            Defaults to 400.
        time_steps (int, optional): The number of time steps. Defaults to 200.
        rannacher_steps (int, optional): The number of Crank-Nicolson steps taken as two
            implicit half steps at the start. Defaults to 2.
        exercise_method (str, optional): The solver of the exercise constraint. Options are:
            "brennan_schwartz" (Brennan-Schwartz), "psor" (projected successive
            over-relaxation). Defaults to "brennan_schwartz".

    Returns:
        float: The value of the shout option.

    Raises:
        ValueError: As get_finite_difference_option.

    Notes:
        References: Thomas, B. (1993). "Something to Shout About." Risk, 6(5), 56-58.
        Windcliff, H., Forsyth, P. A., & Vetzal, K. R. (2001). "Shout Options: A Framework
        for Pricing Contracts Which Can Be Modified by the Investor." Journal of
        Computational and Applied Mathematics, 134(1-2), 213-241.

    As an example:

    ```python
    from financetoolkit.options.numerical import finite_difference_model

    finite_difference_model.get_finite_difference_shout_option(
        stock_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiration=1,
    )
    ```

    Which returns:

    ```
    12.837473185780635
    ```
    """
    _validate(
        stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        scheme,
        price_steps,
        time_steps,
        rannacher_steps,
        exercise_method,
    )
    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        return max(sign * (stock_price - strike_price), 0.0)

    moneyness = np.log(stock_price / strike_price)
    width = _grid_width(risk_free_rate, volatility, time_to_expiration, dividend_yield)
    grid = _log_grid(
        min(moneyness, 0.0) - width, max(moneyness, 0.0) + width, price_steps
    )
    prices = np.exp(grid)

    def exercise(time: float) -> np.ndarray:
        # The Black-Scholes value of an at-the-money option per unit of stock price.
        first = (
            (risk_free_rate - dividend_yield + 0.5 * volatility**2)
            * time
            / (volatility * np.sqrt(time))
        )
        second = first - volatility * np.sqrt(time)
        at_the_money = sign * (
            np.exp(-dividend_yield * time) * ndtr(sign * first)
            - np.exp(-risk_free_rate * time) * ndtr(sign * second)
        )
        return sign * np.exp(-risk_free_rate * time) * (prices - 1) + (
            prices * at_the_money
        )

    def boundaries(time: float) -> tuple[float, float]:
        forward = np.maximum(
            sign
            * (
                prices[[0, -1]] * np.exp(-dividend_yield * time)
                - np.exp(-risk_free_rate * time)
            ),
            0.0,
        )
        value = np.maximum(forward, exercise(time)[[0, -1]])
        return float(value[0]), float(value[1])

    level = _theta_scheme(
        grid,
        np.maximum(sign * (prices - 1), 0.0),
        np.array([time_to_expiration]),
        risk_free_rate,
        volatility,
        dividend_yield,
        scheme,
        time_steps,
        rannacher_steps,
        boundaries,
        obstacle=exercise,
        exercise_low=put_option,
        exercise_method=exercise_method,
    )[0]

    return float(strike_price * CubicSpline(grid, level)(moneyness))
