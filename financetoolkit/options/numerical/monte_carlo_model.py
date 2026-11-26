"""Monte Carlo Model"""

__docformat__ = "google"

import numbers

import numpy as np
from scipy.special import ndtr

from financetoolkit.options.helpers import (
    VARIANCE_REDUCTIONS,
    control_variate,
    draw_standard_normals,
    monte_carlo_estimate,
    simulation_groups,
    validate_numeric_inputs,
    validate_positive_inputs,
)
from financetoolkit.options.options_model import SIMULATION_CHUNK_SIZE
from financetoolkit.options.pricing.black_scholes_model import get_black_scholes
from financetoolkit.utilities.error_model import check_option

# The most prices (paths times time steps times series) a function that keeps whole paths
# holds at once, about 160 MB.
MAX_PATH_VALUES = 20_000_000
LEAST_SQUARES_BASES = ("laguerre", "monomial")
HESTON_SCHEMES = ("quadratic_exponential", "full_truncation")
# The switching level of ψ = s² / m² between the two branches of the quadratic-exponential
# scheme, the value Andersen (2008) recommends.
QUADRATIC_EXPONENTIAL_SWITCH = 1.5


def _check_size(simulations: int, time_steps: int, series: int = 1) -> None:
    """
    Raises a ValueError for paths too large to hold at once.

    Args:
        simulations (int): The number of paths.
        time_steps (int): The number of steps per path.
        series (int, optional): The number of series per path (stocks, or the price and
            variance of the Heston model). Defaults to 1.
    """
    if simulations * (time_steps + 1) * series > MAX_PATH_VALUES:
        raise ValueError(
            f"{simulations:,} paths of {time_steps:,} steps and {series} series is too "
            f"large; keep paths times steps (plus one) times series under "
            f"{MAX_PATH_VALUES:,}."
        )


def _variance_reduction(variance_reduction: str | None) -> None:
    """
    Raises a ValueError for an unknown variance reduction.

    Args:
        variance_reduction (str | None): The variance reduction.
    """
    check_option("variance_reduction", variance_reduction, VARIANCE_REDUCTIONS)


def simulate_gbm_paths(
    stock_price: float | np.ndarray | list[float],
    risk_free_rate: float,
    volatility: float | np.ndarray | list[float],
    time_to_expiration: float,
    dividend_yield: float | np.ndarray | list[float] = 0.0,
    correlation_matrix: np.ndarray | list[list[float]] | None = None,
    time_steps: int = 100,
    simulations: int = 10_000,
    sequence: str = "pseudo_random",
    antithetic: bool = False,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths of one or more correlated stocks under the
    risk-neutral measure: the prices the stocks may follow up to expiration, exact at every time
    step.

    The formula is as follows:

    - S_i(t + Δt) = S_i(t) * e^((r - q_i - σ_i² / 2) * Δt + σ_i * √Δt * X_i), X = L * Z

    Where r is the continuously compounded risk-free rate, q_i the continuous dividend yield
    and σ_i the volatility of stock i, Δt = T / time_steps, Z independent standard normal
    numbers and L the Cholesky factor of the correlation matrix (L * Lᵀ = ρ, Glasserman
    2003, section 2.3). The numbers are pseudo-random (numpy's PCG64 seeded with seed) or
    scrambled Sobol points with a Brownian bridge (16 independently scrambled replications),
    optionally in antithetic pairs, see helpers.draw_standard_normals.

    Shares similar logic:
    - FinancePy, get_gbm_paths: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/process_simulator.py#L267
    - FinancePy, get_assets_paths: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/gbm_process_simulator.py#L163

    Args:
        stock_price (float | np.ndarray | list[float]): The current price of each stock.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float | np.ndarray | list[float]): The volatility of each stock.
        time_to_expiration (float): The horizon in years.
        dividend_yield (float | np.ndarray | list[float], optional): The continuous dividend
            yield of each stock. Defaults to 0.
        correlation_matrix (np.ndarray | list[list[float]] | None, optional): The
            correlations of the returns, positive definite with ones on the diagonal.
            Defaults to None, independent stocks.
        time_steps (int, optional): The number of equal time steps. Defaults to 100.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        antithetic (bool, optional): Whether every path is followed by its antithetic
            path. Defaults to False.
        seed (int | None, optional): The seed of the random numbers. Defaults to 0; None
            draws other paths on every call.

    Returns:
        np.ndarray: The prices, paths by time steps + 1 (starting with today's price) for
        one stock, and paths by time steps + 1 by stocks for several.

    Raises:
        ValueError: If the inputs do not have one value per stock, the correlation matrix
            is not positive definite or the paths are too large to hold
            (MAX_PATH_VALUES).

    Notes:
        References: Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering."
        Springer, chapters 2 and 3.
        - FinancePy draws its own random numbers, which give other paths.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    paths = monte_carlo_model.simulate_gbm_paths(
        stock_price=[100, 50],
        risk_free_rate=0.05,
        volatility=[0.2, 0.3],
        time_to_expiration=1,
        correlation_matrix=[[1, 0.5], [0.5, 1]],
        time_steps=4,
        simulations=10_000,
    )

    paths.mean(axis=0).round(2)
    ```

    Which returns:

    ```
    array([[100.  ,  50.  ],
           [101.27,  50.61],
           [102.48,  51.26],
           [103.76,  51.87],
           [105.32,  52.44]])
    ```
    """
    validate_numeric_inputs(
        risk_free_rate=risk_free_rate, time_to_expiration=time_to_expiration
    )
    single = np.ndim(stock_price) == 0
    prices = np.atleast_1d(np.asarray(stock_price, dtype=float))
    assets = prices.size
    volatilities = np.broadcast_to(np.asarray(volatility, dtype=float), (assets,))
    dividend_yields = np.broadcast_to(
        np.asarray(dividend_yield, dtype=float), (assets,)
    )
    correlation = (
        np.eye(assets)
        if correlation_matrix is None
        else np.asarray(correlation_matrix, dtype=float)
    )
    if correlation.shape != (assets, assets):
        raise ValueError("correlation_matrix must have a row and column per stock.")
    if (prices <= 0).any() or (volatilities < 0).any() or time_to_expiration < 0:
        raise ValueError(
            "Stock prices must be above zero and the volatility and time to expiration "
            "at least zero."
        )
    # The Cholesky factor is unique, unlike an eigenvector decomposition whose signs depend
    # on the linear algebra library, so a seed gives the same paths on every platform.
    try:
        cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("correlation_matrix must be positive definite.") from error
    _check_size(simulations, time_steps, assets)

    time_delta = time_to_expiration / time_steps
    drift = (risk_free_rate - dividend_yields - 0.5 * volatilities**2) * time_delta
    diffusion = volatilities * np.sqrt(time_delta)
    paths = np.empty((simulations, time_steps + 1, assets))
    paths[:, 0] = np.log(prices)
    first = 0

    for normals in draw_standard_normals(
        simulations,
        time_steps,
        assets,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
        chunk_size=SIMULATION_CHUNK_SIZE,
    ):
        rows = normals.shape[0]
        paths[first : first + rows, 1:] = paths[first : first + rows, :1] + np.cumsum(
            drift + diffusion * (normals @ cholesky.T), axis=1
        )
        first += rows

    paths = np.exp(paths)

    return paths[:, :, 0] if single else paths


def simulate_heston_paths(
    stock_price: float,
    variance: float,
    risk_free_rate: float,
    mean_reversion: float,
    long_run_variance: float,
    volatility_of_variance: float,
    correlation: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    time_steps: int = 100,
    simulations: int = 10_000,
    scheme: str = "quadratic_exponential",
    sequence: str = "pseudo_random",
    antithetic: bool = False,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate stock price and variance paths of the Heston (1993) stochastic volatility model
    under the risk-neutral measure: the prices and variances the stock may follow up to
    expiration when its variance reverts to a long-run level.

    The formulas are as follows:

    - dS = (r - q) * S * dt + √v * S * dW_S
    - dv = κ * (θ - v) * dt + ξ * √v * dW_v, dW_S * dW_v = ρ * dt

    Two discretizations are available:

    - Quadratic-exponential (Andersen 2008): the variance
      at the end of a step is drawn from a distribution with the exact mean m and variance
      s² of the CIR process, a scaled non-central chi-squared with one degree of freedom
      a * (b + Z_v)² when ψ = s² / m² ≤ 1.5 and otherwise a point mass at zero with an
      exponential tail; the log price follows with the variance integrated by the
      trapezium rule:
        - m = θ + (v - θ) * e^(-κΔt)
        - s² = v * ξ² * e^(-κΔt) * (1 - e^(-κΔt)) / κ + θ * ξ² * (1 - e^(-κΔt))² / (2κ)
        - ψ ≤ 1.5: b² = 2/ψ - 1 + √(2/ψ) * √(2/ψ - 1), a = m / (1 + b²), v' = a * (b + Z_v)²
        - ψ > 1.5: p = (ψ - 1) / (ψ + 1), β = (1 - p) / m, v' = 0 if U ≤ p, else ln((1 - p) / (1 - U)) / β
        - ln S' = ln S + (r - q) * Δt + K0 + K1 * v + K2 * v' + √(K3 * v + K4 * v') * Z
        - K0 = -ρκθΔt / ξ, K1 = Δt * (κρ / ξ - 1/2) / 2 - ρ / ξ, K2 = Δt * (κρ / ξ - 1/2) / 2 + ρ / ξ
        - K3 = K4 = Δt * (1 - ρ²) / 2
    - Full truncation Euler (Lord, Koekkoek and van Dijk 2010),
      with v⁺ = max(v, 0):
        - v' = v + κ * (θ - v⁺) * Δt + ξ * √(v⁺ * Δt) * Z_v
        - ln S' = ln S + (r - q - v⁺ / 2) * Δt + √(v⁺ * Δt) * (ρ * Z_v + √(1 - ρ²) * Z)

    Where v is the variance, κ the speed of mean reversion, θ the long-run variance, ξ the
    volatility of variance, ρ the correlation of the two Brownian motions, Z_v and Z
    independent standard normal numbers and U = N(Z_v), r the continuously compounded
    risk-free rate and q the continuous dividend yield. The quadratic-exponential scheme
    keeps the variance at or above zero and is accurate at far larger steps.

    Referenced against:
    - QuantLib, AnalyticHestonEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_analytic_heston_engine.html

    Shares similar logic:
    - FinancePy, get_heston_paths: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/process_simulator.py#L126

    Args:
        stock_price (float): The current stock price.
        variance (float): The current variance v (the squared volatility).
        risk_free_rate (float): The continuously compounded risk-free rate.
        mean_reversion (float): The speed of mean reversion κ.
        long_run_variance (float): The long-run variance θ.
        volatility_of_variance (float): The volatility of the variance ξ.
        correlation (float): The correlation ρ of the stock and its variance.
        time_to_expiration (float): The horizon in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        time_steps (int, optional): The number of equal time steps. Defaults to 100.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        scheme (str, optional): The discretization. Options are: "quadratic_exponential"
            (quadratic-exponential), "full_truncation" (full truncation Euler). Defaults to
            "quadratic_exponential".
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        antithetic (bool, optional): Whether every path is followed by its antithetic
            path. Defaults to False.
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: The stock prices and the variances, each paths by
        time steps + 1.

    Raises:
        ValueError: If the scheme is unknown, a parameter is out of range or the paths are
            too large to hold (MAX_PATH_VALUES).

    Notes:
        References: Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic
        Volatility with Applications to Bond and Currency Options." The Review of Financial
        Studies, 6(2), 327-343. Andersen, L. (2008). "Simple and Efficient Simulation of the
        Heston Stochastic Volatility Model." Journal of Computational Finance, 11(3), 1-42.
        Lord, R., Koekkoek, R., & van Dijk, D. (2010). "A Comparison of Biased Simulation
        Schemes for Stochastic Volatility Models." Quantitative Finance, 10(2), 177-194.
        - The option prices of the paths are compared with QuantLib's AnalyticHestonEngine.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    prices, variances = monte_carlo_model.simulate_heston_paths(
        stock_price=100,
        variance=0.04,
        risk_free_rate=0.05,
        mean_reversion=1.5,
        long_run_variance=0.04,
        volatility_of_variance=0.5,
        correlation=-0.7,
        time_to_expiration=1,
        time_steps=50,
    )

    round(float(prices[:, -1].mean()), 2), round(float(variances[:, -1].mean()), 4)
    ```

    Which returns:

    ```
    (104.9, 0.0406)
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        variance=variance,
        risk_free_rate=risk_free_rate,
        mean_reversion=mean_reversion,
        long_run_variance=long_run_variance,
        volatility_of_variance=volatility_of_variance,
        correlation=correlation,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    check_option("scheme", scheme, HESTON_SCHEMES)
    if (
        stock_price <= 0
        or variance < 0
        or mean_reversion <= 0
        or long_run_variance <= 0
        or volatility_of_variance <= 0
        or not -1 <= correlation <= 1
        or time_to_expiration < 0
    ):
        raise ValueError(
            "The stock price, speed of mean reversion, long-run variance and volatility of "
            "variance must be above zero, the variance and time at least zero and the "
            "correlation between -1 and 1."
        )
    _check_size(simulations, time_steps, 2)

    step = time_to_expiration / time_steps
    decay = np.exp(-mean_reversion * step)
    rho, xi, kappa, theta = (
        correlation,
        volatility_of_variance,
        mean_reversion,
        long_run_variance,
    )
    log_prices = np.empty((simulations, time_steps + 1))
    variances = np.empty((simulations, time_steps + 1))
    log_prices[:, 0], variances[:, 0] = np.log(stock_price), variance
    carry = (risk_free_rate - dividend_yield) * step
    first = 0

    for normals in draw_standard_normals(
        simulations,
        time_steps,
        2,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
        chunk_size=SIMULATION_CHUNK_SIZE,
    ):
        rows = slice(first, first + normals.shape[0])
        first += normals.shape[0]
        for index in range(time_steps):
            current = variances[rows, index]
            variance_shock, price_shock = normals[:, index, 0], normals[:, index, 1]

            if scheme == "full_truncation":
                positive = np.maximum(current, 0.0)
                variances[rows, index + 1] = (
                    current
                    + kappa * (theta - positive) * step
                    + xi * np.sqrt(positive * step) * variance_shock
                )
                log_prices[rows, index + 1] = (
                    log_prices[rows, index]
                    + carry
                    - 0.5 * positive * step
                    + np.sqrt(positive * step)
                    * (rho * variance_shock + np.sqrt(1 - rho**2) * price_shock)
                )
                continue

            mean = theta + (current - theta) * decay
            spread = current * xi**2 * decay * (
                1 - decay
            ) / kappa + theta * xi**2 * (1 - decay) ** 2 / (2 * kappa)
            ratio = spread / np.maximum(mean, 1e-300) ** 2
            quadratic = ratio <= QUADRATIC_EXPONENTIAL_SWITCH
            following = np.empty_like(current)

            inverse = 2 / ratio[quadratic]
            shift_squared = inverse - 1 + np.sqrt(inverse) * np.sqrt(inverse - 1)
            following[quadratic] = (
                mean[quadratic]
                / (1 + shift_squared)
                * (np.sqrt(shift_squared) + variance_shock[quadratic]) ** 2
            )

            mass = (ratio[~quadratic] - 1) / (ratio[~quadratic] + 1)
            rate = (1 - mass) / mean[~quadratic]
            uniform = ndtr(variance_shock[~quadratic])
            following[~quadratic] = np.where(
                uniform <= mass,
                0.0,
                np.log((1 - mass) / np.maximum(1 - uniform, 1e-300)) / rate,
            )

            variances[rows, index + 1] = following
            log_prices[rows, index + 1] = (
                log_prices[rows, index]
                + carry
                - rho * kappa * theta * step / xi
                + (0.5 * step * (kappa * rho / xi - 0.5) - rho / xi) * current
                + (0.5 * step * (kappa * rho / xi - 0.5) + rho / xi) * following
                + np.sqrt(0.5 * step * (1 - rho**2) * (current + following))
                * price_shock
            )

    return np.exp(log_prices), variances


def get_monte_carlo_barrier_option(
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
    monitoring_dates: int = 252,
    simulations: int = 10_000,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
    seed: int | None = 0,
) -> tuple[float, float]:
    """
    Calculate the Monte Carlo price of a discretely monitored single-barrier option with a
    rebate: what an option is worth that knocks in or out when the stock price is beyond the
    barrier on one of its monitoring dates.

    The barrier is only checked at the equally spaced monitoring dates, the last at expiration,
    which is how most barrier contracts are written (at the daily close).

    The stock price is simulated exactly at the monitoring dates (simulate_gbm_paths), and
    with t_h the first monitoring date at which the price is at or beyond the barrier.

    The formulas are as follows:

    - Knock-out = e^(-r * t_h) * R if the barrier is hit, else e^(-rT) * max(φ * (S_T - K), 0)
    - Knock-in = e^(-rT) * max(φ * (S_T - K), 0) if the barrier is hit, else e^(-rT) * R

    Where R is the rebate, K the strike price, r the continuously compounded risk-free rate,
    T the time to expiration and φ = 1 for a call and -1 for a put. With a
    control variate, the European option on the same paths, whose
    Black-Scholes price is known, is the control. A discretely monitored barrier is
    crossed less often than a continuous one; its closed-form approximation shifts the
    barrier by e^(±0.5826 * σ * √Δt) (Broadie, Glasserman and Kou 1997, see
    path_dependent_model.get_discrete_monitoring_barrier).

    Shares similar logic:
    - FinancePy, EquityBarrierOption.value_mc: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/equity/equity_barrier_option.py#L132

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
        rebate (float, optional): The rebate, paid at the monitoring date of the hit for a
            knock-out option and at expiration when the barrier is never hit for a
            knock-in option. Defaults to 0.
        monitoring_dates (int, optional): The number of equally spaced monitoring dates,
            which are also the time steps of the paths. Defaults to 252 (daily over a
            year).
        simulations (int, optional): The number of paths. Defaults to 10,000.
        variance_reduction (str | None, optional): The variance reduction. Options are: None
            (none), "antithetic" (antithetic variates), "control_variate" (control variate).
            Defaults to None.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        tuple[float, float]: The price and its standard error.

    Raises:
        ValueError: If the barrier direction, knock type, variance reduction or sequence is
            unknown, or the paths are too large to hold (MAX_PATH_VALUES).

    Notes:
        References: Broadie, M., Glasserman, P., & Kou, S. (1997). "A Continuity Correction
        for Discrete Barrier Options." Mathematical Finance, 7(4), 325-349. Glasserman, P.
        (2003). "Monte Carlo Methods in Financial Engineering." Springer, section 6.4.
        - QuantLib's MCBarrierEngine prices the continuously monitored barrier and is no
          reference for a discrete one.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    price, standard_error = monte_carlo_model.get_monte_carlo_barrier_option(
        stock_price=100,
        strike_price=100,
        barrier=90,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_expiration=1,
        variance_reduction="control_variate",
    )

    round(price, 4), round(standard_error, 4)
    ```

    Which returns:

    ```
    (8.9117, 0.0533)
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        strike_price=strike_price,
        barrier=barrier,
        rebate=rebate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    _variance_reduction(variance_reduction)
    check_option("barrier_direction", barrier_direction, ("down", "up"))
    check_option("knock_type", knock_type, ("out", "in"))

    antithetic = variance_reduction == "antithetic"
    paths = simulate_gbm_paths(
        stock_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        time_steps=monitoring_dates,
        simulations=simulations,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
    )
    monitored = paths[:, 1:]
    crossed = (
        monitored <= barrier if barrier_direction == "down" else monitored >= barrier
    )
    hit = crossed.any(axis=1)
    hit_times = (crossed.argmax(axis=1) + 1) * time_to_expiration / monitoring_dates
    sign = -1 if put_option else 1
    discount = np.exp(-risk_free_rate * time_to_expiration)
    european = discount * np.maximum(sign * (paths[:, -1] - strike_price), 0.0)

    if knock_type == "out":
        values = np.where(hit, rebate * np.exp(-risk_free_rate * hit_times), european)
    else:
        values = np.where(hit, european, discount * rebate)

    if variance_reduction == "control_variate":
        values = control_variate(
            values,
            european,
            float(
                get_black_scholes(
                    stock_price,
                    strike_price,
                    risk_free_rate,
                    volatility,
                    time_to_expiration,
                    dividend_yield,
                    put_option,
                )
            ),
        )

    return monte_carlo_estimate(
        values, simulation_groups(simulations, sequence, antithetic)
    )


def _regression_basis(ratios: np.ndarray, basis: str, degree: int) -> np.ndarray:
    """
    The regressors of the continuation value in the stock price divided by the strike
    price x: a constant with the weighted Laguerre polynomials e^(-x/2) * L_n(x),
    n = 0, ..., degree - 1 (Longstaff and Schwartz, 2001), or the powers x^0, ..., x^degree.

    Args:
        ratios (np.ndarray): The stock prices divided by the strike price.
        basis (str): "laguerre" or "monomial".
        degree (int): The number of basis functions besides the constant.

    Returns:
        np.ndarray: The regressors, paths by degree + 1.
    """
    if basis == "monomial":
        return np.vander(ratios, degree + 1, increasing=True)
    weighted = (
        np.polynomial.laguerre.lagvander(ratios, degree - 1)
        * np.exp(-ratios / 2)[:, None]
    )
    return np.column_stack((np.ones(ratios.size), weighted))


def _longstaff_schwartz(
    payoffs: np.ndarray,
    regressors,
    discount: float,
    european=None,
) -> np.ndarray:
    """
    Values the cash flows of every path of an option that can be exercised at every time
    step by least-squares Monte Carlo (Longstaff and Schwartz, 2001): stepping back from
    expiration, the discounted cash flows of the paths in the money are regressed on the
    basis functions of the prices, and a path is exercised where its exercise value is
    above the fitted continuation value, and above the European value of the remaining life
    where that is known, as holding the option is worth at least that.

    Args:
        payoffs (np.ndarray): The exercise value of every path at every exercise date,
            paths by dates (the last date is expiration).
        regressors (Callable): The basis functions of the paths given at a date.
        discount (float): The discount factor of one time step.
        european (Callable | None, optional): The European value of the remaining life of
            the paths given at a date, a lower bound of the continuation value. Defaults to
            None, which uses the fitted continuation value alone.

    Returns:
        np.ndarray: The cash flow of every path discounted to today.
    """
    cash_flows = payoffs[:, -1].copy()

    for date in range(payoffs.shape[1] - 2, -1, -1):
        cash_flows *= discount
        exercise = payoffs[:, date]
        in_the_money = np.flatnonzero(exercise > 0)
        basis = regressors(date, in_the_money)
        if in_the_money.size <= basis.shape[1]:
            continue
        coefficients = np.linalg.lstsq(basis, cash_flows[in_the_money], rcond=None)[0]
        continuation = basis @ coefficients
        if european is not None:
            continuation = np.maximum(continuation, european(date, in_the_money))
        exercised = in_the_money[exercise[in_the_money] > continuation]
        cash_flows[exercised] = exercise[exercised]

    return cash_flows * discount


def get_longstaff_schwartz_option_prices(
    stock_price: float,
    strike_prices: np.ndarray | list[float],
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    simulations: int = 10_000,
    time_steps: int = 50,
    basis: str = "laguerre",
    degree: int = 3,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the least-squares Monte Carlo price of American options with several strike prices
    on the same paths: what each option is worth when it may be exercised at every time step
    (see get_longstaff_schwartz_option).

    The paths are simulated once and the exercise rule is fitted per strike price, so the prices
    of the strikes share their random numbers and a difference between two of them is far more
    precise than either price.

    The formulas are as follows:

    - C(t) = max(Σ_j β_j * f_j(S(t) / K), BS(S(t), K, T - t)), β fitted by least squares on the in-the-money paths
    - Cash flow(t) = φ * (S(t) - K) if φ * (S(t) - K) > C(t), else e^(-rΔt) * Cash flow(t + Δt)
    - Price = max(φ * (S - K), MEAN(e^(-rΔt) * Cash flow(Δt)))

    Where f_j are the basis functions of x = S / K (see get_longstaff_schwartz_option), K is
    each strike price, r the continuously compounded risk-free rate, Δt = T / time_steps,
    BS the Black-Scholes value of the same type and φ = 1 for a call and -1 for a put.

    Shares similar logic:
    - QuantLib, MCAmericanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_american_engine.html
    - FinancePy, equity_lsmc: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/equity_lsmc.py#L39

    Args:
        stock_price (float): The current stock price.
        strike_prices (np.ndarray | list[float]): The strike prices.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the options are puts. Defaults to False.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        time_steps (int, optional): The number of exercise dates. Defaults to 50.
        basis (str, optional): The basis functions of the regression. Options are:
            "laguerre" (weighted Laguerre polynomials), "monomial" (powers). Defaults to
            "laguerre".
        degree (int, optional): The number of basis functions besides the constant.
            Defaults to 3.
        variance_reduction (str | None, optional): The variance reduction. Options are: None
            (none), "antithetic" (antithetic variates), "control_variate" (control variate).
            Defaults to None.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: The price and its standard error per strike price.

    Raises:
        ValueError: If the basis, variance reduction or sequence is unknown, the degree is
            not a positive integer or the paths are too large to hold (MAX_PATH_VALUES).

    Notes:
        A price at the exercise value (exercising today is better than any rule) has a
        standard error of 0. References: Longstaff, F. A., & Schwartz, E. S. (2001).
        "Valuing American Options by Simulation: A Simple Least-Squares Approach." The Review
        of Financial Studies, 14(1), 113-147. Rasmussen, N. S. (2005). "Control Variates for
        Monte Carlo Valuation of American Options." Journal of Computational Finance, 9(1),
        83-118.
        - The prices of Table 1 of Longstaff and Schwartz (2001) lie within two standard
          errors.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    prices, standard_errors = monte_carlo_model.get_longstaff_schwartz_option_prices(
        stock_price=40,
        strike_prices=[36, 40, 44],
        risk_free_rate=0.06,
        volatility=0.2,
        time_to_expiration=1,
        put_option=True,
        simulations=100_000,
        variance_reduction="antithetic",
    )

    prices.round(4).tolist(), standard_errors.round(4).tolist()
    ```

    Which returns:

    ```
    ([0.9065, 2.3185, 4.6572], [0.0047, 0.0053, 0.0063])
    ```
    """
    validate_numeric_inputs(
        stock_price=stock_price,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiration=time_to_expiration,
        dividend_yield=dividend_yield,
    )
    _variance_reduction(variance_reduction)
    check_option("basis", basis, LEAST_SQUARES_BASES)
    if not isinstance(degree, numbers.Integral) or degree < 1:
        raise ValueError(f"degree must be a positive integer, received {degree!r}.")

    # The regression is on the stock price over the strike price.
    validate_positive_inputs(strike_prices=strike_prices)
    strikes = np.asarray(strike_prices, dtype=float)
    sign = -1 if put_option else 1

    if time_to_expiration == 0:
        return np.maximum(sign * (stock_price - strikes), 0.0), np.zeros(strikes.size)

    antithetic = variance_reduction == "antithetic"
    paths = simulate_gbm_paths(
        stock_price,
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        time_steps=time_steps,
        simulations=simulations,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
    )
    groups = simulation_groups(simulations, sequence, antithetic)
    discount = np.exp(-risk_free_rate * time_to_expiration / time_steps)
    prices = np.empty(strikes.size)
    standard_errors = np.empty(strikes.size)

    for position, strike in enumerate(strikes):
        payoffs = np.maximum(sign * (paths[:, 1:] - strike), 0.0)
        values = _longstaff_schwartz(
            payoffs,
            lambda date, rows, strike=strike: _regression_basis(
                paths[rows, date + 1] / strike, basis, degree
            ),
            discount,
            lambda date, rows, strike=strike: np.asarray(
                get_black_scholes(
                    paths[rows, date + 1],
                    float(strike),
                    risk_free_rate,
                    volatility,
                    time_to_expiration * (1 - (date + 1) / time_steps),
                    dividend_yield,
                    put_option,
                )
            ),
        )
        if variance_reduction == "control_variate":
            values = control_variate(
                values,
                discount**time_steps * payoffs[:, -1],
                float(
                    get_black_scholes(
                        stock_price,
                        float(strike),
                        risk_free_rate,
                        volatility,
                        time_to_expiration,
                        dividend_yield,
                        put_option,
                    )
                ),
            )
        price, standard_error = monte_carlo_estimate(values, groups)
        # Exercising today is the one decision the paths can not make.
        intrinsic = max(sign * (stock_price - strike), 0.0)
        prices[position] = max(price, intrinsic)
        standard_errors[position] = standard_error if price >= intrinsic else 0.0

    return prices, standard_errors


def get_longstaff_schwartz_option(
    stock_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiration: float,
    dividend_yield: float = 0.0,
    put_option: bool = False,
    simulations: int = 10_000,
    time_steps: int = 50,
    basis: str = "laguerre",
    degree: int = 3,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
    seed: int | None = 0,
) -> tuple[float, float]:
    """
    Calculate the price of an American option by least-squares Monte Carlo (Longstaff and
    Schwartz, 2001): what the option is worth when it may be exercised at every time step, with
    the exercise rule fitted by regression on simulated paths.

    Also known as: LSM, LSMC, least squares Monte Carlo, regression Monte Carlo.

    Geometric Brownian Motion paths are simulated at time_steps equally spaced exercise
    dates (simulate_gbm_paths). Stepping back from expiration, the cash flows of the paths
    in the money are discounted to the date and regressed on basis functions of x = S / K,
    and a path is exercised where the exercise value beats the continuation value: the
    fitted one, or the Black-Scholes value of the European option for the remaining life
    where that is higher, as holding the option is worth at least that.

    The formulas are as follows:

    - C(t) = max(Σ_j β_j * f_j(S(t) / K), BS(S(t), K, T - t)), β fitted by least squares on the in-the-money paths
    - Cash flow(t) = φ * (S(t) - K) if φ * (S(t) - K) > C(t), else e^(-rΔt) * Cash flow(t + Δt)
    - Price = max(φ * (S - K), MEAN(e^(-rΔt) * Cash flow(Δt)))

    Where f_j are a constant with the weighted Laguerre polynomials e^(-x/2) * L_n(x),
    n = 0, ..., degree - 1 (the Laguerre basis, as in the paper) or the powers x^0, ...,
    x^degree (the monomial basis), K is the strike price, r the continuously compounded
    risk-free rate, Δt = T / time_steps, BS the Black-Scholes value of the same type and
    φ = 1 for a call and -1 for a put.

    The estimate carries two biases of opposite sign. The exercise rule is fitted on the
    paths it values, as in the paper, which biases the price upwards, while a fitted rule is
    never quite the optimal one, and exercising only at the time steps (a Bermudan option)
    is worth less than at any time, which bias it downwards. Which one wins depends on the
    case: in Longstaff and Schwartz's own puts (S = 36 to 44, K = 40, r = 6%) the price lies
    within two standard errors of the finite difference value, and where early exercise is
    never optimal (a put with r ≤ 0 ≤ q, a call with q = 0) every exercise of the fitted rule
    alone would be an error, which the European floor of the continuation value rules out.
    With a control variate, the European payoff on the same
    paths, whose Black-Scholes price is known, is the control (Rasmussen, 2005).

    Shares similar logic:
    - QuantLib, MCAmericanEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_american_engine.html
    - FinancePy, equity_lsmc: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/equity_lsmc.py#L39

    Args:
        stock_price (float): The current stock price.
        strike_price (float): The strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatility (float): The volatility of the stock.
        time_to_expiration (float): The time to expiration in years.
        dividend_yield (float, optional): The continuous dividend yield. Defaults to 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        time_steps (int, optional): The number of exercise dates. Defaults to 50.
        basis (str, optional): The basis functions of the regression. Options are:
            "laguerre" (weighted Laguerre polynomials), "monomial" (powers). Defaults to
            "laguerre".
        degree (int, optional): The number of basis functions besides the constant.
            Defaults to 3.
        variance_reduction (str | None, optional): The variance reduction. Options are: None
            (none), "antithetic" (antithetic variates), "control_variate" (control variate).
            Defaults to None.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        tuple[float, float]: The price and its standard error.

    Raises:
        ValueError: If the basis, variance reduction or sequence is unknown, the degree is
            not a positive integer or the paths are too large to hold (MAX_PATH_VALUES).

    Notes:
        References: Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American Options by
        Simulation: A Simple Least-Squares Approach." The Review of Financial Studies,
        14(1), 113-147. Rasmussen, N. S. (2005). "Control Variates for Monte Carlo Valuation
        of American Options." Journal of Computational Finance, 9(1), 83-118.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    price, standard_error = monte_carlo_model.get_longstaff_schwartz_option(
        stock_price=36,
        strike_price=40,
        risk_free_rate=0.06,
        volatility=0.2,
        time_to_expiration=1,
        put_option=True,
        simulations=100_000,
        variance_reduction="antithetic",
    )

    round(price, 4), round(standard_error, 4)
    ```

    Which returns:

    ```
    (4.4783, 0.006)
    ```
    """
    prices, standard_errors = get_longstaff_schwartz_option_prices(
        stock_price,
        [strike_price],
        risk_free_rate,
        volatility,
        time_to_expiration,
        dividend_yield,
        put_option,
        simulations,
        time_steps,
        basis,
        degree,
        variance_reduction,
        sequence,
        seed,
    )
    return float(prices[0]), float(standard_errors[0])


def get_longstaff_schwartz_basket_option(
    stock_prices: np.ndarray | list[float],
    weights: np.ndarray | list[float],
    strike_price: float,
    risk_free_rate: float,
    volatilities: np.ndarray | list[float],
    correlation_matrix: np.ndarray | list[list[float]],
    time_to_expiration: float,
    dividend_yields: np.ndarray | list[float] | None = None,
    put_option: bool = False,
    basket_type: str = "weighted",
    simulations: int = 10_000,
    time_steps: int = 50,
    variance_reduction: str | None = None,
    sequence: str = "pseudo_random",
    seed: int | None = 0,
) -> tuple[float, float]:
    """
    Calculate the price of an American option on a basket of correlated stocks by least-squares
    Monte Carlo (Longstaff and Schwartz, 2001): what the option is worth when it may be
    exercised at every time step (see get_longstaff_schwartz_option for the method).

    The basket is the weighted sum B = Σ w_i * S_i or the best of the weighted prices
    B = max(w_i * S_i), and the option pays
    φ * (B - K) when exercised. The correlated paths come from simulate_gbm_paths, and the
    continuation value is regressed on a constant, x_i, x_i * x_j (i ≤ j) and the exercise
    value, with x_i = w_i * S_i / K.

    The formulas are as follows:

    - C(t) = β_0 + Σ β_i * x_i + Σ β_ij * x_i * x_j + β_P * max(φ * (B - K), 0) / K

    Referenced against:
    - QuantLib, MCAmericanBasketEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_m_c_american_basket_engine.html

    Args:
        stock_prices (np.ndarray | list[float]): The current price of each stock.
        weights (np.ndarray | list[float]): The units of each stock in the basket.
        strike_price (float): The strike price.
        risk_free_rate (float): The continuously compounded risk-free rate.
        volatilities (np.ndarray | list[float]): The volatility of each stock.
        correlation_matrix (np.ndarray | list[list[float]]): The correlations of the
            returns, positive definite with ones on the diagonal.
        time_to_expiration (float): The time to expiration in years.
        dividend_yields (np.ndarray | list[float] | None, optional): The continuous dividend
            yield of each stock. Defaults to None, which uses 0.
        put_option (bool, optional): Whether the option is a put. Defaults to False.
        basket_type (str, optional): The basket. Options are: "weighted" (the weighted sum),
            "maximum" (the best of the weighted prices). Defaults to "weighted".
        simulations (int, optional): The number of paths. Defaults to 10,000.
        time_steps (int, optional): The number of exercise dates. Defaults to 50.
        variance_reduction (str | None, optional): The variance reduction. Options are: None
            (none), "antithetic" (antithetic variates). Defaults to None.
        sequence (str, optional): The random numbers. Options are: "pseudo_random"
            (pseudo-random numbers), "sobol" (scrambled Sobol points). Defaults to
            "pseudo_random".
        seed (int | None, optional): The seed of the random numbers. Defaults to 0.

    Returns:
        tuple[float, float]: The price and its standard error.

    Raises:
        ValueError: If the basket type, variance reduction or sequence is unknown, the
            inputs do not have one value per stock or the paths are too large to hold.

    Notes:
        References: Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American Options by
        Simulation: A Simple Least-Squares Approach." The Review of Financial Studies,
        14(1), 113-147. Broadie, M., & Glasserman, P. (1997). "Pricing American-Style
        Securities Using Simulation." Journal of Economic Dynamics and Control, 21(8-9),
        1323-1352. Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering."
        Springer, section 8.6.

    As an example:

    ```python
    from financetoolkit.options.numerical import monte_carlo_model

    price, standard_error = monte_carlo_model.get_longstaff_schwartz_basket_option(
        stock_prices=[100, 100],
        weights=[1, 1],
        strike_price=100,
        risk_free_rate=0.05,
        volatilities=[0.2, 0.2],
        correlation_matrix=[[1, 0], [0, 1]],
        time_to_expiration=1,
        dividend_yields=[0.1, 0.1],
        basket_type="maximum",
        simulations=50_000,
    )

    round(price, 2), round(standard_error, 3)
    ```

    Which returns:

    ```
    (10.3, 0.047)
    ```
    """
    validate_numeric_inputs(
        strike_price=strike_price, time_to_expiration=time_to_expiration
    )
    # The regression is on the basket over the strike price.
    validate_positive_inputs(strike_price=strike_price)
    check_option("variance_reduction", variance_reduction, (None, "antithetic"))
    check_option("basket_type", basket_type, ("weighted", "maximum"))
    prices = np.asarray(stock_prices, dtype=float)
    units = np.broadcast_to(np.asarray(weights, dtype=float), prices.shape)
    sign = -1 if put_option else 1

    def basket(values: np.ndarray) -> np.ndarray:
        weighted = values * units
        return weighted.sum(axis=-1) if basket_type == "weighted" else weighted.max(-1)

    if time_to_expiration == 0:
        return max(sign * (float(basket(prices)) - strike_price), 0.0), 0.0

    antithetic = variance_reduction == "antithetic"
    paths = simulate_gbm_paths(
        prices,
        risk_free_rate,
        volatilities,
        time_to_expiration,
        0.0 if dividend_yields is None else dividend_yields,
        correlation_matrix,
        time_steps=time_steps,
        simulations=simulations,
        sequence=sequence,
        antithetic=antithetic,
        seed=seed,
    )
    payoffs = np.maximum(sign * (basket(paths[:, 1:]) - strike_price), 0.0)
    pairs = np.triu_indices(prices.size)

    def regressors(date: int, rows: np.ndarray) -> np.ndarray:
        ratios = paths[rows, date + 1] * units / strike_price
        return np.column_stack(
            (
                np.ones(rows.size),
                ratios,
                ratios[:, pairs[0]] * ratios[:, pairs[1]],
                payoffs[rows, date] / strike_price,
            )
        )

    values = _longstaff_schwartz(
        payoffs, regressors, np.exp(-risk_free_rate * time_to_expiration / time_steps)
    )
    price, standard_error = monte_carlo_estimate(
        values, simulation_groups(simulations, sequence, antithetic)
    )
    intrinsic = max(sign * (float(basket(prices)) - strike_price), 0.0)

    return (price, standard_error) if price >= intrinsic else (intrinsic, 0.0)
