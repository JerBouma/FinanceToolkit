"""LIBOR Market Model

The LIBOR (forward) market model of Brace, Gatarek and Musiela (1997), Jamshidian (1997) and
Miltersen, Sandmann and Sondermann (1997) models the simple forward rates of a tenor structure
T_0 < T_1 < ... < T_n directly: F_i(t) is the rate of the period from T_i to T_(i+1), with accrual
τ_i = T_(i+1) - T_i, and it is fixed at its reset time T_i. Each forward rate is lognormal,

- dF_i(t) = μ_i(t) F_i(t) dt + σ_i F_i(t) dW_i(t), dW_i dW_j = ρ_ij dt

so a caplet on F_i is priced by Black's formula, the reason the model is the market's standard.
The drift μ_i depends on the measure. These functions work under the spot LIBOR measure, whose
numeraire is a deposit rolled over at the forward rate of every period as it fixes; there the
forwards that have not yet fixed carry the drift

- μ_i(t) = σ_i SUM_(j = q(t))^i (ρ_ij τ_j σ_j F_j(t) / (1 + τ_j F_j(t)))

with q(t) the first forward that has not fixed at t (Brigo and Mercurio, 2006, section 6.3;
Glasserman, 2003, section 3.7).

Every function takes the zero-coupon curve as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between (see
`financetoolkit.fixedincome.helpers.discount_factors`). The forward rates are simple rates over
each accrual, read from that curve as F_i(0) = (P(0, T_i) / P(0, T_(i+1)) - 1) / τ_i. Times are
year fractions from today, without a day count.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option


def _check_tenor(
    times: np.ndarray | list, volatilities: np.ndarray | list | float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check the tenor structure T_0 < ... < T_n (T_0 ≥ 0) and give one volatility per forward.
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or times.size <= 1
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "The times must be at least two increasing dates T_0 < T_1 < ..., T_0 not before today."
        )
    volatilities = np.broadcast_to(
        np.asarray(volatilities, dtype=float), (len(times) - 1,)
    ).copy()
    if np.any(volatilities < 0):
        raise ValueError("The volatilities cannot be negative.")
    return times, volatilities


def get_lmm_factor_loadings(
    times: np.ndarray | list,
    correlation: np.ndarray | None = None,
    correlation_decay: float | None = None,
    factors: int | None = None,
) -> np.ndarray:
    """
    Calculate the factor loadings B of the forward rates of a LIBOR market model: the matrix
    whose rows give each forward's exposure to independent Brownian motions, so that dW_i =
    SUM(B_ik dZ_k) and the correlation of the forwards is B Bᵀ.

    With fewer factors than forwards the correlation is reduced in rank, with Rebonato's
    (2002) method: keep the largest eigenvalues of the correlation matrix and scale every
    row back to unit length, so each forward keeps its volatility.

    The formula is as follows:

    - ρ_ij = e^(-β |T_i - T_j|), with the correlation decay β
    - ρ = V Λ Vᵀ, B̃ = V_k Λ_k^(1/2) with the k largest eigenvalues
    - B_ik = B̃_ik / √(SUM_l B̃_il²)

    One factor gives perfectly correlated forwards (all loadings 1), which is the default.

    For more information, see the following sources:

    - Rebonato, R. (2002). "Modern Pricing of Interest-Rate Derivatives: The LIBOR Market
      Model and Beyond." Princeton University Press, chapter 7.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 6.9 and 6.10.

    Args:
        times (np.ndarray | list): The tenor dates T_0 < T_1 < ... < T_n in years.
        correlation (np.ndarray | None, optional): The correlation matrix of the n forward
            rates, or None for correlation_decay or, without it, perfectly correlated
            forwards. Defaults to None.
        correlation_decay (float | None, optional): The decay β of the correlation
            ρ_ij = e^(-β |T_i - T_j|) on the reset times, instead of a matrix: 0 gives
            perfectly correlated forwards and a larger β less correlation (β = 1 gives
            e^(-1) = 0.37 to forwards a year apart). Defaults to None.
        factors (int | None, optional): The number of factors; None keeps the full rank of
            the correlation. Defaults to None.

    Returns:
        np.ndarray: The loadings, forwards by factors (fewer factors when the correlation
        matrix has a lower rank).

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import lmm_model

    lmm_model.get_lmm_factor_loadings(
        times=[1, 2, 3, 4, 5], correlation_decay=0.2, factors=2
    ).round(4)
    ```

    Which returns:

    ```
    array([[ 0.8774,  0.4797],
           [ 0.9765,  0.2157],
           [ 0.9765, -0.2157],
           [ 0.8774, -0.4797]])
    ```
    """
    times = np.asarray(times, dtype=float)
    resets = times[:-1]
    count = len(resets)
    if correlation is not None and np.ndim(correlation) == 0:
        raise ValueError(
            "correlation takes a correlation matrix of the forwards; pass the decay β of "
            "ρ_ij = e^(-β |T_i - T_j|) as correlation_decay instead."
        )
    if correlation is not None and correlation_decay is not None:
        raise ValueError(
            "Pass either a correlation matrix or a correlation_decay, not both."
        )
    if correlation_decay is not None:
        if correlation_decay < 0:
            raise ValueError("The correlation decay β cannot be negative.")
        matrix = np.exp(
            -float(correlation_decay) * np.abs(resets[:, None] - resets[None, :])
        )
    elif correlation is None:
        matrix = np.ones((count, count))
    else:
        matrix = np.asarray(correlation, dtype=float)
        if matrix.shape != (count, count) or not np.allclose(matrix, matrix.T):
            raise ValueError(
                f"The correlation must be a symmetric {count} by {count} matrix, one row per forward."
            )
        if not np.allclose(np.diag(matrix), 1):
            raise ValueError("The correlation matrix must have ones on its diagonal.")
    if factors is not None and not 1 <= factors <= count:
        raise ValueError(f"The number of factors must lie between 1 and {count}.")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1][: factors or count]
    # Eigenvalues at rounding level are zero: their eigenvectors are arbitrary and would
    # carry 1e-8 loadings that differ between platforms.
    order = order[eigenvalues[order] > 1e-12 * count]
    if len(order) == 0:
        raise ValueError("The correlation matrix has no positive eigenvalue.")
    loadings = eigenvectors[:, order] * np.sqrt(eigenvalues[order])
    # The eigenvectors' signs are arbitrary; make the first clearly nonzero loading of
    # every factor positive, so the paths are the same on every platform.
    first = np.argmax(np.abs(loadings) > 1e-6 * np.abs(loadings).max(axis=0), axis=0)
    loadings = loadings * np.sign(loadings[first, np.arange(len(order))])
    return loadings / np.linalg.norm(loadings, axis=1, keepdims=True)


def _evolve(
    forward_rates: np.ndarray,
    times: np.ndarray,
    volatilities: np.ndarray,
    loadings: np.ndarray,
    simulations: int,
    seed: int | None,
    steps_per_year: int,
    last_reset: int,
) -> np.ndarray:
    """
    Simulate the forwards under the spot LIBOR measure up to the reset time T_last_reset with
    the log-Euler scheme, and return them at every reset time up to it (paths by resets by
    forwards, NaN for the forwards that fixed before).
    """
    count = len(forward_rates)
    accruals = np.diff(times)
    correlation = loadings @ loadings.T
    generator = np.random.default_rng(seed)

    log_forwards = np.tile(np.log(forward_rates), (simulations, 1))
    result = np.full((simulations, last_reset + 1, count), np.nan)
    clock = 0.0
    for reset in range(last_reset + 1):
        length = times[reset] - clock
        steps = (
            max(1, int(np.ceil(length * steps_per_year - 1e-9))) if length > 0 else 0
        )
        for _ in range(steps):
            step = length / steps
            # The first forward that has not fixed: the numeraire deposit runs to its end.
            first = reset
            alive = slice(first, count)
            forwards = np.exp(log_forwards[:, alive])
            weights = (
                accruals[alive]
                * volatilities[alive]
                * forwards
                / (1 + accruals[alive] * forwards)
            )
            # μ_i = σ_i SUM_(j = first)^i ρ_ij w_j, the lower triangle of ρ applied to w.
            drift = volatilities[alive] * (
                weights @ np.tril(correlation[alive, alive]).T
            )
            shocks = (
                generator.standard_normal((simulations, loadings.shape[1]))
                @ loadings[alive].T
            )
            log_forwards[:, alive] += (
                drift - 0.5 * volatilities[alive] ** 2
            ) * step + volatilities[alive] * np.sqrt(step) * shocks
        clock = times[reset]
        result[:, reset, reset:] = np.exp(log_forwards[:, reset:])
    return result


def simulate_lmm_forwards(
    forward_rates: np.ndarray | list,
    times: np.ndarray | list,
    volatilities: np.ndarray | list | float,
    correlation: np.ndarray | None = None,
    correlation_decay: float | None = None,
    factors: int | None = None,
    simulations: int = 10000,
    seed: int | None = 0,
    steps_per_year: int = 12,
) -> np.ndarray:
    """
    Simulate the forward rates of a LIBOR market model under the spot LIBOR measure, from
    today to the last reset time: the paths along which the forward rates of the curve may
    move together.

    Each forward F_i has a constant volatility σ_i and the forwards are correlated through
    factor loadings (`get_lmm_factor_loadings`): one factor makes them move together, more
    factors let the curve twist.

    The scheme is log-Euler with the drift frozen at the start of every step (Glasserman,
    2003, section 3.7): the logarithm of a forward has a constant volatility, so only the
    drift is discretised, and the forwards stay positive.

    The formula is as follows:

    - ln F_i(t + Δ) = ln F_i(t) + (μ_i(t) - σ_i² / 2) Δ + σ_i √Δ SUM_k B_ik Z_k
    - μ_i(t) = σ_i SUM_(j = q(t))^i (ρ_ij τ_j σ_j F_j(t) / (1 + τ_j F_j(t)))

    where Δ is the step, Z_k independent standard normal draws, B the factor loadings, ρ = B Bᵀ
    the correlation they give, τ_j the accrual of forward j and q(t) the first forward not
    yet fixed at t. A forward stops moving at its reset time T_i, when it fixes. The steps
    fall on the reset times, every interval between them split into equal steps of at most
    1 / steps_per_year years.

    The forward rates are simple rates over their accrual; the volatilities are lognormal.

    For more information, see the following sources:

    - Brace, A., Gatarek, D. and Musiela, M. (1997). "The Market Model of Interest Rate
      Dynamics." Mathematical Finance, 7(2), 127-155.
    - Jamshidian, F. (1997). "LIBOR and Swap Market Models and Measures." Finance and
      Stochastics, 1(4), 293-330.
    - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer,
      section 3.7.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 6.3 and 6.10.

    FinancePy's LMM simulations step once per period; the paths match in distribution rather
    than draw by draw.

    Shares similar logic:
    - FinancePy, lmm_simulate_fwds_1f: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L430
    - FinancePy, lmm_simulate_fwds_mf: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L547
    - FinancePy, lmm_simulate_fwds_nf: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L320

    Args:
        forward_rates (np.ndarray | list): The forward rates F_i(0) of the n periods (simple
            rates, decimals).
        times (np.ndarray | list): The tenor dates T_0 < T_1 < ... < T_n in years, T_0 not
            before today; forward i runs from T_i to T_(i+1).
        volatilities (np.ndarray | list | float): The lognormal volatility σ_i of each
            forward, or one for all.
        correlation (np.ndarray | None, optional): The correlation matrix of the forwards,
            or None for correlation_decay or, without it, one factor. Defaults to None.
        correlation_decay (float | None, optional): The decay β of the correlation
            ρ_ij = e^(-β |T_i - T_j|) instead of a matrix, see `get_lmm_factor_loadings`.
            Defaults to None.
        factors (int | None, optional): The number of factors; None keeps the full rank of
            the correlation. Defaults to None.
        simulations (int, optional): The number of paths. Defaults to 10,000.
        seed (int | None, optional): The random seed. Defaults to 0.
        steps_per_year (int, optional): The fewest steps per year between reset times.
            Defaults to 12.

    Returns:
        np.ndarray: The forward rates at the reset times, paths by reset times T_0, ...,
        T_(n-1) by forwards; element [p, k, i] is F_i(T_k) on path p, NaN for the forwards
        that fixed before T_k, so the diagonal [p, i, i] holds the fixings.

    As an example:

    ```python
    from financetoolkit.fixedincome.short_rates import lmm_model

    forwards = lmm_model.simulate_lmm_forwards(
        forward_rates=[0.03, 0.032, 0.034, 0.035], times=[1, 2, 3, 4, 5],
        volatilities=0.2, correlation_decay=0.1, factors=2
    )

    forwards[:, -1, -1].mean()
    ```

    Which returns:

    ```
    0.035617315620775095
    ```
    """
    forward_rates = np.asarray(forward_rates, dtype=float)
    times, volatilities = _check_tenor(times, volatilities)
    if forward_rates.shape != (len(times) - 1,) or np.any(forward_rates <= 0):
        raise ValueError(
            "Pass one positive forward rate per period, one fewer than the times."
        )
    if simulations < 1 or steps_per_year < 1:
        raise ValueError("The simulations and the steps per year must be at least 1.")
    loadings = get_lmm_factor_loadings(times, correlation, correlation_decay, factors)
    return _evolve(
        forward_rates,
        times,
        volatilities,
        loadings,
        simulations,
        seed,
        steps_per_year,
        len(forward_rates) - 1,
    )


def _curve_forwards(
    zero_curve: pd.Series, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The discount factors at the tenor dates and the simple forward rates between them."""
    discounts = discount_factors(zero_curve, times)
    return discounts, (discounts[:-1] / discounts[1:] - 1) / np.diff(times)


def get_lmm_cap(
    zero_curve: pd.Series,
    times: np.ndarray | list,
    volatilities: np.ndarray | list | float,
    strike: float,
    correlation: np.ndarray | None = None,
    correlation_decay: float | None = None,
    factors: int | None = None,
    option_type: str = "caplet",
    simulations: int = 10000,
    seed: int | None = 0,
    steps_per_year: int = 12,
    notional: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate the Monte Carlo prices of the caplets or floorlets of a cap or floor in the
    LIBOR market model: what the protection against a rising or falling forward rate is
    worth when the forwards follow the model.

    The caplet on F_i pays τ_i max(F_i(T_i) - K, 0) at T_(i+1); under the spot LIBOR measure
    it is discounted with the rolled-over deposit, the product of the growth of every period
    up to its payment. A cap is the sum of its caplets. Each caplet depends on its own
    forward only, so its Monte Carlo price converges to Black's formula, P(0, T_(i+1)) τ_i
    Black(F_i(0), K, σ_i, T_i), which makes it a check on the simulation; the model earns
    its keep on products that depend on several forwards (swaptions, `get_lmm_swaption`).

    The formula is as follows:

    - Caplet_i = Notional · P(0, T_0) · E[τ_i max(F_i(T_i) - K, 0) / PROD_(k=0)^i (1 + τ_k F_k(T_k))]
    - Floorlet_i = Notional · P(0, T_0) · E[τ_i max(K - F_i(T_i), 0) / PROD_(k=0)^i (1 + τ_k F_k(T_k))]

    with the forwards simulated by `simulate_lmm_forwards`, P(0, T_0) the discount factor to
    the first tenor date and the expectation the average over the paths. The standard error
    is the sample standard deviation (ddof=1) of the discounted payoffs over √paths.

    The strike is a simple rate over each accrual; the zero curve is annually compounded.

    For more information, see the following sources:

    - Brace, A., Gatarek, D. and Musiela, M. (1997). "The Market Model of Interest Rate
      Dynamics." Mathematical Finance, 7(2), 127-155.
    - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering." Springer,
      section 3.7.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 6.4 and 6.10.

    The prices converge to Black's caplet formula.

    Shares similar logic:
    - FinancePy, lmm_cap_flr_pricer: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L664

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        times (np.ndarray | list): The tenor dates T_0 < T_1 < ... < T_n in years; caplet i
            resets at T_i and pays at T_(i+1). A first tenor date of 0 gives a first caplet
            that is already fixed.
        volatilities (np.ndarray | list | float): The lognormal volatility σ_i of each
            forward (its caplet volatility), or one for all.
        strike (float): The cap or floor rate K (decimal, simple interest).
        correlation (np.ndarray | None, optional): The correlation matrix of the forwards,
            or None for correlation_decay or, without it, one factor. Defaults to None.
        correlation_decay (float | None, optional): The decay β of the correlation
            ρ_ij = e^(-β |T_i - T_j|) instead of a matrix, see `get_lmm_factor_loadings`.
            Defaults to None.
        factors (int | None, optional): The number of factors; None keeps the full rank of
            the correlation. Defaults to None.
        option_type (str, optional): The option. Options are: "caplet" (caplets of a cap),
            "floorlet" (floorlets of a floor). Defaults to "caplet".
        simulations (int, optional): The number of paths. Defaults to 10,000.
        seed (int | None, optional): The random seed. Defaults to 0.
        steps_per_year (int, optional): The fewest steps per year between reset times.
            Defaults to 12.
        notional (float, optional): The notional. Defaults to 1.

    Returns:
        pd.DataFrame: The Price and Standard Error of every caplet or floorlet, indexed by
        its reset time, in the currency of the notional.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import lmm_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    lmm_model.get_lmm_cap(
        zero_curve=zero_curve, times=[1, 2, 3, 4, 5], volatilities=0.2, strike=0.04,
        correlation_decay=0.1, factors=2, notional=1_000_000
    )
    ```

    Which returns:

    ```
                      Price  Standard Error
    Reset Time
    1.0          532.669067       19.037652
    2.0         2022.286709       47.763960
    3.0         3693.751552       72.385872
    4.0         5415.756767       96.762061
    ```
    """
    check_option("option_type", option_type, ("caplet", "floorlet"))
    times, volatilities = _check_tenor(times, volatilities)
    discounts, forward_rates = _curve_forwards(zero_curve, times)
    forwards = simulate_lmm_forwards(
        forward_rates,
        times,
        volatilities,
        correlation,
        correlation_decay,
        factors,
        simulations,
        seed,
        steps_per_year,
    )
    accruals = np.diff(times)
    fixings = np.diagonal(forwards, axis1=1, axis2=2)
    deflators = discounts[0] / np.cumprod(1 + accruals * fixings, axis=1)
    sign = 1.0 if option_type == "caplet" else -1.0
    payoffs = (
        notional * accruals * np.maximum(sign * (fixings - strike), 0.0) * deflators
    )

    return pd.DataFrame(
        {
            "Price": payoffs.mean(axis=0),
            "Standard Error": payoffs.std(axis=0, ddof=1) / np.sqrt(simulations),
        },
        index=pd.Index(times[:-1], name="Reset Time"),
    )


def get_lmm_swaption(
    zero_curve: pd.Series,
    times: np.ndarray | list,
    volatilities: np.ndarray | list | float,
    fixed_rate: float,
    correlation: np.ndarray | None = None,
    correlation_decay: float | None = None,
    factors: int | None = None,
    swaption_type: str = "payer",
    simulations: int = 10000,
    seed: int | None = 0,
    steps_per_year: int = 12,
    notional: float = 1.0,
) -> pd.Series:
    """
    Calculate the Monte Carlo price of a European swaption in the LIBOR market model: what
    the right to enter a swap at the fixed rate is worth when the forwards of the curve move
    together as the model has them.

    The swaption expires at the first tenor date T_0 into a swap that pays the fixed rate K
    against the forward rates on the dates T_1, ..., T_n. At expiry the swap is worth the
    annuity times the gap between the swap rate and K, both functions of all forwards at
    T_0, so the price depends on how the forwards move together: more factors and a lower
    correlation give a lower swap rate volatility and a cheaper swaption. Until T_0 the spot
    LIBOR measure is the T_0-forward measure, whose numeraire is the bond maturing at T_0.

    The formula is as follows:

    - Payer Swaption = Notional · P(0, T_0) · E[A(T_0) max(S(T_0) - K, 0)]
    - Receiver Swaption = Notional · P(0, T_0) · E[A(T_0) max(K - S(T_0), 0)]
    - A(T_0) = SUM_(i=0)^(n-1) τ_i P(T_0, T_(i+1)), P(T_0, T_(i+1)) = PROD_(k=0)^i 1 / (1 + τ_k F_k(T_0))
    - S(T_0) = (1 - P(T_0, T_n)) / A(T_0)

    with the forwards simulated by `simulate_lmm_forwards` and the expectation the average
    over the paths. The standard error is the sample standard deviation (ddof=1) of the
    discounted payoffs over √paths. Rebonato's approximation
    (`get_rebonato_swaption_volatility`) gives the matching Black volatility in closed form.

    The fixed rate is a simple rate over each accrual; the zero curve is annually compounded.

    For more information, see the following sources:

    - Brace, A., Gatarek, D. and Musiela, M. (1997). "The Market Model of Interest Rate
      Dynamics." Mathematical Finance, 7(2), 127-155.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 6.7.

    Referenced against:
    - FinancePy, lmm_swaption_pricer: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L809

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        times (np.ndarray | list): The expiry T_0 > 0 and the payment dates T_1 < ... < T_n
            of the swap in years.
        volatilities (np.ndarray | list | float): The lognormal volatility σ_i of each
            forward, or one for all.
        fixed_rate (float): The fixed rate K of the swap (decimal, simple interest).
        correlation (np.ndarray | None, optional): The correlation matrix of the forwards,
            or None for correlation_decay or, without it, one factor. Defaults to None.
        correlation_decay (float | None, optional): The decay β of the correlation
            ρ_ij = e^(-β |T_i - T_j|) instead of a matrix, see `get_lmm_factor_loadings`.
            Defaults to None.
        factors (int | None, optional): The number of factors; None keeps the full rank of
            the correlation. Defaults to None.
        swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay
            the fixed rate), "receiver" (the right to receive it). Defaults to "payer".
        simulations (int, optional): The number of paths. Defaults to 10,000.
        seed (int | None, optional): The random seed. Defaults to 0.
        steps_per_year (int, optional): The fewest steps per year until the expiry.
            Defaults to 12.
        notional (float, optional): The notional. Defaults to 1.

    Returns:
        pd.Series: The Price and Standard Error, in the currency of the notional.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import lmm_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    lmm_model.get_lmm_swaption(
        zero_curve=zero_curve, times=[2, 3, 4, 5, 6, 7], volatilities=0.2,
        fixed_rate=0.04, correlation_decay=0.1, factors=2, notional=1_000_000
    )
    ```

    Which returns:

    ```
    Price             16368.598779
    Standard Error      283.622494
    ```
    """
    check_option("swaption_type", swaption_type, ("payer", "receiver"))
    times, volatilities = _check_tenor(times, volatilities)
    if times[0] <= 0:
        raise ValueError("The swaption must expire after today.")
    if simulations <= 1 or steps_per_year < 1:
        raise ValueError(
            "The simulations must be at least 2 and the steps per year at least 1."
        )
    discounts, forward_rates = _curve_forwards(zero_curve, times)
    loadings = get_lmm_factor_loadings(times, correlation, correlation_decay, factors)
    forwards = _evolve(
        forward_rates,
        times,
        volatilities,
        loadings,
        simulations,
        seed,
        steps_per_year,
        0,
    )[:, 0, :]
    accruals = np.diff(times)
    bonds = 1 / np.cumprod(1 + accruals * forwards, axis=1)
    annuity = np.sum(accruals * bonds, axis=1)
    swap_rate = (1 - bonds[:, -1]) / annuity
    sign = 1.0 if swaption_type == "payer" else -1.0
    payoffs = (
        notional
        * discounts[0]
        * annuity
        * np.maximum(sign * (swap_rate - fixed_rate), 0.0)
    )

    return pd.Series(
        {
            "Price": payoffs.mean(),
            "Standard Error": payoffs.std(ddof=1) / np.sqrt(simulations),
        }
    )


def get_rebonato_swaption_volatility(
    zero_curve: pd.Series,
    times: np.ndarray | list,
    volatilities: np.ndarray | list | float,
    correlation: np.ndarray | None = None,
    correlation_decay: float | None = None,
    factors: int | None = None,
) -> float:
    """
    Calculate Rebonato's approximation of the Black volatility of a European swaption in the
    LIBOR market model: the swaption volatility that the model's caplet volatilities and
    correlation imply, without simulation.

    The swap rate is a weighted sum of the forwards, S = SUM(w_i F_i), with weights that
    move little compared with the forwards; freezing them at today's values makes the swap
    rate's variance a sum over the forwards' covariances. This is how the model is
    calibrated to both markets at once.

    The formula is as follows:

    - σ_swaption² T_0 = SUM_i SUM_j (w_i w_j F_i F_j ρ_ij ∫_0^T_0 σ_i σ_j dt) / S²
    - w_i = τ_i P(0, T_(i+1)) / SUM_k (τ_k P(0, T_(k+1))), S = SUM_i w_i F_i

    with the forward rates F_i(0), their accruals τ_i, their volatilities σ_i (constant, so
    the integral is σ_i σ_j T_0), the correlation ρ_ij (after the rank reduction of
    `get_lmm_factor_loadings` when fewer factors are given) and the forward swap rate S. The
    sums run over the forwards of the swap, which starts at the expiry T_0 and pays on
    T_1, ..., T_n. The Black price of the swaption is then
    A(0) Black(S, K, σ_swaption, T_0), with A(0) = SUM_i τ_i P(0, T_(i+1)).

    The forward and swap rates are simple rates over each accrual; the zero curve is
    annually compounded.

    For more information, see the following sources:

    - Rebonato, R. (1998). "Interest-Rate Option Models." Wiley, 2nd edition.
    - Rebonato, R. (2002). "Modern Pricing of Interest-Rate Derivatives: The LIBOR Market
      Model and Beyond." Princeton University Press, chapter 8.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 6.15, equation 6.67.

    Shares similar logic:
    - FinancePy, lmm_swaption_vol_approx: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/lmm_mc.py#L64

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        times (np.ndarray | list): The expiry T_0 > 0 and the payment dates T_1 < ... < T_n
            of the swap in years.
        volatilities (np.ndarray | list | float): The lognormal volatility σ_i of each
            forward, or one for all.
        correlation (np.ndarray | None, optional): The correlation matrix of the forwards,
            or None for correlation_decay or, without it, one factor. Defaults to None.
        correlation_decay (float | None, optional): The decay β of the correlation
            ρ_ij = e^(-β |T_i - T_j|) instead of a matrix, see `get_lmm_factor_loadings`.
            Defaults to None.
        factors (int | None, optional): The number of factors; None keeps the full rank of
            the correlation. Defaults to None.

    Returns:
        float: The lognormal (Black) volatility of the swaption.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import lmm_model

    zero_curve = pd.Series([0.03, 0.035, 0.04], index=[1, 5, 10])

    lmm_model.get_rebonato_swaption_volatility(
        zero_curve=zero_curve, times=[2, 3, 4, 5, 6, 7], volatilities=0.2,
        correlation_decay=0.1, factors=2
    )
    ```

    Which returns:

    ```
    0.1905820309198477
    ```
    """
    times, volatilities = _check_tenor(times, volatilities)
    if times[0] <= 0:
        raise ValueError("The swaption must expire after today.")
    discounts, forward_rates = _curve_forwards(zero_curve, times)
    loadings = get_lmm_factor_loadings(times, correlation, correlation_decay, factors)
    weights = np.diff(times) * discounts[1:]
    weights = weights / weights.sum()
    swap_rate = float(weights @ forward_rates)
    scaled = weights * forward_rates * volatilities
    variance = float(scaled @ (loadings @ loadings.T) @ scaled)
    return float(np.sqrt(variance) / swap_rate)
