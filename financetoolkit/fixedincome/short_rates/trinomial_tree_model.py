"""Short-Rate Tree Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from financetoolkit.fixedincome.bonds.bond_model import get_cash_flow_periods
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option

# Ho-Lee and Hull-White model the short rate itself (r = α + x, normal), Black-Karasinski and
# Black-Derman-Toy its logarithm (r = e^(α + x), lognormal). Ho-Lee and Black-Derman-Toy (with
# a constant volatility) are the two without mean reversion.
TREE_MODELS = ("hull_white", "black_karasinski", "black_derman_toy", "ho_lee")
LOGNORMAL_MODELS = ("black_karasinski", "black_derman_toy")
MEAN_REVERTING_MODELS = ("hull_white", "black_karasinski")
# The most steps a tree may have: the nodes per step stay bounded with mean reversion, but grow
# with every step without it, so the work grows with the square of the steps. 5,000 steps (50
# years of 100 steps) price in about a second.
MAX_TREE_STEPS = 5_000
# Times closer than this (in years) are the same date of the tree.
TIME_TOLERANCE = 1e-9
# The shift α is solved to this absolute step, near machine precision for log rates.
SHIFT_TOLERANCE = 1e-14


def _check_tree_inputs(model: str, speed: float, volatility: float) -> float:
    """
    Check the model and its parameters and return the speed the tree uses: zero for the
    models without mean reversion.
    """
    check_option("model", model, TREE_MODELS)
    if not volatility > 0:
        raise ValueError("The volatility must be positive.")
    if model in MEAN_REVERTING_MODELS:
        if speed < 0:
            raise ValueError("The speed of mean reversion cannot be negative.")
        return speed
    return 0.0


def _time_grid(event_times: np.ndarray, steps_per_year: int) -> np.ndarray:
    """
    The times of the tree: today, every event (payment or exercise date) and, between two
    events, equal steps of at most 1 / steps_per_year years, so that every event falls on a
    step of the tree exactly.
    """
    if steps_per_year < 1:
        raise ValueError("The steps per year must be at least 1.")
    events = np.sort(np.asarray(event_times, dtype=float))
    events = events[events > TIME_TOLERANCE]
    events = events[np.diff(events, prepend=0.0) > TIME_TOLERANCE]
    if events.size == 0:
        raise ValueError("The tree needs at least one date after today.")
    bounds = np.concatenate([[0.0], events])
    steps = np.maximum(
        1, np.ceil(np.diff(bounds) * steps_per_year - TIME_TOLERANCE).astype(int)
    )
    if steps.sum() > MAX_TREE_STEPS:
        raise ValueError(
            f"A tree of {steps.sum():,} steps is too large; keep the years times the steps "
            f"per year under {MAX_TREE_STEPS:,}."
        )
    return np.concatenate(
        [[0.0]]
        + [
            np.linspace(start, end, count + 1)[1:]
            for start, end, count in zip(bounds[:-1], bounds[1:], steps, strict=True)
        ]
    )


def _steps_at(times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """The steps of the tree at which the events fall."""
    return np.searchsorted(times, np.asarray(event_times, dtype=float) - TIME_TOLERANCE)


def _fit_lognormal_shift(
    arrow: np.ndarray, states: np.ndarray, dt: float, discount: float
) -> float:
    """
    The shift α of a lognormal step: the root of SUM(Q_j e^(-e^(α + x_j) Δt)) = P(0, t + Δt),
    which falls from SUM(Q_j) = P(0, t) to zero as α rises, so it has one root when the
    forward rate of the step is positive.
    """
    total = arrow.sum()
    if not 0 < discount < total:
        raise ValueError(
            "A lognormal short-rate model needs a positive forward rate at every step of the "
            "tree; the curve has a zero or negative one. Use 'hull_white' or 'ho_lee'."
        )
    forward = np.log(total / discount) / dt
    # Newton's method from the shift that sets the average rate (weighted by the Arrow-Debreu
    # prices) to the forward rate; the error is smooth and monotone, so it converges in a few
    # steps. Brent's method on a widening bracket is the fallback.
    alpha = np.log(forward) - np.log(np.sum(arrow * np.exp(states)) / total)
    for _ in range(50):
        rates = np.exp(alpha + states) * dt
        values = arrow * np.exp(-rates)
        step = (values.sum() - discount) / -np.sum(values * rates)
        if not np.isfinite(step):
            break
        alpha -= step
        if abs(step) < SHIFT_TOLERANCE:
            return float(alpha)

    def error(alpha: float) -> float:
        return float(np.sum(arrow * np.exp(-np.exp(alpha + states) * dt)) - discount)

    guess, width = np.log(forward), 0.5
    while error(guess - width) < 0 or error(guess + width) > 0:
        width *= 2
    return brentq(error, guess - width, guess + width, xtol=1e-15, rtol=1e-15)


def build_short_rate_tree(
    zero_curve: pd.Series,
    event_times: float | list | np.ndarray,
    speed: float,
    volatility: float,
    model: str = "hull_white",
    steps_per_year: int = 50,
    curve_shift: float = 0.0,
) -> dict:
    """
    Build a recombining trinomial tree of the short rate fitted exactly to today's zero
    curve with the two-stage procedure of Hull and White (1994): the rates the short rate
    can take at every date, with their probabilities, on which bonds and options with early
    exercise are priced.

    The first stage builds a tree for a process x with dx = -a x dt + σ dW and x(0) = 0; the
    second shifts the nodes of every step by the α_i that makes the tree price the
    zero-coupon bond maturing at the end of that step exactly, found by forward induction on
    the Arrow-Debreu prices Q (the value today of 1 paid at a node). The steps may differ in
    length, as in Hull and White (1996), so every payment and exercise date is a date of the
    tree.

    The formula is as follows:

    - Stage one, from step i to i + 1 (length Δt): E = x e^(-a Δt),
      V = σ² (1 - e^(-2a Δt)) / (2a) (σ² Δt when a = 0), Δx_{i+1} = √(3V)
    - Node j branches to k - 1, k and k + 1 with k = round(E / Δx_{i+1}), η = E - k Δx_{i+1}:
      p_up = (V + η²) / (2Δx²) + η / (2Δx), p_mid = 1 - (V + η²) / Δx²,
      p_down = (V + η²) / (2Δx²) - η / (2Δx)
    - Stage two: r_{i,j} = α_i + x_j (Hull-White, Ho-Lee) or r_{i,j} = e^(α_i + x_j)
      (Black-Karasinski, Black-Derman-Toy), with α_i solving
      SUM(Q_{i,j} e^(-r_{i,j} Δt)) = P(0, t_{i+1})
    - Normal models: α_i = ln(SUM(Q_{i,j} e^(-x_j Δt)) / P(0, t_{i+1})) / Δt
    - Q_{i+1,k} = SUM(Q_{i,j} p_{j→k} e^(-r_{i,j} Δt)), Q_{0,0} = 1

    The models are:

    - Hull-White (1990): dr = [θ(t) - a r] dt + σ dW, normal rates that revert to a curve
    - Ho-Lee (1986): Hull-White without mean reversion (a = 0)
    - Black-Karasinski (1991): d ln r = [θ(t) - a ln r] dt + σ dW, lognormal rates
    - Black-Derman-Toy (1990): d ln r = θ(t) dt + σ dW, the lognormal model with a constant
      volatility, so without mean reversion (a = 0)

    The branching matches the mean and variance of x over every step exactly, and |η| is at
    most half a node spacing, so the probabilities stay between 1/24 and 2/3 and the mean
    reversion bounds the width of the tree (the branching bends towards the centre where
    Hull and White's j_max ≈ 0.184 / (a Δt) would). The tree reproduces P(0, t) at every
    date of the tree to machine precision. The rates of the tree are continuously
    compounded over their step; the zero curve is annually compounded.

    For more information, see the following sources:

    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Hull, J. and White, A. (1996). "Using Hull-White Interest Rate Trees." Journal of
      Derivatives, 3(3), 26-36.
    - Black, F., Derman, E. and Toy, W. (1990). "A One-Factor Model of Interest Rates and Its
      Application to Treasury Bond Options." Financial Analysts Journal, 46(1), 33-39.
    - Black, F. and Karasinski, P. (1991). "Bond and Option Pricing when Short Rates are
      Lognormal." Financial Analysts Journal, 47(4), 52-59.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, sections 3.3.3 and 3.4.

    Shares similar logic:
    - QuantLib, HullWhite: https://www.quantlib.org/reference/class_quant_lib_1_1_hull_white.html
    - QuantLib, BlackKarasinski: https://www.quantlib.org/reference/class_quant_lib_1_1_black_karasinski.html
    - FinancePy, HWTree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/hw_tree.py#L903
    - FinancePy, BKTree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bk_tree.py#L957
    - FinancePy, BDTTree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bdt_tree.py#L675

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years, interpolated linearly.
        event_times (float | list | np.ndarray): The payment and exercise dates in years; the
            tree runs to the last of them.
        speed (float): The speed of mean reversion a; not used by Ho-Lee and
            Black-Derman-Toy.
        volatility (float): The volatility σ: of the rate for Hull-White and Ho-Lee (0.01 is
            1% a year), of its logarithm for Black-Karasinski and Black-Derman-Toy (0.2 is
            20% of the rate).
        model (str, optional): The short-rate model. Options are: "hull_white" (Hull-White),
            "black_karasinski" (Black-Karasinski), "black_derman_toy" (Black-Derman-Toy),
            "ho_lee" (Ho-Lee). Defaults to "hull_white".
        steps_per_year (int, optional): The steps per year between the events. Defaults to
            50.
        curve_shift (float, optional): A parallel shift of the continuously compounded zero
            curve, for effective durations. Defaults to 0.

    Returns:
        dict: The Times of the tree, and per step the Nodes (the j of x_j = j Δx), the Rates,
        the Branches (the position in the next step of the middle branch of every node) and
        the Probabilities (down, middle, up), and the Arrow-Debreu prices per date.
    """
    speed = _check_tree_inputs(model, speed, volatility)
    times = _time_grid(np.atleast_1d(event_times), steps_per_year)
    discount = discount_factors(zero_curve, times) * np.exp(-curve_shift * times)
    lognormal = model in LOGNORMAL_MODELS

    nodes, states, arrow = np.zeros(1, dtype=int), np.zeros(1), np.ones(1)
    tree: dict = {
        "Times": times,
        "Nodes": [],
        "Rates": [],
        "Branches": [],
        "Probabilities": [],
        "Arrow-Debreu": [arrow],
    }
    for step, dt in enumerate(np.diff(times)):
        if speed > 0:
            expected = states * np.exp(-speed * dt)
            variance = volatility**2 * -np.expm1(-2 * speed * dt) / (2 * speed)
        else:
            expected = states
            variance = volatility**2 * dt
        spacing = np.sqrt(3 * variance)
        middle = np.floor(expected / spacing + 0.5).astype(int)
        eta = expected - middle * spacing
        outer = (variance + eta**2) / (2 * spacing**2)
        drift = eta / (2 * spacing)
        probabilities = np.column_stack([outer - drift, 1 - 2 * outer, outer + drift])

        if lognormal:
            alpha = _fit_lognormal_shift(arrow, states, dt, discount[step + 1])
            rates = np.exp(alpha + states)
        else:
            alpha = (
                np.log(np.sum(arrow * np.exp(-states * dt)) / discount[step + 1]) / dt
            )
            rates = alpha + states

        lowest = middle.min() - 1
        next_nodes = np.arange(lowest, middle.max() + 2)
        branches = middle - lowest
        discounted = arrow * np.exp(-rates * dt)
        arrow = sum(
            np.bincount(
                branches + offset,
                discounted * probabilities[:, offset + 1],
                minlength=next_nodes.size,
            )
            for offset in (-1, 0, 1)
        )

        tree["Nodes"].append(nodes)
        tree["Rates"].append(rates)
        tree["Branches"].append(branches)
        tree["Probabilities"].append(probabilities)
        tree["Arrow-Debreu"].append(arrow)
        nodes, states = next_nodes, next_nodes * spacing

    return tree


def _roll_back(
    tree: dict, step: int, values: np.ndarray, spread: float = 0.0
) -> np.ndarray:
    """The value at the nodes of a step of what is worth `values` at the next step."""
    branches = tree["Branches"][step]
    probabilities = tree["Probabilities"][step]
    dt = tree["Times"][step + 1] - tree["Times"][step]
    expected = (
        probabilities[:, 0] * values[branches - 1]
        + probabilities[:, 1] * values[branches]
        + probabilities[:, 2] * values[branches + 1]
    )
    return expected * np.exp(-(tree["Rates"][step] + spread) * dt)


def get_short_rate_tree(
    zero_curve: pd.Series,
    maturity: float,
    speed: float,
    volatility: float,
    model: str = "hull_white",
    steps_per_year: int = 4,
) -> pd.DataFrame:
    """
    Calculate the short rates of a trinomial tree fitted exactly to today's zero curve: the
    rates the short rate can take by date of the tree (rows) and node (columns, the node j
    of x_j = j Δx, with 0 the centre), see `build_short_rate_tree` for the construction.

    The rate of a node applies from its date to the next date of the tree.

    The formula is as follows:

    - Hull-White and Ho-Lee: r_{i,j} = α_i + x_j
    - Black-Karasinski and Black-Derman-Toy: r_{i,j} = e^(α_i + x_j)
    - α_i solves SUM(Q_{i,j} e^(-r_{i,j} Δt)) = P(0, t_{i+1})

    with x the mean-reverting tree of `build_short_rate_tree` and Q the Arrow-Debreu prices.
    The rates are continuously compounded over their step; the zero curve is annually
    compounded.

    For more information, see the following sources:

    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 32.

    Shares similar logic:
    - QuantLib, HullWhite: https://www.quantlib.org/reference/class_quant_lib_1_1_hull_white.html
    - QuantLib, BlackKarasinski: https://www.quantlib.org/reference/class_quant_lib_1_1_black_karasinski.html

    Args:
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years.
        maturity (float): The years the tree spans.
        speed (float): The speed of mean reversion a; not used by Ho-Lee and
            Black-Derman-Toy.
        volatility (float): The volatility σ: of the rate for Hull-White and Ho-Lee, of its
            logarithm for Black-Karasinski and Black-Derman-Toy.
        model (str, optional): The short-rate model. Options are: "hull_white" (Hull-White),
            "black_karasinski" (Black-Karasinski), "black_derman_toy" (Black-Derman-Toy),
            "ho_lee" (Ho-Lee). Defaults to "hull_white".
        steps_per_year (int, optional): The steps per year. Defaults to 4.

    Returns:
        pd.DataFrame: The short rate per date of the tree (rows, in years) and node (columns).

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import trinomial_tree_model

    zero_curve = pd.Series([0.04, 0.038, 0.037, 0.039, 0.042], index=[0.25, 1, 2, 5, 10])

    trinomial_tree_model.get_short_rate_tree(
        zero_curve, maturity=1, speed=0.1, volatility=0.01
    ).round(4)
    ```

    Which returns:

    |   Time |       -3 |       -2 |       -1 |      0 |        1 |        2 |        3 |
    |-------:|---------:|---------:|---------:|-------:|---------:|---------:|---------:|
    |   0    | nan      | nan      | nan      | 0.0392 | nan      | nan      | nan      |
    |   0.25 | nan      | nan      |   0.0294 | 0.0379 |   0.0465 | nan      | nan      |
    |   0.5  | nan      |   0.0196 |   0.0281 | 0.0367 |   0.0452 |   0.0538 | nan      |
    |   0.75 |   0.0097 |   0.0183 |   0.0268 | 0.0354 |   0.0439 |   0.0525 |   0.0611 |
    """
    tree = build_short_rate_tree(
        zero_curve, maturity, speed, volatility, model, steps_per_year
    )
    columns = np.arange(tree["Nodes"][-1].min(), tree["Nodes"][-1].max() + 1)
    rates = pd.DataFrame(
        np.nan,
        index=pd.Index(tree["Times"][:-1], name="Time"),
        columns=pd.Index(columns, name="Node"),
    )
    for time, nodes, step_rates in zip(
        tree["Times"][:-1], tree["Nodes"], tree["Rates"], strict=True
    ):
        rates.loc[time, nodes] = step_rates
    return rates


def _coupon_schedule(
    years_to_maturity: float, frequency: int
) -> tuple[np.ndarray, float]:
    """
    The coupon dates in years and the fraction of the current coupon period that has
    passed, with the schedule running back from maturity in whole periods.
    """
    if years_to_maturity <= 0:
        raise ValueError("The bond must have at least one coupon left.")
    periods = get_cash_flow_periods(years_to_maturity, frequency)
    return periods / frequency, 1 - periods[0]


def price_callable_bond(
    tree: dict,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 2,
    face_value: float = 100.0,
    call_price: float | None = None,
    years_to_first_call: float = 0.0,
    put_price: float | None = None,
    years_to_first_put: float = 0.0,
    spread: float = 0.0,
) -> float:
    """
    The value of a bond on a short-rate tree of `build_short_rate_tree`, discounting at the
    tree rate plus a spread: the value of all its remaining payments, so including the
    interest accrued since the last coupon (the dirty price). A callable bond is redeemed by
    the issuer whenever its value exceeds the call price on a coupon date from the first call
    date on, so its value at those nodes is capped at the call price. A puttable bond is sold
    back by the investor whenever its value falls below the put price on a coupon date from
    the first put date on, so its value at those nodes is at least the put price:

        - Value = min(Value, Call Price), then max(Value, Put Price), before the coupon

    The call and the put are exercised at a clean price on the coupon dates before maturity
    (Bermudan); the tree must have every coupon date among its dates.

    Args:
        tree (dict): The tree of `build_short_rate_tree`.
        coupon_rate (float): The annual coupon rate.
        years_to_maturity (float): The years to maturity.
        frequency (int, optional): The coupons per year. Defaults to 2.
        face_value (float, optional): The face value. Defaults to 100.
        call_price (float | None, optional): The call price; None for a bond without a call.
            Defaults to None.
        years_to_first_call (float, optional): The years until the bond can first be called.
            Defaults to 0.
        put_price (float | None, optional): The put price; None for a bond without a put.
            Defaults to None.
        years_to_first_put (float, optional): The years until the bond can first be put.
            Defaults to 0.
        spread (float, optional): The spread added to every rate (the OAS), continuously
            compounded. Defaults to 0.

    Returns:
        float: The dirty price.
    """
    coupon_times, _ = _coupon_schedule(years_to_maturity, frequency)
    coupon = face_value * coupon_rate / frequency
    times = tree["Times"]
    coupon_steps = _steps_at(times, coupon_times)
    if (
        coupon_steps.max() >= times.size
        or np.max(np.abs(times[coupon_steps] - coupon_times)) > TIME_TOLERANCE
    ):
        raise ValueError("The tree must have every coupon date among its dates.")
    maturity_step = int(coupon_steps[-1])
    coupon_steps = set(coupon_steps[:-1].tolist())
    value = np.full(tree["Arrow-Debreu"][maturity_step].size, face_value + coupon)
    for step in range(maturity_step - 1, -1, -1):
        value = _roll_back(tree, step, value, spread)
        if step in coupon_steps and step > 0:
            time = times[step]
            if call_price is not None and time >= years_to_first_call - TIME_TOLERANCE:
                value = np.minimum(value, call_price)
            if put_price is not None and time >= years_to_first_put - TIME_TOLERANCE:
                value = np.maximum(value, put_price)
            value = value + coupon
    return float(value[0])


def get_callable_bond_price(
    coupon_rate: float,
    years_to_maturity: float,
    zero_curve: pd.Series,
    speed: float,
    volatility: float,
    model: str = "hull_white",
    call_price: float | None = 100.0,
    years_to_first_call: float = 0.0,
    put_price: float | None = None,
    years_to_first_put: float = 0.0,
    frequency: int = 2,
    face_value: float = 100.0,
    spread: float = 0.0,
    steps_per_year: int = 50,
) -> dict[str, float]:
    """
    Calculate the price of a callable or puttable fixed-coupon bond on a short-rate tree
    fitted to today's zero curve (`build_short_rate_tree`): what the bond is worth when the
    issuer calls it whenever its value exceeds the call price and the investor puts it
    whenever its value falls below the put price, on the coupon dates from their first date
    on (Bermudan exercise at a clean price).

    The value of the embedded options is the straight bond less the callable or puttable
    bond on the same tree, so a put makes it negative.

    The formula is as follows:

    - At maturity: Value = Face Value + Coupon
    - At every earlier date: Value = E[Value next date] · e^(-(r + s) Δt), the expectation
      over the three branches of the tree
    - On a coupon date: Value = min(Value, Call Price), then max(Value, Put Price), then
      Value + Coupon
    - Price = Value today - Accrued Interest, Option Value = Straight Bond Price - Price

    with r the short rate of the node, s the spread and the coupon dates running back from
    maturity in whole periods. The tree has every coupon date among its dates and at most
    1 / steps_per_year years between two. The rates of the tree and the spread are
    continuously compounded, the zero curve annually compounded.

    For more information, see the following sources:

    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Black, F. and Karasinski, P. (1991). "Bond and Option Pricing when Short Rates are
      Lognormal." Financial Analysts Journal, 47(4), 52-59.
    - Fabozzi, F.J. (2016). "Bond Markets, Analysis, and Strategies." Pearson, 9th edition,
      chapter 18.

    Referenced against:
    - QuantLib, CallableFixedRateBond: https://www.quantlib.org/reference/class_quant_lib_1_1_callable_fixed_rate_bond.html
    - QuantLib, TreeCallableFixedRateBondEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_tree_callable_fixed_rate_bond_engine.html
    - FinancePy, BDTTree.callable_puttable_bond_tree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bdt_tree.py#L831

    Shares similar logic:
    - FinancePy, HWTree.callable_puttable_bond_tree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/hw_tree.py#L1366
    - FinancePy, BKTree.callable_puttable_bond_tree: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bk_tree.py#L1167

    Args:
        coupon_rate (float): The annual coupon rate.
        years_to_maturity (float): The years to maturity.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years.
        speed (float): The speed of mean reversion a; not used by Ho-Lee and
            Black-Derman-Toy.
        volatility (float): The volatility σ: of the rate for Hull-White and Ho-Lee (0.01 is
            1% a year), of its logarithm for Black-Karasinski and Black-Derman-Toy.
        model (str, optional): The short-rate model. Options are: "hull_white" (Hull-White),
            "black_karasinski" (Black-Karasinski), "black_derman_toy" (Black-Derman-Toy),
            "ho_lee" (Ho-Lee). Defaults to "hull_white".
        call_price (float | None, optional): The call price; None for a bond without a call.
            Defaults to 100.
        years_to_first_call (float, optional): The years until the bond can first be called.
            Defaults to 0.
        put_price (float | None, optional): The put price; None for a bond without a put.
            Defaults to None.
        years_to_first_put (float, optional): The years until the bond can first be put.
            Defaults to 0.
        frequency (int, optional): The coupons per year. Defaults to 2.
        face_value (float, optional): The face value. Defaults to 100.
        spread (float, optional): The spread over the tree rates (an OAS), continuously
            compounded. Defaults to 0.
        steps_per_year (int, optional): The most steps per year. Defaults to 50.

    Returns:
        dict[str, float]: The (clean) Price, the Straight Bond Price, the Option Value and the
        Accrued Interest.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import trinomial_tree_model

    zero_curve = pd.Series([0.036, 0.035, 0.037, 0.041, 0.047], index=[1, 2, 5, 10, 30])

    trinomial_tree_model.get_callable_bond_price(
        coupon_rate=0.05, years_to_maturity=10, zero_curve=zero_curve, speed=0.05,
        volatility=0.01, years_to_first_call=2
    )
    ```

    Which returns:

    ```
    {'Price': 101.23204068007776,
     'Straight Bond Price': 108.18151154789581,
     'Option Value': 6.94947086781805,
     'Accrued Interest': 0.0}
    ```
    """
    coupon_times, accrued_fraction = _coupon_schedule(years_to_maturity, frequency)
    tree = build_short_rate_tree(
        zero_curve, coupon_times, speed, volatility, model, steps_per_year
    )
    accrued = float(face_value * coupon_rate / frequency * accrued_fraction)
    bond = (coupon_rate, years_to_maturity, frequency, face_value)
    price = (
        price_callable_bond(
            tree,
            *bond,
            call_price,
            years_to_first_call,
            put_price,
            years_to_first_put,
            spread,
        )
        - accrued
    )
    straight = price_callable_bond(tree, *bond, spread=spread) - accrued
    return {
        "Price": price,
        "Straight Bond Price": straight,
        "Option Value": straight - price,
        "Accrued Interest": accrued,
    }


def get_bermudan_swaption(
    option_expiry: float,
    tenor: float,
    fixed_rate: float,
    zero_curve: pd.Series,
    speed: float,
    volatility: float,
    model: str = "hull_white",
    swaption_type: str = "payer",
    fixed_frequency: int = 1,
    notional: float = 1.0,
    exercise: str = "bermudan",
    steps_per_year: int = 50,
) -> float:
    """
    Calculate the price of a Bermudan swaption on a short-rate tree fitted to today's zero
    curve (`build_short_rate_tree`): what the right to enter, on the first exercise date or
    on any fixed payment date after it, the remainder of a swap that ends on a fixed date is
    worth (a co-terminal swaption, "10 non-call 1" for a 1-year expiry and a 10-year tenor
    ends 11 years from now).

    At every exercise date the holder compares the swap's value with the value of waiting.
    With European exercise only the first date counts, which matches the closed form of
    `short_rate_model.get_swaption` on Hull-White.

    The formula is as follows:

    - c_i = K τ_i, plus 1 on the last date T_n; B = the value of the remaining c_i on the tree
    - Payer exercise value at T_k = 1 - B(T_k), receiver exercise value = B(T_k) - 1
    - Option = max(E[Option next date] · e^(-r Δt), exercise value) at T_0, T_1, ..., T_{n-1}
    - Swaption = Notional · Option today

    with T_0 the expiry, T_i the fixed payment dates every 1 / fixed_frequency years after
    it (a short first period when the tenor is not a whole number of periods), K the fixed
    rate and B(T_k) the value at T_k of the fixed payments after T_k. The floating leg is
    worth par on every exercise date (one curve both projects and discounts). The fixed
    rate is a nominal rate paid fixed_frequency times a year with no day count, the tree
    rates continuously compounded and the zero curve annually compounded.

    For more information, see the following sources:

    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Brigo, D. and Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
      Springer, 2nd edition, section 3.11.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 32.

    Referenced against:
    - QuantLib, TreeSwaptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_tree_swaption_engine.html

    Shares similar logic:
    - FinancePy, HWTree.bermudan_swaption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/hw_tree.py#L1225
    - FinancePy, BKTree.bermudan_swaption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bk_tree.py#L1102
    - FinancePy, BDTTree.bermudan_swaption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bdt_tree.py#L787

    Args:
        option_expiry (float): The first exercise date T_0, the start of the swap, in years.
        tenor (float): The length of the swap from T_0 in years.
        fixed_rate (float): The fixed rate K of the swap (decimal).
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years.
        speed (float): The speed of mean reversion a; not used by Ho-Lee and
            Black-Derman-Toy.
        volatility (float): The volatility σ: of the rate for Hull-White and Ho-Lee, of its
            logarithm for Black-Karasinski and Black-Derman-Toy.
        model (str, optional): The short-rate model. Options are: "hull_white" (Hull-White),
            "black_karasinski" (Black-Karasinski), "black_derman_toy" (Black-Derman-Toy),
            "ho_lee" (Ho-Lee). Defaults to "hull_white".
        swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay
            the fixed rate), "receiver" (the right to receive it). Defaults to "payer".
        fixed_frequency (int, optional): The fixed payments per year. Defaults to 1.
        notional (float, optional): The notional. Defaults to 1.
        exercise (str, optional): The exercise dates. Options are: "bermudan" (every fixed
            payment date from the expiry on), "european" (the expiry only). Defaults to
            "bermudan".
        steps_per_year (int, optional): The most steps per year. Defaults to 50.

    Returns:
        float: The price, in the currency of the notional.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import trinomial_tree_model

    zero_curve = pd.Series([0.036, 0.035, 0.037, 0.041, 0.047], index=[1, 2, 5, 10, 30])

    trinomial_tree_model.get_bermudan_swaption(
        option_expiry=1, tenor=10, fixed_rate=0.045, zero_curve=zero_curve, speed=0.05,
        volatility=0.01, notional=1_000_000
    )
    ```

    Which returns:

    ```
    43556.03632591781
    ```
    """
    payment_times = swap_payment_times(option_expiry, tenor, fixed_frequency)
    tree = build_short_rate_tree(
        zero_curve,
        np.concatenate([[option_expiry], payment_times]),
        speed,
        volatility,
        model,
        steps_per_year,
    )
    return notional * price_bermudan_swaption(
        tree, option_expiry, tenor, fixed_rate, swaption_type, fixed_frequency, exercise
    )


def swap_payment_times(
    option_expiry: float, tenor: float, fixed_frequency: int
) -> np.ndarray:
    """
    Calculate the fixed payment dates in years of a swap that starts at the expiry.

    Args:
        option_expiry (float): The expiry of the swaption in years, the start of the swap.
        tenor (float): The tenor of the swap in years.
        fixed_frequency (int): The fixed payments per year.

    Returns:
        np.ndarray: The fixed payment dates in years.
    """
    if option_expiry <= 0 or tenor <= 0:
        raise ValueError("The expiry and the tenor of the swaption must be positive.")
    return (
        option_expiry + get_cash_flow_periods(tenor, fixed_frequency) / fixed_frequency
    )


def price_bermudan_swaption(
    tree: dict,
    option_expiry: float,
    tenor: float,
    fixed_rate: float,
    swaption_type: str = "payer",
    fixed_frequency: int = 1,
    exercise: str = "bermudan",
) -> float:
    """
    Calculate the value per 1 of notional of a co-terminal Bermudan (or European) swaption
    on a short-rate tree of `build_short_rate_tree` that has the expiry and every fixed
    payment date among its dates, see `get_bermudan_swaption`.

    One tree can value several swaptions.

    Args:
        tree (dict): The tree of `build_short_rate_tree`.
        option_expiry (float): The first exercise date, the start of the swap, in years.
        tenor (float): The length of the swap from the expiry in years.
        fixed_rate (float): The fixed rate of the swap (decimal).
        swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay
            the fixed rate), "receiver" (the right to receive it). Defaults to "payer".
        fixed_frequency (int, optional): The fixed payments per year. Defaults to 1.
        exercise (str, optional): The exercise dates. Options are: "bermudan" (every fixed
            payment date from the expiry on), "european" (the expiry only). Defaults to
            "bermudan".

    Returns:
        float: The value per 1 of notional.
    """
    check_option("swaption_type", swaption_type, ("payer", "receiver"))
    check_option("exercise", exercise, ("bermudan", "european"))
    payment_times = swap_payment_times(option_expiry, tenor, fixed_frequency)
    cash_flows = fixed_rate * np.diff(payment_times, prepend=option_expiry)
    times = tree["Times"]
    event_times = np.concatenate([[option_expiry], payment_times])
    event_steps = _steps_at(times, event_times)
    if (
        event_steps.max() >= times.size
        or np.max(np.abs(times[event_steps] - event_times)) > TIME_TOLERANCE
    ):
        raise ValueError("The tree must have the expiry and every payment date.")
    payment_steps = dict(
        zip(event_steps[1:-1].tolist(), cash_flows[:-1].tolist(), strict=True)
    )
    expiry_step, last_step = int(event_steps[0]), int(event_steps[-1])
    exercise_steps = (
        {expiry_step, *payment_steps} if exercise == "bermudan" else {expiry_step}
    )
    sign = 1.0 if swaption_type == "payer" else -1.0

    bond = np.full(tree["Arrow-Debreu"][last_step].size, 1 + cash_flows[-1])
    option = np.zeros_like(bond)
    for step in range(last_step - 1, -1, -1):
        bond = _roll_back(tree, step, bond)
        option = _roll_back(tree, step, option)
        if step in exercise_steps:
            option = np.maximum(option, sign * (1 - bond))
        if step in payment_steps:
            bond = bond + payment_steps[step]
    return float(option[0])


def get_bond_option(
    option_expiry: float,
    strike: float,
    coupon_rate: float,
    years_to_maturity: float,
    zero_curve: pd.Series,
    speed: float,
    volatility: float,
    model: str = "hull_white",
    option_type: str = "call",
    exercise: str = "american",
    frequency: int = 2,
    face_value: float = 100.0,
    steps_per_year: int = 50,
) -> float:
    """
    Calculate the price of an American or European option on a fixed-coupon bond, struck at
    a clean price, on a short-rate tree fitted to today's zero curve
    (`build_short_rate_tree`): what the right to buy or sell the bond at the strike is
    worth.

    An American option can be exercised at any date of the tree up to its expiry (today
    included), a European one at its expiry only. The holder of the option does not receive
    the coupons paid before exercise.

    The formula is as follows:

    - B = the value on the tree of the coupons and the face value paid after a date
    - Clean Price = B - Accrued Interest, Accrued Interest = Coupon · (t - t_previous) · f
    - Call exercise value = Clean Price - Strike, put exercise value = Strike - Clean Price
    - Option = max(E[Option next date] · e^(-r Δt), exercise value) on the exercise dates

    with t_previous the last coupon date before t and f the coupons per year. On a coupon
    date B holds the payments after it, so the clean price is the price just after the
    coupon. The rates of the tree are continuously compounded, the zero curve annually
    compounded.

    For more information, see the following sources:

    - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure
      Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.
    - Black, F., Derman, E. and Toy, W. (1990). "A One-Factor Model of Interest Rates and Its
      Application to Treasury Bond Options." Financial Analysts Journal, 46(1), 33-39.
    - Hull, J.C. (2022). "Options, Futures, and Other Derivatives." Pearson, 11th edition,
      chapter 32.

    European options agree with Jamshidian's (1989) closed form on Hull-White.

    Referenced against:
    - FinancePy, HWTree.bond_option: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/hw_tree.py#L1273

    Shares similar logic:
    - FinancePy, BKTree.bond_option: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bk_tree.py#L1042
    - FinancePy, BDTTree.bond_option: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/bdt_tree.py#L743

    Args:
        option_expiry (float): The expiry of the option in years.
        strike (float): The strike, a clean price per face value.
        coupon_rate (float): The annual coupon rate of the bond.
        years_to_maturity (float): The years from today to the maturity of the bond, which
            must lie after the expiry.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals) by
            maturity in years.
        speed (float): The speed of mean reversion a; not used by Ho-Lee and
            Black-Derman-Toy.
        volatility (float): The volatility σ: of the rate for Hull-White and Ho-Lee, of its
            logarithm for Black-Karasinski and Black-Derman-Toy.
        model (str, optional): The short-rate model. Options are: "hull_white" (Hull-White),
            "black_karasinski" (Black-Karasinski), "black_derman_toy" (Black-Derman-Toy),
            "ho_lee" (Ho-Lee). Defaults to "hull_white".
        option_type (str, optional): The option. Options are: "call" (the right to buy the
            bond), "put" (the right to sell it). Defaults to "call".
        exercise (str, optional): The exercise. Options are: "american" (any date up to the
            expiry), "european" (the expiry only). Defaults to "american".
        frequency (int, optional): The coupons per year. Defaults to 2.
        face_value (float, optional): The face value. Defaults to 100.
        steps_per_year (int, optional): The most steps per year. Defaults to 50.

    Returns:
        float: The price of the option on one bond of the face value.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.short_rates import trinomial_tree_model

    zero_curve = pd.Series([0.036, 0.035, 0.037, 0.041, 0.047], index=[1, 2, 5, 10, 30])

    trinomial_tree_model.get_bond_option(
        option_expiry=2, strike=100, coupon_rate=0.04, years_to_maturity=10,
        zero_curve=zero_curve, speed=0.05, volatility=0.01, option_type="put"
    )
    ```

    Which returns:

    ```
    3.63606036239939
    ```
    """
    check_option("option_type", option_type, ("call", "put"))
    check_option("exercise", exercise, ("american", "european"))
    if not 0 < option_expiry < years_to_maturity:
        raise ValueError(
            "The option must expire after today and before the bond matures."
        )

    coupon_times, _ = _coupon_schedule(years_to_maturity, frequency)
    coupon = face_value * coupon_rate / frequency
    tree = build_short_rate_tree(
        zero_curve,
        np.concatenate([[option_expiry], coupon_times]),
        speed,
        volatility,
        model,
        steps_per_year,
    )
    times = tree["Times"]
    coupon_steps = set(_steps_at(times, coupon_times[:-1]).tolist())
    expiry_step = int(_steps_at(times, [option_expiry])[0])
    # The accrued interest at every date of the tree: the share of the coupon period since
    # the last coupon date (on a coupon date the coupon is paid, so nothing has accrued).
    previous = np.concatenate([[coupon_times[0] - 1 / frequency], coupon_times])
    last_coupon = previous[np.searchsorted(previous, times + TIME_TOLERANCE) - 1]
    accrued = coupon * (times - last_coupon) * frequency
    sign = 1.0 if option_type == "call" else -1.0

    bond = np.full(tree["Arrow-Debreu"][-1].size, face_value + coupon)
    option = np.zeros(tree["Arrow-Debreu"][expiry_step].size)
    for step in range(len(tree["Rates"]) - 1, -1, -1):
        bond = _roll_back(tree, step, bond)
        if step < expiry_step:
            option = _roll_back(tree, step, option)
        if step == expiry_step or (step < expiry_step and exercise == "american"):
            option = np.maximum(option, sign * (bond - accrued[step] - strike))
        if step in coupon_steps:
            bond = bond + coupon
    return float(option[0])
