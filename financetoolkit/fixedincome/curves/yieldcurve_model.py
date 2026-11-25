"""Yield Curve Model Module"""

import numbers

import numpy as np
import pandas as pd

from financetoolkit.fixedincome.bonds import bond_model


def get_forward_rate(
    near_rate: float | pd.Series,
    far_rate: float | pd.Series,
    near_maturity: float | pd.Series,
    far_maturity: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the implied forward rate between two points on a yield curve: the rate for
    borrowing between the two maturities that today's curve locks in.

    The forward rate is the interest rate, implied by the current (spot) yield curve,
    for a loan that starts at a future date. It answers the question "given what the
    market is pricing today for a short-dated and a long-dated zero-coupon bond, what
    rate must apply to the period between those two maturities so that investing at
    the near maturity and rolling over at the forward rate yields the same result as
    investing directly at the far maturity?". It is derived purely from no-arbitrage
    pricing and does not require any forecast of future rates.

    Also known as: implied forward rate, forward-forward rate.

    The formula is as follows:

        Forward Rate = ((1 + far_rate)^far_maturity / (1 + near_rate)^near_maturity)
        ^(1 / (far_maturity - near_maturity)) - 1

    Both the spot rates supplied and the forward rate returned are treated as **effective
    annual** rates, i.e. compounded once per year, which is why the maturities appear
    directly as exponents. Note that this is a different convention from
    `get_par_yield` and `bond_model.get_bond_price_from_curve`, which read the same kind
    of spot curve as nominal annual rates compounded `frequency` times per year. The two
    coincide for annual compounding; at any other frequency a curve should not be passed to both
    without first restating it, since a 4% nominal semi-annual rate is a 4.04% effective
    annual one.

    For more information, see: https://en.wikipedia.org/wiki/Forward_rate

    Args:
        near_rate (float | pd.Series): The zero/spot rate for the nearer maturity (in decimal).
        far_rate (float | pd.Series): The zero/spot rate for the further maturity (in decimal).
        near_maturity (float | pd.Series): The nearer maturity, in years.
        far_maturity (float | pd.Series): The further maturity, in years.

    Returns:
        float | pd.Series: The implied forward rate between the near and far maturities (in decimal).

    Raises:
        TypeError: If any of the inputs is not a float, int or pandas Series.
        ValueError: If the far maturity is not greater than the near maturity, for any
            element when they are Series.
    """
    for value in (near_rate, far_rate, near_maturity, far_maturity):
        if not isinstance(value, int | float | pd.Series):
            raise TypeError(
                "Expected near_rate, far_rate, near_maturity and far_maturity to be "
                f"a float, int or pandas Series, received {type(value)} instead."
            )

    # A pair of a Series checks every element, so no row turns into a forward rate over a
    # negative or zero period.
    far, near = (
        far_maturity.align(near_maturity)
        if isinstance(far_maturity, pd.Series) and isinstance(near_maturity, pd.Series)
        else (far_maturity, near_maturity)
    )
    if np.any(np.asarray(far <= near)):
        raise ValueError(
            "The far maturity must be greater than the near maturity to derive "
            "a forward rate."
        )

    compounded_far = (1 + far_rate) ** far_maturity
    compounded_near = (1 + near_rate) ** near_maturity
    period = far_maturity - near_maturity

    forward_rate = (compounded_far / compounded_near) ** (1 / period) - 1

    return forward_rate


def get_par_yield(
    spot_rates: pd.Series,
    years_to_maturity: float,
    frequency: int = 1,
    par_value: float = 100,
) -> float:
    """
    Calculate the par yield for a given maturity given a zero-coupon (spot) yield curve: the
    coupon rate at which a bond of that maturity prices at par on the curve.

    The par yield is the coupon rate that would need to be attached to a newly-issued
    bond of the given maturity so that, when its cash flows are discounted using the
    prevailing zero/spot curve, its price equals its par value exactly. It is used to
    construct the "par yield curve" that is often quoted for government bonds (e.g. the
    on-the-run Treasury curve), as opposed to the theoretical spot curve which is
    typically bootstrapped rather than directly observed.

    Also known as: par rate, par coupon rate.

    The formula is as follows:

        Par Yield = m * (1 - DF(t_n)) / (SUM_(i=1)^(n) [ DF(t_i) ] - a)

    where m = frequency, DF(t) = 1 / (1 + spot_rate(t / m) / m)^t is the discount factor
    for a cash flow received after t periods, spot_rate(τ) is obtained by linearly
    interpolating the provided spot curve at time τ (in years), n is the number of
    remaining coupons, t_i is the time of the i-th coupon in periods and a is the elapsed
    fraction of the current coupon period (see `bond_model.get_cash_flow_periods`), so
    t_n = years_to_maturity * m. Spot rates are treated as nominal annual rates
    compounded at `frequency`, consistent with `bond_model.get_bond_price_from_curve`.

    For a maturity that is a whole number of periods a = 0 and t_i = i, the textbook
    formula. Otherwise the coupon is the one at which the clean (quoted) price, the
    dirty price minus the accrued coupon, equals par, the same price `bond_model.get_z_spread`
    compares against.

    For more information, see: https://en.wikipedia.org/wiki/Par_yield

    Args:
        spot_rates (pd.Series): The zero-coupon (spot) yield curve, indexed by maturity
            in years (in decimal). The rates for maturities that fall in between two
            index values are linearly interpolated.
        years_to_maturity (float): The maturity, in years, of the bond to derive the par yield for.
        frequency (int): The number of coupon payments per year. Defaults to 1.
        par_value (float): The face value of the bond. Defaults to 100. Note that the
            par value cancels out algebraically and does not affect the result — it is
            kept as an argument for consistency with the other bond functions.

    Returns:
        float: The par yield for the given maturity (in decimal).

    Raises:
        TypeError: If spot_rates is not a pandas Series or years_to_maturity, frequency
            or par_value are not a float or int.
        ValueError: If years_to_maturity or frequency are not positive.
    """
    if not isinstance(spot_rates, pd.Series):
        raise TypeError(
            f"Expected spot_rates to be a pandas Series, received {type(spot_rates)} instead."
        )
    if not isinstance(years_to_maturity, int | float):
        raise TypeError(
            f"Expected years_to_maturity to be a float or int, received {type(years_to_maturity)} instead."
        )
    if not isinstance(frequency, numbers.Integral):
        raise TypeError(
            f"Expected frequency to be an int, received {type(frequency)} instead."
        )
    if not isinstance(par_value, int | float):
        raise TypeError(
            f"Expected par_value to be a float or int, received {type(par_value)} instead."
        )
    if years_to_maturity <= 0:
        raise ValueError("years_to_maturity must be greater than 0.")
    if frequency <= 0:
        raise ValueError("frequency must be greater than 0.")

    spot_rates_sorted = spot_rates.sort_index()

    # The same cash flow times as the bond pricing, so a bond with this coupon prices at par on the same curve.  # noqa: E501
    periods = bond_model.get_cash_flow_periods(years_to_maturity, frequency)
    accrued_fraction = bond_model.get_accrued_fraction(years_to_maturity, frequency)

    # Nominal annual rates compounded at `frequency`, so discount per (fractional) period.
    period_rates = np.interp(
        periods / frequency,
        spot_rates_sorted.index.to_numpy(dtype=float),
        spot_rates_sorted.to_numpy(dtype=float),
    )
    discount_factors = (1 + period_rates / frequency) ** -periods

    # The accrued part of the next coupon is paid by the buyer on top of par, so it is taken off the coupon annuity.  # noqa: E501
    par_yield = (
        frequency
        * (1 - discount_factors[-1])
        / (discount_factors.sum() - accrued_fraction)
    )

    return float(par_yield)


def get_carry_and_roll_down(
    spot_rates: pd.Series,
    maturities: list | np.ndarray | float,
    horizon: float = 0.25,
    short_rate: float | None = None,
) -> pd.DataFrame:
    """
    Calculate the carry and roll-down of zero-coupon bonds on a yield curve: the return in
    excess of financing that a bond earns over a horizon if the curve does not change.

    A bond held over the horizon earns its own yield (the carry over the financing rate)
    and, as it ages, is priced at the yield of a shorter maturity. On an upward sloping
    curve that yield is lower, so the price rises (the roll-down). Koijen, Moskowitz,
    Pedersen and Vrugt (2018) define the carry of any asset as its return when market
    conditions, here the yield curve, stay the same; for a bond it is the yield spread over
    the short rate plus the roll-down. The bonds are zero-coupon bonds, so each one is
    repriced exactly on the unchanged curve rather than approximated with its duration.

    The formula is as follows:

        Carry = (1 + z(T))^h - (1 + r)^h
        Roll-Down = (1 + z(T))^T / (1 + z(T - h))^(T - h) - (1 + z(T))^h
        Total = Carry + Roll-Down = P(T - h) / P(T) - (1 + r)^h

    where z(m) is the zero-coupon rate of maturity m, annually compounded and interpolated
    linearly between the maturities of `spot_rates` (flat beyond the shortest and the
    longest), P(m) = (1 + z(m))^(-m) the price of a zero-coupon bond paying 1 at m, T the
    maturity, h the horizon and r the financing rate over the horizon, annually compounded:
    `short_rate`, or by default the zero-coupon rate of the horizon, z(h), so that a bond
    maturing at the horizon has no carry and no roll-down. The carry is what the bond earns
    over the financing if its yield stays z(T); the roll-down is the price change when its
    yield moves from z(T) to z(T - h), the yield the unchanged curve has at its remaining
    maturity, valued at the horizon. Every measure is a return over the horizon, not
    annualized.

    To first order, Carry ≈ (z(T) - r) × h and Roll-Down ≈ -D × (z(T - h) - z(T)), with
    D = (T - h) / (1 + z(T)) the modified duration (annual compounding) of the bond at the
    horizon, the approximation of Koijen et al. (2018).

    For more information, see the following sources:

    - Koijen, R.S.J., Moskowitz, T.J., Pedersen, L.H. and Vrugt, E.B. (2018). "Carry."
      Journal of Financial Economics, 127(2), 197-225.
    - Ilmanen, A. (2011). "Expected Returns: An Investor's Guide to Harvesting Market
      Rewards." Wiley, chapter 9.

    Args:
        spot_rates (pd.Series): The zero-coupon (spot) yield curve as decimals, annually
            compounded, indexed by maturity in years.
        maturities (list | np.ndarray | float): The maturities of the bonds, in years.
        horizon (float, optional): The holding period, in years. Defaults to 0.25 (three
            months).
        short_rate (float | None, optional): The financing rate over the horizon, as an
            annually compounded decimal. Defaults to None, the zero-coupon rate of the
            horizon on the curve.

    Returns:
        pd.DataFrame: The Carry, Roll-Down and Total (decimals, returns over the horizon)
        by maturity in years. A maturity shorter than the horizon is NaN, as the bond is
        repaid before the horizon ends.

    Raises:
        TypeError: If spot_rates is not a pandas Series or horizon or short_rate are not
            a float or int.
        ValueError: If the horizon is not positive.
    """
    if not isinstance(spot_rates, pd.Series):
        raise TypeError(
            f"Expected spot_rates to be a pandas Series, received {type(spot_rates)} instead."
        )
    if not isinstance(horizon, int | float):
        raise TypeError(
            f"Expected horizon to be a float or int, received {type(horizon)} instead."
        )
    if short_rate is not None and not isinstance(short_rate, int | float):
        raise TypeError(
            f"Expected short_rate to be a float, int or None, received {type(short_rate)} instead."
        )
    if horizon <= 0:
        raise ValueError("The horizon must be greater than 0.")

    curve = spot_rates.dropna().sort_index()
    curve_maturities = curve.index.to_numpy(dtype=float)
    curve_rates = curve.to_numpy(dtype=float)
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))

    def zero_rate(maturity: np.ndarray | float) -> np.ndarray:
        return np.interp(maturity, curve_maturities, curve_rates)

    financing = (
        1 + (zero_rate(horizon) if short_rate is None else short_rate)
    ) ** horizon

    # A bond repaid before the horizon has no price at the horizon; at the horizon itself its
    # remaining maturity is zero, which the price (1 + z)^0 = 1 handles.
    rolled = np.maximum(maturities - horizon, 0.0)
    held = (1 + zero_rate(maturities)) ** horizon
    carry = held - financing
    roll_down = (1 + zero_rate(maturities)) ** maturities / (
        1 + zero_rate(rolled)
    ) ** rolled - held

    result = pd.DataFrame(
        {"Carry": carry, "Roll-Down": roll_down, "Total": carry + roll_down},
        index=pd.Index(maturities, name="Maturity"),
    )
    # A tolerance, so a label such as '3M' (3 / 12 years) counts as the horizon of 0.25 years.
    result.loc[maturities < horizon - 1e-9] = np.nan

    return result


def get_yield_curve_spread(
    long_yield: float | pd.Series, short_yield: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the spread between two points on the yield curve: how much more the longer
    maturity yields than the shorter one, the slope of the curve.

    The yield curve spread measures the difference in yield between a longer-dated and
    a shorter-dated instrument (e.g. the widely followed 10-year minus 2-year Treasury
    spread). A positive spread indicates a "normal" upward-sloping curve, while a
    negative spread ("inversion") has historically been used as a leading indicator of
    economic slowdown.

    Also known as: term spread, yield curve slope.

    The formula is as follows:

        Yield Curve Spread = Long-Term Yield - Short-Term Yield

    For more information, see: https://en.wikipedia.org/wiki/Yield_spread

    Args:
        long_yield (float | pd.Series): The yield of the longer-dated instrument (in decimal).
        short_yield (float | pd.Series): The yield of the shorter-dated instrument (in decimal).

    Returns:
        float | pd.Series: The yield curve spread (in decimal).

    Raises:
        TypeError: If long_yield or short_yield are not a float, int or pandas Series.
    """
    for value in (long_yield, short_yield):
        if not isinstance(value, int | float | pd.Series):
            raise TypeError(
                "Expected long_yield and short_yield to be a float, int or pandas "
                f"Series, received {type(value)} instead."
            )

    return long_yield - short_yield


def get_yield_curve_butterfly(
    short_yield: float | pd.Series,
    middle_yield: float | pd.Series,
    long_yield: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate the butterfly ("fly") of three points on the yield curve: twice how far the
    yield of the middle maturity (the body) lies above the average yield of the two wings, a
    measure of the curvature of the curve. Only when the body lies midway between the wings
    is that average the straight line through them; in 2s5s10s it is not, so a straight
    curve there has a butterfly of its own. A positive butterfly means the body yields more
    than the wings on average (it is cheap against them), a negative one that it yields
    less.

    Also known as: fly, curve butterfly, 2s5s10s, curvature.

    The formula is as follows:

        Butterfly = 2 * Middle Yield - Short Yield - Long Yield

    The wings are weighted equally, as butterflies are quoted (e.g. 2s5s10s); a duration- or
    regression-weighted butterfly trade has other weights.

    For more information, see Tuckman and Serrat (2022), Fixed Income Securities, 4th
    edition, chapter 6.

    Args:
        short_yield (float | pd.Series): The yield of the short wing (in decimal).
        middle_yield (float | pd.Series): The yield of the body (in decimal).
        long_yield (float | pd.Series): The yield of the long wing (in decimal).

    Returns:
        float | pd.Series: The butterfly (in decimal).

    Raises:
        TypeError: If a yield is not a float, int or pandas Series.
    """
    for value in (short_yield, middle_yield, long_yield):
        if not isinstance(value, int | float | pd.Series):
            raise TypeError(
                "Expected the yields to be a float, int or pandas Series, received "
                f"{type(value)} instead."
            )

    return 2 * middle_yield - short_yield - long_yield


def get_breakeven_inflation_rate(
    nominal_yield: float | pd.Series, real_yield: float | pd.Series
) -> float | pd.Series:
    """
    Calculate the breakeven inflation rate implied by a nominal and a real
    (inflation-protected) yield: the average inflation at which the two bonds return the
    same, the inflation the market expects.

    The breakeven inflation rate is the rate of inflation that would make an investor
    indifferent between holding a nominal bond and an inflation-protected bond (e.g. a
    U.S. Treasury Inflation-Protected Security, or TIPS) of the same maturity. It is
    widely used as a market-implied measure of expected inflation over the relevant
    horizon.

    Also known as: TIPS breakeven spread, inflation breakeven.

    The formula is as follows:

        Breakeven Inflation Rate = Nominal Yield - Real Yield

    For more information, see: https://en.wikipedia.org/wiki/Treasury_Inflation-Protected_Securities

    Args:
        nominal_yield (float | pd.Series): The yield of the nominal (non-inflation-protected)
            bond (in decimal).
        real_yield (float | pd.Series): The yield of the inflation-protected bond of the
            same maturity (in decimal).

    Returns:
        float | pd.Series: The breakeven inflation rate (in decimal).

    Raises:
        TypeError: If nominal_yield or real_yield are not a float, int or pandas Series.
    """
    for value in (nominal_yield, real_yield):
        if not isinstance(value, int | float | pd.Series):
            raise TypeError(
                "Expected nominal_yield and real_yield to be a float, int or pandas "
                f"Series, received {type(value)} instead."
            )

    return nominal_yield - real_yield
